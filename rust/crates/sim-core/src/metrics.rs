//! Per-request records and the summary row every simulator run emits.

use serde::Serialize;

/// What happened to one request.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct RequestRecord {
    pub id: u64,
    pub arrival_ms: f64,
    pub first_token_ms: Option<f64>,
    pub done_ms: Option<f64>,
    pub output_tokens: u32,
    pub rejected: bool,
}

impl RequestRecord {
    pub fn e2e_ms(&self) -> Option<f64> {
        self.done_ms.map(|d| d - self.arrival_ms)
    }

    pub fn ttft_ms(&self) -> Option<f64> {
        self.first_token_ms.map(|f| f - self.arrival_ms)
    }
}

/// Mean and tail of one metric, in ms.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Percentiles {
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

impl Percentiles {
    pub fn of(values: &mut [f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        values.sort_by(f64::total_cmp);
        let q = |p: f64| {
            // Linear interpolation, as numpy.percentile does by default.
            let x = p * (values.len() - 1) as f64;
            let (lo, hi) = (x.floor() as usize, x.ceil() as usize);
            values[lo] + (values[hi] - values[lo]) * (x - lo as f64)
        };
        Some(Self {
            mean: values.iter().sum::<f64>() / values.len() as f64,
            p50: q(0.50),
            p95: q(0.95),
            p99: q(0.99),
        })
    }
}

/// One JSONL row: the same fields from every runner.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Summary {
    pub requests: usize,
    pub completed: usize,
    pub rejected: usize,
    /// From the first arrival to the last completion.
    pub duration_s: f64,
    pub throughput_rps: f64,
    pub output_tokens_per_s: f64,
    pub ttft_ms: Option<Percentiles>,
    /// Mean time per output token after the first, per request.
    pub tpot_ms: Option<Percentiles>,
    /// Mean gap between consecutive tokens over all gaps of all requests,
    /// as vLLM's benchmark reports ITL.
    pub itl_mean_ms: Option<f64>,
    pub e2e_ms: Option<Percentiles>,
    /// Share of completed requests whose end-to-end time exceeded the SLO.
    pub slo_miss_rate: Option<f64>,
}

impl Summary {
    pub fn from_records(records: &[RequestRecord], slo_ms: Option<f64>) -> Self {
        let done: Vec<&RequestRecord> = records.iter().filter(|r| r.done_ms.is_some()).collect();
        let start = records
            .iter()
            .map(|r| r.arrival_ms)
            .fold(f64::INFINITY, f64::min);
        let end = done
            .iter()
            .filter_map(|r| r.done_ms)
            .fold(f64::NEG_INFINITY, f64::max);
        let duration_s = if done.is_empty() {
            0.0
        } else {
            (end - start) / 1e3
        };
        let out_tokens: u64 = done.iter().map(|r| r.output_tokens as u64).sum();
        let per_s = |x: f64| {
            if duration_s > 0.0 {
                x / duration_s
            } else {
                0.0
            }
        };

        let mut ttft: Vec<f64> = done.iter().filter_map(|r| r.ttft_ms()).collect();
        let mut e2e: Vec<f64> = done.iter().filter_map(|r| r.e2e_ms()).collect();
        let mut tpot = Vec::new();
        let (mut gap_ms, mut gaps) = (0.0, 0u64);
        for r in &done {
            if let (Some(f), Some(d)) = (r.first_token_ms, r.done_ms) {
                if r.output_tokens > 1 {
                    let n = (r.output_tokens - 1) as f64;
                    tpot.push((d - f) / n);
                    gap_ms += d - f;
                    gaps += (r.output_tokens - 1) as u64;
                }
            }
        }
        Self {
            requests: records.len(),
            completed: done.len(),
            rejected: records.iter().filter(|r| r.rejected).count(),
            duration_s,
            throughput_rps: per_s(done.len() as f64),
            output_tokens_per_s: per_s(out_tokens as f64),
            ttft_ms: Percentiles::of(&mut ttft),
            tpot_ms: Percentiles::of(&mut tpot),
            itl_mean_ms: (gaps > 0).then(|| gap_ms / gaps as f64),
            e2e_ms: Percentiles::of(&mut e2e),
            slo_miss_rate: slo_ms
                .filter(|_| !e2e.is_empty())
                .map(|slo| e2e.iter().filter(|&&x| x > slo).count() as f64 / e2e.len() as f64),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_interpolate_like_numpy() {
        let mut v: Vec<f64> = (1..=100).map(f64::from).collect();
        let p = Percentiles::of(&mut v).unwrap();
        assert!((p.p50 - 50.5).abs() < 1e-9);
        assert!((p.p99 - 99.01).abs() < 1e-9);
        assert!((p.mean - 50.5).abs() < 1e-9);
    }
}
