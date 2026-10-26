//! Request streams: synthetic arrival processes and length distributions,
//! plus replay of the Azure LLM inference trace format.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Exp, LogNormal};
use serde::{Deserialize, Serialize};

use crate::engine::{Request, Work};
use crate::Error;

/// One request and when it arrives.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Arrival {
    pub t_ms: f64,
    pub req: Request,
}

/// A distribution of positive integers (token counts, step counts).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "dist", rename_all = "snake_case")]
pub enum LengthDist {
    Fixed {
        value: u32,
    },
    Uniform {
        min: u32,
        max: u32,
    },
    /// Lognormal with the given mean and coefficient of variation,
    /// clipped to [min, max].
    LogNormal {
        mean: f64,
        cv: f64,
        min: u32,
        max: u32,
    },
}

impl LengthDist {
    fn sample(&self, rng: &mut ChaCha8Rng) -> u32 {
        match *self {
            LengthDist::Fixed { value } => value,
            LengthDist::Uniform { min, max } => rng.gen_range(min..=max.max(min)),
            LengthDist::LogNormal { mean, cv, min, max } => {
                let s2 = (1.0 + cv * cv).ln();
                let d = LogNormal::new(mean.max(1e-9).ln() - s2 / 2.0, s2.sqrt())
                    .expect("valid lognormal");
                (d.sample(rng).round() as u32).clamp(min, max.max(min))
            }
        }
    }
}

/// When requests arrive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "process", rename_all = "snake_case")]
pub enum ArrivalProcess {
    /// Everything at t = 0 (a benchmark's "QPS inf").
    Burst,
    /// Constant-rate Poisson.
    Poisson { rate_per_s: f64 },
    /// Piecewise-constant Poisson: `(start_s, rate_per_s)` pairs, sorted.
    Steps { steps: Vec<(f64, f64)> },
    /// Two-state Markov-modulated Poisson: bursts at `high`, calm at `low`,
    /// exponential dwell with the given mean in each state.
    Mmpp {
        low: f64,
        high: f64,
        mean_dwell_s: f64,
    },
    /// Sinusoidal day curve: mean * (1 + amplitude * sin(2 pi t / period)).
    Diurnal {
        mean_rate_per_s: f64,
        amplitude: f64,
        period_s: f64,
    },
}

/// What each request asks for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkShape {
    /// `max_total` caps prompt + output, as benchmark dataset filters do.
    Llm {
        prompt: LengthDist,
        output: LengthDist,
        max_total: Option<u32>,
    },
    Image {
        steps: LengthDist,
        width: u32,
        height: u32,
    },
    Generic,
}

impl WorkShape {
    /// Lengths shaped like the ShareGPT sample vLLM's benchmarks use:
    /// mean 202 prompt and 179 output tokens (vLLM v0.6.0 blog), heavy
    /// tailed, prompts at most 1024 and totals at most 2048 tokens, outputs
    /// at least 4 tokens (the benchmark's filters).
    pub fn sharegpt_like() -> Self {
        WorkShape::Llm {
            prompt: LengthDist::LogNormal {
                mean: 202.0,
                cv: 1.1,
                min: 4,
                max: 1024,
            },
            output: LengthDist::LogNormal {
                mean: 179.0,
                cv: 1.0,
                min: 4,
                max: 2044,
            },
            max_total: Some(2048),
        }
    }

    fn sample(&self, rng: &mut ChaCha8Rng) -> Work {
        match self {
            WorkShape::Llm {
                prompt,
                output,
                max_total,
            } => {
                let p = prompt.sample(rng);
                let mut o = output.sample(rng);
                if let Some(cap) = max_total {
                    o = o.min(cap.saturating_sub(p).max(1));
                }
                Work::Llm {
                    prompt_tokens: p,
                    output_tokens: o,
                }
            }
            WorkShape::Image {
                steps,
                width,
                height,
            } => Work::Image {
                steps: steps.sample(rng),
                width: *width,
                height: *height,
            },
            WorkShape::Generic => Work::Generic,
        }
    }
}

/// A synthetic workload: `count` requests from `arrivals`, shaped by `work`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkloadSpec {
    pub arrivals: ArrivalProcess,
    pub work: WorkShape,
    pub count: usize,
    #[serde(default)]
    pub seed: u64,
}

impl WorkloadSpec {
    /// Draw the requests. The same spec and seed give the same stream.
    pub fn generate(&self) -> Vec<Arrival> {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let times = arrival_times(&self.arrivals, self.count, &mut rng);
        times
            .into_iter()
            .enumerate()
            .map(|(i, t_ms)| Arrival {
                t_ms,
                req: Request {
                    id: i as u64,
                    work: self.work.sample(&mut rng),
                },
            })
            .collect()
    }
}

fn arrival_times(p: &ArrivalProcess, n: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
    match p {
        ArrivalProcess::Burst => vec![0.0; n],
        ArrivalProcess::Poisson { rate_per_s } => {
            let gap = Exp::new(rate_per_s.max(1e-12)).expect("positive rate");
            let mut t = 0.0;
            (0..n)
                .map(|_| {
                    t += gap.sample(rng);
                    t * 1e3
                })
                .collect()
        }
        ArrivalProcess::Steps { steps } => {
            let max = steps.iter().map(|s| s.1).fold(0.0, f64::max);
            thinned(n, max, rng, |t| {
                steps
                    .iter()
                    .rev()
                    .find(|(start, _)| t >= *start)
                    .map_or(0.0, |s| s.1)
            })
        }
        ArrivalProcess::Diurnal {
            mean_rate_per_s,
            amplitude,
            period_s,
        } => {
            let max = mean_rate_per_s * (1.0 + amplitude.abs());
            thinned(n, max, rng, |t| {
                let phase = 2.0 * std::f64::consts::PI * t / period_s.max(1e-9);
                (mean_rate_per_s * (1.0 + amplitude * phase.sin())).max(0.0)
            })
        }
        ArrivalProcess::Mmpp {
            low,
            high,
            mean_dwell_s,
        } => {
            let dwell = Exp::new(1.0 / mean_dwell_s.max(1e-9)).expect("positive dwell");
            let mut out = Vec::with_capacity(n);
            let (mut t, mut hi) = (0.0, false);
            let mut switch = dwell.sample(rng);
            while out.len() < n {
                let rate = if hi { *high } else { *low };
                let next = if rate > 0.0 {
                    t + Exp::new(rate).expect("positive rate").sample(rng)
                } else {
                    f64::INFINITY
                };
                if next < switch {
                    t = next;
                    out.push(t * 1e3);
                } else {
                    t = switch;
                    hi = !hi;
                    switch = t + dwell.sample(rng);
                }
            }
            out
        }
    }
}

/// Non-homogeneous Poisson by thinning against a bound `max` on `rate(t)`.
fn thinned(n: usize, max: f64, rng: &mut ChaCha8Rng, rate: impl Fn(f64) -> f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    if max <= 0.0 {
        return out;
    }
    let gap = Exp::new(max).expect("positive rate");
    let mut t = 0.0;
    while out.len() < n {
        t += gap.sample(rng);
        if rng.gen::<f64>() * max <= rate(t) {
            out.push(t * 1e3);
        }
    }
    out
}

/// Replay an Azure LLM inference trace (`TIMESTAMP,ContextTokens,
/// GeneratedTokens`, timestamps like `2023-11-16 18:15:46.6805900`).
/// Times are shifted so the first request arrives at 0 and divided by
/// `speedup`.
pub fn load_azure_trace(csv: &str, speedup: f64) -> Result<Vec<Arrival>, Error> {
    let mut rows = Vec::new();
    for (i, line) in csv.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || (i == 0 && line.starts_with("TIMESTAMP")) {
            continue;
        }
        let mut cols = line.split(',');
        let (Some(ts), Some(ctx), Some(gen)) = (cols.next(), cols.next(), cols.next()) else {
            return Err(Error::Trace(format!("line {}: expected 3 columns", i + 1)));
        };
        let bad = |what: &str| Error::Trace(format!("line {}: bad {what}", i + 1));
        let t = parse_timestamp(ts.trim()).ok_or_else(|| bad("timestamp"))?;
        let p: u32 = ctx.trim().parse().map_err(|_| bad("ContextTokens"))?;
        let o: u32 = gen.trim().parse().map_err(|_| bad("GeneratedTokens"))?;
        rows.push((t, p, o));
    }
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    let t0 = rows.first().map_or(0.0, |r| r.0);
    let speedup = if speedup > 0.0 { speedup } else { 1.0 };
    Ok(rows
        .into_iter()
        .enumerate()
        .map(|(i, (t, p, o))| Arrival {
            t_ms: (t - t0) * 1e3 / speedup,
            req: Request {
                id: i as u64,
                work: Work::Llm {
                    prompt_tokens: p.max(1),
                    output_tokens: o.max(1),
                },
            },
        })
        .collect())
}

/// Seconds since the Unix epoch for `YYYY-MM-DD HH:MM:SS[.fff]`.
fn parse_timestamp(s: &str) -> Option<f64> {
    let (date, time) = s.split_once([' ', 'T'])?;
    let mut d = date.split('-');
    let y: i64 = d.next()?.parse().ok()?;
    let m: i64 = d.next()?.parse().ok()?;
    let day: i64 = d.next()?.parse().ok()?;
    let mut t = time.trim_end_matches('Z').split(':');
    let hh: f64 = t.next()?.parse().ok()?;
    let mm: f64 = t.next()?.parse().ok()?;
    let ss: f64 = t.next()?.parse().ok()?;
    // Days from civil date (Howard Hinnant's algorithm).
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days as f64 * 86_400.0 + hh * 3600.0 + mm * 60.0 + ss)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let spec = WorkloadSpec {
            arrivals: ArrivalProcess::Poisson { rate_per_s: 10.0 },
            work: WorkShape::sharegpt_like(),
            count: 100,
            seed: 7,
        };
        assert_eq!(spec.generate(), spec.generate());
    }

    #[test]
    fn poisson_rate_is_close() {
        let spec = WorkloadSpec {
            arrivals: ArrivalProcess::Poisson { rate_per_s: 20.0 },
            work: WorkShape::Generic,
            count: 20_000,
            seed: 1,
        };
        let a = spec.generate();
        let rate = a.len() as f64 / (a.last().unwrap().t_ms / 1e3);
        assert!((rate - 20.0).abs() < 0.5, "rate {rate}");
    }

    #[test]
    fn steps_follow_the_schedule() {
        let spec = WorkloadSpec {
            arrivals: ArrivalProcess::Steps {
                steps: vec![(0.0, 10.0), (100.0, 40.0)],
            },
            work: WorkShape::Generic,
            count: 5_000,
            seed: 3,
        };
        let a = spec.generate();
        let before = a.iter().filter(|x| x.t_ms < 100_000.0).count() as f64 / 100.0;
        assert!((before - 10.0).abs() < 1.0, "rate before step {before}");
    }

    #[test]
    fn sharegpt_like_means_and_caps() {
        let spec = WorkloadSpec {
            arrivals: ArrivalProcess::Burst,
            work: WorkShape::sharegpt_like(),
            count: 50_000,
            seed: 0,
        };
        let (mut sp, mut so) = (0u64, 0u64);
        for a in spec.generate() {
            let Work::Llm {
                prompt_tokens,
                output_tokens,
            } = a.req.work
            else {
                panic!()
            };
            assert!(prompt_tokens <= 1024 && prompt_tokens + output_tokens <= 2048);
            sp += prompt_tokens as u64;
            so += output_tokens as u64;
        }
        let (mp, mo) = (sp as f64 / 50_000.0, so as f64 / 50_000.0);
        assert!((mp - 202.0).abs() < 15.0, "mean prompt {mp}");
        assert!((mo - 179.0).abs() < 15.0, "mean output {mo}");
    }

    #[test]
    fn azure_trace_replays_relative_times() {
        let csv = "TIMESTAMP,ContextTokens,GeneratedTokens\n\
                   2023-11-16 18:15:46.5000000,374,44\n\
                   2023-11-16 18:15:47.0000000,396,109\n";
        let a = load_azure_trace(csv, 1.0).unwrap();
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].t_ms, 0.0);
        assert!((a[1].t_ms - 500.0).abs() < 1e-6);
        assert_eq!(
            a[1].req.work,
            Work::Llm {
                prompt_tokens: 396,
                output_tokens: 109
            }
        );
    }
}
