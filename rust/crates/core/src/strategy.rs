//! The seven routing strategies of `swarmpilot/scheduler/algorithms/`.
//!
//! Selection is split from randomness: every strategy that draws takes
//! its draws from a [`Draws`] source, so the Python Scheduler can pass in
//! the numbers its own generator drew (and get identical decisions), the
//! golden vectors can replay recorded draws, and a Rust data plane can
//! plug in its own generator.

use crate::interp::interp;
use crate::sketch::{Charge, Ledger, Sketch};

/// Samples the probabilistic strategy draws per decision.
pub const PROBABILISTIC_SAMPLES: usize = 10;

/// One instance's prediction for a task.
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    pub instance_id: String,
    pub predicted_time_ms: f64,
    pub error_margin_ms: Option<f64>,
    /// `(quantile, value)` pairs in any order; `None` or empty when absent.
    pub quantiles: Option<Vec<(f64, f64)>>,
}

impl Prediction {
    /// The charge this prediction adds to `instance_id`'s sketch.
    pub fn charge(&self) -> Charge {
        Charge::of(
            &self.instance_id,
            self.predicted_time_ms,
            self.error_margin_ms,
            self.quantiles.as_deref(),
        )
    }
}

/// A source of random draws, mirroring the numpy calls the Python
/// strategies make.
pub trait Draws {
    /// `rng.random(n)`: `n` uniforms in `[0, 1)`.
    fn random(&mut self, n: usize) -> Vec<f64>;
    /// `rng.integers(high)`: uniform in `0..high`.
    fn integers(&mut self, high: usize) -> usize;
    /// `rng.choice(n, size=2, replace=False)`: two distinct indices.
    fn choice2(&mut self, n: usize) -> (usize, usize);
}

/// The predictor could not answer (no model, missing features, a bad
/// experiment-mode parameter); the message says why.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictError(pub String);

impl std::fmt::Display for PredictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PredictError {}

/// A predictor that can fail: `(prediction type, instance indices)`.
pub type FallibleFn<'a> =
    dyn FnMut(&'static str, &[usize]) -> Result<Vec<Prediction>, PredictError> + 'a;

/// Strategy names, as `get_strategy` takes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrategyKind {
    AdaptiveBootstrap,
    MinTime,
    Po2,
    Probabilistic,
    Random,
    RoundRobin,
    Serverless,
}

impl StrategyKind {
    /// Parse a strategy name; unknown names are adaptive_bootstrap, as in
    /// `get_strategy`.
    pub fn from_name(name: &str) -> Self {
        match name {
            "min_time" => StrategyKind::MinTime,
            "po2" => StrategyKind::Po2,
            "probabilistic" => StrategyKind::Probabilistic,
            "random" => StrategyKind::Random,
            "round_robin" => StrategyKind::RoundRobin,
            "serverless" => StrategyKind::Serverless,
            _ => StrategyKind::AdaptiveBootstrap,
        }
    }

    /// The sketch kind the strategy reads (`queue_info_type_for`).
    pub fn sketch_kind(self) -> crate::SketchKind {
        match self {
            StrategyKind::MinTime | StrategyKind::Po2 | StrategyKind::Serverless => {
                crate::SketchKind::ExpectError
            }
            _ => crate::SketchKind::Probabilistic,
        }
    }

    /// The prediction type the strategy asks the predictor for.
    pub fn prediction_type(self) -> &'static str {
        match self {
            StrategyKind::Probabilistic | StrategyKind::AdaptiveBootstrap => "quantile",
            _ => "expect_error",
        }
    }

    /// Whether the chosen prediction is charged to the queue sketch.
    pub fn charges(self) -> bool {
        !matches!(self, StrategyKind::Random | StrategyKind::RoundRobin)
    }
}

/// min_time and serverless: least `queue expected + queue error + task`.
///
/// `queues[k]` is the sketch of `preds[k]`'s instance, if any.
pub fn select_min_expected(preds: &[Prediction], queues: &[Option<&Sketch>]) -> Option<usize> {
    let mut best = None;
    let mut best_total = f64::INFINITY;
    for (k, pred) in preds.iter().enumerate() {
        let total = match queues[k] {
            Some(Sketch::ExpectError {
                expected_time_ms,
                error_margin_ms,
            }) => expected_time_ms + error_margin_ms + pred.predicted_time_ms,
            _ => pred.predicted_time_ms,
        };
        if total < best_total {
            best_total = total;
            best = Some(k);
        }
    }
    best
}

/// po2: the shorter queue of two distinct random instances, every task
/// counting as one unit. The caller charges the winner 1.0 ms, as the
/// Python strategy overwrites every prediction's time with 1.0.
pub fn select_po2(n: usize, queues: &[Option<&Sketch>], draws: &mut dyn Draws) -> Option<usize> {
    match n {
        0 => None,
        1 => Some(0),
        _ => {
            let (i, j) = draws.choice2(n);
            let total = |k: usize| match queues[k] {
                Some(Sketch::ExpectError {
                    expected_time_ms, ..
                }) => expected_time_ms + 1.0,
                _ => 1.0,
            };
            Some(if total(i) <= total(j) { i } else { j })
        }
    }
}

/// probabilistic: Monte Carlo over shared random percentiles; the
/// instance with the least `task + queue` time in the most samples wins.
pub fn select_probabilistic(
    preds: &[Prediction],
    queues: &[Option<&Sketch>],
    percentiles: &[f64],
) -> Option<usize> {
    if preds.is_empty() {
        return None;
    }
    // Python keys predictions by instance id: the first position of an
    // id, with the values of its last prediction.
    let mut slots: Vec<usize> = Vec::new();
    let mut value_of: Vec<usize> = Vec::new();
    for (k, p) in preds.iter().enumerate() {
        match slots
            .iter()
            .position(|&s| preds[s].instance_id == p.instance_id)
        {
            Some(pos) => value_of[pos] = k,
            None => {
                slots.push(k);
                value_of.push(k);
            }
        }
    }
    let samples = percentiles.len();
    let mut totals = vec![vec![0.0; samples]; slots.len()];
    for (row, (&slot, &vk)) in slots.iter().zip(&value_of).enumerate() {
        let pred = &preds[vk];
        let task: Vec<f64> = match pred.quantiles.as_deref().filter(|q| !q.is_empty()) {
            Some(q) => {
                let mut q = q.to_vec();
                q.sort_by(|a, b| a.0.total_cmp(&b.0));
                let xp: Vec<f64> = q.iter().map(|p| p.0).collect();
                let fp: Vec<f64> = q.iter().map(|p| p.1).collect();
                percentiles.iter().map(|&x| interp(x, &xp, &fp)).collect()
            }
            None => vec![pred.predicted_time_ms; samples],
        };
        // The queue of the instance in the slot's own position.
        let queue: Vec<f64> = match queues[slot] {
            Some(Sketch::Probabilistic { quantiles, values }) => percentiles
                .iter()
                .map(|&x| interp(x, quantiles, values))
                .collect(),
            _ => vec![0.0; samples],
        };
        for (cell, (t, q)) in totals[row].iter_mut().zip(task.iter().zip(&queue)) {
            *cell = t + q;
        }
    }
    let mut wins = vec![0usize; slots.len()];
    // Column-wise: each sample's winner across instances.
    #[allow(clippy::needless_range_loop)]
    for s in 0..samples {
        // np.argmin: the first minimum (a NaN counts as the minimum).
        let mut best = 0;
        for row in 1..slots.len() {
            let (b, v) = (totals[best][s], totals[row][s]);
            if !b.is_nan() && (v < b || v.is_nan()) {
                best = row;
            }
        }
        wins[best] += 1;
    }
    // np.argmax: the first maximum.
    let mut top = 0;
    for row in 1..wins.len() {
        if wins[row] > wins[top] {
            top = row;
        }
    }
    Some(slots[top])
}

/// What [`Router::route`] decided.
#[derive(Clone, Debug, PartialEq)]
pub struct Routed {
    /// Index into the instances passed to `route`.
    pub selected: usize,
    /// The chosen instance's prediction, as the strategy saw it.
    pub prediction: Option<Prediction>,
    /// The charge added to the ledger, if any.
    pub charged: Option<Charge>,
    /// Prediction types requested, in order.
    pub prediction_types: Vec<&'static str>,
}

/// A strategy with its state: the whole `schedule_task` of the Python
/// strategies, sketches and charging included.
#[derive(Clone, Debug)]
pub struct Router {
    pub kind: StrategyKind,
    counter: u64,
}

impl Router {
    pub fn new(kind: StrategyKind) -> Self {
        Router { kind, counter: 0 }
    }

    /// Route one task.
    ///
    /// - `instances`: candidate instance ids, in order.
    /// - `warm[k]`: whether `instances[k]`'s platform has a trained model
    ///   (only adaptive_bootstrap reads it).
    /// - `predict(kind, subset)`: predictions for those instance indices.
    pub fn route(
        &mut self,
        ledger: &mut Ledger,
        instances: &[String],
        warm: &[bool],
        predict: &mut dyn FnMut(&'static str, &[usize]) -> Vec<Prediction>,
        draws: &mut dyn Draws,
        task_id: Option<&str>,
    ) -> Routed {
        let mut ok = |ptype: &'static str, subset: &[usize]| Ok(predict(ptype, subset));
        self.try_route(ledger, instances, warm, &mut ok, draws, task_id)
    }

    /// [`Router::route`] with a predictor that can fail (no model, missing
    /// features), handled as the Python Scheduler handles it:
    ///
    /// - round_robin and random use placeholder predictions of 0 ms;
    /// - adaptive_bootstrap, when its round-robin turn landed on a warm
    ///   instance, keeps that round-robin pick;
    /// - every other strategy falls back to the first instance with no
    ///   prediction (the proxy's fallback), and nothing is charged.
    pub fn try_route(
        &mut self,
        ledger: &mut Ledger,
        instances: &[String],
        warm: &[bool],
        predict: &mut FallibleFn<'_>,
        draws: &mut dyn Draws,
        task_id: Option<&str>,
    ) -> Routed {
        let all: Vec<usize> = (0..instances.len()).collect();
        let fallback = |selected: usize, ptype: &'static str| Routed {
            selected,
            prediction: None,
            charged: None,
            prediction_types: vec![ptype],
        };
        if self.kind != StrategyKind::AdaptiveBootstrap {
            return self
                .schedule(self.kind, ledger, instances, &all, predict, draws, task_id)
                .unwrap_or_else(|_| fallback(0, self.kind.prediction_type()));
        }
        let hot: Vec<usize> = all.iter().copied().filter(|&k| warm[k]).collect();
        if hot.len() == instances.len() {
            return self
                .schedule(
                    StrategyKind::Probabilistic,
                    ledger,
                    instances,
                    &all,
                    predict,
                    draws,
                    task_id,
                )
                .unwrap_or_else(|_| fallback(0, "quantile"));
        }
        let idx = (self.counter % instances.len() as u64) as usize;
        self.counter += 1;
        if warm[idx] {
            return self
                .schedule(
                    StrategyKind::Probabilistic,
                    ledger,
                    instances,
                    &hot,
                    predict,
                    draws,
                    task_id,
                )
                .unwrap_or_else(|_| fallback(idx, "quantile"));
        }
        Routed {
            selected: idx,
            prediction: None,
            charged: None,
            prediction_types: Vec::new(),
        }
    }

    /// `SchedulingStrategy.schedule_task` for one plain strategy over the
    /// instance indices `subset`.
    #[allow(clippy::too_many_arguments)]
    fn schedule(
        &mut self,
        kind: StrategyKind,
        ledger: &mut Ledger,
        instances: &[String],
        subset: &[usize],
        predict: &mut FallibleFn<'_>,
        draws: &mut dyn Draws,
        task_id: Option<&str>,
    ) -> Result<Routed, PredictError> {
        let ptype = kind.prediction_type();
        let mut preds = match predict(ptype, subset) {
            Ok(p) => p,
            // round_robin and random do not need real predictions.
            Err(_) if !kind.charges() => subset
                .iter()
                .map(|&k| Prediction {
                    instance_id: instances[k].clone(),
                    predicted_time_ms: 0.0,
                    error_margin_ms: None,
                    quantiles: None,
                })
                .collect(),
            Err(e) => return Err(e),
        };
        let queues: Vec<Option<&Sketch>> = preds
            .iter()
            .map(|p| ledger.sketch(&p.instance_id))
            .collect();
        let pick = match kind {
            StrategyKind::MinTime | StrategyKind::Serverless => {
                select_min_expected(&preds, &queues)
            }
            StrategyKind::Po2 => select_po2(preds.len(), &queues, draws),
            StrategyKind::Probabilistic | StrategyKind::AdaptiveBootstrap => {
                if preds.is_empty() {
                    None
                } else {
                    let pct = draws.random(PROBABILISTIC_SAMPLES);
                    select_probabilistic(&preds, &queues, &pct)
                }
            }
            StrategyKind::Random => (!preds.is_empty()).then(|| draws.integers(preds.len())),
            StrategyKind::RoundRobin => (!preds.is_empty()).then(|| {
                let k = (self.counter % preds.len() as u64) as usize;
                self.counter += 1;
                k
            }),
        };
        if kind == StrategyKind::Po2 {
            for p in &mut preds {
                p.predicted_time_ms = 1.0;
            }
        }
        // Fall back to the first candidate, as the Python base class does.
        let selected_id = match pick {
            Some(k) => preds[k].instance_id.clone(),
            None => instances[subset[0]].clone(),
        };
        let prediction = preds.iter().find(|p| p.instance_id == selected_id).cloned();
        let mut charged = None;
        if let Some(p) = &prediction {
            if kind.charges() {
                let c = p.charge();
                ledger.charge(task_id, c.clone());
                charged = Some(c);
            }
        }
        let selected = subset
            .iter()
            .copied()
            .find(|&k| instances[k] == selected_id)
            .unwrap_or(subset[0]);
        Ok(Routed {
            selected,
            prediction,
            charged,
            prediction_types: vec![ptype],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pred(id: &str, t: f64) -> Prediction {
        Prediction {
            instance_id: id.into(),
            predicted_time_ms: t,
            error_margin_ms: Some(1.0),
            quantiles: None,
        }
    }

    #[test]
    fn min_expected_adds_queue() {
        let q = Sketch::ExpectError {
            expected_time_ms: 100.0,
            error_margin_ms: 5.0,
        };
        let preds = [pred("a", 10.0), pred("b", 50.0)];
        assert_eq!(select_min_expected(&preds, &[Some(&q), None]), Some(1));
        assert_eq!(select_min_expected(&preds, &[None, None]), Some(0));
        assert_eq!(select_min_expected(&[], &[]), None);
    }

    #[test]
    fn probabilistic_prefers_short_queue() {
        let preds = [pred("a", 10.0), pred("b", 10.0)];
        let q = Sketch::Probabilistic {
            quantiles: vec![0.1, 0.9],
            values: vec![50.0, 90.0],
        };
        let pct = [0.2, 0.5, 0.8];
        assert_eq!(
            select_probabilistic(&preds, &[Some(&q), None], &pct),
            Some(1)
        );
        // Ties go to the first instance.
        assert_eq!(select_probabilistic(&preds, &[None, None], &pct), Some(0));
    }

    struct Fixed;

    impl Draws for Fixed {
        fn random(&mut self, n: usize) -> Vec<f64> {
            vec![0.5; n]
        }
        fn integers(&mut self, _high: usize) -> usize {
            0
        }
        fn choice2(&mut self, _n: usize) -> (usize, usize) {
            (0, 1)
        }
    }

    fn setup(kind: StrategyKind) -> (Router, Ledger, Vec<String>) {
        let mut ledger = Ledger::new(kind.sketch_kind(), &crate::DEFAULT_QUANTILES);
        let ids: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        for id in &ids {
            ledger.register(id).unwrap();
        }
        (Router::new(kind), ledger, ids)
    }

    fn failing(_: &'static str, _: &[usize]) -> Result<Vec<Prediction>, PredictError> {
        Err(PredictError("Model not found".into()))
    }

    #[test]
    fn failed_prediction_falls_back_to_first_instance_uncharged() {
        for kind in [
            StrategyKind::MinTime,
            StrategyKind::Probabilistic,
            StrategyKind::Po2,
            StrategyKind::Serverless,
        ] {
            let (mut router, mut ledger, ids) = setup(kind);
            let r = router.try_route(
                &mut ledger,
                &ids,
                &[false; 3],
                &mut failing,
                &mut Fixed,
                Some("t"),
            );
            assert_eq!(r.selected, 0, "{kind:?}");
            assert!(r.prediction.is_none() && r.charged.is_none());
            assert!(ledger.get_charge("t").is_none());
        }
    }

    #[test]
    fn round_robin_and_random_ignore_failed_predictions() {
        let (mut router, mut ledger, ids) = setup(StrategyKind::RoundRobin);
        let picks: Vec<usize> = (0..4)
            .map(|_| {
                router
                    .try_route(
                        &mut ledger,
                        &ids,
                        &[false; 3],
                        &mut failing,
                        &mut Fixed,
                        None,
                    )
                    .selected
            })
            .collect();
        assert_eq!(picks, [0, 1, 2, 0]);
        let (mut router, mut ledger, ids) = setup(StrategyKind::Random);
        let r = router.try_route(
            &mut ledger,
            &ids,
            &[false; 3],
            &mut failing,
            &mut Fixed,
            None,
        );
        assert_eq!(r.prediction.map(|p| p.predicted_time_ms), Some(0.0));
    }

    #[test]
    fn adaptive_keeps_round_robin_pick_when_warm_prediction_fails() {
        let (mut router, mut ledger, ids) = setup(StrategyKind::AdaptiveBootstrap);
        let warm = [false, true, true];
        let picks: Vec<usize> = (0..3)
            .map(|_| {
                router
                    .try_route(
                        &mut ledger,
                        &ids,
                        &warm,
                        &mut failing,
                        &mut Fixed,
                        Some("t"),
                    )
                    .selected
            })
            .collect();
        // Turn 0 is cold (plain round robin); turns 1 and 2 are warm, and
        // their failed predictions keep the round-robin pick.
        assert_eq!(picks, [0, 1, 2]);
        assert!(ledger.get_charge("t").is_none());
        // All warm and failing: the first instance.
        let r = router.try_route(
            &mut ledger,
            &ids,
            &[true; 3],
            &mut failing,
            &mut Fixed,
            None,
        );
        assert_eq!(r.selected, 0);
    }
}
