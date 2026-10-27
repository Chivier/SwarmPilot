//! Routing: the strategy, the queue sketches and the predictor.
//!
//! This is the Python proxy's `schedule_task` call with the predictor
//! client behind it, on the Rust core: [`Router::try_route`] runs the
//! strategy, the [`Ledger`] keeps the queue sketches, and predictions come
//! from the portable model files or experiment mode exactly as
//! `PredictorClient.predict` makes them (grouped by platform, in the order
//! platforms first appear).

use std::collections::{HashMap, HashSet};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{Map, Value};
use swarmpilot_core::experiment::{self, Params, Synthetic};
use swarmpilot_core::mlp::Output;
use swarmpilot_core::strategy::{PredictError, Router};
use swarmpilot_core::{Draws, Ledger, Prediction, StrategyKind};

use crate::state::{Instance, ModelCache, Snapshot};

/// What one request brings to routing.
pub struct RequestInfo {
    /// The adapter's predictor features.
    pub features: HashMap<String, f64>,
    /// `exp_runtime`, `exp_cv`, `exp_modes`, `exp_skewness` from the
    /// client's body, when present.
    pub exp: Map<String, Value>,
}

impl RequestInfo {
    pub const EXP_KEYS: [&'static str; 4] = ["exp_runtime", "exp_cv", "exp_modes", "exp_skewness"];

    pub fn new(features: HashMap<String, f64>, client_body: &Value) -> Self {
        let mut exp = Map::new();
        if let Some(obj) = client_body.as_object() {
            for key in Self::EXP_KEYS {
                if let Some(v) = obj.get(key) {
                    exp.insert(key.to_string(), v.clone());
                }
            }
        }
        RequestInfo { features, exp }
    }

    /// The task's metadata values a model or training can use: features
    /// and numeric `exp_*` fields.
    pub fn numeric_metadata(&self) -> HashMap<String, f64> {
        let mut out = self.features.clone();
        for (k, v) in &self.exp {
            match v {
                Value::Number(n) => {
                    if let Some(x) = n.as_f64() {
                        out.insert(k.clone(), x);
                    }
                }
                Value::Bool(b) => {
                    out.insert(k.clone(), f64::from(u8::from(*b)));
                }
                _ => {}
            }
        }
        out
    }
}

/// The routing decision for one request.
pub struct Decision {
    pub instance_id: String,
    pub prediction: Option<Prediction>,
}

/// numpy-style draws from the gateway's own generator.
struct RngDraws<'a>(&'a mut StdRng);

impl Draws for RngDraws<'_> {
    fn random(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.0.gen::<f64>()).collect()
    }

    fn integers(&mut self, high: usize) -> usize {
        self.0.gen_range(0..high)
    }

    fn choice2(&mut self, n: usize) -> (usize, usize) {
        let a = self.0.gen_range(0..n);
        let mut b = self.0.gen_range(0..n - 1);
        if b >= a {
            b += 1;
        }
        (a, b)
    }
}

/// Shadow mode: the draws the Python strategy made for the decision,
/// replayed in order. A draw of another shape, or one too many, marks the
/// replay as diverged and comes from the gateway's generator instead.
struct ReplayDraws {
    recorded: std::collections::VecDeque<Value>,
    rng: StdRng,
    diverged: bool,
}

impl ReplayDraws {
    fn next(&mut self, key: &str) -> Option<Value> {
        match self.recorded.pop_front() {
            Some(Value::Object(mut m)) if m.contains_key(key) => m.remove(key),
            _ => {
                self.diverged = true;
                None
            }
        }
    }
}

impl Draws for ReplayDraws {
    fn random(&mut self, n: usize) -> Vec<f64> {
        let xs: Option<Vec<f64>> = self.next("random").and_then(|v| {
            v.as_array()
                .map(|a| a.iter().filter_map(Value::as_f64).collect())
        });
        match xs {
            Some(xs) if xs.len() == n => xs,
            _ => {
                self.diverged = true;
                RngDraws(&mut self.rng).random(n)
            }
        }
    }

    fn integers(&mut self, high: usize) -> usize {
        match self.next("integers").and_then(|v| v.as_u64()) {
            Some(k) if (k as usize) < high => k as usize,
            _ => {
                self.diverged = true;
                RngDraws(&mut self.rng).integers(high)
            }
        }
    }

    fn choice2(&mut self, n: usize) -> (usize, usize) {
        let pair = self.next("choice2").and_then(|v| {
            let a = v.as_array()?;
            Some((a.first()?.as_u64()? as usize, a.get(1)?.as_u64()? as usize))
        });
        match pair {
            Some((a, b)) if a < n && b < n && a != b => (a, b),
            _ => {
                self.diverged = true;
                RngDraws(&mut self.rng).choice2(n)
            }
        }
    }
}

/// Strategy, sketches and generator, behind one lock.
pub struct Routing {
    router: Router,
    ledger: Ledger,
    grid: Vec<f64>,
    rng: StdRng,
    registered: HashSet<String>,
}

impl Routing {
    pub fn new(seed: Option<u64>) -> Self {
        let kind = StrategyKind::AdaptiveBootstrap;
        let grid = swarmpilot_core::DEFAULT_QUANTILES.to_vec();
        Routing {
            router: Router::new(kind),
            ledger: Ledger::new(kind.sketch_kind(), &grid),
            grid,
            rng: match seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_entropy(),
            },
            registered: HashSet::new(),
        }
    }

    /// Follow a new snapshot: strategy, sketch grid, instance set and
    /// capacities. Charges of tasks still running are kept.
    pub fn sync(&mut self, snap: &Snapshot) {
        let grid = if snap.sketch_quantiles.is_empty() {
            self.grid.clone()
        } else {
            snap.sketch_quantiles.clone()
        };
        if snap.strategy != self.router.kind {
            // POST /v1/strategy/set makes a new strategy object.
            self.router = Router::new(snap.strategy);
        }
        if snap.strategy.sketch_kind() != self.ledger.kind() || grid != self.grid {
            self.ledger.reset(snap.strategy.sketch_kind(), Some(&grid));
            self.grid = grid;
        }
        let ids: HashSet<String> = snap.instances.iter().map(|i| i.id.clone()).collect();
        for gone in self
            .registered
            .difference(&ids)
            .cloned()
            .collect::<Vec<_>>()
        {
            self.ledger.remove(&gone);
            self.registered.remove(&gone);
        }
        for inst in &snap.instances {
            if self.registered.insert(inst.id.clone()) {
                let _ = self.ledger.register(&inst.id);
            }
            self.ledger.set_capacity(&inst.id, i64::from(inst.servers));
        }
    }

    /// Choose an instance among `candidates` (active, in registry order).
    pub fn route(
        &mut self,
        snap: &Snapshot,
        models: &ModelCache,
        candidates: &[&Instance],
        req: &RequestInfo,
        task_id: &str,
    ) -> Decision {
        self.route_with(snap, models, candidates, req, task_id, None)
            .0
    }

    /// [`Routing::route`], with the draws replayed from `recorded` when
    /// given (shadow mode); also says whether the replay diverged.
    pub fn route_with(
        &mut self,
        snap: &Snapshot,
        models: &ModelCache,
        candidates: &[&Instance],
        req: &RequestInfo,
        task_id: &str,
        recorded: Option<Vec<Value>>,
    ) -> (Decision, bool) {
        let ids: Vec<String> = candidates.iter().map(|i| i.id.clone()).collect();
        let warm: Vec<bool> = if self.router.kind == StrategyKind::AdaptiveBootstrap {
            let mut by_platform: HashMap<&str, bool> = HashMap::new();
            candidates
                .iter()
                .map(|i| {
                    *by_platform
                        .entry(i.platform_key.as_str())
                        .or_insert_with(|| is_warm(snap, models, i, req))
                })
                .collect()
        } else {
            vec![false; ids.len()]
        };
        let mut predict = |ptype: &'static str, subset: &[usize]| {
            let chosen: Vec<&Instance> = subset.iter().map(|&k| candidates[k]).collect();
            predictions(snap, models, &chosen, ptype, req)
        };
        let (routed, diverged) = match recorded {
            Some(recorded) => {
                let mut replay = ReplayDraws {
                    recorded: recorded.into(),
                    rng: StdRng::seed_from_u64(self.rng.gen()),
                    diverged: false,
                };
                let routed = self.router.try_route(
                    &mut self.ledger,
                    &ids,
                    &warm,
                    &mut predict,
                    &mut replay,
                    Some(task_id),
                );
                // Draws left over: Python drew more than the gateway did.
                (routed, replay.diverged || !replay.recorded.is_empty())
            }
            None => (
                self.router.try_route(
                    &mut self.ledger,
                    &ids,
                    &warm,
                    &mut predict,
                    &mut RngDraws(&mut self.rng),
                    Some(task_id),
                ),
                false,
            ),
        };
        (
            Decision {
                instance_id: ids[routed.selected].clone(),
                prediction: routed.prediction,
            },
            diverged,
        )
    }

    /// The task finished (or failed): take its charge off its sketch.
    pub fn release(&mut self, task_id: &str) {
        self.ledger.release(task_id);
    }
}

/// adaptive_bootstrap's `_is_warm`: experiment mode, or a trained
/// quantile model for the platform.
fn is_warm(snap: &Snapshot, models: &ModelCache, inst: &Instance, req: &RequestInfo) -> bool {
    let platform = snap.platforms.get(&inst.platform_key);
    if experiment::applies(&req.exp, platform.is_some_and(|p| p.experiment)) {
        return true;
    }
    platform.is_some_and(|p| models.get(&p.quantile_key).is_some())
}

/// `PredictorClient.predict`: one prediction per platform, copied to
/// every instance on it, platforms in order of first appearance.
fn predictions(
    snap: &Snapshot,
    models: &ModelCache,
    instances: &[&Instance],
    ptype: &'static str,
    req: &RequestInfo,
) -> Result<Vec<Prediction>, PredictError> {
    let mut groups: Vec<(&str, Vec<&Instance>)> = Vec::new();
    for inst in instances {
        match groups.iter_mut().find(|(k, _)| *k == inst.platform_key) {
            Some((_, members)) => members.push(inst),
            None => groups.push((inst.platform_key.as_str(), vec![inst])),
        }
    }
    let mut out = Vec::with_capacity(instances.len());
    for (key, members) in groups {
        let (time, error, quantiles) = predict_platform(snap, models, key, ptype, req)?;
        out.extend(members.iter().map(|inst| Prediction {
            instance_id: inst.id.clone(),
            predicted_time_ms: time,
            error_margin_ms: error,
            quantiles: quantiles.clone(),
        }));
    }
    Ok(out)
}

type Predicted = (f64, Option<f64>, Option<Vec<(f64, f64)>>);

fn predict_platform(
    snap: &Snapshot,
    models: &ModelCache,
    key: &str,
    ptype: &'static str,
    req: &RequestInfo,
) -> Result<Predicted, PredictError> {
    let platform = snap.platforms.get(key);
    if experiment::applies(&req.exp, platform.is_some_and(|p| p.experiment)) {
        let levels = (ptype == "quantile").then_some(snap.sketch_quantiles.as_slice());
        return match Params::parse(&req.exp)?.predict(ptype, levels, &snap.normals)? {
            Synthetic::ExpectError {
                expected_ms,
                error_ms,
            } => Ok((expected_ms, Some(error_ms), None)),
            Synthetic::Quantiles(pairs) => quantile_prediction(pairs),
        };
    }
    let platform =
        platform.ok_or_else(|| PredictError("Model not found: unknown platform".into()))?;
    if snap.preprocessor_rules {
        return Err(PredictError(
            "the gateway cannot run this model's preprocessor chain".into(),
        ));
    }
    let model_key = if ptype == "quantile" {
        &platform.quantile_key
    } else {
        &platform.expect_error_key
    };
    let model = models
        .get(model_key)
        .ok_or_else(|| PredictError(format!("Model not found: {model_key}")))?;
    let mut features = req.numeric_metadata();
    features.extend(
        platform
            .hardware_features
            .iter()
            .map(|(k, v)| (k.clone(), *v)),
    );
    match model
        .predict_features(&features)
        .map_err(|e| PredictError(e.to_string()))?
    {
        Output::ExpectError {
            expected_ms,
            error_ms,
        } if ptype == "expect_error" => Ok((expected_ms, Some(error_ms), None)),
        Output::Quantiles(pairs) if ptype == "quantile" => quantile_prediction(pairs),
        _ => Err(PredictError(format!(
            "model {model_key} is not a {ptype} model"
        ))),
    }
}

/// The median (or the first quantile) as the predicted time.
fn quantile_prediction(pairs: Vec<(f64, f64)>) -> Result<Predicted, PredictError> {
    let median = pairs
        .iter()
        .find(|(q, _)| *q == 0.5)
        .or_else(|| pairs.first())
        .map(|(_, v)| *v)
        .ok_or_else(|| PredictError("no quantiles predicted".into()))?;
    Ok((median, None, Some(pairs)))
}
