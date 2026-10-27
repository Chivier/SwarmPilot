//! Queue sketches: what each instance still has to work through.
//!
//! Port of `swarmpilot/scheduler/registry/queue_sketch.py` and of the
//! charge ledger in `InstanceRegistry`. A sketch summarises the predicted
//! runtime of every routed, unfinished task on an instance. Each task's
//! contribution is a [`Charge`]; the [`Ledger`] keeps one per task, adds
//! it when the task is routed and subtracts exactly the same charge when
//! the task finishes, fails or moves.

use std::collections::HashMap;

/// Quantile grid of probabilistic sketches, below the median too.
pub const DEFAULT_QUANTILES: [f64; 8] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];

/// Which summary a sketch keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SketchKind {
    /// A runtime value per quantile.
    Probabilistic,
    /// A sum of expected times and a root-sum-square of error margins.
    ExpectError,
}

impl SketchKind {
    /// Parse the Python name (`"probabilistic"` or `"expect_error"`).
    ///
    /// Anything else is probabilistic, as in `InstanceRegistry`.
    pub fn from_name(name: &str) -> Self {
        if name == "expect_error" {
            SketchKind::ExpectError
        } else {
            SketchKind::Probabilistic
        }
    }

    /// The Python name.
    pub fn name(self) -> &'static str {
        match self {
            SketchKind::Probabilistic => "probabilistic",
            SketchKind::ExpectError => "expect_error",
        }
    }
}

/// One task's predicted runtime, as added to an instance's sketch.
#[derive(Clone, Debug, PartialEq)]
pub struct Charge {
    pub instance_id: String,
    pub time_ms: f64,
    pub error_ms: f64,
    /// `(quantile, value)` pairs sorted by quantile; `None` when the
    /// prediction had none.
    pub quantiles: Option<Vec<(f64, f64)>>,
}

impl Charge {
    /// Build a charge from prediction fields (`Charge.of`).
    pub fn of(
        instance_id: &str,
        time_ms: f64,
        error_ms: Option<f64>,
        quantiles: Option<&[(f64, f64)]>,
    ) -> Self {
        let quantiles = quantiles.filter(|q| !q.is_empty()).map(|q| {
            let mut q = q.to_vec();
            q.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
            q
        });
        Charge {
            instance_id: instance_id.to_string(),
            time_ms,
            // Python: float(error_ms or 0.0)
            error_ms: error_ms.unwrap_or(0.0),
            quantiles,
        }
    }

    /// The same charge on another instance.
    pub fn moved_to(&self, instance_id: &str) -> Self {
        Charge {
            instance_id: instance_id.to_string(),
            ..self.clone()
        }
    }

    /// The task's runtime at each quantile of `grid`.
    ///
    /// Quantiles between the predicted ones are interpolated linearly;
    /// quantiles outside them take the nearest predicted value.
    pub fn at(&self, grid: &[f64]) -> Vec<f64> {
        let Some(qs) = self.quantiles.as_deref() else {
            return vec![self.time_ms; grid.len()];
        };
        let (first, last) = (qs[0], qs[qs.len() - 1]);
        grid.iter()
            .map(|&g| {
                if g <= first.0 {
                    return first.1;
                }
                if g >= last.0 {
                    return last.1;
                }
                for w in qs.windows(2) {
                    let ((q0, v0), (q1, v1)) = (w[0], w[1]);
                    if g <= q1 {
                        return v0 + (v1 - v0) * (g - q0) / (q1 - q0);
                    }
                }
                unreachable!("g is inside the predicted quantiles")
            })
            .collect()
    }
}

/// An instance's queue sketch.
#[derive(Clone, Debug, PartialEq)]
pub enum Sketch {
    Probabilistic {
        quantiles: Vec<f64>,
        values: Vec<f64>,
    },
    ExpectError {
        expected_time_ms: f64,
        error_margin_ms: f64,
    },
}

/// Python's `max(a, b)`: `b` only if it is strictly greater.
fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

impl Sketch {
    /// An empty sketch of `kind` over `grid`.
    pub fn empty(kind: SketchKind, grid: &[f64]) -> Self {
        match kind {
            SketchKind::Probabilistic => Sketch::Probabilistic {
                quantiles: grid.to_vec(),
                values: vec![0.0; grid.len()],
            },
            SketchKind::ExpectError => Sketch::ExpectError {
                expected_time_ms: 0.0,
                error_margin_ms: 0.0,
            },
        }
    }

    /// Which kind this sketch is.
    pub fn kind(&self) -> SketchKind {
        match self {
            Sketch::Probabilistic { .. } => SketchKind::Probabilistic,
            Sketch::ExpectError { .. } => SketchKind::ExpectError,
        }
    }

    /// An empty sketch of the same kind and quantiles (`empty_like`).
    pub fn empty_like(&self) -> Self {
        match self {
            Sketch::Probabilistic { quantiles, .. } => {
                Sketch::empty(SketchKind::Probabilistic, quantiles)
            }
            Sketch::ExpectError { .. } => Sketch::empty(SketchKind::ExpectError, &[]),
        }
    }

    /// Add (`sign = 1`) or remove (`sign = -1`) a charge (`apply`).
    ///
    /// The charge is divided by `servers`, the requests the instance
    /// serves at once. Values never go below zero.
    pub fn apply(&self, charge: &Charge, sign: i32, servers: u32) -> Self {
        let scale = f64::from(sign) / f64::from(servers.max(1));
        match self {
            Sketch::Probabilistic { quantiles, values } => {
                let task = charge.at(quantiles);
                let mut floor = 0.0;
                let values = values
                    .iter()
                    .zip(task)
                    .map(|(&v, t)| {
                        // Stay non-negative and non-decreasing.
                        floor = py_max(floor, v + scale * t);
                        floor
                    })
                    .collect();
                Sketch::Probabilistic {
                    quantiles: quantiles.clone(),
                    values,
                }
            }
            Sketch::ExpectError {
                expected_time_ms,
                error_margin_ms,
            } => {
                let e = scale * charge.error_ms;
                let var = error_margin_ms * error_margin_ms + f64::from(sign) * (e * e);
                Sketch::ExpectError {
                    expected_time_ms: py_max(0.0, expected_time_ms + scale * charge.time_ms),
                    error_margin_ms: py_max(0.0, var).sqrt(),
                }
            }
        }
    }

    /// True when every value is zero.
    pub fn is_empty(&self) -> bool {
        match self {
            Sketch::Probabilistic { values, .. } => values.iter().all(|v| *v == 0.0),
            Sketch::ExpectError {
                expected_time_ms, ..
            } => *expected_time_ms == 0.0,
        }
    }
}

/// Error from [`Ledger::register`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("Instance {0} already exists")]
pub struct AlreadyRegistered(pub String);

/// Per-instance sketches and the open charge of every task.
///
/// Charges are replayed in the order Python's dict would hold them
/// (insertion order, a re-charged or moved task going to the end), so a
/// rebuilt sketch sums its charges in the same order as Python does.
#[derive(Clone, Debug)]
pub struct Ledger {
    kind: SketchKind,
    grid: Vec<f64>,
    order: Vec<String>,
    sketches: HashMap<String, Sketch>,
    charges: HashMap<String, (u64, Charge)>,
    next_seq: u64,
    counts: HashMap<String, i64>,
    servers: HashMap<String, u32>,
}

impl Ledger {
    /// An empty ledger keeping sketches of `kind` over `grid`.
    pub fn new(kind: SketchKind, grid: &[f64]) -> Self {
        Ledger {
            kind,
            grid: grid.to_vec(),
            order: Vec::new(),
            sketches: HashMap::new(),
            charges: HashMap::new(),
            next_seq: 0,
            counts: HashMap::new(),
            servers: HashMap::new(),
        }
    }

    /// The sketch kind new instances get.
    pub fn kind(&self) -> SketchKind {
        self.kind
    }

    /// The quantile grid of probabilistic sketches.
    pub fn grid(&self) -> &[f64] {
        &self.grid
    }

    /// Add an instance with an empty sketch.
    pub fn register(&mut self, instance_id: &str) -> Result<(), AlreadyRegistered> {
        if self.sketches.contains_key(instance_id) {
            return Err(AlreadyRegistered(instance_id.to_string()));
        }
        self.order.push(instance_id.to_string());
        self.sketches.insert(
            instance_id.to_string(),
            Sketch::empty(self.kind, &self.grid),
        );
        Ok(())
    }

    /// Drop an instance, its sketch, capacity and every charge on it.
    pub fn remove(&mut self, instance_id: &str) -> bool {
        if self.sketches.remove(instance_id).is_none() {
            return false;
        }
        self.order.retain(|i| i != instance_id);
        self.counts.remove(instance_id);
        self.charges
            .retain(|_, (_, c)| c.instance_id != instance_id);
        self.servers.remove(instance_id);
        true
    }

    /// Registered instances, in registration order.
    pub fn instance_ids(&self) -> &[String] {
        &self.order
    }

    /// An instance's sketch.
    pub fn sketch(&self, instance_id: &str) -> Option<&Sketch> {
        self.sketches.get(instance_id)
    }

    /// A task's open charge.
    pub fn get_charge(&self, task_id: &str) -> Option<&Charge> {
        self.charges.get(task_id).map(|(_, c)| c)
    }

    /// Requests the instance serves at once (1 when not set).
    pub fn capacity(&self, instance_id: &str) -> u32 {
        self.servers.get(instance_id).copied().unwrap_or(1)
    }

    fn apply_charge(&mut self, charge: &Charge, sign: i32) {
        let Some(sketch) = self.sketches.get(&charge.instance_id) else {
            return;
        };
        let count = self.counts.get(&charge.instance_id).copied().unwrap_or(0) + i64::from(sign);
        self.counts.insert(charge.instance_id.clone(), count.max(0));
        // An empty sketch is exactly empty, not a rounding residue.
        let next = if count <= 0 {
            sketch.empty_like()
        } else {
            sketch.apply(charge, sign, self.capacity(&charge.instance_id))
        };
        self.sketches.insert(charge.instance_id.clone(), next);
    }

    fn store(&mut self, task_id: &str, charge: Charge) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.charges.insert(task_id.to_string(), (seq, charge));
    }

    /// Charges in Python dict order.
    fn ordered_charges(&self) -> Vec<Charge> {
        let mut all: Vec<_> = self.charges.values().collect();
        all.sort_by_key(|(seq, _)| *seq);
        all.into_iter().map(|(_, c)| c.clone()).collect()
    }

    /// Add a task's charge; a task charged before has its old charge
    /// removed first. `None` is a charge that is never released.
    pub fn charge(&mut self, task_id: Option<&str>, charge: Charge) {
        if let Some(task_id) = task_id {
            if let Some((_, old)) = self.charges.remove(task_id) {
                self.apply_charge(&old, -1);
            }
            self.store(task_id, charge.clone());
        }
        self.apply_charge(&charge, 1);
    }

    /// Remove a finished task's charge and return it.
    pub fn release(&mut self, task_id: &str) -> Option<Charge> {
        let (_, old) = self.charges.remove(task_id)?;
        self.apply_charge(&old, -1);
        Some(old)
    }

    /// Move a task's charge, unchanged, to another instance.
    pub fn move_charge(&mut self, task_id: &str, instance_id: &str) -> bool {
        let Some((_, old)) = self.charges.remove(task_id) else {
            return false;
        };
        self.apply_charge(&old, -1);
        let new = old.moved_to(instance_id);
        self.store(task_id, new.clone());
        self.apply_charge(&new, 1);
        true
    }

    /// Set how many requests an instance serves at once and rebuild its
    /// sketch from its open charges.
    pub fn set_capacity(&mut self, instance_id: &str, servers: i64) {
        let servers = servers.clamp(1, i64::from(u32::MAX)) as u32;
        let Some(sketch) = self.sketches.get(instance_id) else {
            return;
        };
        if self.capacity(instance_id) == servers {
            return;
        }
        let empty = sketch.empty_like();
        self.servers.insert(instance_id.to_string(), servers);
        self.sketches.insert(instance_id.to_string(), empty);
        self.counts.remove(instance_id);
        for charge in self.ordered_charges() {
            if charge.instance_id == instance_id {
                self.apply_charge(&charge, 1);
            }
        }
    }

    /// Switch every sketch to `kind` (and `grid`, when given) and rebuild
    /// it from the open charges (`reset_queues`).
    pub fn reset(&mut self, kind: SketchKind, grid: Option<&[f64]>) {
        self.kind = kind;
        if let Some(grid) = grid.filter(|g| !g.is_empty()) {
            let mut g = grid.to_vec();
            g.sort_by(f64::total_cmp);
            g.dedup();
            self.grid = g;
        }
        for id in &self.order {
            self.sketches
                .insert(id.clone(), Sketch::empty(kind, &self.grid));
        }
        self.counts.clear();
        for charge in self.ordered_charges() {
            self.apply_charge(&charge, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charge_interpolates_and_clamps() {
        let c = Charge::of("i", 5.0, None, Some(&[(0.9, 20.0), (0.5, 10.0)]));
        // 14.999999999999998, as Python computes it too.
        let mid = 10.0 + (20.0 - 10.0) * (0.7 - 0.5) / (0.9 - 0.5);
        assert_eq!(c.at(&[0.1, 0.5, 0.7, 0.99]), vec![10.0, 10.0, mid, 20.0]);
        let flat = Charge::of("i", 5.0, Some(1.0), None);
        assert_eq!(flat.at(&[0.1, 0.9]), vec![5.0, 5.0]);
    }

    #[test]
    fn ledger_returns_to_empty() {
        let mut l = Ledger::new(SketchKind::Probabilistic, &DEFAULT_QUANTILES);
        l.register("a").unwrap();
        l.register("b").unwrap();
        assert!(l.register("a").is_err());
        l.charge(Some("t1"), Charge::of("a", 10.0, None, None));
        l.charge(Some("t2"), Charge::of("a", 30.0, None, None));
        l.set_capacity("a", 2);
        assert_eq!(
            l.sketch("a").unwrap(),
            &Sketch::Probabilistic {
                quantiles: DEFAULT_QUANTILES.to_vec(),
                values: vec![20.0; 8]
            }
        );
        assert!(l.move_charge("t1", "b"));
        l.release("t1");
        l.release("t2");
        assert!(l.sketch("a").unwrap().is_empty());
        assert!(l.sketch("b").unwrap().is_empty());
    }
}
