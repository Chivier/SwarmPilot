//! Python bindings of the SwarmPilot scheduling core.
//!
//! The Python Scheduler calls these when `SCHEDULER_CORE=rust`
//! (`swarmpilot/scheduler/core.py`). Each function is a drop-in for one
//! Python function or method and takes plain lists and tuples, so the
//! call boundary carries no Python model classes.

use std::borrow::Cow;

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyMapping, PyString};

use std::collections::HashMap;

use swarmpilot_core::extract::{Doc, Extractor as CoreExtractor};
use swarmpilot_core::mlp::{Output, RuntimePredictor as CorePredictor, Tensor};
use swarmpilot_core::sketch::{Charge, Ledger as CoreLedger, Sketch, SketchKind};
use swarmpilot_core::strategy::{self, Prediction};

type Pairs = Vec<(f64, f64)>;

/// A Python request body, read with the Python extractor's type tests.
#[derive(Clone)]
struct PyDoc<'py>(Bound<'py, PyAny>);

impl<'py> Doc for PyDoc<'py> {
    fn get(&self, key: &str) -> Option<Self> {
        if let Ok(d) = self.0.cast::<PyDict>() {
            return d.get_item(key).ok().flatten().map(PyDoc);
        }
        let m = self.0.cast::<PyMapping>().ok()?;
        m.get_item(key).ok().map(PyDoc)
    }
    fn elements(&self) -> Option<Vec<Self>> {
        let l = self.0.cast_exact::<PyList>().ok()?;
        Some(l.iter().map(PyDoc).collect())
    }
    fn item(&self, index: usize) -> Option<Self> {
        let l = self.0.cast_exact::<PyList>().ok()?;
        l.get_item(index).ok().map(PyDoc)
    }
    fn number(&self) -> Option<f64> {
        let v = &self.0;
        if v.is_instance_of::<PyBool>() {
            return None;
        }
        if !(v.is_instance_of::<PyInt>() || v.is_instance_of::<PyFloat>()) {
            return None;
        }
        // An int too large for a float counts as missing.
        v.extract::<f64>().ok().filter(|x| x.is_finite())
    }
    fn sized_len(&self) -> Option<usize> {
        let v = &self.0;
        if v.is_instance_of::<PyList>()
            || v.is_instance_of::<PyString>()
            || v.is_instance_of::<PyMapping>()
        {
            return v.len().ok();
        }
        None
    }
    fn exact_str_len(&self) -> Option<usize> {
        if self.0.is_exact_instance_of::<PyString>() {
            return self.0.len().ok();
        }
        None
    }
    fn text(&self) -> Option<Cow<'_, str>> {
        let s = self.0.cast::<PyString>().ok()?;
        // Strings with lone surrogates never hold "WxH".
        s.to_str().ok().map(|t| Cow::Owned(t.to_string()))
    }
}

/// A compiled feature extractor expression (`compile_extractor`).
#[pyclass(frozen, module = "swarmpilot_core")]
struct Extractor {
    inner: CoreExtractor,
}

#[pymethods]
impl Extractor {
    #[new]
    #[pyo3(signature = (expr, default = 0.0))]
    fn new(expr: &str, default: f64) -> PyResult<Self> {
        CoreExtractor::compile(expr, default)
            .map(|inner| Extractor { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// The feature value for a parsed body.
    fn __call__(&self, body: Bound<'_, PyAny>) -> f64 {
        self.inner.eval(&PyDoc(body))
    }

    /// The feature value for a raw JSON body.
    fn eval_json(&self, body: &str) -> PyResult<f64> {
        let v: serde_json::Value =
            serde_json::from_str(body).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(self.inner.eval(&&v))
    }
}

/// `Charge.at`: a charge's runtime at each quantile of `grid`.
#[pyfunction]
#[pyo3(signature = (time_ms, quantiles, grid))]
fn charge_at(time_ms: f64, quantiles: Option<Pairs>, grid: Vec<f64>) -> Vec<f64> {
    Charge::of("", time_ms, None, quantiles.as_deref()).at(&grid)
}

/// `apply` on a probabilistic sketch; returns the new values.
#[pyfunction]
#[pyo3(signature = (values, grid, time_ms, quantiles, sign, servers = 1))]
fn apply_probabilistic(
    values: Vec<f64>,
    grid: Vec<f64>,
    time_ms: f64,
    quantiles: Option<Pairs>,
    sign: i32,
    servers: u32,
) -> PyResult<Vec<f64>> {
    if values.len() != grid.len() {
        return Err(PyValueError::new_err("values and grid differ in length"));
    }
    let sketch = Sketch::Probabilistic {
        quantiles: grid,
        values,
    };
    match sketch.apply(
        &Charge::of("", time_ms, None, quantiles.as_deref()),
        sign,
        servers,
    ) {
        Sketch::Probabilistic { values, .. } => Ok(values),
        Sketch::ExpectError { .. } => unreachable!(),
    }
}

/// `apply` on an expect-error sketch; returns (expected, error margin).
#[pyfunction]
#[pyo3(signature = (expected_time_ms, error_margin_ms, time_ms, error_ms, sign, servers = 1))]
fn apply_expect_error(
    expected_time_ms: f64,
    error_margin_ms: f64,
    time_ms: f64,
    error_ms: Option<f64>,
    sign: i32,
    servers: u32,
) -> (f64, f64) {
    let sketch = Sketch::ExpectError {
        expected_time_ms,
        error_margin_ms,
    };
    match sketch.apply(&Charge::of("", time_ms, error_ms, None), sign, servers) {
        Sketch::ExpectError {
            expected_time_ms,
            error_margin_ms,
        } => (expected_time_ms, error_margin_ms),
        Sketch::Probabilistic { .. } => unreachable!(),
    }
}

fn preds(times: &[f64]) -> Vec<Prediction> {
    times
        .iter()
        .map(|&t| Prediction {
            instance_id: String::new(),
            predicted_time_ms: t,
            error_margin_ms: None,
            quantiles: None,
        })
        .collect()
}

/// min_time / serverless: index of the least queue + error + task time.
///
/// `queues[k]` is `(expected_time_ms, error_margin_ms)` of an
/// expect-error sketch, or None.
#[pyfunction]
fn select_min_expected(
    times: Vec<f64>,
    queues: Vec<Option<(f64, f64)>>,
) -> PyResult<Option<usize>> {
    if times.len() != queues.len() {
        return Err(PyValueError::new_err("times and queues differ in length"));
    }
    let sketches: Vec<Option<Sketch>> = queues
        .into_iter()
        .map(|q| {
            q.map(|(e, m)| Sketch::ExpectError {
                expected_time_ms: e,
                error_margin_ms: m,
            })
        })
        .collect();
    let refs: Vec<Option<&Sketch>> = sketches.iter().map(Option::as_ref).collect();
    Ok(strategy::select_min_expected(&preds(&times), &refs))
}

/// po2 between the drawn indices `i` and `j`; `queues[k]` is the
/// expected time of an expect-error sketch, or None.
#[pyfunction]
fn select_po2(queues: Vec<Option<f64>>, i: usize, j: usize) -> PyResult<usize> {
    if i >= queues.len() || j >= queues.len() {
        return Err(PyValueError::new_err("index out of range"));
    }
    struct Fixed(usize, usize);
    impl strategy::Draws for Fixed {
        fn random(&mut self, _: usize) -> Vec<f64> {
            unreachable!()
        }
        fn integers(&mut self, _: usize) -> usize {
            unreachable!()
        }
        fn choice2(&mut self, _: usize) -> (usize, usize) {
            (self.0, self.1)
        }
    }
    let sketches: Vec<Option<Sketch>> = queues
        .into_iter()
        .map(|q| {
            q.map(|e| Sketch::ExpectError {
                expected_time_ms: e,
                error_margin_ms: 0.0,
            })
        })
        .collect();
    let refs: Vec<Option<&Sketch>> = sketches.iter().map(Option::as_ref).collect();
    // n >= 2 here: one instance never draws.
    Ok(strategy::select_po2(refs.len().max(2), &refs, &mut Fixed(i, j)).unwrap_or(0))
}

/// probabilistic: index of the winning prediction.
///
/// `queues[k]` is `(grid, values)` of a probabilistic sketch, or None.
#[pyfunction]
fn select_probabilistic(
    ids: Vec<String>,
    times: Vec<f64>,
    quantiles: Vec<Option<Pairs>>,
    queues: Vec<Option<(Vec<f64>, Vec<f64>)>>,
    percentiles: Vec<f64>,
) -> PyResult<Option<usize>> {
    let n = ids.len();
    if times.len() != n || quantiles.len() != n || queues.len() != n {
        return Err(PyValueError::new_err("arguments differ in length"));
    }
    let preds: Vec<Prediction> = ids
        .into_iter()
        .zip(times)
        .zip(quantiles)
        .map(|((id, t), q)| Prediction {
            instance_id: id,
            predicted_time_ms: t,
            error_margin_ms: None,
            quantiles: q,
        })
        .collect();
    let mut sketches = Vec::with_capacity(n);
    for q in queues {
        sketches.push(match q {
            Some((grid, values)) => {
                if grid.is_empty() || grid.len() != values.len() {
                    return Err(PyValueError::new_err("bad queue sketch"));
                }
                Some(Sketch::Probabilistic {
                    quantiles: grid,
                    values,
                })
            }
            None => None,
        });
    }
    let refs: Vec<Option<&Sketch>> = sketches.iter().map(Option::as_ref).collect();
    Ok(strategy::select_probabilistic(&preds, &refs, &percentiles))
}

/// The charge ledger of `InstanceRegistry` with its sketches.
#[pyclass(module = "swarmpilot_core")]
struct Ledger {
    inner: CoreLedger,
}

fn sketch_tuple(py: Python<'_>, s: &Sketch) -> PyResult<Py<PyAny>> {
    Ok(match s {
        Sketch::Probabilistic { quantiles, values } => {
            ("probabilistic", quantiles.clone(), values.clone())
                .into_pyobject(py)?
                .into_any()
                .unbind()
        }
        Sketch::ExpectError {
            expected_time_ms,
            error_margin_ms,
        } => ("expect_error", *expected_time_ms, *error_margin_ms)
            .into_pyobject(py)?
            .into_any()
            .unbind(),
    })
}

#[pymethods]
impl Ledger {
    #[new]
    #[pyo3(signature = (kind = "probabilistic", grid = None))]
    fn new(kind: &str, grid: Option<Vec<f64>>) -> Self {
        let grid = grid.unwrap_or_else(|| swarmpilot_core::DEFAULT_QUANTILES.to_vec());
        Ledger {
            inner: CoreLedger::new(SketchKind::from_name(kind), &grid),
        }
    }

    fn register(&mut self, instance_id: &str) -> PyResult<()> {
        self.inner
            .register(instance_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn remove(&mut self, instance_id: &str) -> PyResult<()> {
        if self.inner.remove(instance_id) {
            Ok(())
        } else {
            Err(PyKeyError::new_err(instance_id.to_string()))
        }
    }

    /// `("probabilistic", grid, values)`, `("expect_error", expected,
    /// error)` or None.
    fn sketch(&self, py: Python<'_>, instance_id: &str) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .sketch(instance_id)
            .map(|s| sketch_tuple(py, s))
            .transpose()
    }

    #[pyo3(signature = (task_id, instance_id, time_ms, error_ms = None, quantiles = None))]
    fn charge(
        &mut self,
        task_id: Option<&str>,
        instance_id: &str,
        time_ms: f64,
        error_ms: Option<f64>,
        quantiles: Option<Pairs>,
    ) {
        self.inner.charge(
            task_id,
            Charge::of(instance_id, time_ms, error_ms, quantiles.as_deref()),
        );
    }

    /// Release a task; returns the instance it was charged to.
    fn release(&mut self, task_id: &str) -> Option<String> {
        self.inner.release(task_id).map(|c| c.instance_id)
    }

    fn move_charge(&mut self, task_id: &str, instance_id: &str) -> bool {
        self.inner.move_charge(task_id, instance_id)
    }

    fn set_capacity(&mut self, instance_id: &str, servers: i64) {
        self.inner.set_capacity(instance_id, servers);
    }

    fn capacity(&self, instance_id: &str) -> u32 {
        self.inner.capacity(instance_id)
    }

    #[pyo3(signature = (kind, grid = None))]
    fn reset(&mut self, kind: &str, grid: Option<Vec<f64>>) {
        self.inner
            .reset(SketchKind::from_name(kind), grid.as_deref());
    }
}

/// A trained runtime predictor (`ExpectErrorPredictor` or
/// `QuantilePredictor`) evaluated in Rust.
#[pyclass(frozen, module = "swarmpilot_core")]
struct RuntimePredictor {
    inner: CorePredictor,
}

fn model_err(e: swarmpilot_core::mlp::ModelError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pymethods]
impl RuntimePredictor {
    /// Build from `get_model_state()`: `state_json` is every field but
    /// the state dict, `tensors` maps state-dict names to `(shape,
    /// little-endian float32 bytes)`.
    #[staticmethod]
    #[pyo3(signature = (state_json, tensors, norm_f32 = false))]
    fn from_state(
        state_json: &str,
        tensors: HashMap<String, (Vec<usize>, Vec<u8>)>,
        norm_f32: bool,
    ) -> PyResult<Self> {
        let state: serde_json::Value =
            serde_json::from_str(state_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut parsed = HashMap::new();
        for (name, (shape, bytes)) in tensors {
            if bytes.len() % 4 != 0 {
                return Err(PyValueError::new_err(format!("{name}: not float32 bytes")));
            }
            let data = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            parsed.insert(name, Tensor { shape, data });
        }
        CorePredictor::from_state(&state, &parsed, norm_f32)
            .map(|inner| RuntimePredictor { inner })
            .map_err(model_err)
    }

    /// Load the portable model file written by `ModelStorage`.
    #[staticmethod]
    fn load_portable(sidecar: std::path::PathBuf) -> PyResult<Self> {
        CorePredictor::load_portable(&sidecar)
            .map(|inner| RuntimePredictor { inner })
            .map_err(model_err)
    }

    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.inner.feature_names().to_vec()
    }

    /// `("expect_error", expected_ms, error_ms)` or `("quantile",
    /// [(q, ms), ...])` for float32 feature values in model order.
    fn predict(&self, py: Python<'_>, values: Vec<f32>) -> PyResult<Py<PyAny>> {
        Ok(match self.inner.predict(&values).map_err(model_err)? {
            Output::ExpectError {
                expected_ms,
                error_ms,
            } => ("expect_error", expected_ms, error_ms)
                .into_pyobject(py)?
                .into_any()
                .unbind(),
            Output::Quantiles(q) => ("quantile", q).into_pyobject(py)?.into_any().unbind(),
        })
    }
}

#[pymodule]
#[pyo3(name = "swarmpilot_core")]
fn init_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Extractor>()?;
    m.add_class::<Ledger>()?;
    m.add_class::<RuntimePredictor>()?;
    m.add_function(wrap_pyfunction!(charge_at, m)?)?;
    m.add_function(wrap_pyfunction!(apply_probabilistic, m)?)?;
    m.add_function(wrap_pyfunction!(apply_expect_error, m)?)?;
    m.add_function(wrap_pyfunction!(select_min_expected, m)?)?;
    m.add_function(wrap_pyfunction!(select_po2, m)?)?;
    m.add_function(wrap_pyfunction!(select_probabilistic, m)?)?;
    Ok(())
}
