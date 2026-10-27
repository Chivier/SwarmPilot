//! Experiment-mode predictions (`swarmpilot/predictor/utils/experiment.py`).
//!
//! A request that carries `exp_runtime`, or an instance whose platform is
//! all `"exp"`, gets a synthetic prediction instead of a trained model's.
//! The Python code draws 2,000 samples from `numpy.random.RandomState(42)`
//! every time; every one of those draws is `loc + scale * z` or
//! `exp(loc + scale * z)` for the same fixed sequence `z`, so the caller
//! passes that sequence in (`normals`, recorded once from numpy) and the
//! rest is recomputed here with the same arithmetic: `x ** 2` as `powf(2)`
//! (libm `pow`, as CPython), `numpy.maximum` (NaN wins) and
//! `numpy.quantile`'s linear method, and CPython 3.12's compensated
//! `sum()` (Python 3.11 sums naively and can differ in the last bit).

use serde_json::{Map, Value};

use crate::strategy::PredictError;

/// Samples experiment mode draws per prediction.
pub const SAMPLES: usize = 2000;

/// Quantile levels when the strategy passes none.
pub const DEFAULT_QUANTILES: [f64; 4] = [0.5, 0.9, 0.95, 0.99];

/// A Python number as the request's JSON gave it. The mixture formulas
/// mix ints and floats, and Python keeps int arithmetic exact, so the
/// difference shows in the last bit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Num {
    Int(i128),
    Float(f64),
}

impl Num {
    pub fn f(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Float(x) => x,
        }
    }

    fn mul(self, o: Num) -> Result<Num, PredictError> {
        Ok(match (self, o) {
            (Num::Int(a), Num::Int(b)) => {
                Num::Int(a.checked_mul(b).ok_or_else(|| err("integer overflow"))?)
            }
            (a, b) => Num::Float(a.f() * b.f()),
        })
    }

    fn sub(self, o: Num) -> Result<Num, PredictError> {
        Ok(match (self, o) {
            (Num::Int(a), Num::Int(b)) => {
                Num::Int(a.checked_sub(b).ok_or_else(|| err("integer overflow"))?)
            }
            (a, b) => Num::Float(a.f() - b.f()),
        })
    }

    /// `x ** 2`: exact for ints, libm `pow` for floats.
    fn square(self) -> Result<Num, PredictError> {
        Ok(match self {
            Num::Int(a) => Num::Int(a.checked_mul(a).ok_or_else(|| err("integer overflow"))?),
            Num::Float(x) => Num::Float(x.powf(2.0)),
        })
    }
}

/// One mode of a multimodal (`exp_modes`) distribution.
#[derive(Clone, Debug, PartialEq)]
pub struct Mode {
    pub mean: Num,
    pub weight: Num,
    pub cv: Num,
    pub skewness: f64,
}

/// The `exp_*` fields of one request.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub runtime: f64,
    pub cv: f64,
    pub skewness: f64,
    pub modes: Option<Vec<Mode>>,
}

/// What experiment mode predicts.
#[derive(Clone, Debug, PartialEq)]
pub enum Synthetic {
    ExpectError {
        expected_ms: f64,
        error_ms: f64,
    },
    /// `(quantile, value)` in the order asked for.
    Quantiles(Vec<(f64, f64)>),
}

fn err(msg: impl Into<String>) -> PredictError {
    PredictError(msg.into())
}

/// A Python number: int, float or bool.
fn number(v: &Value, what: &str) -> Result<f64, PredictError> {
    match v {
        Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| err(format!("{what} is out of range"))),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        _ => Err(err(format!("{what} must be numeric"))),
    }
}

fn get_or(obj: &Map<String, Value>, key: &str, default: f64) -> Result<f64, PredictError> {
    obj.get(key).map_or(Ok(default), |v| number(v, key))
}

/// A Python number keeping int-ness (bool is an int).
fn py_num(v: &Value, what: &str) -> Result<Num, PredictError> {
    match v {
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => Ok(Num::Int(i as i128)),
            (None, Some(u)) => Ok(Num::Int(u as i128)),
            _ => Ok(Num::Float(number(v, what)?)),
        },
        Value::Bool(b) => Ok(Num::Int(*b as i128)),
        _ => Err(err(format!("{what} must be numeric"))),
    }
}

fn num_or(obj: &Map<String, Value>, key: &str, default: f64) -> Result<Num, PredictError> {
    obj.get(key)
        .map_or(Ok(Num::Float(default)), |v| py_num(v, key))
}

/// Whether experiment mode applies (`is_experiment_mode`).
pub fn applies(fields: &Map<String, Value>, platform_is_exp: bool) -> bool {
    fields.contains_key("exp_runtime") || platform_is_exp
}

impl Params {
    /// Read the `exp_*` fields of a request body.
    pub fn parse(fields: &Map<String, Value>) -> Result<Self, PredictError> {
        let runtime = match fields.get("exp_runtime") {
            None => return Err(err("Experiment mode requires 'exp_runtime' in features")),
            Some(v) => number(v, "exp_runtime")?,
        };
        if runtime <= 0.0 || runtime.is_nan() {
            return Err(err(format!("exp_runtime must be positive, got {runtime}")));
        }
        let cv = get_or(fields, "exp_cv", 0.30)?;
        let skewness = get_or(fields, "exp_skewness", 0.0)?;
        // `modes is not None and len(modes) > 0`: an empty list, string or
        // object means no modes.
        let modes = match fields.get("exp_modes") {
            None | Some(Value::Null) => None,
            Some(Value::Array(items)) if !items.is_empty() => Some(
                items
                    .iter()
                    .map(|m| {
                        let m = m
                            .as_object()
                            .ok_or_else(|| err("exp_modes entries must be objects"))?;
                        Ok(Mode {
                            mean: py_num(
                                m.get("mean").ok_or_else(|| err("mode without mean"))?,
                                "mean",
                            )?,
                            weight: num_or(m, "weight", 1.0)?,
                            cv: num_or(m, "cv", 0.20)?,
                            skewness: get_or(m, "skewness", 0.0)?,
                        })
                    })
                    .collect::<Result<Vec<_>, PredictError>>()?,
            ),
            Some(Value::Array(_)) => None,
            Some(Value::String(s)) if s.is_empty() => None,
            Some(Value::Object(o)) if o.is_empty() => None,
            Some(_) => return Err(err("exp_modes must be a list")),
        };
        Ok(Params {
            runtime,
            cv,
            skewness,
            modes,
        })
    }

    /// `generate_experiment_prediction`.
    pub fn predict(
        &self,
        prediction_type: &str,
        quantiles: Option<&[f64]>,
        normals: &[f64],
    ) -> Result<Synthetic, PredictError> {
        match prediction_type {
            "expect_error" => {
                let cv = match &self.modes {
                    Some(modes) => mixture_cv(modes, self.cv)?,
                    None => self.cv,
                };
                Ok(Synthetic::ExpectError {
                    expected_ms: self.runtime,
                    error_ms: self.runtime * cv,
                })
            }
            "quantile" => {
                let levels = quantiles.unwrap_or(&DEFAULT_QUANTILES);
                let mut samples = self.samples(normals)?;
                for s in &mut samples {
                    // numpy.maximum(s, 1.0): NaN stays NaN.
                    if !s.is_nan() && *s < 1.0 {
                        *s = 1.0;
                    }
                }
                let values = np_quantile(samples, levels)?;
                Ok(Synthetic::Quantiles(
                    levels.iter().copied().zip(values).collect(),
                ))
            }
            other => Err(err(format!("Unknown prediction type: {other}"))),
        }
    }

    fn samples(&self, z: &[f64]) -> Result<Vec<f64>, PredictError> {
        if z.len() < SAMPLES {
            return Err(err("experiment mode draws are missing"));
        }
        if let Some(modes) = &self.modes {
            return mixture_samples(modes, z);
        }
        let mut out = Vec::with_capacity(SAMPLES);
        if self.skewness <= 0.0 {
            let sigma = self.runtime * self.cv;
            normal_scale(sigma)?;
            out.extend(z[..SAMPLES].iter().map(|z| self.runtime + sigma * z));
        } else {
            let (mu, sigma) = lognormal_params(self.runtime, self.cv, self.skewness);
            normal_scale(sigma)?;
            out.extend(z[..SAMPLES].iter().map(|z| (mu + sigma * z).exp()));
        }
        Ok(out)
    }
}

/// numpy's check on `scale`: negative or NaN raises.
fn normal_scale(scale: f64) -> Result<(), PredictError> {
    if scale.is_nan() || scale < 0.0 {
        return Err(err("scale < 0"));
    }
    Ok(())
}

/// `(mu_ln, sigma_ln)` of the log-normal with this mean, CV and skewness.
fn lognormal_params(mean: f64, cv: f64, skewness: f64) -> (f64, f64) {
    let effective_cv = cv * (1.0 + 0.3 * skewness);
    let sigma = (1.0 + effective_cv.powf(2.0)).ln().sqrt();
    (mean.ln() - sigma.powf(2.0) / 2.0, sigma)
}

/// CPython 3.12+ `sum()`: ints add exactly until the first float, which
/// is added plainly; from there on floats are added with Neumaier
/// compensation and ints plainly, as floats. Python 3.11 sums floats naively and can
/// differ in the last bit.
fn py_sum(values: impl Iterator<Item = Result<Num, PredictError>>) -> Result<Num, PredictError> {
    let mut int_sum: i128 = 0;
    let mut float: Option<(f64, f64)> = None;
    for v in values {
        let v = v?;
        match (&mut float, v) {
            (None, Num::Int(i)) => {
                int_sum = int_sum
                    .checked_add(i)
                    .ok_or_else(|| err("integer overflow"))?;
            }
            (None, Num::Float(x)) => float = Some((int_sum as f64 + x, 0.0)),
            (Some((total, _)), Num::Int(i)) => *total += i as f64,
            (Some((total, c)), Num::Float(x)) => {
                let t = *total + x;
                if total.abs() >= x.abs() {
                    *c += (*total - t) + x;
                } else {
                    *c += (x - t) + *total;
                }
                *total = t;
            }
        }
    }
    Ok(match float {
        None => Num::Int(int_sum),
        Some((total, c)) if c != 0.0 && c.is_finite() => Num::Float(total + c),
        Some((total, _)) => Num::Float(total),
    })
}

/// `generate_multimodal_samples` with numpy's draws in order.
fn mixture_samples(modes: &[Mode], z: &[f64]) -> Result<Vec<f64>, PredictError> {
    let total = py_sum(modes.iter().map(|m| Ok(m.weight)))?.f();
    let mut counts: Vec<i64> = modes
        .iter()
        .map(|m| (m.weight.f() / total * SAMPLES as f64) as i64)
        .collect();
    let diff = SAMPLES as i64 - counts.iter().sum::<i64>();
    if diff > 0 {
        counts[0] += diff;
    }
    let mut out = Vec::with_capacity(SAMPLES);
    let mut next = 0usize;
    for (m, &n) in modes.iter().zip(&counts) {
        if n <= 0 {
            continue;
        }
        let n = n as usize;
        if next + n > z.len() {
            return Err(err("exp_modes weights need more draws than recorded"));
        }
        let draws = &z[next..next + n];
        next += n;
        let mean = m.mean.f();
        if m.skewness <= 0.0 {
            let sigma = m.mean.mul(m.cv)?.f();
            normal_scale(sigma)?;
            out.extend(draws.iter().map(|z| mean + sigma * z));
        } else {
            let (mu, sigma) = lognormal_params(mean, m.cv.f(), m.skewness);
            normal_scale(sigma)?;
            out.extend(draws.iter().map(|z| (mu + sigma * z).exp()));
        }
    }
    // numpy shuffles the samples next; quantiles do not depend on order.
    Ok(out)
}

/// The expect-error CV of a mixture: within- plus between-mode spread.
fn mixture_cv(modes: &[Mode], cv: f64) -> Result<f64, PredictError> {
    let total = py_sum(modes.iter().map(|m| Ok(m.weight)))?.f();
    let mean = py_sum(modes.iter().map(|m| m.mean.mul(m.weight)))?.f() / total;
    let within = py_sum(
        modes
            .iter()
            .map(|m| m.weight.mul(m.mean.mul(m.cv)?.square()?)),
    )?
    .f() / total;
    let mean_num = Num::Float(mean);
    let between = py_sum(
        modes
            .iter()
            .map(|m| m.weight.mul(m.mean.sub(mean_num)?.square()?)),
    )?
    .f() / total;
    let std = (within + between).sqrt();
    Ok(if mean > 0.0 { std / mean } else { cv })
}

/// `numpy.quantile(samples, q)` with the default linear method.
pub fn np_quantile(mut samples: Vec<f64>, q: &[f64]) -> Result<Vec<f64>, PredictError> {
    if q.iter().any(|&x| !(0.0..=1.0).contains(&x)) {
        return Err(err("Quantiles must be in the range [0, 1]"));
    }
    if samples.is_empty() {
        return Err(err("no samples"));
    }
    if samples.iter().any(|s| s.is_nan()) {
        return Ok(vec![f64::NAN; q.len()]);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    let last = (n - 1) as f64;
    Ok(q.iter()
        .map(|&q| {
            let vi = last * q;
            let (prev, next) = if vi >= last {
                (n - 1, n - 1)
            } else {
                let p = vi.floor();
                (p as usize, p as usize + 1)
            };
            let gamma = vi - prev as f64;
            let (a, b) = (samples[prev], samples[next]);
            let diff = b - a;
            if gamma >= 0.5 {
                b - diff * (1.0 - gamma)
            } else {
                a + diff * gamma
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn quantile_matches_numpy_linear() {
        // numpy.quantile([1, 2, 3, 4], [0, 0.25, 0.5, 0.9, 1])
        let got = np_quantile(vec![4.0, 1.0, 3.0, 2.0], &[0.0, 0.25, 0.5, 0.9, 1.0]).unwrap();
        assert_eq!(got, vec![1.0, 1.75, 2.5, 3.7, 4.0]);
        assert!(np_quantile(vec![1.0], &[1.5]).is_err());
    }

    #[test]
    fn parse_follows_python_truthiness() {
        let f = |v: Value| Params::parse(v.as_object().unwrap());
        assert!(f(json!({})).is_err());
        assert!(f(json!({"exp_runtime": 0})).is_err());
        assert!(f(json!({"exp_runtime": "5"})).is_err());
        assert_eq!(f(json!({"exp_runtime": true})).unwrap().runtime, 1.0);
        assert_eq!(
            f(json!({"exp_runtime": 5, "exp_modes": []})).unwrap().modes,
            None
        );
        assert_eq!(
            f(json!({"exp_runtime": 5, "exp_modes": ""})).unwrap().modes,
            None
        );
        assert!(f(json!({"exp_runtime": 5, "exp_modes": "ab"})).is_err());
        assert!(f(json!({"exp_runtime": 5, "exp_modes": [{"weight": 1}]})).is_err());
    }

    #[test]
    fn expect_error_uses_cv() {
        let p = Params::parse(
            json!({"exp_runtime": 100, "exp_cv": 0.5})
                .as_object()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            p.predict("expect_error", None, &[]).unwrap(),
            Synthetic::ExpectError {
                expected_ms: 100.0,
                error_ms: 50.0
            }
        );
    }
}
