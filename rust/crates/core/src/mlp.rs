//! Runtime predictor inference: the MLPs of `swarmpilot/predictor/`.
//!
//! [`RuntimePredictor`] reproduces `ExpectErrorPredictor.predict` and
//! `QuantilePredictor.predict` after the feature dict has been turned into
//! an ordered vector: z-score normalisation, the MLP (Linear + ReLU
//! layers), and for quantile models the base + softplus-delta transform,
//! target de-normalisation and the optional exp(). Arithmetic follows the
//! dtypes torch and numpy use (float32 network, float64 cumsum
//! accumulator), so results agree with torch to about 1e-6 relative; the
//! remaining difference is the summation order inside torch's matmul.
//!
//! Models load from the portable file the trainer writes (plan item 1.7):
//! `{key}.safetensors` plus the `{key}.json` sidecar.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;
use sha2::{Digest, Sha256};

/// A model file or state could not be used.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Invalid(String),
}

fn invalid<T>(msg: impl Into<String>) -> Result<T, ModelError> {
    Err(ModelError::Invalid(msg.into()))
}

/// A float32 tensor: shape and row-major values.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Read every tensor of a safetensors file. Only F32 tensors are
/// accepted; the trainer writes nothing else.
pub fn read_safetensors(bytes: &[u8]) -> Result<HashMap<String, Tensor>, ModelError> {
    if bytes.len() < 8 {
        return invalid("safetensors: file too short");
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let header_end = 8u64
        .checked_add(n)
        .filter(|&e| e <= bytes.len() as u64)
        .ok_or_else(|| ModelError::Invalid("safetensors: bad header length".into()))?
        as usize;
    let header: serde_json::Map<String, Value> = serde_json::from_slice(&bytes[8..header_end])?;
    let data = &bytes[header_end..];
    let mut out = HashMap::new();
    for (name, info) in header {
        if name == "__metadata__" {
            continue;
        }
        if info["dtype"] != "F32" {
            return invalid(format!("safetensors: {name} is {}, not F32", info["dtype"]));
        }
        let shape: Vec<usize> = serde_json::from_value(info["shape"].clone())?;
        let offsets: [usize; 2] = serde_json::from_value(info["data_offsets"].clone())?;
        let count = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
        let (start, end) = (offsets[0], offsets[1]);
        if start > end
            || end > data.len()
            || count.and_then(|c| c.checked_mul(4)) != Some(end - start)
        {
            return invalid(format!("safetensors: bad extent for {name}"));
        }
        let values = data[start..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        out.insert(
            name,
            Tensor {
                shape,
                data: values,
            },
        );
    }
    Ok(out)
}

/// One `nn.Linear` layer.
#[derive(Clone, Debug)]
struct Linear {
    in_dim: usize,
    out_dim: usize,
    /// `out_dim x in_dim`, row-major, as torch stores it.
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Linear {
    fn forward(&self, x: &[f32], relu: bool) -> Vec<f32> {
        (0..self.out_dim)
            .map(|o| {
                let row = &self.weight[o * self.in_dim..(o + 1) * self.in_dim];
                let mut acc = 0.0f32;
                for (w, v) in row.iter().zip(x) {
                    acc += w * v;
                }
                let y = acc + self.bias[o];
                if relu {
                    y.max(0.0)
                } else {
                    y
                }
            })
            .collect()
    }
}

/// The `MLP` module: Linear layers with ReLU between them.
#[derive(Clone, Debug)]
pub struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    /// Build from `model_config` and the `network.{i}.weight/bias`
    /// tensors of a torch state dict.
    pub fn from_tensors(
        config: &Value,
        tensors: &HashMap<String, Tensor>,
    ) -> Result<Self, ModelError> {
        let dim = |k: &str| {
            config[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| ModelError::Invalid(format!("model_config.{k} missing")))
        };
        let hidden: Vec<usize> = serde_json::from_value(config["hidden_layers"].clone())?;
        let mut dims = vec![dim("input_dim")?];
        dims.extend(&hidden);
        dims.push(dim("output_dim")?);
        let mut layers = Vec::new();
        for (k, pair) in dims.windows(2).enumerate() {
            let (in_dim, out_dim) = (pair[0], pair[1]);
            // nn.Sequential numbers Linear layers 0, 2, 4, ... (ReLUs between).
            let get = |part: &str, shape: &[usize]| -> Result<Vec<f32>, ModelError> {
                let name = format!("network.{}.{part}", 2 * k);
                match tensors.get(&name) {
                    Some(t) if t.shape == shape => Ok(t.data.clone()),
                    Some(t) => invalid(format!("{name} has shape {:?}, want {shape:?}", t.shape)),
                    None => invalid(format!("tensor {name} missing")),
                }
            };
            layers.push(Linear {
                in_dim,
                out_dim,
                weight: get("weight", &[out_dim, in_dim])?,
                bias: get("bias", &[out_dim])?,
            });
        }
        if tensors.len() != 2 * layers.len() {
            return invalid("unexpected extra tensors");
        }
        Ok(Mlp { layers })
    }

    pub fn input_dim(&self) -> usize {
        self.layers[0].in_dim
    }

    pub fn output_dim(&self) -> usize {
        self.layers[self.layers.len() - 1].out_dim
    }

    /// Run the network on one input row.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let last = self.layers.len() - 1;
        let mut h = x.to_vec();
        for (k, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, k != last);
        }
        h
    }
}

/// What a predictor outputs.
#[derive(Clone, Debug)]
enum Head {
    ExpectError {
        mean_error: f64,
    },
    Quantile {
        quantiles: Vec<f64>,
        delta_scale: f32,
        target_mean: f32,
        target_std: f32,
        log_transform: bool,
    },
}

/// A prediction.
#[derive(Clone, Debug, PartialEq)]
pub enum Output {
    ExpectError {
        expected_ms: f64,
        error_ms: f64,
    },
    /// `(quantile, runtime_ms)` in quantile order.
    Quantiles(Vec<(f64, f64)>),
}

/// torch's softplus (beta 1, threshold 20) in float32.
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// A trained runtime predictor.
#[derive(Clone, Debug)]
pub struct RuntimePredictor {
    mlp: Mlp,
    head: Head,
    feature_names: Vec<String>,
    feature_mean: Vec<f64>,
    feature_std: Vec<f64>,
    norm_f32: bool,
}

fn f64_list(state: &Value, key: &str) -> Result<Vec<f64>, ModelError> {
    serde_json::from_value(state[key].clone())
        .map_err(|e| ModelError::Invalid(format!("state.{key}: {e}")))
}

fn f64_field(state: &Value, key: &str) -> Result<f64, ModelError> {
    state[key]
        .as_f64()
        .ok_or_else(|| ModelError::Invalid(format!("state.{key} missing")))
}

impl RuntimePredictor {
    /// Build from a predictor's `get_model_state()` (without the state
    /// dict) and its tensors.
    ///
    /// `norm_f32` says the feature mean and std are float32 arrays, as
    /// right after training; a model loaded from disk holds float64 ones.
    pub fn from_state(
        state: &Value,
        tensors: &HashMap<String, Tensor>,
        norm_f32: bool,
    ) -> Result<Self, ModelError> {
        let mlp = Mlp::from_tensors(&state["model_config"], tensors)?;
        let feature_names: Vec<String> = serde_json::from_value(state["feature_names"].clone())?;
        let feature_mean = f64_list(state, "feature_mean")?;
        let feature_std = f64_list(state, "feature_std")?;
        let n = feature_names.len();
        if feature_mean.len() != n || feature_std.len() != n || mlp.input_dim() != n {
            return invalid("feature names, scaling and input_dim disagree");
        }
        let head = if state.get("quantiles").is_some_and(|q| !q.is_null()) {
            let quantiles = f64_list(state, "quantiles")?;
            if quantiles.len() != mlp.output_dim() {
                return invalid("quantiles and output_dim disagree");
            }
            Head::Quantile {
                quantiles,
                delta_scale: state
                    .get("delta_scale")
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0) as f32,
                target_mean: f64_field(state, "target_mean")? as f32,
                target_std: f64_field(state, "target_std")? as f32,
                log_transform: state
                    .get("log_transform_enabled")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            }
        } else {
            if mlp.output_dim() != 1 {
                return invalid("an expect_error model has one output");
            }
            Head::ExpectError {
                mean_error: f64_field(state, "mean_error")?,
            }
        };
        Ok(RuntimePredictor {
            mlp,
            head,
            feature_names,
            feature_mean,
            feature_std,
            norm_f32,
        })
    }

    /// Load the portable model file (the JSON sidecar and the weights it
    /// names), checking the weights' SHA-256.
    pub fn load_portable(sidecar: &Path) -> Result<Self, ModelError> {
        let doc: Value = serde_json::from_slice(&std::fs::read(sidecar)?)?;
        if doc["format"] != "swarmpilot-predictor" || doc["format_version"] != 1 {
            return invalid("not a swarmpilot-predictor v1 file");
        }
        if doc.get("activation").is_some_and(|a| a != "relu") {
            return invalid("only relu models are supported");
        }
        let name = doc["weights_file"]
            .as_str()
            .filter(|n| !n.contains('/') && !n.contains('\\') && !n.starts_with('.'))
            .ok_or_else(|| ModelError::Invalid("bad weights_file".into()))?;
        let bytes = std::fs::read(sidecar.with_file_name(name))?;
        let digest: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if doc["weights_sha256"] != digest.as_str() {
            return invalid("weights do not match weights_sha256");
        }
        Self::from_state(&doc["state"], &read_safetensors(&bytes)?, false)
    }

    /// Feature names, in input order.
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Predict from feature values in `feature_names` order, already
    /// rounded to float32 as `np.array(..., dtype=float32)` does.
    pub fn predict(&self, x: &[f32]) -> Result<Output, ModelError> {
        if x.len() != self.feature_names.len() {
            return invalid("wrong number of features");
        }
        let input: Vec<f32> = x
            .iter()
            .zip(self.feature_mean.iter().zip(&self.feature_std))
            .map(|(&v, (&m, &s))| {
                if self.norm_f32 {
                    (v - m as f32) / s as f32
                } else {
                    ((f64::from(v) - m) / s) as f32
                }
            })
            .collect();
        let raw = self.mlp.forward(&input);
        Ok(match &self.head {
            Head::ExpectError { mean_error } => Output::ExpectError {
                expected_ms: f64::from(raw[0]),
                error_ms: *mean_error,
            },
            Head::Quantile {
                quantiles,
                delta_scale,
                target_mean,
                target_std,
                log_transform,
            } => {
                let base = raw[0];
                let mut acc = 0.0f64;
                let mut normed = vec![base];
                for &d in &raw[1..] {
                    // torch's CPU cumsum accumulates float32 in float64.
                    acc += f64::from(softplus(d) * delta_scale);
                    normed.push(base + acc as f32);
                }
                let e = std::f32::consts::E;
                Output::Quantiles(
                    quantiles
                        .iter()
                        .zip(normed)
                        .map(|(&q, p)| {
                            let mut v = p * target_std + target_mean;
                            if *log_transform {
                                v = e.powf(v);
                            }
                            (q, f64::from(v))
                        })
                        .collect(),
                )
            }
        })
    }

    /// Predict from a feature map, as the gateway will: every name the
    /// model was trained on must be present.
    pub fn predict_features(&self, features: &HashMap<String, f64>) -> Result<Output, ModelError> {
        let x = self
            .feature_names
            .iter()
            .map(|n| {
                features
                    .get(n)
                    .map(|&v| v as f32)
                    .ok_or_else(|| ModelError::Invalid(format!("missing feature {n}")))
            })
            .collect::<Result<Vec<f32>, _>>()?;
        self.predict(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tensors() -> HashMap<String, Tensor> {
        let t = |shape: Vec<usize>, data: Vec<f32>| Tensor { shape, data };
        HashMap::from([
            ("network.0.weight".into(), t(vec![2, 1], vec![1.0, -1.0])),
            ("network.0.bias".into(), t(vec![2], vec![0.0, 0.5])),
            (
                "network.2.weight".into(),
                t(vec![2, 2], vec![1.0, 1.0, 0.0, 2.0]),
            ),
            ("network.2.bias".into(), t(vec![2], vec![0.25, -30.0])),
        ])
    }

    #[test]
    fn quantile_head() {
        let state = json!({
            "model_config": {"input_dim": 1, "output_dim": 2, "hidden_layers": [2]},
            "feature_names": ["x"], "feature_mean": [1.0], "feature_std": [2.0],
            "quantiles": [0.5, 0.9], "target_mean": 10.0, "target_std": 2.0,
        });
        let p = RuntimePredictor::from_state(&state, &tensors(), false).unwrap();
        // x=5 -> z=2 -> hidden relu(2, -1.5)=(2, 0) -> raw (2.25, -30)
        let Output::Quantiles(q) = p.predict(&[5.0]).unwrap() else {
            panic!()
        };
        assert_eq!(q[0], (0.5, 14.5));
        // softplus(-30) is below float32 resolution at 2.25.
        assert_eq!(q[1], (0.9, 14.5));
    }

    #[test]
    fn rejects_bad_shapes() {
        let mut t = tensors();
        t.get_mut("network.2.bias").unwrap().shape = vec![3];
        let state = json!({
            "model_config": {"input_dim": 1, "output_dim": 2, "hidden_layers": [2]},
            "feature_names": ["x"], "feature_mean": [0.0], "feature_std": [1.0],
            "mean_error": 1.0,
        });
        assert!(RuntimePredictor::from_state(&state, &t, false).is_err());
        assert!(read_safetensors(&[1, 0, 0, 0, 0, 0, 0, 0xff]).is_err());
    }
}
