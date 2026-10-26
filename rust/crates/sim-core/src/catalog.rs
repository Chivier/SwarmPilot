//! GPU, model and engine catalogs loaded from `swarmpilot/sim/data/*.yaml`.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::Error;

const GPUS_YAML: &str = include_str!("../../../../swarmpilot/sim/data/gpus.yaml");
const MODELS_YAML: &str = include_str!("../../../../swarmpilot/sim/data/models.yaml");
const ENGINES_YAML: &str = include_str!("../../../../swarmpilot/sim/data/engines.yaml");

/// Datasheet numbers for one GPU model.
#[derive(Debug, Clone, Deserialize)]
pub struct GpuSpec {
    pub memory_gb: f64,
    pub bandwidth_tbps: f64,
    pub tflops_bf16: f64,
    #[serde(default)]
    pub nvlink: bool,
    #[serde(default)]
    pub source: String,
}

/// One model: an LLM's shape, or a diffusion/generic latency anchor.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ModelSpec {
    Llm {
        params_b: f64,
        layers: u32,
        kv_heads: u32,
        head_dim: u32,
        #[serde(default)]
        aliases: Vec<String>,
    },
    Diffusion {
        reference_gpu: String,
        ms_per_step_1024: f64,
        vae_ms: f64,
        max_batch: u32,
        batch_alpha: f64,
        #[serde(default)]
        source: String,
        #[serde(default)]
        aliases: Vec<String>,
    },
    Generic {
        reference_gpu: String,
        mean_ms: f64,
        cv: f64,
        concurrency: u32,
        #[serde(default)]
        source: String,
        #[serde(default)]
        aliases: Vec<String>,
    },
}

impl ModelSpec {
    /// Other names the model goes by, such as Hugging Face repo ids.
    pub fn aliases(&self) -> &[String] {
        match self {
            ModelSpec::Llm { aliases, .. }
            | ModelSpec::Diffusion { aliases, .. }
            | ModelSpec::Generic { aliases, .. } => aliases,
        }
    }
}

/// Efficiency factors and limits for one engine. LLM engines use every
/// field; diffusion and generic engines only the lifecycle ones and
/// `flops_scaling_exponent`.
#[derive(Debug, Clone, Deserialize)]
pub struct EngineTuning {
    #[serde(default)]
    pub mbu: f64,
    #[serde(default)]
    pub mfu: f64,
    #[serde(default)]
    pub step_overhead_ms: f64,
    #[serde(default)]
    pub request_overhead_ms: f64,
    #[serde(default)]
    pub request_overhead_per_extra_rank_ms: f64,
    #[serde(default)]
    pub tp_comm_coef: f64,
    #[serde(default = "default_mem_util")]
    pub gpu_memory_utilization: f64,
    #[serde(default)]
    pub activation_reserve_gb: f64,
    #[serde(default = "default_block_size")]
    pub block_size: u32,
    #[serde(default = "default_max_num_seqs")]
    pub max_num_seqs: u32,
    #[serde(default = "default_max_batched_tokens")]
    pub max_num_batched_tokens: u32,
    #[serde(default)]
    pub cold_start_s: f64,
    #[serde(default = "default_load_gbps")]
    pub load_gbps: f64,
    #[serde(default = "default_flops_exp")]
    pub flops_scaling_exponent: f64,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub uncalibrated: bool,
}

fn default_mem_util() -> f64 {
    0.9
}
fn default_block_size() -> u32 {
    16
}
fn default_max_num_seqs() -> u32 {
    256
}
fn default_max_batched_tokens() -> u32 {
    8192
}
fn default_load_gbps() -> f64 {
    2.0
}
fn default_flops_exp() -> f64 {
    1.0
}

/// All three catalogs together.
#[derive(Debug, Clone)]
pub struct Catalog {
    pub gpus: BTreeMap<String, GpuSpec>,
    pub models: BTreeMap<String, ModelSpec>,
    pub engines: BTreeMap<String, EngineTuning>,
}

impl Catalog {
    /// The catalogs compiled into the binary.
    pub fn builtin() -> Self {
        Self::from_yaml(GPUS_YAML, MODELS_YAML, ENGINES_YAML)
            .expect("built-in simulator catalogs must parse")
    }

    /// Parse catalogs from YAML text.
    pub fn from_yaml(gpus: &str, models: &str, engines: &str) -> Result<Self, Error> {
        Ok(Self {
            gpus: serde_yaml::from_str(gpus)?,
            models: serde_yaml::from_str(models)?,
            engines: serde_yaml::from_str(engines)?,
        })
    }

    /// Load `gpus.yaml`, `models.yaml` and `engines.yaml` from a directory.
    pub fn from_dir(dir: &Path) -> Result<Self, Error> {
        let read = |name: &str| std::fs::read_to_string(dir.join(name));
        Self::from_yaml(
            &read("gpus.yaml")?,
            &read("models.yaml")?,
            &read("engines.yaml")?,
        )
    }

    pub fn gpu(&self, name: &str) -> Result<&GpuSpec, Error> {
        self.gpus
            .get(name)
            .ok_or_else(|| Error::Unknown("GPU", name.to_string()))
    }

    /// Look a model up by catalog name or by any of its aliases.
    pub fn model(&self, name: &str) -> Result<(&str, &ModelSpec), Error> {
        if let Some((k, m)) = self.models.get_key_value(name) {
            return Ok((k.as_str(), m));
        }
        self.models
            .iter()
            .find(|(_, m)| m.aliases().iter().any(|a| a == name))
            .map(|(k, m)| (k.as_str(), m))
            .ok_or_else(|| Error::Unknown("model", name.to_string()))
    }

    pub fn engine(&self, name: &str) -> Result<&EngineTuning, Error> {
        self.engines
            .get(name)
            .ok_or_else(|| Error::Unknown("engine", name.to_string()))
    }
}
