//! Engine performance profiles: catalog entries turned into the numbers an
//! engine model runs on, for one (engine, model, GPU, TP) combination.

use serde::Serialize;

use crate::catalog::{Catalog, ModelSpec};
use crate::Error;

/// What a simulated engine instance runs on.
#[derive(Debug, Clone, Serialize)]
pub struct EngineProfile {
    pub engine: String,
    pub model: String,
    pub gpu: String,
    pub tp: u32,
    /// Time from launch to healthy: engine init plus weight loading.
    pub cold_start_s: f64,
    pub source: String,
    pub uncalibrated: bool,
    pub kind: ProfileKind,
}

/// Engine-family specific parameters.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ProfileKind {
    Llm(LlmProfile),
    Diffusion(DiffusionProfile),
    Generic(GenericProfile),
}

/// Continuous-batching LLM engine.
#[derive(Debug, Clone, Serialize)]
pub struct LlmProfile {
    /// Fixed CPU and launch cost of every step.
    pub step_overhead_ms: f64,
    /// API server delay before a request reaches the scheduler.
    pub request_overhead_ms: f64,
    /// Time to stream the weights once (paid by every step).
    pub weight_ms: f64,
    /// Attention read cost per context token of each decoding sequence.
    pub kv_ms_per_token: f64,
    /// Compute cost per batched token (prefill or decode).
    pub compute_ms_per_token: f64,
    /// KV-cache capacity in tokens.
    pub kv_capacity_tokens: u64,
    pub block_size: u32,
    pub max_num_seqs: u32,
    pub max_num_batched_tokens: u32,
}

/// Diffusion engine: a step loop over a small batch.
#[derive(Debug, Clone, Serialize)]
pub struct DiffusionProfile {
    /// Denoising step time at 1024x1024, batch 1.
    pub ms_per_step_1024: f64,
    pub vae_ms: f64,
    pub max_batch: u32,
    /// Batch of n costs (1 + alpha * (n - 1)) times batch 1.
    pub batch_alpha: f64,
}

/// Opaque engine: a latency distribution and a concurrency limit.
#[derive(Debug, Clone, Serialize)]
pub struct GenericProfile {
    pub mean_ms: f64,
    pub cv: f64,
    pub concurrency: u32,
}

const BF16_BYTES: f64 = 2.0;

impl EngineProfile {
    /// Build the profile for `engine` serving `model` on `tp` GPUs of type
    /// `gpu`. Fails when a name is unknown, the engine does not serve that
    /// kind of model, or the model does not fit in memory.
    pub fn resolve(
        catalog: &Catalog,
        engine: &str,
        model: &str,
        gpu: &str,
        tp: u32,
    ) -> Result<Self, Error> {
        if tp == 0 {
            return Err(Error::Invalid("tp must be at least 1".into()));
        }
        let tuning = catalog.engine(engine)?;
        let g = catalog.gpu(gpu)?;
        let (model_name, m) = catalog.model(model)?;
        let tpf = tp as f64;

        let (kind, weight_gb) = match (engine, m) {
            (
                "vllm" | "sglang",
                ModelSpec::Llm {
                    params_b,
                    layers,
                    kv_heads,
                    head_dim,
                    ..
                },
            ) => {
                let bw = g.bandwidth_tbps * 1e12;
                let flops = g.tflops_bf16 * 1e12;
                let tp_eff = 1.0 / (1.0 + tuning.tp_comm_coef * (tpf - 1.0));
                let weight_bytes = params_b * 1e9 * BF16_BYTES / tpf;
                // KV heads are split across TP ranks, and replicated once
                // there are more ranks than heads.
                let kv_heads_per_gpu = (*kv_heads as f64 / tpf).ceil().max(1.0);
                let kv_bytes_per_token =
                    *layers as f64 * kv_heads_per_gpu * *head_dim as f64 * 2.0 * BF16_BYTES;
                let free = g.memory_gb * 1e9 * tuning.gpu_memory_utilization
                    - weight_bytes
                    - tuning.activation_reserve_gb * 1e9;
                if free <= 0.0 {
                    return Err(Error::DoesNotFit {
                        model: model_name.to_string(),
                        gpu: gpu.to_string(),
                        tp,
                    });
                }
                let p = LlmProfile {
                    step_overhead_ms: tuning.step_overhead_ms,
                    request_overhead_ms: tuning.request_overhead_ms
                        + tuning.request_overhead_per_extra_rank_ms * (tpf - 1.0),
                    weight_ms: weight_bytes / (bw * tuning.mbu * tp_eff) * 1e3,
                    kv_ms_per_token: kv_bytes_per_token / (bw * tuning.mbu) * 1e3,
                    compute_ms_per_token: 2.0 * params_b * 1e9
                        / tpf
                        / (flops * tuning.mfu * tp_eff)
                        * 1e3,
                    kv_capacity_tokens: (free / kv_bytes_per_token) as u64,
                    block_size: tuning.block_size.max(1),
                    max_num_seqs: tuning.max_num_seqs,
                    max_num_batched_tokens: tuning.max_num_batched_tokens,
                };
                (ProfileKind::Llm(p), weight_bytes / 1e9)
            }
            (
                "diffusers",
                ModelSpec::Diffusion {
                    reference_gpu,
                    ms_per_step_1024,
                    vae_ms,
                    max_batch,
                    batch_alpha,
                    ..
                },
            ) => {
                let s = flops_scale(catalog, reference_gpu, gpu, tuning.flops_scaling_exponent)?;
                let p = DiffusionProfile {
                    ms_per_step_1024: ms_per_step_1024 * s,
                    vae_ms: vae_ms * s,
                    max_batch: *max_batch,
                    batch_alpha: *batch_alpha,
                };
                // SDXL-class pipelines are around 7 GB in FP16.
                (ProfileKind::Diffusion(p), 7.0)
            }
            (
                "generic",
                ModelSpec::Generic {
                    reference_gpu,
                    mean_ms,
                    cv,
                    concurrency,
                    ..
                },
            ) => {
                let s = flops_scale(catalog, reference_gpu, gpu, tuning.flops_scaling_exponent)?;
                let p = GenericProfile {
                    mean_ms: mean_ms * s,
                    cv: *cv,
                    concurrency: *concurrency,
                };
                (ProfileKind::Generic(p), 1.0)
            }
            _ => {
                return Err(Error::Invalid(format!(
                    "engine {engine} cannot serve model {model_name}"
                )))
            }
        };

        let model_source = match m {
            ModelSpec::Diffusion { source, .. } | ModelSpec::Generic { source, .. } => {
                source.as_str()
            }
            ModelSpec::Llm { .. } => "",
        };
        let source = if model_source.is_empty() {
            format!("{}; {}", tuning.source, g.source)
        } else {
            format!("{}; {}; {}", model_source, tuning.source, g.source)
        };
        Ok(Self {
            engine: engine.to_string(),
            model: model_name.to_string(),
            gpu: gpu.to_string(),
            tp,
            cold_start_s: tuning.cold_start_s + weight_gb / tuning.load_gbps.max(1e-9),
            source,
            uncalibrated: tuning.uncalibrated || gpu_needs_scaling(engine, gpu),
            kind,
        })
    }

    /// Single-request latency in ms at an otherwise idle engine, for a
    /// quick sanity check of a profile.
    pub fn idle_latency_ms(&self, prompt_tokens: u32, output_tokens: u32, steps: u32) -> f64 {
        match &self.kind {
            ProfileKind::Llm(p) => {
                let prefill = p.request_overhead_ms
                    + p.step_overhead_ms
                    + (p.weight_ms).max(prompt_tokens as f64 * p.compute_ms_per_token);
                let decode: f64 = (1..output_tokens.max(1))
                    .map(|i| {
                        p.step_overhead_ms
                            + p.weight_ms
                            + (prompt_tokens + i) as f64 * p.kv_ms_per_token
                    })
                    .sum();
                prefill + decode
            }
            ProfileKind::Diffusion(p) => steps as f64 * p.ms_per_step_1024 + p.vae_ms,
            ProfileKind::Generic(p) => p.mean_ms,
        }
    }
}

/// The vLLM fit points are all H100; every other GPU is extrapolated from
/// datasheet ratios, so its profile is marked uncalibrated.
fn gpu_needs_scaling(engine: &str, gpu: &str) -> bool {
    matches!(engine, "vllm" | "sglang") && gpu != "H100-SXM-80GB"
}

fn flops_scale(catalog: &Catalog, reference: &str, gpu: &str, exp: f64) -> Result<f64, Error> {
    let r = catalog.gpu(reference)?;
    let g = catalog.gpu(gpu)?;
    Ok((r.tflops_bf16 / g.tflops_bf16).powf(exp))
}
