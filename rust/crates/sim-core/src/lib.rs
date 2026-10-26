//! Core models of the SwarmPilot cluster simulator.
//!
//! - [`catalog`]: GPU, model and engine data from `swarmpilot/sim/data/`.
//! - [`profile`]: an (engine, model, GPU, TP) combination turned into the
//!   numbers an engine model runs on.
//! - [`engine`]: event-driven engine models (continuous-batching LLM,
//!   diffusion, generic) behind one [`engine::EngineModel`] trait.
//! - [`spec`]: cluster specs and instance placement.
//! - [`workload`]: arrival processes, length distributions, trace replay.
//! - [`metrics`]: per-request records and the summary row.
//! - [`runner`]: drive one engine with a request stream.

pub mod catalog;
pub mod engine;
pub mod metrics;
pub mod profile;
pub mod runner;
pub mod spec;
pub mod workload;

pub use catalog::Catalog;
pub use profile::EngineProfile;
pub use spec::ClusterSpec;

/// Errors from loading or validating simulator inputs.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("yaml: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("unknown {0}: {1}")]
    Unknown(&'static str, String),
    #[error("{0}")]
    Invalid(String),
    #[error("{model} does not fit on {tp} x {gpu}")]
    DoesNotFit { model: String, gpu: String, tp: u32 },
    #[error("trace: {0}")]
    Trace(String),
}
