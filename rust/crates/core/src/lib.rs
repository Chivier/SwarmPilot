//! The SwarmPilot scheduling core, ported from the Python Scheduler.
//!
//! Every module mirrors one Python module and must behave the same way;
//! the golden vectors in `tests/golden/` and the differential tests in
//! `tests/core/` check that it does.
//!
//! - [`sketch`]: queue sketches, charges and the per-task charge ledger
//!   (`swarmpilot/scheduler/registry/queue_sketch.py` and the charge part
//!   of `instance_registry.py`).
//! - [`strategy`]: the seven routing strategies
//!   (`swarmpilot/scheduler/algorithms/`).
//! - [`extract`]: the declarative feature extractor grammar
//!   (`swarmpilot/engines/extractors.py`).
//! - [`mlp`]: runtime predictor inference (`swarmpilot/predictor/predictor/`)
//!   from the portable safetensors + JSON model file.
//! - [`interp`] and [`fsum`]: bit-compatible ports of `numpy.interp` and
//!   `math.fsum`, which the Python code relies on.

pub mod extract;
pub mod fsum;
pub mod interp;
pub mod mlp;
pub mod sketch;
pub mod strategy;

pub use extract::{Doc, Extractor, ExtractorError};
pub use sketch::{Charge, Ledger, Sketch, SketchKind, DEFAULT_QUANTILES};
pub use strategy::{Draws, Prediction, StrategyKind};
