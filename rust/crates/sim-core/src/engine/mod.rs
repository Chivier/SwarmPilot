//! Event-driven engine models. Time is in milliseconds on whatever clock
//! the caller runs: virtual in the discrete-event sim, wall clock in the
//! emulated cluster.
//!
//! The caller submits requests, asks for the time of the engine's next
//! internal event, and advances the engine to a time, collecting the
//! events that happened up to it.

mod diffusion;
mod generic;
mod llm;

pub use diffusion::DiffusionEngine;
pub use generic::GenericEngine;
pub use llm::LlmEngine;

use serde::{Deserialize, Serialize};

use crate::profile::{EngineProfile, ProfileKind};

/// What a request asks for.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Work {
    /// Text generation: prompt length and number of tokens to generate.
    Llm {
        prompt_tokens: u32,
        output_tokens: u32,
    },
    /// Image generation: denoising steps and output size.
    Image { steps: u32, width: u32, height: u32 },
    /// An opaque call to a generic engine.
    Generic,
}

/// One request handed to an engine.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Request {
    pub id: u64,
    pub work: Work,
}

/// Something an engine did for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// First output token (LLM only).
    FirstToken,
    /// A later output token (LLM only, when token events are on).
    Token,
    /// The request finished.
    Done,
    /// The request was rejected (wrong kind of work for this engine).
    Rejected,
}

/// An engine event at time `t_ms`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub id: u64,
    pub t_ms: f64,
    pub kind: EventKind,
}

/// Live load numbers an engine would report on `/metrics`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Load {
    pub running: u32,
    pub waiting: u32,
    /// Fraction of KV cache (LLM) or batch slots in use.
    pub utilization: f64,
}

/// A simulated engine instance.
pub trait EngineModel: Send {
    /// Hand the engine a request at time `now`.
    fn submit(&mut self, now: f64, req: Request);

    /// Time of the next internal event, or `None` when idle.
    fn next_event(&self) -> Option<f64>;

    /// Process everything due at or before `now`, appending events.
    fn advance(&mut self, now: f64, out: &mut Vec<Event>);

    /// Current load.
    fn load(&self) -> Load;
}

/// Build the engine model for a profile. `seed` drives any randomness
/// (only the generic engine samples).
pub fn build(profile: &EngineProfile, seed: u64, emit_tokens: bool) -> Box<dyn EngineModel> {
    match &profile.kind {
        ProfileKind::Llm(p) => Box::new(LlmEngine::new(p.clone(), emit_tokens)),
        ProfileKind::Diffusion(p) => Box::new(DiffusionEngine::new(p.clone())),
        ProfileKind::Generic(p) => Box::new(GenericEngine::new(p.clone(), seed)),
    }
}
