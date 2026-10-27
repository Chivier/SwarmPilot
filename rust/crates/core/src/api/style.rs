//! The style interface (`ApiStyle` in `swarmpilot/api/base.py`).

use serde_json::{Map, Value};

use super::base::{Context, Env, TrResult};
use super::canonical::{Event, Request, Response};
use super::py::{self, PyResult};

/// What a stream decoder does after one SSE event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    /// Read the next event.
    Continue,
    /// Stop reading and finish the stream (`break`).
    Break,
    /// The stream is over, nothing follows (`return`).
    Return,
}

/// An engine's SSE events to canonical events.
pub trait Decoder: Send {
    fn on_data(&mut self, data: &str, env: &mut dyn Env, out: &mut Vec<Event>) -> PyResult<Flow>;
    /// The engine's stream ended or the decoder broke out of it.
    fn finish(&mut self, env: &mut dyn Env, out: &mut Vec<Event>) -> PyResult<()>;
}

/// Canonical events to a client's SSE bytes.
pub trait Encoder: Send {
    /// Encode one event; true when the client's stream is complete.
    fn on(&mut self, ev: &Event, env: &mut dyn Env, out: &mut Vec<u8>) -> PyResult<bool>;
}

pub trait Style: Sync {
    fn name(&self) -> &'static str;
    /// The route this style owns (and generates on).
    fn path(&self) -> &'static str;
    /// Extension fields that change the answer when left out.
    fn semantic_extensions(&self) -> &'static [&'static str] {
        &[]
    }
    /// Extension fields the style itself honours on the way back.
    fn handled_extensions(&self) -> &'static [&'static str] {
        &[]
    }

    fn decode_request(
        &self,
        body: &Map<String, Value>,
        ctx: &mut Context,
        env: &mut dyn Env,
    ) -> TrResult<Request>;
    fn encode_response(&self, resp: &Response, req: &Request, env: &mut dyn Env)
        -> PyResult<Value>;
    /// Start a client stream (the generator's first step, which takes
    /// its ids and timestamps).
    fn encoder(&self, req: &Request, env: &mut dyn Env) -> Box<dyn Encoder>;
    fn encode_error(
        &self,
        status: u16,
        message: &str,
        error_type: Option<&str>,
        param: Option<&str>,
    ) -> Value;

    fn encode_request(&self, req: &Request, ctx: &mut Context) -> TrResult<(String, Value)>;
    fn decode_response(&self, body: &Map<String, Value>, env: &mut dyn Env) -> PyResult<Response>;
    fn decoder(&self) -> Box<dyn Decoder>;

    /// `decode_error`: the message of an engine's error body.
    fn decode_error(&self, body: &Value) -> String {
        if let Value::Object(b) = body {
            let e = py::get(b, "error");
            if let Value::Object(eo) = e {
                if py::truthy(py::get(eo, "message")) {
                    return py::str_of(py::get(eo, "message"));
                }
            }
            if let Value::String(s) = e {
                return s.clone();
            }
            if py::truthy(py::get(b, "message")) {
                return py::str_of(py::get(b, "message"));
            }
            if py::truthy(py::get(b, "detail")) {
                return py::str_of(py::get(b, "detail"));
            }
        }
        "Engine request failed".into()
    }
}
