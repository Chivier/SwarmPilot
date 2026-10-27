//! Translate one request between a client's API style and an engine's
//! (`swarmpilot/api/translate.py`).

use serde_json::{Map, Value};

use super::anthropic_messages::AnthropicMessages;
use super::base::{Context, Env, TrError, TrResult};
use super::canonical::{events_from_response, Event, Request};
use super::openai_chat::OpenAiChat;
use super::openai_responses::OpenAiResponses;
use super::py::{PyErr, PyResult};
use super::sse::SseParser;
use super::style::{Decoder, Encoder, Flow, Style};

pub const DROPPED_HEADER: &str = "x-swarmpilot-dropped";

/// The built-in styles, in engine preference order.
static STYLES: [&dyn Style; 3] = [&OpenAiChat, &AnthropicMessages, &OpenAiResponses];

/// The style called `name`.
pub fn find_style(name: &str) -> Option<&'static dyn Style> {
    STYLES.iter().copied().find(|s| s.name() == name)
}

/// The style that owns `path` (leading slashes ignored).
pub fn style_for_path(path: &str) -> Option<&'static dyn Style> {
    let path = path.trim_start_matches('/');
    STYLES.iter().copied().find(|s| s.path() == path)
}

/// How one request travels to the engine and its answer back.
pub struct Translation {
    pub client: &'static dyn Style,
    pub engine: &'static dyn Style,
    /// None when the body passes through.
    pub request: Option<Request>,
    pub path: String,
    pub body: Value,
    pub ctx: Context,
}

impl Translation {
    pub fn passthrough(&self) -> bool {
        self.request.is_none()
    }

    /// Whether the client asked for a stream.
    pub fn stream(&self) -> bool {
        match &self.request {
            Some(r) => r.stream,
            None => self.body.get("stream") == Some(&Value::Bool(true)),
        }
    }

    /// The `x-swarmpilot-dropped` value, if anything was dropped.
    pub fn dropped_header(&self) -> Option<String> {
        (!self.ctx.dropped.is_empty()).then(|| self.ctx.dropped.join(","))
    }

    /// An engine's non-streamed answer in the client's style.
    pub fn response(&self, status: u16, body: &Value, env: &mut dyn Env) -> PyResult<(u16, Value)> {
        let Some(req) = &self.request else {
            return Ok((status, body.clone()));
        };
        if status >= 400 {
            let msg = self.engine.decode_error(body);
            return Ok((status, self.client.encode_error(status, &msg, None, None)));
        }
        let Value::Object(b) = body else {
            return Ok((
                502,
                self.client.encode_error(
                    502,
                    "Engine returned a malformed response",
                    Some("bad_gateway"),
                    None,
                ),
            ));
        };
        let resp = self.engine.decode_response(b, env)?;
        Ok((status, self.client.encode_response(&resp, req, env)?))
    }

    /// A complete answer streamed to a client that asked for a stream: the
    /// bytes, and the error that broke the stream after them, if any.
    pub fn stream_from_body(
        &self,
        body: &Map<String, Value>,
        env: &mut dyn Env,
    ) -> PyResult<(Vec<u8>, Option<PyErr>)> {
        let req = self.request.as_ref().expect("translated request");
        let resp = self.engine.decode_response(body, env)?;
        let mut enc = self.client.encoder(req, env);
        let mut out = Vec::new();
        for ev in events_from_response(&resp) {
            match enc.on(&ev, env, &mut out) {
                Ok(true) => break,
                Ok(false) => {}
                Err(e) => return Ok((out, Some(e))),
            }
        }
        Ok((out, None))
    }

    /// The stream converter for an engine's SSE answer.
    pub fn stream_translator(&self) -> StreamTranslator {
        StreamTranslator {
            client: self.client,
            request: self.request.clone().expect("translated request"),
            parser: SseParser::default(),
            decoder: self.engine.decoder(),
            encoder: None,
            decoder_done: false,
            finished: false,
            broken: None,
        }
    }

    /// SwarmPilot's own error in the client's style.
    pub fn error(&self, status: u16, message: &str, error_type: Option<&str>) -> Value {
        self.client.encode_error(status, message, error_type, None)
    }
}

/// An engine's SSE bytes in, the client's SSE bytes out. Mirrors the
/// Python generator chain `encode_stream(_guard(decode_stream(...)))`:
/// the client encoder starts before the first engine event is read, a
/// failing decoder becomes an `Engine stream failed` error event, and a
/// failing encoder breaks the client's stream.
pub struct StreamTranslator {
    client: &'static dyn Style,
    request: Request,
    parser: SseParser,
    decoder: Box<dyn Decoder>,
    encoder: Option<Box<dyn Encoder>>,
    decoder_done: bool,
    finished: bool,
    broken: Option<PyErr>,
}

impl StreamTranslator {
    /// The client's stream is complete: no more bytes follow.
    pub fn finished(&self) -> bool {
        self.finished
    }

    /// The encoder failed: the client's stream must be cut off.
    pub fn broken(&self) -> Option<&PyErr> {
        self.broken.as_ref()
    }

    fn start(&mut self, env: &mut dyn Env) {
        if self.encoder.is_none() {
            self.encoder = Some(self.client.encoder(&self.request, env));
        }
    }

    fn emit(&mut self, events: Vec<Event>, env: &mut dyn Env, out: &mut Vec<u8>) {
        let enc = self.encoder.as_mut().expect("started");
        for ev in events {
            if self.finished {
                return;
            }
            match enc.on(&ev, env, out) {
                Ok(done) => self.finished = done,
                Err(e) => {
                    self.broken = Some(e);
                    self.finished = true;
                }
            }
        }
    }

    fn on_data(&mut self, data: &str, env: &mut dyn Env, out: &mut Vec<u8>) {
        let mut events = Vec::new();
        let flow = self.decoder.on_data(data, env, &mut events);
        self.emit(events, env, out);
        match flow {
            Ok(Flow::Continue) => {}
            Ok(Flow::Break) => self.decoder_end(env, out),
            Ok(Flow::Return) => self.decoder_done = true,
            Err(_) => self.decoder_failed(env, out),
        }
    }

    fn decoder_end(&mut self, env: &mut dyn Env, out: &mut Vec<u8>) {
        if self.finished || self.decoder_done {
            return;
        }
        self.decoder_done = true;
        let mut events = Vec::new();
        let r = self.decoder.finish(env, &mut events);
        self.emit(events, env, out);
        if r.is_err() {
            self.decoder_failed(env, out);
        }
    }

    fn decoder_failed(&mut self, env: &mut dyn Env, out: &mut Vec<u8>) {
        self.decoder_done = true;
        self.emit(vec![Event::error("Engine stream failed")], env, out);
    }

    /// One chunk of the engine's body.
    pub fn feed(&mut self, chunk: &[u8], env: &mut dyn Env) -> Vec<u8> {
        let mut out = Vec::new();
        if self.finished {
            return out;
        }
        self.start(env);
        let mut events = Vec::new();
        self.parser.feed(chunk, &mut events);
        for (_, data) in events {
            if self.finished || self.decoder_done {
                break;
            }
            self.on_data(&data, env, &mut out);
        }
        if self.decoder_done {
            // The decoder is over, so is the client's stream.
            self.finished = true;
        }
        out
    }

    /// The engine's body ended.
    pub fn end(&mut self, env: &mut dyn Env) -> Vec<u8> {
        let mut out = Vec::new();
        if self.finished {
            return out;
        }
        self.start(env);
        let mut events = Vec::new();
        self.parser.finish(&mut events);
        for (_, data) in events {
            if self.finished || self.decoder_done {
                break;
            }
            self.on_data(&data, env, &mut out);
        }
        self.decoder_end(env, &mut out);
        self.finished = true;
        out
    }

    /// Reading the engine's body failed.
    pub fn fail(&mut self, env: &mut dyn Env) -> Vec<u8> {
        let mut out = Vec::new();
        if self.finished {
            return out;
        }
        self.start(env);
        if !self.decoder_done {
            self.decoder_failed(env, &mut out);
        }
        self.finished = true;
        out
    }
}

/// `plan_request`: how a POST to `path` reaches an engine that speaks
/// `native_styles`. None when the path belongs to no style or the engine
/// declares none.
pub fn plan_request(
    path: &str,
    body: &Value,
    native_styles: &[String],
    strict: bool,
    default_max_tokens: i128,
    env: &mut dyn Env,
) -> TrResult<Option<Translation>> {
    let Some(client) = style_for_path(path) else {
        return Ok(None);
    };
    if native_styles.is_empty() {
        return Ok(None);
    }
    let mut ctx = Context::new(strict, default_max_tokens);
    if native_styles.iter().any(|n| n == client.name()) {
        return Ok(Some(Translation {
            client,
            engine: client,
            request: None,
            path: path.to_string(),
            body: body.clone(),
            ctx,
        }));
    }
    let Value::Object(b) = body else {
        return Err(super::base::invalid(
            "The request body must be a JSON object",
            None,
        ));
    };
    let Some(engine) = engine_style(native_styles) else {
        return Err(TrError::Translation {
            message: format!("This model's engine cannot serve /{path}"),
            param: None,
            status: 404,
            error_type: "not_found_error".into(),
        });
    };
    let req = client.decode_request(b, &mut ctx, env)?;
    if let Some(ext) = req.extension(client.name()) {
        for key in ext.keys() {
            if client.handled_extensions().contains(&key.as_str()) {
                continue;
            }
            ctx.drop_field(key, client.semantic_extensions().contains(&key.as_str()))?;
        }
    }
    let (epath, ebody) = engine.encode_request(&req, &mut ctx)?;
    Ok(Some(Translation {
        client,
        engine,
        request: Some(req),
        path: epath,
        body: ebody,
        ctx,
    }))
}

/// The engine's preferred style; all built-in styles are of the `llm`
/// family. Unknown names are skipped, as `find_style` gives None.
fn engine_style(native_styles: &[String]) -> Option<&'static dyn Style> {
    STYLES
        .iter()
        .copied()
        .find(|s| native_styles.iter().any(|n| n == s.name()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::base::CountingEnv;
    use crate::api::py;
    use serde_json::json;

    fn env() -> CountingEnv {
        CountingEnv { next: 0, now: 7 }
    }

    #[test]
    fn messages_client_on_chat_engine() {
        let body = json!({"model": "m", "max_tokens": 5, "stream": true, "top_k": 3,
            "messages": [{"role": "user", "content": "hi"}]});
        let natives = vec!["openai_chat".to_string()];
        let tr = plan_request("/v1/messages", &body, &natives, true, 4096, &mut env())
            .unwrap()
            .unwrap();
        assert_eq!(tr.path, "v1/chat/completions");
        assert_eq!(
            py::dumps(&tr.body),
            r#"{"messages":[{"role":"user","content":"hi"}],"model":"m","max_tokens":5,"top_k":3,"stream":true,"stream_options":{"include_usage":true}}"#
        );
        // Split anywhere, the client gets the same bytes.
        let sse = b"data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"h\xc3\xa9\"}}]}\n\ndata: [DONE]\n\n";
        let whole = {
            let mut st = tr.stream_translator();
            let mut e = env();
            let mut out = st.feed(sse, &mut e);
            out.extend(st.end(&mut e));
            out
        };
        assert!(String::from_utf8_lossy(&whole)
            .ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
        for cut in 0..sse.len() {
            let mut st = tr.stream_translator();
            let mut e = env();
            let mut out = st.feed(&sse[..cut], &mut e);
            out.extend(st.feed(&sse[cut..], &mut e));
            out.extend(st.end(&mut e));
            assert_eq!(out, whole, "cut at {cut}");
        }
    }

    #[test]
    fn refused_and_unknown() {
        let natives = vec!["anthropic_messages".to_string()];
        let body = json!({"messages": [], "n": 2});
        match plan_request(
            "v1/chat/completions",
            &body,
            &natives,
            true,
            4096,
            &mut env(),
        ) {
            Err(TrError::Translation { param, status, .. }) => {
                assert_eq!((param.as_deref(), status), (Some("n"), 400))
            }
            _ => panic!("n=2 must be refused"),
        }
        assert!(
            plan_request("v1/other", &body, &natives, true, 4096, &mut env())
                .unwrap()
                .is_none()
        );
        let unknown = vec!["custom".to_string()];
        assert!(matches!(
            plan_request("v1/messages", &body, &unknown, true, 4096, &mut env()),
            Err(TrError::Translation { status: 404, .. })
        ));
    }
}
