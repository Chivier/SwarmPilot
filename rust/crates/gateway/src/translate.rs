//! API style translation on the data plane: the translated half of the
//! Python proxy's `_body_response` and `_stream_response`
//! (`swarmpilot/scheduler/proxy/router.py`), on `swarmpilot_core::api`.

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use futures_util::stream::{self, StreamExt};
use serde_json::{json, Value};
use swarmpilot_core::api::base::SystemEnv;
use swarmpilot_core::api::py;
use swarmpilot_core::api::translate::{Translation, DROPPED_HEADER};

use crate::dispatch::RelayStream;

/// Starlette's answer to an exception the handler did not catch.
pub fn internal_error() -> Response {
    let mut resp = Response::new(Body::from("Internal Server Error"));
    *resp.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
    resp.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    resp
}

/// A Starlette `media_type`: `text/*` types get `; charset=utf-8`.
pub fn media_type(mt: &str) -> HeaderValue {
    let v = if mt.starts_with("text/") && !mt.contains("charset=") {
        format!("{mt}; charset=utf-8")
    } else {
        mt.to_string()
    };
    HeaderValue::from_str(&v).unwrap_or_else(|_| HeaderValue::from_static("text/event-stream"))
}

/// Engine headers for a translated answer: filtered, without the
/// engine's content type, plus the dropped-field list. None when that
/// list cannot be a header (Python fails encoding it as latin-1).
fn headers_for(tr: &Translation, engine: &HeaderMap) -> Option<HeaderMap> {
    let mut out = HeaderMap::new();
    crate::gateway::copy_headers(engine, &mut out);
    out.remove("content-type");
    if let Some(dropped) = tr.dropped_header() {
        let latin1: Option<Vec<u8>> = dropped
            .chars()
            .map(|c| u8::try_from(u32::from(c)).ok())
            .collect();
        let value = HeaderValue::from_bytes(&latin1?).ok()?;
        out.insert(DROPPED_HEADER, value);
    }
    Some(out)
}

fn with_headers(mut resp: Response, status: u16, headers: HeaderMap) -> Response {
    *resp.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
    for (k, v) in &headers {
        resp.headers_mut().append(k.clone(), v.clone());
    }
    resp
}

/// An engine's one-body answer to a translated request.
///
/// `stream_job`: the engine was asked to stream (the body then comes as
/// is; otherwise a falsy body becomes `{}`, as `result.result or {}`).
pub fn body(
    tr: &Translation,
    status: u16,
    engine_headers: &HeaderMap,
    raw: &Bytes,
    stream_job: bool,
) -> Response {
    let Some(mut headers) = headers_for(tr, engine_headers) else {
        return internal_error();
    };
    let parsed: Value = serde_json::from_slice(raw).unwrap_or_else(|_| json!({}));
    let body = if !stream_job && !py::truthy(&parsed) {
        json!({})
    } else {
        parsed
    };
    let mut env = SystemEnv;
    if let (true, Value::Object(b)) = (tr.stream() && status < 400, &body) {
        // The client asked for a stream; the engine answered with one body.
        let Ok((bytes, broken)) = tr.stream_from_body(b, &mut env) else {
            return internal_error();
        };
        headers.insert("content-type", media_type("text/event-stream"));
        let mut parts: Vec<Result<Bytes, std::io::Error>> = vec![Ok(Bytes::from(bytes))];
        if let Some(e) = broken {
            tracing::error!(error = %e, "translating the answer failed mid-stream");
            parts.push(Err(std::io::Error::other(e.0)));
        }
        return with_headers(
            Response::new(Body::from_stream(stream::iter(parts))),
            status,
            headers,
        );
    }
    match tr.response(status, &body, &mut env) {
        Ok((status, content)) => {
            headers.insert("content-type", HeaderValue::from_static("application/json"));
            with_headers(
                Response::new(Body::from(py::dumps(&content))),
                status,
                headers,
            )
        }
        Err(e) => {
            tracing::error!(error = %e, "translating the answer failed");
            internal_error()
        }
    }
}

enum Phase {
    Running,
    /// The client's stream is complete; read the engine's to its end.
    Draining,
    /// The encoder failed: cut the client's stream off.
    Broken(String),
    Done,
}

/// An engine's SSE answer to a translated request.
pub fn stream(
    tr: &Translation,
    status: u16,
    engine_headers: &HeaderMap,
    chunks: RelayStream,
) -> Response {
    let Some(mut headers) = headers_for(tr, engine_headers) else {
        return internal_error();
    };
    let ct = engine_headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("text/event-stream");
    headers.insert("content-type", media_type(ct));
    let st = tr.stream_translator();
    let body = stream::unfold(
        (chunks, st, Phase::Running),
        |(mut chunks, mut st, mut phase)| async move {
            let mut env = SystemEnv;
            loop {
                match phase {
                    Phase::Done => return None,
                    Phase::Broken(msg) => {
                        return Some((Err(std::io::Error::other(msg)), (chunks, st, Phase::Done)));
                    }
                    Phase::Draining => {
                        while let Some(Ok(_)) = chunks.next().await {}
                        return None;
                    }
                    Phase::Running => {
                        let (out, ended) = match chunks.next().await {
                            Some(Ok(b)) => (st.feed(&b, &mut env), false),
                            Some(Err(e)) => {
                                tracing::warn!(error = %e, "engine stream failed");
                                (st.fail(&mut env), true)
                            }
                            None => (st.end(&mut env), true),
                        };
                        phase = match (st.finished(), st.broken()) {
                            (_, Some(e)) => {
                                tracing::error!(error = %e, "translating the stream failed");
                                Phase::Broken(e.0.clone())
                            }
                            _ if ended => Phase::Done,
                            (true, None) => Phase::Draining,
                            (false, None) => Phase::Running,
                        };
                        if !out.is_empty() {
                            return Some((Ok(Bytes::from(out)), (chunks, st, phase)));
                        }
                    }
                }
            }
        },
    );
    with_headers(Response::new(Body::from_stream(body)), status, headers)
}
