//! One virtual engine instance served over HTTP on the wall clock.
//!
//! The engine model from `sim-core` runs on virtual time; a driver task
//! sleeps until the model's next event, advances it, and hands events to
//! the waiting HTTP requests. `time_scale` maps virtual to wall time
//! (0.1 runs ten times faster than real).
//!
//! Routes follow what the engine adapters call:
//! - OpenAI: `POST /v1/chat/completions`, `POST /v1/completions` (JSON or
//!   SSE with `"stream": true`), `GET /v1/models`.
//! - Diffusion: `POST /v1/images/generations`.
//! - Lifecycle: `GET /health` (503 until the cold start is over),
//!   `GET /version`, `GET /get_server_info` (SGLang), `GET /metrics`
//!   (vLLM or SGLang metric names).
//! - Generic engines answer any other `POST` with a small JSON body.
//!
//! A request body may set `sim_output_tokens` to fix the generated length.

use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::{Request as HttpRequest, State};
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::{self, Stream};
use futures_util::StreamExt;
use rand::Rng;
use serde_json::{json, Value};
use swarmpilot_sim_core::engine::{self, EngineModel, Event, EventKind, Request, Work};
use swarmpilot_sim_core::profile::{EngineProfile, ProfileKind};
use tokio::sync::{mpsc, watch, Notify};

const DEFAULT_OUTPUT_TOKENS: u32 = 128;
const DEFAULT_STEPS: u32 = 30;
/// A 1x1 transparent PNG.
const PNG_1X1: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

/// Injected misbehaviour, set through the control API.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Faults {
    /// Share of requests answered with `error_status` right away.
    #[serde(default)]
    pub error_rate: f64,
    #[serde(default = "default_error_status")]
    pub error_status: u16,
    /// Accept requests but never answer them.
    #[serde(default)]
    pub stall: bool,
    /// Run this many times slower than the profile.
    #[serde(default = "one")]
    pub slowdown: f64,
}

fn default_error_status() -> u16 {
    500
}
fn one() -> f64 {
    1.0
}

/// Virtual time that can change speed without jumping.
#[derive(Debug, Clone, Copy)]
struct Clock {
    anchor_wall: Instant,
    anchor_v: f64,
    /// Wall ms per virtual ms.
    rate: f64,
}

impl Clock {
    fn now(&self) -> f64 {
        self.anchor_v + self.anchor_wall.elapsed().as_secs_f64() * 1e3 / self.rate
    }

    fn wall_at(&self, v: f64) -> Instant {
        let ms = ((v - self.anchor_v) * self.rate).max(0.0);
        self.anchor_wall + Duration::from_secs_f64(ms / 1e3)
    }

    fn set_rate(&mut self, rate: f64) {
        self.anchor_v = self.now();
        self.anchor_wall = Instant::now();
        self.rate = rate;
    }
}

struct Inner {
    engine: Box<dyn EngineModel>,
    waiters: HashMap<u64, mpsc::UnboundedSender<Event>>,
    clock: Clock,
}

/// A running virtual engine.
pub struct VirtualEngine {
    pub instance_id: String,
    pub addr: SocketAddr,
    pub profile: EngineProfile,
    /// The model id the instance was launched with.
    pub served_model: String,
    time_scale: f64,
    inner: Mutex<Inner>,
    notify: Notify,
    next_id: AtomicU64,
    ready_at: Instant,
    faults: Mutex<Faults>,
    crashed: AtomicBool,
    completed: AtomicU64,
    shutdown: watch::Sender<bool>,
}

impl VirtualEngine {
    /// Bind `addr` and start serving. The engine reports healthy after the
    /// profile's cold start (scaled by `time_scale`).
    pub async fn start(
        instance_id: String,
        addr: SocketAddr,
        profile: EngineProfile,
        served_model: String,
        time_scale: f64,
        seed: u64,
    ) -> std::io::Result<Arc<Self>> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        let addr = listener.local_addr()?;
        let scale = time_scale.max(1e-6);
        let (shutdown, rx) = watch::channel(false);
        let engine = Arc::new(Self {
            instance_id,
            addr,
            served_model,
            time_scale: scale,
            inner: Mutex::new(Inner {
                engine: engine::build(&profile, seed, true),
                waiters: HashMap::new(),
                clock: Clock {
                    anchor_wall: Instant::now(),
                    anchor_v: 0.0,
                    rate: scale,
                },
            }),
            ready_at: Instant::now() + Duration::from_secs_f64(profile.cold_start_s * scale),
            profile,
            notify: Notify::new(),
            next_id: AtomicU64::new(0),
            faults: Mutex::new(Faults {
                slowdown: 1.0,
                error_status: 500,
                ..Faults::default()
            }),
            crashed: AtomicBool::new(false),
            completed: AtomicU64::new(0),
            shutdown,
        });
        tokio::spawn(drive(engine.clone(), rx.clone()));
        let app = router(engine.clone());
        let mut stop = rx;
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = stop.wait_for(|s| *s).await;
                })
                .await;
        });
        Ok(engine)
    }

    pub fn port(&self) -> u16 {
        self.addr.port()
    }

    pub fn is_ready(&self) -> bool {
        Instant::now() >= self.ready_at && !self.is_crashed()
    }

    pub fn is_crashed(&self) -> bool {
        self.crashed.load(Ordering::SeqCst)
    }

    pub fn completed(&self) -> u64 {
        self.completed.load(Ordering::Relaxed)
    }

    pub fn faults(&self) -> Faults {
        self.faults.lock().unwrap().clone()
    }

    pub fn set_faults(&self, f: Faults) {
        let rate = self.time_scale * f.slowdown.max(1e-6);
        self.inner.lock().unwrap().clock.set_rate(rate);
        *self.faults.lock().unwrap() = f;
        self.notify.notify_one();
    }

    /// Stop serving. Open requests are dropped.
    pub fn stop(&self) {
        let _ = self.shutdown.send(true);
        self.inner.lock().unwrap().waiters.clear();
    }

    /// Die as a process would: stop serving and report failure upstream.
    pub fn crash(&self) {
        self.crashed.store(true, Ordering::SeqCst);
        self.stop();
    }

    pub fn load(&self) -> engine::Load {
        self.inner.lock().unwrap().engine.load()
    }

    fn submit(&self, work: Work) -> mpsc::UnboundedReceiver<Event> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::unbounded_channel();
        {
            let mut g = self.inner.lock().unwrap();
            g.waiters.insert(id, tx);
            let now = g.clock.now();
            g.engine.submit(now, Request { id, work });
        }
        self.notify.notify_one();
        rx
    }

    fn version(&self) -> &'static str {
        match self.profile.engine.as_str() {
            "vllm" => "0.11.0",
            "sglang" => "0.5.3",
            "diffusers" => "0.35.0",
            _ => "1.0.0",
        }
    }
}

async fn drive(e: Arc<VirtualEngine>, mut stop: watch::Receiver<bool>) {
    let far = Duration::from_secs(3600);
    let mut events = Vec::new();
    loop {
        let notified = e.notify.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        let wake = {
            let g = e.inner.lock().unwrap();
            g.engine
                .next_event()
                .map_or_else(|| Instant::now() + far, |v| g.clock.wall_at(v))
        };
        tokio::select! {
            _ = tokio::time::sleep_until(wake.into()) => {}
            _ = &mut notified => {}
            _ = stop.wait_for(|s| *s) => return,
        }
        let mut g = e.inner.lock().unwrap();
        let now = g.clock.now();
        g.engine.advance(now, &mut events);
        for ev in events.drain(..) {
            let last = matches!(ev.kind, EventKind::Done | EventKind::Rejected);
            if let Some(tx) = g.waiters.get(&ev.id) {
                let _ = tx.send(ev);
            }
            if last {
                g.waiters.remove(&ev.id);
                if ev.kind == EventKind::Done {
                    e.completed.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

fn router(e: Arc<VirtualEngine>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/version", get(version))
        .route("/get_server_info", get(version))
        .route("/server_info", get(version))
        .route("/v1/models", get(models))
        .route("/metrics", get(metrics))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .route("/v1/images/generations", post(images))
        .fallback(fallback)
        .with_state(e)
}

type S = State<Arc<VirtualEngine>>;

fn error(status: StatusCode, msg: &str) -> Response {
    (
        status,
        Json(json!({"error": {"message": msg, "type": "sim_error"}})),
    )
        .into_response()
}

async fn health(State(e): S) -> Response {
    if e.is_ready() {
        Json(json!({"status": "ok"})).into_response()
    } else {
        error(StatusCode::SERVICE_UNAVAILABLE, "loading")
    }
}

async fn version(State(e): S) -> Response {
    Json(json!({
        "version": e.version(),
        "sim": {
            "engine": e.profile.engine,
            "model": e.profile.model,
            "gpu": e.profile.gpu,
            "tp": e.profile.tp,
            "uncalibrated": e.profile.uncalibrated,
        }
    }))
    .into_response()
}

async fn models(State(e): S) -> Response {
    Json(json!({
        "object": "list",
        "data": [{"id": e.served_model, "object": "model", "owned_by": "swarmpilot-sim"}]
    }))
    .into_response()
}

async fn metrics(State(e): S) -> Response {
    let l = e.load();
    let (run, wait, usage) = match e.profile.engine.as_str() {
        "sglang" => (
            "sglang:num_running_reqs",
            "sglang:num_queue_reqs",
            "sglang:token_usage",
        ),
        _ => (
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:kv_cache_usage_perc",
        ),
    };
    let body = format!(
        "# TYPE {run} gauge\n{run} {}\n# TYPE {wait} gauge\n{wait} {}\n# TYPE {usage} gauge\n{usage} {}\n\
         # TYPE swarmpilot_sim_completed_total counter\nswarmpilot_sim_completed_total {}\n",
        l.running,
        l.waiting,
        l.utilization,
        e.completed()
    );
    ([("content-type", "text/plain; version=0.0.4")], body).into_response()
}

/// Check readiness and injected faults. `Err` is the response to send.
async fn admit(e: &VirtualEngine) -> Result<(), Response> {
    if !e.is_ready() {
        return Err(error(StatusCode::SERVICE_UNAVAILABLE, "loading"));
    }
    let f = e.faults();
    if f.error_rate > 0.0 && rand::thread_rng().gen::<f64>() < f.error_rate {
        let status =
            StatusCode::from_u16(f.error_status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        return Err(error(status, "injected fault"));
    }
    if f.stall {
        // Hold the request until the client gives up.
        std::future::pending::<()>().await;
    }
    Ok(())
}

fn chars(v: &Value) -> usize {
    match v {
        Value::String(s) => s.chars().count(),
        Value::Array(a) => a.iter().map(chars).sum(),
        Value::Object(o) => o.get("content").or_else(|| o.get("text")).map_or(0, chars),
        _ => 0,
    }
}

/// Token counts from an OpenAI body: about 4 characters per prompt token
/// (a list of token ids counts as is); output from `sim_output_tokens`,
/// then `max_tokens`/`max_completion_tokens`, else 128.
fn llm_work(body: &Value, chat: bool) -> Work {
    let prompt = if chat {
        body.get("messages").map_or(0, chars).div_ceil(4)
    } else {
        match body.get("prompt") {
            Some(Value::Array(a)) if a.iter().all(Value::is_number) => a.len(),
            Some(p) => chars(p).div_ceil(4),
            None => 0,
        }
    };
    let out = ["sim_output_tokens", "max_tokens", "max_completion_tokens"]
        .iter()
        .find_map(|k| body.get(*k).and_then(Value::as_u64))
        .unwrap_or(DEFAULT_OUTPUT_TOKENS as u64);
    Work::Llm {
        prompt_tokens: prompt.max(1) as u32,
        output_tokens: out.clamp(1, u32::MAX as u64) as u32,
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

async fn chat(State(e): S, Json(body): Json<Value>) -> Response {
    openai(e, body, true).await
}

async fn completions(State(e): S, Json(body): Json<Value>) -> Response {
    openai(e, body, false).await
}

async fn openai(e: Arc<VirtualEngine>, body: Value, chat: bool) -> Response {
    if let Err(r) = admit(&e).await {
        return r;
    }
    if matches!(e.profile.kind, ProfileKind::Diffusion(_)) {
        return error(StatusCode::NOT_FOUND, "this engine serves images");
    }
    let work = match e.profile.kind {
        ProfileKind::Generic(_) => Work::Generic,
        _ => llm_work(&body, chat),
    };
    let (prompt_tokens, output_tokens) = match work {
        Work::Llm {
            prompt_tokens,
            output_tokens,
        } => (prompt_tokens, output_tokens),
        _ => (1, 1),
    };
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let id = format!("sim-{}", e.next_id.load(Ordering::Relaxed));
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&e.served_model)
        .to_string();
    let rx = e.submit(work);
    let usage = json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
    });
    if stream {
        return Sse::new(token_stream(rx, id, model, chat, usage)).into_response();
    }
    let mut rx = rx;
    loop {
        match rx.recv().await {
            Some(ev) if ev.kind == EventKind::Done => break,
            Some(ev) if ev.kind == EventKind::Rejected => {
                return error(StatusCode::BAD_REQUEST, "request does not fit this engine");
            }
            Some(_) => continue,
            None => return error(StatusCode::SERVICE_UNAVAILABLE, "engine stopped"),
        }
    }
    let text = "x ".repeat(output_tokens as usize);
    let choice = if chat {
        json!({"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "length"})
    } else {
        json!({"index": 0, "text": text, "finish_reason": "length"})
    };
    Json(json!({
        "id": id,
        "object": if chat { "chat.completion" } else { "text_completion" },
        "created": unix_now(),
        "model": model,
        "choices": [choice],
        "usage": usage,
    }))
    .into_response()
}

fn token_stream(
    rx: mpsc::UnboundedReceiver<Event>,
    id: String,
    model: String,
    chat: bool,
    usage: Value,
) -> impl Stream<Item = Result<SseEvent, Infallible>> {
    let object = if chat {
        "chat.completion.chunk"
    } else {
        "text_completion"
    };
    let chunk = move |finish: Option<&str>, with_usage: bool| {
        let choice = if chat {
            json!({"index": 0, "delta": {"content": if finish.is_some() { "" } else { "x " }}, "finish_reason": finish})
        } else {
            json!({"index": 0, "text": if finish.is_some() { "" } else { "x " }, "finish_reason": finish})
        };
        let mut v = json!({"id": id, "object": object, "created": unix_now(), "model": model, "choices": [choice]});
        if with_usage {
            v["usage"] = usage.clone();
        }
        SseEvent::default().data(v.to_string())
    };
    // Token chunks until Done (or the engine stops), then [DONE].
    stream::unfold(Some(rx), move |rx| {
        let chunk = chunk.clone();
        async move {
            let mut rx = rx?;
            match rx.recv().await {
                Some(ev) => match ev.kind {
                    EventKind::FirstToken | EventKind::Token => {
                        Some((Ok(chunk(None, false)), Some(rx)))
                    }
                    EventKind::Done => Some((Ok(chunk(Some("length"), true)), None)),
                    EventKind::Rejected => Some((
                        Ok(SseEvent::default().data(
                            json!({"error": {"message": "request does not fit this engine"}})
                                .to_string(),
                        )),
                        None,
                    )),
                },
                None => None,
            }
        }
    })
    .chain(stream::once(async {
        Ok(SseEvent::default().data("[DONE]"))
    }))
}

fn image_work(body: &Value) -> Work {
    let n = body.get("n").and_then(Value::as_u64).unwrap_or(1).max(1) as u32;
    let (mut w, mut h) = (1024u32, 1024u32);
    if let Some(s) = body.get("size").and_then(Value::as_str) {
        if let Some((a, b)) = s.to_lowercase().split_once('x') {
            if let (Ok(a), Ok(b)) = (a.parse(), b.parse()) {
                (w, h) = (a, b);
            }
        }
    }
    if let Some(x) = body.get("width").and_then(Value::as_u64) {
        w = x as u32;
    }
    if let Some(x) = body.get("height").and_then(Value::as_u64) {
        h = x as u32;
    }
    let steps = body
        .get("num_inference_steps")
        .and_then(Value::as_u64)
        .map_or(DEFAULT_STEPS, |s| s as u32);
    // n images cost about n times the pixels.
    Work::Image {
        steps,
        width: w,
        height: h.saturating_mul(n),
    }
}

async fn images(State(e): S, Json(body): Json<Value>) -> Response {
    if let Err(r) = admit(&e).await {
        return r;
    }
    let work = match e.profile.kind {
        ProfileKind::Diffusion(_) => image_work(&body),
        ProfileKind::Generic(_) => Work::Generic,
        ProfileKind::Llm(_) => return error(StatusCode::NOT_FOUND, "this engine serves text"),
    };
    let n = body.get("n").and_then(Value::as_u64).unwrap_or(1).max(1);
    let mut rx = e.submit(work);
    while let Some(ev) = rx.recv().await {
        if ev.kind == EventKind::Done {
            let data: Vec<Value> = (0..n).map(|_| json!({"b64_json": PNG_1X1})).collect();
            return Json(json!({"created": unix_now(), "data": data})).into_response();
        }
        if ev.kind == EventKind::Rejected {
            return error(StatusCode::BAD_REQUEST, "request does not fit this engine");
        }
    }
    error(StatusCode::SERVICE_UNAVAILABLE, "engine stopped")
}

/// Generic engines serve any other POST route.
async fn fallback(State(e): S, req: HttpRequest) -> Response {
    if req.method() != axum::http::Method::POST
        || !matches!(e.profile.kind, ProfileKind::Generic(_))
    {
        return error(StatusCode::NOT_FOUND, "no such route");
    }
    if let Err(r) = admit(&e).await {
        return r;
    }
    let mut rx = e.submit(Work::Generic);
    while let Some(ev) = rx.recv().await {
        if ev.kind == EventKind::Done {
            return Json(json!({"ok": true, "path": req.uri().path()})).into_response();
        }
    }
    error(StatusCode::SERVICE_UNAVAILABLE, "engine stopped")
}
