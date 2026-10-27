//! The data plane: the Python proxy's request flow on the Rust core.
//!
//! For each request: pick the active instances, compute the adapter's
//! features, route with the strategy, queue on the chosen instance and
//! relay its answer. Errors that SwarmPilot itself raises use the error
//! shape of the API style the client called.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};

use crate::dispatch::{Job, OnFinish, OnUnreachable, Outcome, Reply, Settings, TaskCtx, Worker};
use crate::features;
use crate::pb;
use crate::route::{RequestInfo, Routing};
use crate::state::{ModelCache, Snapshot};

/// Header a client sets on calls made on behalf of another model's task.
const PARENT_HEADER: &str = "x-swarmpilot-parent";

/// A local UNHEALTHY mark lasts until the Scheduler's state shows the
/// instance as not active, or this long at most.
const UNHEALTHY_MARK: Duration = Duration::from_secs(10);

/// Arrivals since the last completion batch.
#[derive(Default)]
pub struct Arrivals {
    pub count: u64,
    pub calls_from: HashMap<String, u64>,
}

pub struct Gateway {
    snapshot: RwLock<Option<Arc<Snapshot>>>,
    models: RwLock<ModelCache>,
    routing: Mutex<Routing>,
    workers: RwLock<HashMap<String, Arc<Worker>>>,
    unhealthy: Mutex<HashMap<String, Instant>>,
    /// The engine client and the read timeout it was built with.
    client: Mutex<Option<(f64, reqwest::Client)>>,
    next_task: AtomicU64,
    pub completions: mpsc::UnboundedSender<pb::Completion>,
    pub arrivals: Mutex<Arrivals>,
    pub unreachable: mpsc::UnboundedSender<pb::UnreachableReport>,
}

impl Gateway {
    pub fn new(
        seed: Option<u64>,
        completions: mpsc::UnboundedSender<pb::Completion>,
        unreachable: mpsc::UnboundedSender<pb::UnreachableReport>,
    ) -> Arc<Self> {
        Arc::new(Gateway {
            snapshot: RwLock::new(None),
            models: RwLock::new(ModelCache::default()),
            routing: Mutex::new(Routing::new(seed)),
            workers: RwLock::new(HashMap::new()),
            unhealthy: Mutex::new(HashMap::new()),
            client: Mutex::new(None),
            next_task: AtomicU64::new(0),
            completions,
            arrivals: Mutex::new(Arrivals::default()),
            unreachable,
        })
    }

    pub fn snapshot(&self) -> Option<Arc<Snapshot>> {
        self.snapshot.read().unwrap().clone()
    }

    /// The engine client: pooled connections, the Scheduler's connect
    /// (30 s) and read (`http_timeout_s`) timeouts. Workers created after
    /// a timeout change use the new one.
    fn engine_client(&self, read_timeout_s: f64) -> reqwest::Client {
        let mut slot = self.client.lock().unwrap();
        if let Some((t, c)) = slot.as_ref() {
            if *t == read_timeout_s {
                return c.clone();
            }
        }
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(1024)
            .tcp_nodelay(true)
            .connect_timeout(Duration::from_secs(30))
            .read_timeout(Duration::from_secs_f64(read_timeout_s))
            .build()
            .expect("HTTP client");
        *slot = Some((read_timeout_s, client.clone()));
        client
    }

    /// Take a new Scheduler state into use.
    pub fn apply(self: &Arc<Self>, snap: Snapshot) {
        let snap = Arc::new(snap);
        self.routing.lock().unwrap().sync(&snap);
        self.refresh_models_for(&snap);
        {
            let mut marks = self.unhealthy.lock().unwrap();
            marks.retain(|id, at| {
                at.elapsed() < UNHEALTHY_MARK && snap.instance(id).is_some_and(|i| i.active)
            });
        }
        let settings = Settings {
            max_retries: snap.max_retries,
            retry_delay: Duration::from_secs_f64(snap.retry_delay_s),
        };
        let client = self.engine_client(snap.http_timeout_s);
        let mut orphans = Vec::new();
        {
            let mut workers = self.workers.write().unwrap();
            let ids: Vec<String> = workers.keys().cloned().collect();
            for id in ids {
                let keep = snap.instance(&id).is_some_and(|i| {
                    workers[&id].same_target(&i.endpoint, &i.extra_headers, &i.software)
                });
                if !keep {
                    let w = workers.remove(&id).unwrap();
                    orphans.extend(w.stop().into_iter().map(|j| (id.clone(), j)));
                }
            }
            for inst in &snap.instances {
                match workers.get(&inst.id) {
                    Some(w) => w.set_max_in_flight(inst.max_in_flight),
                    None => {
                        let w = Worker::new(
                            inst.id.clone(),
                            inst.endpoint.clone(),
                            inst.extra_headers.clone(),
                            inst.software.clone(),
                            snap.adapter(&inst.software),
                            inst.max_in_flight,
                            client.clone(),
                            settings.clone(),
                            self.on_unreachable(),
                            self.on_finish(),
                        );
                        workers.insert(inst.id.clone(), w);
                    }
                }
            }
        }
        *self.snapshot.write().unwrap() = Some(snap.clone());
        for (from, job) in orphans {
            self.reroute(job, &from, "was removed");
        }
    }

    fn refresh_models_for(&self, snap: &Snapshot) {
        let mut models = self.models.write().unwrap();
        models.set_dir(&snap.storage_dir);
        models.refresh(
            snap.platforms
                .values()
                .flat_map(|p| [p.quantile_key.as_str(), p.expect_error_key.as_str()]),
        );
    }

    /// Pick up retrained models (called every second).
    pub fn refresh_models(&self) {
        if let Some(snap) = self.snapshot() {
            self.refresh_models_for(&snap);
        }
    }

    fn on_finish(self: &Arc<Self>) -> OnFinish {
        let me = Arc::downgrade(self);
        Arc::new(move |ctx: &Arc<TaskCtx>, worker: &str, outcome: Outcome| {
            if let Some(me) = me.upgrade() {
                me.finished(ctx, worker, outcome);
            }
        })
    }

    fn on_unreachable(self: &Arc<Self>) -> OnUnreachable {
        let me = Arc::downgrade(self);
        Arc::new(move |id: String, error: String, jobs: Vec<Job>| {
            if let Some(me) = me.upgrade() {
                me.instance_unreachable(id, error, jobs);
            }
        })
    }

    /// `TaskResultCallback.handle_result`, gateway half: release the
    /// charge here, report the rest to the Scheduler.
    fn finished(&self, ctx: &Arc<TaskCtx>, worker: &str, outcome: Outcome) {
        self.routing.lock().unwrap().release(&ctx.task_id);
        let prediction = ctx.assigned.lock().unwrap().1.clone();
        let names = self
            .snapshot()
            .and_then(|s| {
                s.instance(worker)
                    .map(|i| s.adapter(&i.software).feature_names.clone())
            })
            .unwrap_or_default();
        let metadata = ctx.info.numeric_metadata();
        let features = names
            .iter()
            .filter_map(|n| metadata.get(n).map(|v| (n.clone(), *v)))
            .collect();
        let completion = pb::Completion {
            task_id: ctx.task_id.clone(),
            instance_id: worker.to_string(),
            model_id: ctx.model_id.clone(),
            path: ctx.path.clone(),
            features,
            predicted_time_ms: prediction.as_ref().map(|p| p.predicted_time_ms),
            predicted_error_margin_ms: prediction.as_ref().and_then(|p| p.error_margin_ms),
            predicted_quantiles: prediction
                .and_then(|p| p.quantiles)
                .map(|qs| qs.into_iter().map(|(q, v)| (q.to_string(), v)).collect())
                .unwrap_or_default(),
            succeeded: outcome.succeeded,
            http_status: u32::from(outcome.http_status),
            execution_time_ms: outcome.execution_time_ms,
            error: outcome.error,
        };
        let _ = self.completions.send(completion);
    }

    /// `_handle_unreachable`: stop routing to the instance, tell the
    /// Scheduler, and route its unsent work elsewhere.
    fn instance_unreachable(&self, id: String, error: String, jobs: Vec<Job>) {
        tracing::warn!(instance = %id, %error, unsent = jobs.len(), "instance unreachable");
        self.unhealthy
            .lock()
            .unwrap()
            .insert(id.clone(), Instant::now());
        let _ = self.unreachable.send(pb::UnreachableReport {
            instance_id: id.clone(),
            error,
        });
        for job in jobs {
            self.reroute(job, &id, "is unreachable");
        }
    }

    fn candidates<'a>(&self, snap: &'a Snapshot) -> Vec<&'a crate::state::Instance> {
        let marks = self.unhealthy.lock().unwrap();
        snap.instances
            .iter()
            .filter(|i| i.active && !marks.contains_key(&i.id))
            .collect()
    }

    /// `_reroute_orphaned_tasks` for one job.
    fn reroute(&self, mut job: Job, from: &str, reason: &str) {
        let Some(snap) = self.snapshot() else { return };
        let candidates: Vec<_> = self
            .candidates(&snap)
            .into_iter()
            .filter(|i| i.id != from && i.model_id == job.ctx.model_id)
            .collect();
        let target = if candidates.is_empty() {
            None
        } else {
            let decision = {
                let models = self.models.read().unwrap();
                self.routing.lock().unwrap().route(
                    &snap,
                    &models,
                    &candidates,
                    &job.ctx.info,
                    &job.ctx.task_id,
                )
            };
            let id = decision.instance_id.clone();
            *job.ctx.assigned.lock().unwrap() = (decision.instance_id, decision.prediction);
            self.workers.read().unwrap().get(&id).cloned()
        };
        let msg = format!(
            "Instance {from} {reason} and the task could not be re-routed: no other active instance"
        );
        match target {
            Some(w) => {
                tracing::info!(task = %job.ctx.task_id, from, to = %w.id, "re-routed");
                if let Err(mut job) = w.enqueue(job) {
                    self.fail(&mut job, msg);
                }
            }
            None => self.fail(&mut job, msg),
        }
    }

    fn fail(&self, job: &mut Job, msg: String) {
        if let Some(tx) = job.reply.take() {
            let _ = tx.send(Reply::Failed(msg.clone()));
        }
        let worker = job.ctx.assigned.lock().unwrap().0.clone();
        self.finished(
            &job.ctx,
            &worker,
            Outcome {
                succeeded: false,
                http_status: 0,
                execution_time_ms: 0.0,
                error: msg,
            },
        );
    }

    fn record_arrival(&self, parent: Option<&str>) {
        let mut a = self.arrivals.lock().unwrap();
        a.count += 1;
        if let Some(p) = parent.filter(|p| !p.is_empty()) {
            *a.calls_from.entry(p.to_string()).or_default() += 1;
        }
    }
}

/// Python truthiness of a JSON value.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// SwarmPilot's own error in the style of the route called
/// (`_error_response`).
fn error_response(
    snap: Option<&Snapshot>,
    path: &str,
    status: u16,
    message: &str,
    error_type: &str,
) -> Response {
    let style = snap
        .and_then(|s| s.style_paths.get(path))
        .map(String::as_str);
    let body = match style {
        Some("anthropic_messages") => {
            let kind = match status {
                400 => "invalid_request_error",
                401 => "authentication_error",
                403 => "permission_error",
                404 => "not_found_error",
                413 => "request_too_large",
                429 => "rate_limit_error",
                503 | 529 => "overloaded_error",
                _ => "api_error",
            };
            json!({"type": "error", "error": {"type": kind, "message": message}})
        }
        Some(_) => {
            json!({"error": {"message": message, "type": error_type, "param": null, "code": null}})
        }
        None => json!({"error": {"message": message, "type": error_type}}),
    };
    json_response(status, &HeaderMap::new(), Bytes::from(body.to_string()))
}

fn json_response(status: u16, engine_headers: &HeaderMap, body: Bytes) -> Response {
    let mut resp = Response::new(Body::from(body));
    *resp.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
    let headers = resp.headers_mut();
    copy_headers(engine_headers, headers);
    if !headers.contains_key("content-type") {
        headers.insert("content-type", HeaderValue::from_static("application/json"));
    }
    resp
}

/// Engine response headers minus hop-by-hop, length and encoding ones.
fn copy_headers(from: &HeaderMap, to: &mut HeaderMap) {
    for (k, v) in from {
        if matches!(
            k.as_str(),
            "content-length"
                | "transfer-encoding"
                | "content-encoding"
                | "connection"
                | "keep-alive"
        ) {
            continue;
        }
        to.append(k.clone(), v.clone());
    }
}

/// Request headers passed to the engine.
fn forward_headers(from: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (k, v) in from {
        // The Python proxy drops the first four. accept-encoding goes too:
        // the gateway relays bodies as they are, so the engine must not
        // compress them (aiohttp decompressed them for the Python proxy).
        if matches!(
            k.as_str(),
            "host" | "connection" | "transfer-encoding" | "content-length" | "accept-encoding"
        ) {
            continue;
        }
        out.append(HeaderName::from(k), v.clone());
    }
    out
}

/// Every request that no gateway route takes.
pub async fn proxy(State(gw): State<Arc<Gateway>>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let method = parts.method.clone();
    let path = parts.uri.path().trim_start_matches('/').to_string();
    // The Python proxy's catch-all route takes these seven methods only.
    if !matches!(
        method,
        Method::GET
            | Method::POST
            | Method::PUT
            | Method::DELETE
            | Method::PATCH
            | Method::OPTIONS
            | Method::HEAD
    ) {
        return json_response(
            405,
            &HeaderMap::new(),
            Bytes::from_static(br#"{"detail":"Method Not Allowed"}"#),
        );
    }
    // The Python proxy calls the engine on the path alone and drops the
    // query string; the gateway does the same until both change together.
    let query: Option<String> = None;
    let snap = match gw.snapshot() {
        Some(s) => s,
        None => {
            return error_response(
                None,
                &path,
                503,
                "No backend instances available",
                "service_unavailable",
            )
        }
    };
    let candidates = gw.candidates(&snap);
    if candidates.is_empty() {
        return error_response(
            Some(&snap),
            &path,
            503,
            "No backend instances available",
            "service_unavailable",
        );
    }
    let raw = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(b) => b,
        Err(_) => {
            return error_response(
                Some(&snap),
                &path,
                400,
                "Could not read the request body",
                "invalid_request_error",
            )
        }
    };
    let has_body = matches!(method, Method::POST | Method::PUT | Method::PATCH);
    let body: Value = if has_body {
        serde_json::from_slice(&raw).unwrap_or_else(|_| json!({}))
    } else {
        json!({})
    };
    let first = candidates[0];
    let adapter = snap.adapter(&first.software);
    if method == Method::POST {
        if let Some(style) = snap.style_paths.get(&path) {
            if !adapter.native_styles.is_empty() && !adapter.native_styles.contains(style) {
                return error_response(
                    Some(&snap),
                    &path,
                    501,
                    &format!("The gateway does not translate {style} requests yet; call this model in its engine's API style"),
                    "not_implemented",
                );
            }
        }
    }
    let task_id = format!(
        "gw-{:x}-{:x}",
        std::process::id(),
        gw.next_task.fetch_add(1, Ordering::Relaxed)
    );
    let info = RequestInfo::new(features::extract(&adapter, &body), &body);
    let decision = {
        let models = gw.models.read().unwrap();
        gw.routing
            .lock()
            .unwrap()
            .route(&snap, &models, &candidates, &info, &task_id)
    };
    let worker = gw
        .workers
        .read()
        .unwrap()
        .get(&decision.instance_id)
        .cloned();
    let Some(worker) = worker else {
        gw.routing.lock().unwrap().release(&task_id);
        return error_response(
            Some(&snap),
            &path,
            503,
            "Selected instance not ready",
            "service_unavailable",
        );
    };
    gw.record_arrival(
        parts
            .headers
            .get(PARENT_HEADER)
            .and_then(|v| v.to_str().ok()),
    );
    let stream = body.get("stream").is_some_and(truthy);
    // The Python proxy sends `json=body`: a body that is not a JSON object
    // reaches the engine as `{}`.
    let send_body = has_body.then(|| {
        if body.as_object().is_some_and(|o| !o.is_empty()) || raw_is_empty_object(&raw) {
            raw.clone()
        } else {
            Bytes::from_static(b"{}")
        }
    });
    let ctx = Arc::new(TaskCtx {
        task_id: task_id.clone(),
        model_id: first.model_id.clone(),
        path: path.clone(),
        info,
        assigned: Mutex::new((decision.instance_id, decision.prediction)),
    });
    let (tx, rx) = oneshot::channel();
    let job = Job {
        ctx,
        method,
        path: path.clone(),
        query,
        headers: forward_headers(&parts.headers),
        body: send_body,
        stream,
        priority: adapter.is_lightweight(&path),
        reply: Some(tx),
    };
    if let Err(mut job) = worker.enqueue(job) {
        gw.fail(&mut job, "Failed to enqueue request".into());
        return error_response(
            Some(&snap),
            &path,
            503,
            "Failed to enqueue request",
            "service_unavailable",
        );
    }
    let reply = match tokio::time::timeout(Duration::from_secs_f64(snap.timeout_s), rx).await {
        Ok(Ok(r)) => r,
        Ok(Err(_)) => Reply::Failed("request dropped".into()),
        Err(_) => {
            tracing::error!(task = %task_id, "timed out after {}s", snap.timeout_s);
            return error_response(
                Some(&snap),
                &path,
                504,
                "Request timed out",
                "gateway_timeout",
            );
        }
    };
    match reply {
        Reply::Failed(msg) => error_response(Some(&snap), &path, 502, &msg, "bad_gateway"),
        Reply::Body {
            status,
            headers,
            body,
        } => json_response(status, &headers, body),
        Reply::Stream {
            status,
            headers,
            chunks,
        } => {
            let mut resp = Response::new(Body::from_stream(chunks));
            *resp.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::OK);
            copy_headers(&headers, resp.headers_mut());
            if !resp.headers().contains_key("content-type") {
                resp.headers_mut().insert(
                    "content-type",
                    HeaderValue::from_static("text/event-stream"),
                );
            }
            resp.into_response()
        }
    }
}

fn raw_is_empty_object(raw: &Bytes) -> bool {
    serde_json::from_slice::<Value>(raw).is_ok_and(|v| v.as_object().is_some_and(|o| o.is_empty()))
}

/// `GET /_gateway/health`: the gateway's own state.
pub async fn health(State(gw): State<Arc<Gateway>>) -> Response {
    let snap = gw.snapshot();
    let body = json!({
        "status": if snap.is_some() { "ok" } else { "waiting_for_state" },
        "state_version": snap.as_ref().map(|s| s.version),
        "model_id": snap.as_ref().map(|s| s.model_id.clone()),
        "strategy": snap.as_ref().map(|s| s.strategy_name.clone()),
        "instances": snap.as_ref().map(|s| s.instances.len()),
    });
    let status = if snap.is_some() { 200 } else { 503 };
    json_response(status, &HeaderMap::new(), Bytes::from(body.to_string()))
}
