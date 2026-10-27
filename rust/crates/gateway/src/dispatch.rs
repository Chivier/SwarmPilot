//! Per-instance dispatch (`AsyncWorkerQueue` of the Python Scheduler).
//!
//! Each instance has a FIFO queue and sends at most `max_in_flight`
//! requests at once; lightweight (metadata) requests jump the queue and
//! ignore the limit. A request's slot is held until its answer has been
//! read to the end, streamed or not. A refused connection hands the
//! request and everything still queued for that instance back to the
//! gateway to route elsewhere.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::Method;
use tokio::sync::oneshot;

use crate::route::RequestInfo;
use crate::state::Adapter;
use swarmpilot_core::Prediction;

/// What the caller gets back for one request.
pub enum Reply {
    /// One complete body (non-streamed requests, errors, engines that
    /// ignored `stream`).
    Body {
        status: u16,
        headers: HeaderMap,
        body: Bytes,
    },
    /// An SSE answer, relayed chunk by chunk.
    Stream {
        status: u16,
        headers: HeaderMap,
        chunks: RelayStream,
    },
    /// No answer from the engine; becomes a 502.
    Failed(String),
}

/// How a finished request went, for the completion report.
pub struct Outcome {
    pub succeeded: bool,
    pub http_status: u16,
    pub execution_time_ms: f64,
    pub error: String,
}

/// What routing and reporting know about one request; shared by the job
/// and the gateway so a re-route can update it.
pub struct TaskCtx {
    pub task_id: String,
    pub model_id: String,
    /// Path the engine is called on (no leading slash, no query).
    pub path: String,
    pub info: RequestInfo,
    /// Instance and prediction of the latest routing decision.
    pub assigned: Mutex<(String, Option<Prediction>)>,
}

/// Called once per request with the worker id when its answer is fully
/// read or abandoned.
pub type OnFinish = Arc<dyn Fn(&Arc<TaskCtx>, &str, Outcome) + Send + Sync>;

/// One proxied request.
pub struct Job {
    pub ctx: Arc<TaskCtx>,
    pub method: Method,
    /// Path and query, without the leading slash.
    pub path: String,
    pub query: Option<String>,
    pub headers: HeaderMap,
    pub body: Option<Bytes>,
    /// The client asked for a stream (`body["stream"]` is truthy).
    pub stream: bool,
    pub priority: bool,
    pub reply: Option<oneshot::Sender<Reply>>,
}

/// Settings shared by every worker.
#[derive(Clone)]
pub struct Settings {
    pub max_retries: u32,
    pub retry_delay: Duration,
}

/// Called with the instance id and its unsent jobs when it is unreachable.
pub type OnUnreachable = Arc<dyn Fn(String, String, Vec<Job>) + Send + Sync>;

struct Queues {
    normal: VecDeque<Job>,
    priority: VecDeque<Job>,
    running: usize,
    max_in_flight: usize,
    stopped: bool,
}

/// The queue of one instance.
pub struct Worker {
    pub id: String,
    endpoint: String,
    extra_headers: Vec<(String, String)>,
    software: String,
    adapter: Arc<Adapter>,
    client: reqwest::Client,
    settings: Settings,
    on_unreachable: OnUnreachable,
    on_finish: OnFinish,
    queues: Mutex<Queues>,
}

impl Worker {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        endpoint: String,
        extra_headers: Vec<(String, String)>,
        software: String,
        adapter: Arc<Adapter>,
        max_in_flight: usize,
        client: reqwest::Client,
        settings: Settings,
        on_unreachable: OnUnreachable,
        on_finish: OnFinish,
    ) -> Arc<Self> {
        Arc::new(Worker {
            id,
            endpoint: endpoint.trim_end_matches('/').to_string(),
            extra_headers,
            software,
            adapter,
            client,
            settings,
            on_unreachable,
            on_finish,
            queues: Mutex::new(Queues {
                normal: VecDeque::new(),
                priority: VecDeque::new(),
                running: 0,
                max_in_flight: max_in_flight.max(1),
                stopped: false,
            }),
        })
    }

    /// Whether this worker still sends to the same place the same way.
    pub fn same_target(
        &self,
        endpoint: &str,
        extra_headers: &[(String, String)],
        software: &str,
    ) -> bool {
        self.endpoint == endpoint.trim_end_matches('/')
            && self.extra_headers == extra_headers
            && self.software == software
    }

    pub fn set_max_in_flight(self: &Arc<Self>, n: usize) {
        self.queues.lock().unwrap().max_in_flight = n.max(1);
        self.pump();
    }

    /// Queue a job; hands it back if the worker was stopped.
    #[allow(clippy::result_large_err)]
    pub fn enqueue(self: &Arc<Self>, job: Job) -> Result<(), Job> {
        {
            let mut q = self.queues.lock().unwrap();
            if q.stopped {
                return Err(job);
            }
            if job.priority {
                q.priority.push_back(job);
            } else {
                q.normal.push_back(job);
            }
        }
        self.pump();
        Ok(())
    }

    /// Stop taking jobs and return the unsent ones, priority first.
    pub fn stop(&self) -> Vec<Job> {
        let mut q = self.queues.lock().unwrap();
        q.stopped = true;
        Self::take_pending(&mut q)
    }

    fn take_pending(q: &mut Queues) -> Vec<Job> {
        q.priority.drain(..).chain(q.normal.drain(..)).collect()
    }

    fn pump(self: &Arc<Self>) {
        let mut start = Vec::new();
        {
            let mut q = self.queues.lock().unwrap();
            loop {
                let job = if let Some(j) = q.priority.pop_front() {
                    j
                } else if q.running < q.max_in_flight {
                    match q.normal.pop_front() {
                        Some(j) => j,
                        None => break,
                    }
                } else {
                    break;
                };
                q.running += 1;
                start.push(job);
            }
        }
        for job in start {
            let me = self.clone();
            tokio::spawn(async move { me.execute(job).await });
        }
    }

    fn slot_freed(self: &Arc<Self>) {
        self.queues.lock().unwrap().running -= 1;
        self.pump();
    }

    fn url(&self, job: &Job) -> String {
        match &job.query {
            Some(q) => format!("{}/{}?{}", self.endpoint, job.path, q),
            None => format!("{}/{}", self.endpoint, job.path),
        }
    }

    fn retryable(&self, job: &Job) -> bool {
        match self.adapter.route(job.path.split('?').next().unwrap_or("")) {
            Some(r) => r.retry_safe,
            None => matches!(
                job.method,
                Method::GET | Method::HEAD | Method::OPTIONS | Method::PUT | Method::DELETE
            ),
        }
    }

    fn request(&self, job: &Job) -> reqwest::RequestBuilder {
        let mut headers = HeaderMap::new();
        for (k, v) in &self.extra_headers {
            if let (Ok(k), Ok(v)) = (HeaderName::try_from(k.as_str()), HeaderValue::from_str(v)) {
                headers.insert(k, v);
            }
        }
        // Task headers win on conflict.
        for name in job.headers.keys() {
            headers.remove(name);
        }
        for (k, v) in &job.headers {
            headers.append(k.clone(), v.clone());
        }
        let mut rb = self
            .client
            .request(job.method.clone(), self.url(job))
            .headers(headers);
        if let Some(body) = &job.body {
            rb = rb.body(body.clone());
        }
        rb
    }

    async fn execute(self: Arc<Self>, mut job: Job) {
        let started = Instant::now();
        let mut attempt = 0u32;
        let resp = loop {
            match self.request(&job).send().await {
                Ok(r) => break r,
                Err(e) if e.is_connect() => {
                    // Never reached the engine: another instance takes it
                    // and everything queued behind it.
                    let pending = {
                        let mut q = self.queues.lock().unwrap();
                        q.running -= 1;
                        Self::take_pending(&mut q)
                    };
                    let mut jobs = vec![job];
                    jobs.extend(pending);
                    (self.on_unreachable)(self.id.clone(), e.to_string(), jobs);
                    self.pump();
                    return;
                }
                Err(e) => {
                    let lost = e.is_request() && !e.is_timeout();
                    attempt += 1;
                    if lost && attempt < self.settings.max_retries && self.retryable(&job) {
                        let delay = self.settings.retry_delay * 2u32.pow(attempt - 1);
                        tracing::warn!(worker = %self.id, error = %e, ?delay, "connection lost, retrying");
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    let msg = error_text(&e);
                    self.finish_failed(&mut job, started, msg);
                    return;
                }
            }
        };
        let status = resp.status().as_u16();
        let headers = resp.headers().clone();
        let sse = headers
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.contains("text/event-stream"));
        if !job.stream || status >= 400 || !sse {
            let body = match resp.bytes().await {
                Ok(b) => normalise_json(b),
                Err(e) => {
                    let msg = error_text(&e);
                    self.finish_failed(&mut job, started, msg);
                    return;
                }
            };
            let outcome = Outcome {
                succeeded: status < 400,
                http_status: status,
                execution_time_ms: started.elapsed().as_secs_f64() * 1000.0,
                error: if status < 400 {
                    String::new()
                } else {
                    format!("HTTP {status}")
                },
            };
            if let Some(tx) = job.reply.take() {
                let _ = tx.send(Reply::Body {
                    status,
                    headers,
                    body,
                });
            }
            self.finish(&mut job, outcome);
            return;
        }
        let guard = FinishGuard {
            worker: Some(self.clone()),
            ctx: job.ctx.clone(),
            started,
            status,
            ok: false,
        };
        let relay = RelayStream {
            inner: Box::pin(resp.bytes_stream()),
            guard,
        };
        match job.reply.take() {
            Some(tx) => {
                if let Err(Reply::Stream { mut chunks, .. }) = tx.send(Reply::Stream {
                    status,
                    headers,
                    chunks: relay,
                }) {
                    // Nobody is listening any more: drain so the engine's
                    // slot is freed.
                    while chunks.next().await.is_some() {}
                }
            }
            None => {
                let mut relay = relay;
                while relay.next().await.is_some() {}
            }
        }
    }

    fn finish_failed(self: &Arc<Self>, job: &mut Job, started: Instant, msg: String) {
        tracing::warn!(task = %job.ctx.task_id, worker = %self.id, error = %msg, "request failed");
        if let Some(tx) = job.reply.take() {
            let _ = tx.send(Reply::Failed(msg.clone()));
        }
        self.finish(
            job,
            Outcome {
                succeeded: false,
                http_status: 0,
                execution_time_ms: started.elapsed().as_secs_f64() * 1000.0,
                error: msg,
            },
        );
    }

    fn finish(self: &Arc<Self>, job: &mut Job, outcome: Outcome) {
        (self.on_finish)(&job.ctx, &self.id, outcome);
        self.slot_freed();
    }
}

fn error_text(e: &reqwest::Error) -> String {
    let mut msg = e.to_string();
    let mut src = std::error::Error::source(e);
    while let Some(s) = src {
        msg.push_str(": ");
        msg.push_str(&s.to_string());
        src = s.source();
    }
    msg
}

/// The engine's body as the Python proxy returns it: JSON as is, an
/// empty body as `{}`, anything else as `{"error": <text>}`.
fn normalise_json(raw: Bytes) -> Bytes {
    if raw.is_empty() {
        return Bytes::from_static(b"{}");
    }
    if serde_json::from_slice::<serde::de::IgnoredAny>(&raw).is_ok() {
        return raw;
    }
    let text = String::from_utf8_lossy(&raw);
    Bytes::from(serde_json::json!({ "error": text }).to_string())
}

/// Reports the end of a streamed request exactly once: read to the end
/// (success) or dropped early (client gone).
struct FinishGuard {
    worker: Option<Arc<Worker>>,
    ctx: Arc<TaskCtx>,
    started: Instant,
    status: u16,
    ok: bool,
}

impl Drop for FinishGuard {
    fn drop(&mut self) {
        if let Some(w) = self.worker.take() {
            (w.on_finish)(
                &self.ctx,
                &w.id,
                Outcome {
                    succeeded: self.ok && self.status < 400,
                    http_status: self.status,
                    execution_time_ms: self.started.elapsed().as_secs_f64() * 1000.0,
                    error: if self.ok {
                        String::new()
                    } else {
                        "stream closed before the end".into()
                    },
                },
            );
            w.slot_freed();
        }
    }
}

type ByteStream = Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>;

/// An engine's SSE body, relayed; frees the slot when it ends or drops.
pub struct RelayStream {
    inner: ByteStream,
    guard: FinishGuard,
}

impl Stream for RelayStream {
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(None) => {
                self.guard.ok = true;
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(std::io::Error::other(e)))),
            Poll::Ready(Some(Ok(b))) => Poll::Ready(Some(Ok(b))),
            Poll::Pending => Poll::Pending,
        }
    }
}
