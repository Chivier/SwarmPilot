//! Virtual engines over HTTP, and a sim worker against a fake PyLet head.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::{Path, State};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};
use swarmpilot_sim_cluster::engine_server::{Faults, VirtualEngine};
use swarmpilot_sim_cluster::pylet::{NodeSpec, SimNode};
use swarmpilot_sim_core::{Catalog, EngineProfile};

const LOCAL: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
const SCALE: f64 = 0.01;

fn profile(engine: &str, model: &str) -> EngineProfile {
    EngineProfile::resolve(&Catalog::builtin(), engine, model, "H100-SXM-80GB", 1).unwrap()
}

async fn engine(engine: &str, model: &str, warm: bool) -> Arc<VirtualEngine> {
    let mut p = profile(engine, model);
    if warm {
        p.cold_start_s = 0.0;
    }
    VirtualEngine::start(
        "t".into(),
        SocketAddr::new(LOCAL, 0),
        p,
        model.into(),
        SCALE,
        1,
    )
    .await
    .unwrap()
}

fn url(e: &VirtualEngine, path: &str) -> String {
    format!("http://127.0.0.1:{}{path}", e.port())
}

async fn until<F: Fn() -> bool>(what: &str, f: F) {
    for _ in 0..200 {
        if f() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("timed out waiting for {what}");
}

#[tokio::test]
async fn cold_start_then_healthy() {
    let e = engine("vllm", "llama-3.1-8b", false);
    let e = e.await;
    let c = reqwest::Client::new();
    // 33 s cold start at 0.01 is about 0.33 s.
    assert_eq!(
        c.get(url(&e, "/health")).send().await.unwrap().status(),
        503
    );
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(
        c.get(url(&e, "/health")).send().await.unwrap().status(),
        200
    );
}

#[tokio::test]
async fn chat_completion_json_and_stream() {
    let e = engine("vllm", "llama-3.1-8b", true).await;
    let c = reqwest::Client::new();
    let body =
        json!({"model": "m", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 12});
    let r: Value = c
        .post(url(&e, "/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["usage"]["completion_tokens"], 12);
    assert_eq!(r["choices"][0]["finish_reason"], "length");

    let mut body = body;
    body["stream"] = json!(true);
    let text = c
        .post(url(&e, "/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let chunks = text.lines().filter(|l| l.starts_with("data: {")).count();
    assert!(chunks >= 12, "{text}");
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");

    let m = c
        .get(url(&e, "/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(m.contains("vllm:num_requests_running"), "{m}");
    assert!(m.contains("swarmpilot_sim_completed_total 2"), "{m}");
}

#[tokio::test]
async fn images_only_on_diffusion_engines() {
    let llm = engine("vllm", "llama-3.1-8b", true).await;
    let sd = engine("diffusers", "sdxl", true).await;
    let c = reqwest::Client::new();
    let body = json!({"prompt": "a cat", "num_inference_steps": 2});
    let r = c
        .post(url(&llm, "/v1/images/generations"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
    let r: Value = c
        .post(url(&sd, "/v1/images/generations"))
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(r["data"][0]["b64_json"]
        .as_str()
        .is_some_and(|s| !s.is_empty()));
}

#[tokio::test]
async fn faults_and_crash() {
    let e = engine("sglang", "llama-3.1-8b", true).await;
    let c = reqwest::Client::new();
    let body = json!({"model": "m", "prompt": "hi", "max_tokens": 2});
    e.set_faults(Faults {
        error_rate: 1.0,
        error_status: 503,
        slowdown: 1.0,
        ..Faults::default()
    });
    let r = c
        .post(url(&e, "/v1/completions"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 503);
    e.set_faults(Faults {
        slowdown: 1.0,
        error_status: 500,
        ..Faults::default()
    });
    let r = c
        .post(url(&e, "/v1/completions"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

    e.crash();
    assert!(e.is_crashed());
    let fresh = reqwest::Client::new();
    let mut refused = false;
    for _ in 0..40 {
        if fresh.get(url(&e, "/health")).send().await.is_err() {
            refused = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert!(refused, "crashed engine still serving");
}

// -- A sim worker against a fake head ------------------------------------

#[derive(Default)]
struct Head {
    desired: Vec<Value>,
    registrations: Vec<Value>,
    reports: Vec<Value>,
    gen: u64,
}

type Shared = Arc<Mutex<Head>>;

async fn register(State(h): State<Shared>, Json(body): Json<Value>) -> Json<Value> {
    let mut h = h.lock().unwrap();
    h.registrations.push(body);
    Json(json!({"worker_token": format!("tok-{}", h.registrations.len())}))
}

async fn heartbeat(
    State(h): State<Shared>,
    Path(_id): Path<String>,
    Json(body): Json<Value>,
) -> Json<Value> {
    // A short long-poll keeps the loop turning.
    tokio::time::sleep(Duration::from_millis(20)).await;
    let mut h = h.lock().unwrap();
    h.reports = body["instances"].as_array().cloned().unwrap_or_default();
    h.gen += 1;
    Json(json!({"gen": h.gen, "desired_instances": h.desired}))
}

async fn fake_head() -> (String, Shared) {
    let state: Shared = Arc::default();
    let app = Router::new()
        .route("/workers", post(register))
        .route("/workers/{id}/heartbeat", post(heartbeat))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind((LOCAL, 0)).await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await });
    (format!("http://{addr}"), state)
}

fn free_port() -> u16 {
    std::net::TcpListener::bind((LOCAL, 0))
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn report(h: &Shared, id: &str) -> Option<Value> {
    h.lock()
        .unwrap()
        .reports
        .iter()
        .find(|r| r["instance_id"] == id)
        .cloned()
}

#[tokio::test]
async fn worker_runs_reports_and_reregisters() {
    let (head, h) = fake_head().await;
    // A block of free ports is hard to guarantee; take the worker port and
    // allow a small range above it, which the bind retry tolerates.
    let base = free_port();
    let node = SimNode::new(
        NodeSpec {
            id: "n0".into(),
            gpu: "H100-SXM-80GB".into(),
            gpus: 8,
            cpus: 64,
            memory_mb: 1 << 20,
            http_port: base,
            port_max: base.saturating_add(20),
        },
        Arc::new(Catalog::builtin()),
        LOCAL,
        "127.0.0.1".into(),
        SCALE,
    );
    tokio::spawn(node.clone().serve_http());
    tokio::spawn(node.clone().run(head, reqwest::Client::new()));

    until("registration", || {
        h.lock().unwrap().registrations.len() == 1
    })
    .await;
    let reg = h.lock().unwrap().registrations[0].clone();
    assert_eq!(reg["worker_id"], "n0");
    assert_eq!(reg["resources"]["gpu_units"], 8);
    assert_eq!(reg["http_port"], base);

    h.lock().unwrap().desired = vec![
        json!({"instance_id": "a", "attempt": 1, "gpu_indices": [0],
               "command": "vllm serve meta-llama/Llama-3.1-8B-Instruct --port $PORT", "env": {}}),
        json!({"instance_id": "b", "attempt": 1, "gpu_indices": [1],
               "command": "vllm serve no-such/model --port $PORT", "env": {}}),
    ];
    until("a RUNNING", || {
        report(&h, "a").is_some_and(|r| r["status"] == "RUNNING")
    })
    .await;
    until("b FAILED", || {
        report(&h, "b").is_some_and(|r| r["status"] == "FAILED")
    })
    .await;
    let port = report(&h, "a").unwrap()["port"].as_u64().unwrap();
    assert!(port > base as u64 && port <= base as u64 + 20);

    let c = reqwest::Client::new();
    let logs: Value = c
        .get(format!("http://127.0.0.1:{base}/logs/a"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(logs.to_string().contains("content"), "{logs}");

    node.engine("a").unwrap().crash();
    node.poke();
    until("a FAILED 137", || {
        report(&h, "a").is_some_and(|r| r["status"] == "FAILED" && r["exit_code"] == 137)
    })
    .await;

    // The head stops wanting both: the worker drops them.
    h.lock().unwrap().desired.clear();
    until("instances dropped", || node.instances().is_empty()).await;

    node.set_down(true);
    tokio::time::sleep(Duration::from_millis(100)).await;
    node.set_down(false);
    until("re-registration", || {
        h.lock().unwrap().registrations.len() == 2
    })
    .await;
    let regs = h.lock().unwrap().registrations.clone();
    assert_ne!(regs[0]["boot_id"], regs[1]["boot_id"]);
}

#[test]
fn engine_model_from_env_override() {
    let env = HashMap::from([("SWARMPILOT_SIM_ENGINE".to_string(), "generic".to_string())]);
    let l = swarmpilot_sim_cluster::command::parse("python -m private_engine", &env);
    assert_eq!(
        (l.engine.as_str(), l.model.as_str()),
        ("generic", "generic-small")
    );
}
