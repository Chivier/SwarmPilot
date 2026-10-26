//! A simulated PyLet worker: one per node of the cluster spec.
//!
//! It speaks PyLet's worker protocol (pinned to pylet 0.5) to a real PyLet
//! head: register with `POST /workers`, then long-poll
//! `POST /workers/{id}/heartbeat` with reports on its instances and get
//! back the instances it should run. Where a real worker would exec the
//! command, this one starts a virtual engine on the next free port of its
//! range and reports it `RUNNING`; the engine itself answers `/health`
//! with 503 until its cold start is over, as a real engine would.
//!
//! Like the real worker it also serves `GET /logs/{instance_id}` and
//! `GET /health` on its own HTTP port.

use std::collections::{BTreeSet, HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::{Path, State};
use axum::routing::get;
use axum::{Json, Router};
use base64::Engine as _;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use swarmpilot_sim_core::{Catalog, EngineProfile};
use tokio::sync::{watch, Notify};

use crate::command;
use crate::engine_server::VirtualEngine;

/// One simulated node.
#[derive(Debug, Clone, Serialize)]
pub struct NodeSpec {
    pub id: String,
    pub gpu: String,
    pub gpus: u32,
    pub cpus: u32,
    pub memory_mb: u64,
    pub http_port: u16,
    /// Instance ports: `http_port + 1 ..= port_max`.
    pub port_max: u16,
}

#[derive(Debug, Clone, Deserialize)]
struct DesiredInstance {
    instance_id: String,
    attempt: u32,
    command: String,
    #[serde(default)]
    gpu_indices: Vec<u32>,
    #[serde(default)]
    env: HashMap<String, String>,
    #[serde(default = "assigned")]
    expected_status: String,
}

fn assigned() -> String {
    "ASSIGNED".into()
}

#[derive(Debug, Deserialize)]
struct HeartbeatResponse {
    gen: u64,
    desired_instances: Vec<DesiredInstance>,
}

#[derive(Debug, Serialize)]
struct InstanceReport {
    instance_id: String,
    attempt: u32,
    status: &'static str,
    port: Option<u16>,
    exit_code: Option<i32>,
}

/// One instance as this node knows it.
pub struct LocalInstance {
    pub status: &'static str,
    pub port: Option<u16>,
    pub exit_code: Option<i32>,
    pub engine: Option<Arc<VirtualEngine>>,
    log: String,
}

#[derive(Default)]
struct NodeState {
    instances: HashMap<(String, u32), LocalInstance>,
    last_desired: Vec<DesiredInstance>,
    used_ports: BTreeSet<u16>,
    /// Last port handed out; allocation continues after it, so a port a
    /// dead instance just freed is not reused at once (a stale endpoint
    /// would then reach a different engine and hide the failure).
    last_port: u16,
}

/// A node and its worker loop.
pub struct SimNode {
    pub spec: NodeSpec,
    catalog: Arc<Catalog>,
    bind_ip: IpAddr,
    advertise_host: String,
    time_scale: f64,
    state: Mutex<NodeState>,
    changed: Notify,
    down: watch::Sender<bool>,
}

/// Snapshot of one instance for the control API.
#[derive(Debug, Clone, Serialize)]
pub struct InstanceView {
    pub instance_id: String,
    pub attempt: u32,
    pub node: String,
    pub status: &'static str,
    pub port: Option<u16>,
    pub exit_code: Option<i32>,
    pub profile: Option<Value>,
    pub ready: bool,
    pub load: Option<Value>,
    pub faults: Option<Value>,
}

impl SimNode {
    pub fn new(
        spec: NodeSpec,
        catalog: Arc<Catalog>,
        bind_ip: IpAddr,
        advertise_host: String,
        time_scale: f64,
    ) -> Arc<Self> {
        Arc::new(Self {
            spec,
            catalog,
            bind_ip,
            advertise_host,
            time_scale,
            state: Mutex::new(NodeState::default()),
            changed: Notify::new(),
            down: watch::channel(false).0,
        })
    }

    pub fn is_down(&self) -> bool {
        *self.down.borrow()
    }

    /// Take the node off the network: stop every engine and stop
    /// heartbeating, so the head eventually marks the worker offline.
    pub fn set_down(&self, down: bool) {
        if down {
            let mut st = self.state.lock().unwrap();
            for inst in st.instances.values() {
                if let Some(e) = &inst.engine {
                    e.stop();
                }
            }
            st.instances.clear();
            st.last_desired.clear();
            st.used_ports.clear();
        }
        let _ = self.down.send(down);
        self.changed.notify_one();
    }

    /// Wake the heartbeat loop to report a local change now.
    pub fn poke(&self) {
        self.changed.notify_one();
    }

    pub fn instances(&self) -> Vec<InstanceView> {
        let st = self.state.lock().unwrap();
        let mut out: Vec<InstanceView> = st
            .instances
            .iter()
            .map(|((id, attempt), i)| InstanceView {
                instance_id: id.clone(),
                attempt: *attempt,
                node: self.spec.id.clone(),
                status: effective_status(i).0,
                port: i.port,
                exit_code: effective_status(i).1,
                profile: i.engine.as_ref().map(|e| json!(e.profile)),
                ready: i.engine.as_ref().is_some_and(|e| e.is_ready()),
                load: i.engine.as_ref().map(|e| json!(e.load())),
                faults: i.engine.as_ref().map(|e| json!(e.faults())),
            })
            .collect();
        out.sort_by(|a, b| a.instance_id.cmp(&b.instance_id));
        out
    }

    pub fn engine(&self, instance_id: &str) -> Option<Arc<VirtualEngine>> {
        let st = self.state.lock().unwrap();
        st.instances
            .iter()
            .filter(|((id, _), _)| id == instance_id)
            .max_by_key(|((_, a), _)| *a)
            .and_then(|(_, i)| i.engine.clone())
    }

    fn reports(&self) -> Vec<InstanceReport> {
        let st = self.state.lock().unwrap();
        let mut out: Vec<InstanceReport> = st
            .instances
            .iter()
            .map(|((id, attempt), i)| {
                let (status, exit_code) = effective_status(i);
                InstanceReport {
                    instance_id: id.clone(),
                    attempt: *attempt,
                    status,
                    port: i.port,
                    exit_code,
                }
            })
            .collect();
        for d in &st.last_desired {
            let key = (d.instance_id.clone(), d.attempt);
            if d.expected_status != "ASSIGNED" && !st.instances.contains_key(&key) {
                out.push(InstanceReport {
                    instance_id: d.instance_id.clone(),
                    attempt: d.attempt,
                    status: "NOT_FOUND",
                    port: None,
                    exit_code: None,
                });
            }
        }
        out
    }

    async fn reconcile(self: &Arc<Self>, desired: Vec<DesiredInstance>) {
        let keys: BTreeSet<(String, u32)> = desired
            .iter()
            .map(|d| (d.instance_id.clone(), d.attempt))
            .collect();
        let mut changed = false;
        {
            let mut st = self.state.lock().unwrap();
            let mut drop_keys = Vec::new();
            for (key, inst) in st.instances.iter_mut() {
                if keys.contains(key) {
                    continue;
                }
                match effective_status(inst).0 {
                    "RUNNING" => {
                        if let Some(e) = inst.engine.take() {
                            e.stop();
                        }
                        inst.status = "CANCELLED";
                        inst.exit_code = Some(0);
                        changed = true;
                    }
                    _ => drop_keys.push(key.clone()),
                }
            }
            for key in drop_keys {
                if let Some(i) = st.instances.remove(&key) {
                    if let Some(p) = i.port {
                        st.used_ports.remove(&p);
                    }
                }
            }
        }
        for d in &desired {
            let key = (d.instance_id.clone(), d.attempt);
            let known = self.state.lock().unwrap().instances.contains_key(&key);
            if !known && d.expected_status == "ASSIGNED" {
                self.start(d).await;
                changed = true;
            }
        }
        self.state.lock().unwrap().last_desired = desired;
        if changed {
            self.poke();
        }
    }

    fn alloc_port(&self) -> Option<u16> {
        let mut st = self.state.lock().unwrap();
        let (lo, hi) = (self.spec.http_port + 1, self.spec.port_max);
        let from = st.last_port.clamp(lo - 1, hi);
        let p = (from + 1..=hi)
            .chain(lo..=from)
            .find(|p| !st.used_ports.contains(p))?;
        st.used_ports.insert(p);
        st.last_port = p;
        Some(p)
    }

    async fn start(self: &Arc<Self>, d: &DesiredInstance) {
        let launch = command::parse(&d.command, &d.env);
        let tp = launch.tp.unwrap_or((d.gpu_indices.len() as u32).max(1));
        let mut log = format!(
            "[swarmpilot-sim] node {} ({} x {}), command: {}\n",
            self.spec.id, self.spec.gpus, self.spec.gpu, d.command
        );
        let fail = |log: String, reason: String| LocalInstance {
            status: "FAILED",
            port: None,
            exit_code: Some(1),
            engine: None,
            log: format!("{log}[swarmpilot-sim] failed: {reason}\n"),
        };
        let result = if !d.gpu_indices.is_empty() && (d.gpu_indices.len() as u32) < tp {
            Err(format!("tp {tp} needs {tp} GPUs, got {:?}", d.gpu_indices))
        } else {
            EngineProfile::resolve(
                &self.catalog,
                &launch.engine,
                &launch.model,
                &self.spec.gpu,
                tp,
            )
            .map_err(|e| e.to_string())
        };
        let inst = match result {
            Err(reason) => fail(log, reason),
            Ok(profile) => {
                let mut h = DefaultHasher::new();
                (&d.instance_id, d.attempt).hash(&mut h);
                log.push_str(&format!(
                    "[swarmpilot-sim] {} {} on {} x{} (cold start {:.1}s, {})\n",
                    profile.engine,
                    profile.model,
                    profile.gpu,
                    profile.tp,
                    profile.cold_start_s,
                    if profile.uncalibrated {
                        "uncalibrated"
                    } else {
                        "calibrated"
                    },
                ));
                match self
                    .bind_engine(&d.instance_id, profile, &launch.model, h.finish())
                    .await
                {
                    Ok((port, e)) => LocalInstance {
                        status: "RUNNING",
                        port: Some(port),
                        exit_code: None,
                        engine: Some(e),
                        log,
                    },
                    Err(reason) => fail(log, reason),
                }
            }
        };
        self.state
            .lock()
            .unwrap()
            .instances
            .insert((d.instance_id.clone(), d.attempt), inst);
    }

    /// Start an engine on a free port of this node's range. A port some
    /// other process holds (an outgoing connection's ephemeral port, say)
    /// is skipped, as a real engine would fail and be restarted elsewhere.
    async fn bind_engine(
        &self,
        instance_id: &str,
        profile: EngineProfile,
        model: &str,
        seed: u64,
    ) -> Result<(u16, Arc<VirtualEngine>), String> {
        let mut last_err = String::from("no free port");
        for _ in 0..16 {
            let Some(port) = self.alloc_port() else { break };
            let addr = SocketAddr::new(self.bind_ip, port);
            match VirtualEngine::start(
                instance_id.to_string(),
                addr,
                profile.clone(),
                model.to_string(),
                self.time_scale,
                seed,
            )
            .await
            {
                Ok(e) => return Ok((port, e)),
                Err(err) => {
                    // The allocator's cursor has moved past this port, so
                    // the next try takes the one after it.
                    self.state.lock().unwrap().used_ports.remove(&port);
                    last_err = format!("bind {addr}: {err}");
                    if err.kind() != std::io::ErrorKind::AddrInUse {
                        break;
                    }
                }
            }
        }
        Err(last_err)
    }

    fn log_of(&self, instance_id: &str) -> String {
        let st = self.state.lock().unwrap();
        st.instances
            .iter()
            .filter(|((id, _), _)| id == instance_id)
            .map(|(_, i)| i.log.clone())
            .collect()
    }

    /// Serve this node's worker HTTP port.
    pub async fn serve_http(self: Arc<Self>) -> std::io::Result<()> {
        let addr = SocketAddr::new(self.bind_ip, self.spec.http_port);
        let listener = tokio::net::TcpListener::bind(addr).await?;
        let app = Router::new()
            .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
            .route("/logs/{instance_id}", get(logs))
            .with_state(self);
        axum::serve(listener, app).await
    }

    /// Register with the head and heartbeat until the process exits.
    /// Re-registers with a fresh boot id after the node comes back up.
    pub async fn run(self: Arc<Self>, head: String, client: reqwest::Client) {
        let head = head.trim_end_matches('/').to_string();
        let mut down = self.down.subscribe();
        loop {
            let _ = down.wait_for(|d| !*d).await;
            let boot_id = format!("{:032x}", rand::thread_rng().gen::<u128>());
            let token = loop {
                match self.register(&head, &client, &boot_id).await {
                    Ok(t) => break t,
                    Err(e) => {
                        eprintln!("swarmpilot-sim: {} register failed: {e}", self.spec.id);
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
            };
            let mut gen = 0u64;
            while !self.is_down() {
                let notified = self.changed.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                let body = json!({
                    "worker_token": token,
                    "boot_id": boot_id,
                    "last_seen_gen": gen,
                    "instances": self.reports(),
                });
                let url = format!("{head}/workers/{}/heartbeat", self.spec.id);
                let req = client
                    .post(&url)
                    .json(&body)
                    .timeout(Duration::from_secs(35))
                    .send();
                tokio::select! {
                    r = req => match r {
                        Ok(resp) if resp.status().is_success() => {
                            match resp.json::<HeartbeatResponse>().await {
                                Ok(hb) => {
                                    gen = hb.gen;
                                    self.reconcile(hb.desired_instances).await;
                                }
                                Err(e) => eprintln!("swarmpilot-sim: {} bad heartbeat: {e}", self.spec.id),
                            }
                        }
                        Ok(resp) => {
                            eprintln!("swarmpilot-sim: {} heartbeat {}", self.spec.id, resp.status());
                            if resp.status().as_u16() == 401 || resp.status().as_u16() == 404 {
                                break; // token rejected: register again
                            }
                            tokio::time::sleep(Duration::from_secs(1)).await;
                        }
                        Err(e) if e.is_timeout() => {}
                        Err(e) => {
                            eprintln!("swarmpilot-sim: {} heartbeat failed: {e}", self.spec.id);
                            tokio::time::sleep(Duration::from_secs(1)).await;
                        }
                    },
                    // A local change cancels the long-poll and reports now.
                    _ = &mut notified => {}
                }
            }
        }
    }

    async fn register(
        &self,
        head: &str,
        client: &reqwest::Client,
        boot_id: &str,
    ) -> Result<String, String> {
        let body = json!({
            "worker_id": self.spec.id,
            "host": self.advertise_host,
            "resources": {
                "cpu_cores": self.spec.cpus,
                "gpu_units": self.spec.gpus,
                "memory_mb": self.spec.memory_mb,
            },
            "boot_id": boot_id,
            "http_port": self.spec.http_port,
        });
        let resp = client
            .post(format!("{head}/workers"))
            .json(&body)
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("status {}", resp.status()));
        }
        let v: Value = resp.json().await.map_err(|e| e.to_string())?;
        Ok(v.get("worker_token")
            .and_then(Value::as_str)
            .unwrap_or("legacy")
            .to_string())
    }
}

/// A crashed engine reports as a failed process (exit 137, like SIGKILL).
fn effective_status(i: &LocalInstance) -> (&'static str, Option<i32>) {
    match &i.engine {
        Some(e) if i.status == "RUNNING" && e.is_crashed() => ("FAILED", Some(137)),
        _ => (i.status, i.exit_code),
    }
}

async fn logs(State(node): State<Arc<SimNode>>, Path(instance_id): Path<String>) -> Json<Value> {
    let text = node.log_of(&instance_id);
    let b64 = base64::engine::general_purpose::STANDARD.encode(text.as_bytes());
    Json(json!({
        "available_offset": 0,
        "total_size": text.len(),
        "content": b64,
        "size": text.len(),
    }))
}
