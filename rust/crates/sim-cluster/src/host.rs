//! The emulated cluster in one process: a sim worker per node of the spec,
//! plus a control API for inspection and fault injection.
//!
//! Control API (JSON):
//! - `GET /nodes`, `GET /instances`: what runs where, readiness, load.
//! - `POST /instances/{id}/faults` with
//!   `{"error_rate", "error_status", "stall", "slowdown"}`.
//! - `POST /instances/{id}/crash`: the engine dies; its worker reports the
//!   instance FAILED (exit 137).
//! - `POST /nodes/{id}/down`, `POST /nodes/{id}/up`: lose a node (engines
//!   stop, heartbeats stop) and bring it back with a fresh boot id.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;
use swarmpilot_sim_core::{Catalog, ClusterSpec};

use crate::engine_server::Faults;
use crate::pylet::{NodeSpec, SimNode};

/// Options for [`Host::new`].
#[derive(Debug, Clone)]
pub struct HostOptions {
    pub bind_ip: IpAddr,
    /// Host name the head and the Planner use to reach the engines.
    pub advertise_host: String,
    /// First node's worker HTTP port; node i uses `port_start + i * gap`.
    pub port_start: u16,
    pub port_gap: u16,
    pub time_scale: f64,
}

pub struct Host {
    pub nodes: Vec<Arc<SimNode>>,
}

impl Host {
    /// One node per spec node, with port ranges laid out from `port_start`.
    pub fn new(
        spec: &ClusterSpec,
        catalog: Arc<Catalog>,
        opt: &HostOptions,
    ) -> Result<Arc<Self>, String> {
        if opt.port_gap < 2 {
            return Err("port gap must be at least 2 (worker port plus one engine)".into());
        }
        let mut nodes = Vec::new();
        let mut i: u32 = 0;
        for g in &spec.node_groups {
            catalog.gpu(&g.gpu).map_err(|e| e.to_string())?;
            for id in g.node_ids() {
                let base = opt.port_start as u32 + i * opt.port_gap as u32;
                let max = base + opt.port_gap as u32 - 1;
                if max > u16::MAX as u32 {
                    return Err(format!("port range for node {id} exceeds 65535"));
                }
                nodes.push(SimNode::new(
                    NodeSpec {
                        id,
                        gpu: g.gpu.clone(),
                        gpus: g.gpus_per_node,
                        cpus: g.cpus_per_node,
                        memory_mb: g.memory_mb_per_node,
                        http_port: base as u16,
                        port_max: max as u16,
                    },
                    catalog.clone(),
                    opt.bind_ip,
                    opt.advertise_host.clone(),
                    opt.time_scale,
                ));
                i += 1;
            }
        }
        Ok(Arc::new(Self { nodes }))
    }

    fn node(&self, id: &str) -> Option<&Arc<SimNode>> {
        self.nodes.iter().find(|n| n.spec.id == id)
    }

    fn find_instance(&self, id: &str) -> Option<&Arc<SimNode>> {
        self.nodes.iter().find(|n| n.engine(id).is_some())
    }

    /// Start every node's worker loop and HTTP port, and the control API.
    pub async fn run(self: Arc<Self>, head: String, control: SocketAddr) -> std::io::Result<()> {
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(self.nodes.len() + 8)
            .build()
            .map_err(std::io::Error::other)?;
        for n in &self.nodes {
            let node = n.clone();
            tokio::spawn(async move {
                if let Err(e) = node.clone().serve_http().await {
                    eprintln!("swarmpilot-sim: {} http port: {e}", node.spec.id);
                }
            });
            tokio::spawn(n.clone().run(head.clone(), client.clone()));
        }
        let listener = tokio::net::TcpListener::bind(control).await?;
        let app = Router::new()
            .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
            .route("/nodes", get(nodes))
            .route("/instances", get(instances))
            .route("/instances/{id}/faults", post(set_faults))
            .route("/instances/{id}/crash", post(crash))
            .route("/nodes/{id}/down", post(node_down))
            .route("/nodes/{id}/up", post(node_up))
            .with_state(self);
        axum::serve(listener, app).await
    }
}

type S = State<Arc<Host>>;

fn not_found(what: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({"error": format!("no such {what}")})),
    )
        .into_response()
}

async fn nodes(State(h): S) -> Response {
    let v: Vec<_> = h
        .nodes
        .iter()
        .map(|n| {
            json!({
                "id": n.spec.id,
                "gpu": n.spec.gpu,
                "gpus": n.spec.gpus,
                "http_port": n.spec.http_port,
                "down": n.is_down(),
                "instances": n.instances().len(),
            })
        })
        .collect();
    Json(v).into_response()
}

async fn instances(State(h): S) -> Response {
    let v: Vec<_> = h.nodes.iter().flat_map(|n| n.instances()).collect();
    Json(v).into_response()
}

async fn set_faults(State(h): S, Path(id): Path<String>, Json(f): Json<Faults>) -> Response {
    let Some(e) = h.find_instance(&id).and_then(|n| n.engine(&id)) else {
        return not_found("instance");
    };
    e.set_faults(f);
    Json(json!({"ok": true, "faults": e.faults()})).into_response()
}

async fn crash(State(h): S, Path(id): Path<String>) -> Response {
    let Some(node) = h.find_instance(&id) else {
        return not_found("instance");
    };
    if let Some(e) = node.engine(&id) {
        e.crash();
    }
    node.poke();
    Json(json!({"ok": true})).into_response()
}

async fn node_down(State(h): S, Path(id): Path<String>) -> Response {
    let Some(n) = h.node(&id) else {
        return not_found("node");
    };
    n.set_down(true);
    Json(json!({"ok": true})).into_response()
}

async fn node_up(State(h): S, Path(id): Path<String>) -> Response {
    let Some(n) = h.node(&id) else {
        return not_found("node");
    };
    n.set_down(false);
    Json(json!({"ok": true})).into_response()
}
