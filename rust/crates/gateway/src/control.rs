//! The control channel to the Python Scheduler: gRPC on a Unix socket.
//!
//! One task watches the state and reconnects when the Scheduler goes
//! away; the gateway keeps serving from its last snapshot meanwhile.
//! Completions and arrival counts go up in batches every 100 ms.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use hyper_util::rt::TokioIo;
use tokio::net::UnixStream;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::{Channel, Endpoint, Uri};
use tower::service_fn;

use crate::gateway::Gateway;
use crate::pb;
use crate::pb::gateway_control_client::GatewayControlClient;
use crate::state::Snapshot;

pub const PROTOCOL_VERSION: u32 = 1;
const BATCH_EVERY: Duration = Duration::from_millis(100);
const BATCH_MAX: usize = 512;

async fn connect(socket: PathBuf) -> Result<Channel, tonic::transport::Error> {
    // The URI is required by tonic but never used: every connection goes
    // to the socket.
    Endpoint::try_from("http://[::]:50051")?
        .connect_with_connector(service_fn(move |_: Uri| {
            let socket = socket.clone();
            async move { Ok::<_, std::io::Error>(TokioIo::new(UnixStream::connect(socket).await?)) }
        }))
        .await
}

/// Watch the state for as long as the gateway runs.
pub async fn watch(
    gw: Arc<Gateway>,
    socket: PathBuf,
    gateway_id: String,
    batches: Arc<Mutex<mpsc::Receiver<pb::CompletionBatch>>>,
) {
    let mut first = true;
    loop {
        match session(&gw, &socket, &gateway_id, batches.clone()).await {
            Ok(()) => tracing::warn!("Scheduler closed the state stream; reconnecting"),
            Err(e) => {
                if first || gw.snapshot().is_some() {
                    tracing::warn!(error = %e, socket = %socket.display(), "control channel down; retrying");
                }
            }
        }
        first = false;
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

async fn session(
    gw: &Arc<Gateway>,
    socket: &Path,
    gateway_id: &str,
    batches: Arc<Mutex<mpsc::Receiver<pb::CompletionBatch>>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let channel = connect(socket.to_path_buf()).await?;
    let mut client = GatewayControlClient::new(channel);
    let mut states = client
        .watch_state(pb::WatchStateRequest {
            gateway_id: gateway_id.to_string(),
            protocol_version: PROTOCOL_VERSION,
        })
        .await?
        .into_inner();
    tracing::info!(socket = %socket.display(), "connected to the Scheduler");

    // Completions flow over their own call on the same connection.
    let mut reporter = client.clone();
    let (tx, rx) = mpsc::channel::<pb::CompletionBatch>(64);
    let forward = tokio::spawn(async move {
        let mut batches = batches.lock().await;
        while let Some(b) = batches.recv().await {
            if tx.send(b).await.is_err() {
                break;
            }
        }
    });
    let report =
        tokio::spawn(async move { reporter.report_completions(ReceiverStream::new(rx)).await });

    let result = async {
        while let Some(state) = states.message().await? {
            if state.protocol_version > PROTOCOL_VERSION {
                tracing::warn!(
                    got = state.protocol_version,
                    "Scheduler speaks a newer control protocol; unknown fields are ignored"
                );
            }
            let version = state.version;
            match Snapshot::from_proto(state) {
                Ok(snap) => {
                    tracing::info!(
                        version,
                        instances = snap.instances.len(),
                        strategy = %snap.strategy_name,
                        "state"
                    );
                    gw.apply(snap);
                }
                Err(e) => tracing::error!(version, error = %e, "rejected state"),
            }
        }
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    }
    .await;
    forward.abort();
    report.abort();
    result
}

/// Batch completions and arrivals every 100 ms.
pub async fn batcher(
    gw: Arc<Gateway>,
    mut completions: mpsc::UnboundedReceiver<pb::Completion>,
    out: mpsc::Sender<pb::CompletionBatch>,
) {
    let mut tick = tokio::time::interval(BATCH_EVERY);
    let mut pending = Vec::new();
    loop {
        tokio::select! {
            c = completions.recv() => match c {
                Some(c) => {
                    pending.push(c);
                    if pending.len() < BATCH_MAX {
                        continue;
                    }
                }
                None => return,
            },
            _ = tick.tick() => {}
        }
        let arrivals = std::mem::take(&mut *gw.arrivals.lock().unwrap());
        if pending.is_empty() && arrivals.count == 0 {
            continue;
        }
        let batch = pb::CompletionBatch {
            completions: std::mem::take(&mut pending),
            arrivals: arrivals.count,
            calls_from: arrivals.calls_from,
        };
        if out.try_send(batch).is_err() {
            tracing::warn!("Scheduler is not taking completions; dropping a batch");
        }
    }
}

/// Send unreachable reports as they happen.
pub async fn unreachable_reporter(
    socket: PathBuf,
    mut reports: mpsc::UnboundedReceiver<pb::UnreachableReport>,
) {
    let mut client: Option<GatewayControlClient<Channel>> = None;
    while let Some(report) = reports.recv().await {
        for _ in 0..3 {
            if client.is_none() {
                client = connect(socket.clone())
                    .await
                    .ok()
                    .map(GatewayControlClient::new);
            }
            let Some(c) = client.as_mut() else {
                tokio::time::sleep(Duration::from_millis(200)).await;
                continue;
            };
            match c.report_unreachable(report.clone()).await {
                Ok(_) => break,
                Err(e) => {
                    tracing::warn!(error = %e, "cannot report an unreachable instance");
                    client = None;
                }
            }
        }
    }
}
