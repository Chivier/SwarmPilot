//! `swarmpilot-gateway`: the Scheduler's data plane in Rust.
//!
//! Serves the public port in place of the Python Scheduler's catch-all
//! proxy. Routing state comes from the Scheduler over gRPC on a Unix
//! socket (`proto/swarmpilot/gateway/v1/control.proto`); everything the
//! Scheduler needs back (training samples, statistics, unreachable
//! instances) is reported over the same channel. See `docs/rust_core.md`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::DefaultBodyLimit;
use axum::routing::get;
use axum::Router;
use clap::Parser;
use tokio::sync::{mpsc, Mutex};

mod control;
mod dispatch;
mod features;
mod gateway;
mod route;
mod state;

pub mod pb {
    tonic::include_proto!("swarmpilot.gateway.v1");
}

#[derive(Parser)]
#[command(version, about = "SwarmPilot gateway: routing and dispatch in Rust")]
struct Args {
    /// Address of the public port.
    #[arg(long, env = "GATEWAY_LISTEN", default_value = "0.0.0.0:8080")]
    listen: SocketAddr,
    /// The Scheduler's control socket (its SCHEDULER_GATEWAY_SOCKET).
    #[arg(long, env = "SCHEDULER_GATEWAY_SOCKET")]
    control: PathBuf,
    /// Name in Scheduler logs.
    #[arg(long, env = "GATEWAY_ID", default_value = "gateway")]
    id: String,
    /// Seed of the routing generator (random, po2, probabilistic); unset
    /// draws one from the OS.
    #[arg(long, env = "GATEWAY_SEED")]
    seed: Option<u64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("GATEWAY_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run(args))
}

async fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let (completions_tx, completions_rx) = mpsc::unbounded_channel();
    let (unreachable_tx, unreachable_rx) = mpsc::unbounded_channel();
    let (batches_tx, batches_rx) = mpsc::channel(256);
    let gw = gateway::Gateway::new(args.seed, completions_tx, unreachable_tx);

    tokio::spawn(control::watch(
        gw.clone(),
        args.control.clone(),
        args.id.clone(),
        Arc::new(Mutex::new(batches_rx)),
    ));
    tokio::spawn(control::batcher(gw.clone(), completions_rx, batches_tx));
    tokio::spawn(control::unreachable_reporter(
        args.control.clone(),
        unreachable_rx,
    ));
    let refresher = gw.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(1));
        loop {
            tick.tick().await;
            refresher.refresh_models();
        }
    });

    let app = Router::new()
        .route("/_gateway/health", get(gateway::health))
        .fallback(gateway::proxy)
        .layer(DefaultBodyLimit::disable())
        .with_state(gw);
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    tracing::info!(listen = %args.listen, control = %args.control.display(), "swarmpilot-gateway started");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
