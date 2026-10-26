//! `swarmpilot-sim`: inspect cluster specs and engine profiles, and replay
//! request streams through one simulated engine.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use swarmpilot_sim_core::engine;
use swarmpilot_sim_core::metrics::Summary;
use swarmpilot_sim_core::runner::run_single;
use swarmpilot_sim_core::workload::{
    load_azure_trace, ArrivalProcess, LengthDist, WorkShape, WorkloadSpec,
};
use swarmpilot_sim_core::{Catalog, ClusterSpec, EngineProfile, Error};

#[derive(Parser)]
#[command(name = "swarmpilot-sim", version, about)]
struct Cli {
    /// Directory with gpus.yaml, models.yaml and engines.yaml (default:
    /// the catalogs built into the binary).
    #[arg(long, global = true)]
    catalog: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Validate a cluster spec and print its placement summary as JSON.
    Plan {
        /// Spec file (default: the built-in 256-instance spec).
        spec: Option<PathBuf>,
        /// Print every instance, not just the totals.
        #[arg(long)]
        instances: bool,
    },
    /// Print the resolved profile for one engine/model/GPU/TP as JSON.
    Profile {
        #[arg(long)]
        engine: String,
        #[arg(long)]
        model: String,
        #[arg(long)]
        gpu: String,
        #[arg(long, default_value_t = 1)]
        tp: u32,
    },
    /// Replay a request stream through one simulated engine and print the
    /// summary row as JSON.
    Bench {
        #[arg(long)]
        engine: String,
        #[arg(long)]
        model: String,
        #[arg(long)]
        gpu: String,
        #[arg(long, default_value_t = 1)]
        tp: u32,
        /// Poisson arrival rate; 0 sends everything at once.
        #[arg(long, default_value_t = 0.0)]
        qps: f64,
        #[arg(long, default_value_t = 200)]
        requests: usize,
        /// Fixed prompt length (default: ShareGPT-like lengths).
        #[arg(long)]
        prompt: Option<u32>,
        /// Fixed output length (with --prompt).
        #[arg(long, default_value_t = 128)]
        output: u32,
        /// Denoising steps for image engines.
        #[arg(long, default_value_t = 30)]
        steps: u32,
        /// Replay an Azure LLM inference trace CSV instead.
        #[arg(long)]
        trace: Option<PathBuf>,
        /// Divide trace times by this.
        #[arg(long, default_value_t = 1.0)]
        speedup: f64,
        /// End-to-end SLO for the miss rate, in ms.
        #[arg(long)]
        slo_ms: Option<f64>,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("swarmpilot-sim: {e}");
            ExitCode::FAILURE
        }
    }
}

fn print(v: &impl serde::Serialize) {
    println!("{}", serde_json::to_string_pretty(v).expect("serializable"));
}

fn run(cli: Cli) -> Result<(), Error> {
    let catalog = match &cli.catalog {
        Some(dir) => Catalog::from_dir(dir)?,
        None => Catalog::builtin(),
    };
    match cli.cmd {
        Cmd::Plan { spec, instances } => {
            let spec = match spec {
                Some(p) => ClusterSpec::load(&p)?,
                None => ClusterSpec::default_256(),
            };
            let plan = spec.plan(&catalog)?;
            if instances {
                print(&plan);
            } else {
                print(&spec.summarize(&plan));
            }
        }
        Cmd::Profile {
            engine,
            model,
            gpu,
            tp,
        } => {
            print(&EngineProfile::resolve(
                &catalog, &engine, &model, &gpu, tp,
            )?);
        }
        Cmd::Bench {
            engine,
            model,
            gpu,
            tp,
            qps,
            requests,
            prompt,
            output,
            steps,
            trace,
            speedup,
            slo_ms,
            seed,
        } => {
            let profile = EngineProfile::resolve(&catalog, &engine, &model, &gpu, tp)?;
            let arrivals = match trace {
                Some(path) => load_azure_trace(&std::fs::read_to_string(path)?, speedup)?,
                None => {
                    let work = match engine.as_str() {
                        "diffusers" => WorkShape::Image {
                            steps: LengthDist::Fixed { value: steps },
                            width: 1024,
                            height: 1024,
                        },
                        "generic" => WorkShape::Generic,
                        _ => match prompt {
                            Some(p) => WorkShape::Llm {
                                prompt: LengthDist::Fixed { value: p },
                                output: LengthDist::Fixed { value: output },
                                max_total: None,
                            },
                            None => WorkShape::sharegpt_like(),
                        },
                    };
                    let arrivals = if qps > 0.0 {
                        ArrivalProcess::Poisson { rate_per_s: qps }
                    } else {
                        ArrivalProcess::Burst
                    };
                    WorkloadSpec {
                        arrivals,
                        work,
                        count: requests,
                        seed,
                    }
                    .generate()
                }
            };
            let mut e = engine::build(&profile, seed, false);
            let records = run_single(e.as_mut(), &arrivals);
            print(&Summary::from_records(&records, slo_ms));
        }
    }
    Ok(())
}
