//! The vLLM profile against the public numbers it was fit to.
//!
//! Source: vLLM nightly performance benchmark on 8xH100, buildkite
//! vllm/performance-benchmark build 9565. Latency tests send a batch of 8
//! requests with 32 prompt and 128 output tokens; serving tests send 200
//! ShareGPT prompts with Poisson arrivals at the given QPS (0 = all at
//! once). The 70B runs there are Llama-3-70B; Llama-3.1-70B has the same
//! architecture, so the catalog's llama-3.1-70b stands in. We do not have
//! the exact ShareGPT sample, so serving runs use
//! ShareGPT-like lengths and average over seeds.
//!
//! Run with `cargo test --release -p swarmpilot-sim-core --test calibration
//! -- --nocapture` to print the comparison table.

use swarmpilot_sim_core::engine::{self, Work};
use swarmpilot_sim_core::metrics::Summary;
use swarmpilot_sim_core::runner::run_single;
use swarmpilot_sim_core::workload::{Arrival, ArrivalProcess, WorkShape, WorkloadSpec};
use swarmpilot_sim_core::{Catalog, EngineProfile};

const GPU: &str = "H100-SXM-80GB";
const SEEDS: u64 = 10;

struct Serving {
    qps: f64,
    rps: f64,
    ttft_ms: f64,
    itl_ms: f64,
}

struct Anchor {
    model: &'static str,
    tp: u32,
    latency_ms: f64,
    serving: [Serving; 4],
}

const ANCHORS: [Anchor; 2] = [
    Anchor {
        model: "llama-3.1-8b",
        tp: 1,
        latency_ms: 997.542,
        serving: [
            Serving {
                qps: 1.0,
                rps: 1.00494,
                ttft_ms: 24.6827,
                itl_ms: 7.62798,
            },
            Serving {
                qps: 4.0,
                rps: 3.80428,
                ttft_ms: 25.2503,
                itl_ms: 8.09316,
            },
            Serving {
                qps: 16.0,
                rps: 11.4896,
                ttft_ms: 38.2191,
                itl_ms: 10.2977,
            },
            Serving {
                qps: 0.0,
                rps: 19.4787,
                ttft_ms: 1186.77,
                itl_ms: 14.2914,
            },
        ],
    },
    Anchor {
        model: "llama-3.1-70b",
        tp: 4,
        latency_ms: 2444.47,
        serving: [
            Serving {
                qps: 1.0,
                rps: 0.984501,
                ttft_ms: 62.4149,
                itl_ms: 19.3936,
            },
            Serving {
                qps: 4.0,
                rps: 3.33967,
                ttft_ms: 72.0854,
                itl_ms: 22.171,
            },
            Serving {
                qps: 16.0,
                rps: 7.29345,
                ttft_ms: 130.278,
                itl_ms: 28.2932,
            },
            Serving {
                qps: 0.0,
                rps: 8.89715,
                ttft_ms: 2828.76,
                itl_ms: 30.6683,
            },
        ],
    },
];

fn profile(model: &str, tp: u32) -> EngineProfile {
    EngineProfile::resolve(&Catalog::builtin(), "vllm", model, GPU, tp).unwrap()
}

fn run(p: &EngineProfile, arrivals: &[Arrival]) -> Summary {
    let mut e = engine::build(p, 0, false);
    Summary::from_records(&run_single(e.as_mut(), arrivals), None)
}

fn latency_ms(p: &EngineProfile) -> f64 {
    let arrivals: Vec<Arrival> = (0..8)
        .map(|i| Arrival {
            t_ms: 0.0,
            req: engine::Request {
                id: i,
                work: Work::Llm {
                    prompt_tokens: 32,
                    output_tokens: 128,
                },
            },
        })
        .collect();
    run(p, &arrivals).e2e_ms.unwrap().mean
}

/// Mean over seeds of (throughput, mean TTFT, mean ITL).
fn serving(p: &EngineProfile, qps: f64) -> (f64, f64, f64) {
    let (mut rps, mut ttft, mut itl) = (0.0, 0.0, 0.0);
    for seed in 0..SEEDS {
        let spec = WorkloadSpec {
            arrivals: if qps > 0.0 {
                ArrivalProcess::Poisson { rate_per_s: qps }
            } else {
                ArrivalProcess::Burst
            },
            work: WorkShape::sharegpt_like(),
            count: 200,
            seed,
        };
        let s = run(p, &spec.generate());
        rps += s.throughput_rps;
        ttft += s.ttft_ms.unwrap().mean;
        itl += s.itl_mean_ms.unwrap();
    }
    let n = SEEDS as f64;
    (rps / n, ttft / n, itl / n)
}

fn err(sim: f64, real: f64) -> f64 {
    (sim - real) / real
}

#[test]
fn vllm_h100_matches_nightly_benchmark() {
    let mut worst: Vec<(String, f64, f64)> = Vec::new();
    println!(
        "{:<26} {:>10} {:>10} {:>8}",
        "point", "measured", "simulated", "error"
    );
    for a in &ANCHORS {
        let p = profile(a.model, a.tp);
        let mut row = |name: String, real: f64, sim: f64, tol: f64| {
            println!(
                "{name:<26} {real:>10.2} {sim:>10.2} {:>+7.1}%",
                100.0 * err(sim, real)
            );
            worst.push((name, err(sim, real), tol));
        };
        let tag = format!("{} tp{}", a.model.trim_start_matches("llama-3.1-"), a.tp);
        row(
            format!("{tag} latency ms"),
            a.latency_ms,
            latency_ms(&p),
            0.05,
        );
        for s in &a.serving {
            let (rps, ttft, itl) = serving(&p, s.qps);
            let q = if s.qps > 0.0 {
                format!("qps{}", s.qps)
            } else {
                "qps-inf".into()
            };
            // Fit points: the latency test and QPS 1 (low load). Higher
            // loads are checked with looser bounds. At QPS inf the run
            // lasts as long as its longest output, so throughput swings
            // with the length sample (12 to 19 req/s across seeds for 8B)
            // and gets the loosest bound.
            let (t_ttft, t_itl, t_rps) = match s.qps as u32 {
                1 => (0.10, 0.05, 0.10),
                0 => (0.35, 0.25, 0.30),
                _ => (0.40, 0.25, 0.20),
            };
            row(format!("{tag} {q} rps"), s.rps, rps, t_rps);
            row(format!("{tag} {q} ttft ms"), s.ttft_ms, ttft, t_ttft);
            row(format!("{tag} {q} itl ms"), s.itl_ms, itl, t_itl);
        }
    }
    let bad: Vec<_> = worst.iter().filter(|(_, e, tol)| e.abs() > *tol).collect();
    assert!(bad.is_empty(), "outside tolerance: {bad:?}");
}

#[test]
fn diffusion_profile_matches_its_anchor() {
    // SDXL, 1024x1024, 30 steps, batch 1 on H100: about 2.45 s.
    let p = EngineProfile::resolve(&Catalog::builtin(), "diffusers", "sdxl", GPU, 1).unwrap();
    let ms = p.idle_latency_ms(0, 0, 30);
    assert!((ms - 2450.0).abs() / 2450.0 < 0.10, "{ms}");
    let arrivals = vec![Arrival {
        t_ms: 0.0,
        req: engine::Request {
            id: 0,
            work: Work::Image {
                steps: 30,
                width: 1024,
                height: 1024,
            },
        },
    }];
    let e2e = run(&p, &arrivals).e2e_ms.unwrap().mean;
    assert!((e2e - ms).abs() < 1e-6);
}

#[test]
fn slower_gpus_are_slower() {
    // Decode is bandwidth-bound: per-token time follows bandwidth.
    let c = Catalog::builtin();
    let lat = |gpu: &str| {
        EngineProfile::resolve(&c, "vllm", "qwen2.5-7b", gpu, 1)
            .unwrap()
            .idle_latency_ms(256, 128, 0)
    };
    let order = [
        "H100-SXM-80GB",
        "A100-SXM-80GB",
        "A100-40GB",
        "L40S-48GB",
        "A10G-24GB",
    ];
    for w in order.windows(2) {
        assert!(lat(w[0]) < lat(w[1]), "{} vs {}", w[0], w[1]);
    }
}
