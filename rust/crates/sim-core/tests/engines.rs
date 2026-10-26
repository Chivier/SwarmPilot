//! Behaviour of the engine models on small, hand-checked cases.

use swarmpilot_sim_core::engine::{
    DiffusionEngine, EngineModel, EventKind, GenericEngine, LlmEngine, Request, Work,
};
use swarmpilot_sim_core::metrics::{RequestRecord, Summary};
use swarmpilot_sim_core::profile::{DiffusionProfile, GenericProfile, LlmProfile};
use swarmpilot_sim_core::runner::run_single;
use swarmpilot_sim_core::workload::{Arrival, ArrivalProcess, WorkShape, WorkloadSpec};

/// 10 ms per step at any batch size, no compute bound, no overhead.
fn flat_llm(kv_tokens: u64, max_num_seqs: u32) -> LlmProfile {
    LlmProfile {
        step_overhead_ms: 0.0,
        request_overhead_ms: 0.0,
        weight_ms: 10.0,
        kv_ms_per_token: 0.0,
        compute_ms_per_token: 0.0,
        kv_capacity_tokens: kv_tokens,
        block_size: 16,
        max_num_seqs,
        max_num_batched_tokens: 8192,
    }
}

fn llm(id: u64, t_ms: f64, prompt: u32, output: u32) -> Arrival {
    Arrival {
        t_ms,
        req: Request {
            id,
            work: Work::Llm {
                prompt_tokens: prompt,
                output_tokens: output,
            },
        },
    }
}

fn run(e: &mut dyn EngineModel, a: &[Arrival]) -> Vec<RequestRecord> {
    run_single(e, a)
}

#[test]
fn one_request_takes_one_step_per_token() {
    let mut e = LlmEngine::new(flat_llm(100_000, 256), false);
    let r = run(&mut e, &[llm(0, 0.0, 100, 5)]);
    // Prefill step yields token 1, then 4 decode steps.
    assert_eq!(r[0].first_token_ms, Some(10.0));
    assert_eq!(r[0].done_ms, Some(50.0));
}

#[test]
fn batching_shares_steps() {
    let mut e = LlmEngine::new(flat_llm(100_000, 256), false);
    let a: Vec<_> = (0..8).map(|i| llm(i, 0.0, 100, 5)).collect();
    let r = run(&mut e, &a);
    // All 8 prefill in one step and decode together.
    assert!(r.iter().all(|x| x.done_ms == Some(50.0)), "{r:?}");
}

#[test]
fn max_num_seqs_queues_the_rest() {
    let mut e = LlmEngine::new(flat_llm(100_000, 2), false);
    let a: Vec<_> = (0..4).map(|i| llm(i, 0.0, 10, 3)).collect();
    let r = run(&mut e, &a);
    assert_eq!(r[0].done_ms, Some(30.0));
    assert_eq!(r[1].done_ms, Some(30.0));
    // The next two are admitted once the first pair is done.
    assert_eq!(r[2].first_token_ms, Some(40.0));
    assert_eq!(r[3].done_ms, Some(60.0));
}

#[test]
fn token_budget_chunks_long_prefills() {
    let mut p = flat_llm(100_000, 256);
    p.max_num_batched_tokens = 512;
    let mut e = LlmEngine::new(p, false);
    let r = run(&mut e, &[llm(0, 0.0, 1500, 2)]);
    // 1500 tokens at 512 per step: 3 prefill steps, then 1 decode.
    assert_eq!(r[0].first_token_ms, Some(30.0));
    assert_eq!(r[0].done_ms, Some(40.0));
}

#[test]
fn full_kv_cache_preempts_and_still_finishes_everything() {
    // Room for 8 blocks of 16 tokens: two sequences growing to 76 tokens
    // (5 blocks) each cannot both stay resident.
    let mut e = LlmEngine::new(flat_llm(128, 256), false);
    let a: Vec<_> = (0..2).map(|i| llm(i, 0.0, 16, 60)).collect();
    let r = run(&mut e, &a);
    assert!(
        r.iter().all(|x| x.done_ms.is_some() && !x.rejected),
        "{r:?}"
    );
    // The older one is never preempted, so it finishes in 60 steps; the
    // newer one is evicted, recomputed and finishes later.
    assert_eq!(r[0].done_ms, Some(600.0));
    assert!(r[1].done_ms.unwrap() > 600.0, "{r:?}");
    // Its first token was not re-emitted by the recompute.
    assert_eq!(r[1].first_token_ms, Some(10.0));
}

#[test]
fn request_larger_than_cache_is_rejected() {
    let mut e = LlmEngine::new(flat_llm(64, 256), false);
    let r = run(&mut e, &[llm(0, 0.0, 60, 20), llm(1, 0.0, 10, 2)]);
    assert!(r[0].rejected);
    assert_eq!(r[1].done_ms, Some(20.0));
}

#[test]
fn request_overhead_delays_admission() {
    let mut p = flat_llm(100_000, 256);
    p.request_overhead_ms = 5.0;
    let mut e = LlmEngine::new(p, false);
    let r = run(&mut e, &[llm(0, 0.0, 10, 2)]);
    assert_eq!(r[0].first_token_ms, Some(15.0));
    assert_eq!(r[0].done_ms, Some(25.0));
}

#[test]
fn token_events_cover_every_output_token() {
    let mut e = LlmEngine::new(flat_llm(100_000, 256), true);
    e.submit(0.0, llm(0, 0.0, 10, 6).req);
    let mut out = Vec::new();
    while let Some(t) = e.next_event() {
        e.advance(t, &mut out);
    }
    let first = out
        .iter()
        .filter(|x| x.kind == EventKind::FirstToken)
        .count();
    let tokens = out.iter().filter(|x| x.kind == EventKind::Token).count();
    assert_eq!((first, tokens), (1, 5));
    assert_eq!(out.last().unwrap().kind, EventKind::Done);
}

#[test]
fn llm_engine_rejects_image_work() {
    let mut e = LlmEngine::new(flat_llm(100_000, 256), false);
    let a = [Arrival {
        t_ms: 0.0,
        req: Request {
            id: 0,
            work: Work::Image {
                steps: 10,
                width: 512,
                height: 512,
            },
        },
    }];
    assert!(run(&mut e, &a)[0].rejected);
}

#[test]
fn every_request_finishes_under_load_and_runs_repeat() {
    let p = flat_llm(20_000, 64);
    let spec = WorkloadSpec {
        arrivals: ArrivalProcess::Poisson { rate_per_s: 50.0 },
        work: WorkShape::sharegpt_like(),
        count: 500,
        seed: 4,
    };
    let a = spec.generate();
    let r1 = run(&mut LlmEngine::new(p.clone(), false), &a);
    let r2 = run(&mut LlmEngine::new(p, false), &a);
    assert_eq!(r1, r2);
    let s = Summary::from_records(&r1, None);
    assert_eq!(s.completed + s.rejected, 500);
    for x in r1.iter().filter(|x| !x.rejected) {
        assert!(x.first_token_ms.unwrap() >= x.arrival_ms);
        assert!(x.done_ms.unwrap() >= x.first_token_ms.unwrap());
    }
}

fn image(id: u64, steps: u32) -> Arrival {
    Arrival {
        t_ms: 0.0,
        req: Request {
            id,
            work: Work::Image {
                steps,
                width: 1024,
                height: 1024,
            },
        },
    }
}

#[test]
fn diffusion_batches_up_to_max_batch() {
    let p = DiffusionProfile {
        ms_per_step_1024: 10.0,
        vae_ms: 0.0,
        max_batch: 2,
        batch_alpha: 0.5,
    };
    let mut e = DiffusionEngine::new(p);
    let r = run(&mut e, &[image(0, 10), image(1, 10), image(2, 10)]);
    // Batch of 2 costs 1.5x per step: 150 ms; the third runs alone after.
    assert_eq!(r[0].done_ms, Some(150.0));
    assert_eq!(r[1].done_ms, Some(150.0));
    assert_eq!(r[2].done_ms, Some(250.0));
}

#[test]
fn diffusion_time_scales_with_pixels() {
    let p = DiffusionProfile {
        ms_per_step_1024: 10.0,
        vae_ms: 0.0,
        max_batch: 1,
        batch_alpha: 0.0,
    };
    let mut e = DiffusionEngine::new(p);
    let a = [Arrival {
        t_ms: 0.0,
        req: Request {
            id: 0,
            work: Work::Image {
                steps: 10,
                width: 512,
                height: 512,
            },
        },
    }];
    assert_eq!(run(&mut e, &a)[0].done_ms, Some(25.0));
}

#[test]
fn generic_engine_respects_concurrency_and_mean() {
    let p = GenericProfile {
        mean_ms: 100.0,
        cv: 0.5,
        concurrency: 4,
    };
    let mut e = GenericEngine::new(p, 9);
    let a: Vec<_> = (0..4000)
        .map(|i| Arrival {
            t_ms: i as f64 * 50.0,
            req: Request {
                id: i,
                work: Work::Generic,
            },
        })
        .collect();
    let r = run(&mut e, &a);
    let mean = r
        .iter()
        .map(|x| x.done_ms.unwrap() - x.arrival_ms)
        .sum::<f64>()
        / 4000.0;
    // Light load (2 busy of 4 on average): little queueing.
    assert!((mean - 100.0).abs() < 10.0, "mean {mean}");
    let mut e1 = GenericEngine::new(
        GenericProfile {
            mean_ms: 100.0,
            cv: 0.0,
            concurrency: 1,
        },
        0,
    );
    let burst: Vec<_> = (0..3)
        .map(|i| Arrival {
            t_ms: 0.0,
            req: Request {
                id: i,
                work: Work::Generic,
            },
        })
        .collect();
    let r = run(&mut e1, &burst);
    let done: Vec<_> = r.iter().map(|x| x.done_ms.unwrap().round()).collect();
    assert_eq!(done, vec![100.0, 200.0, 300.0]);
}
