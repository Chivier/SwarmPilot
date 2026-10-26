# Baselines at 8a9cee4

Recorded 2026-10-26 on a 4-vCPU cloud container (Python 3.13, no GPU) with
the Phase 0 scripts. Product code is 8a9cee4 plus the event-loop lag probe.
Engine (GPU) baselines from `bench_engine.py` still have to be taken on the
GPU pool, one file per engine and GPU type.

## Proxy overhead (`proxy-8a9cee4.json`)

One Scheduler instance in front of a fake engine that answers instantly,
`round_robin`, log level INFO, 10 s per point.

| Stream | Concurrency | Direct req/s | Proxy req/s | Throughput ratio | Added P50 | Added P99 |
|--------|-------------|-------------:|------------:|-----------------:|----------:|----------:|
| no  | 1   | 1500 | 159 | 0.106 | 5.4 ms   | 9.3 ms   |
| no  | 64  | 4071 | 178 | 0.044 | 335 ms   | 400 ms   |
| no  | 512 | 4082 | 142 | 0.035 | 2817 ms  | 3134 ms  |
| yes | any | 1044–2441 | 0 | 0 | every request fails with 502 | |

The proxy sends one blocking request at a time per instance, so throughput
is capped near 1/RTT and latency at concurrency is queueing. `stream: true`
fails because the worker parses the SSE body as JSON.

## Routing simulation (`sim_routing-8a9cee4.jsonl`)

8 replicas, 2,000 tasks, 80% load. P95 in seconds.

| Strategy | As shipped | Sketch fix only | Reseed fix only |
|----------|-----------:|----------------:|----------------:|
| oracle least-work (reference) | 7.18 | | |
| round_robin | 42.0 | | |
| random | 2543.7 (one replica gets 100%) | | 31.8 |
| po2 | 1040.8 (two replicas) | | 19.3 |
| min_time | 9.73 | 7.57 | |
| probabilistic | 9.09 | 7.92 | 12.86 |

Phase 2 target: default strategy P95 at least 13% better than 9.09 s and no
worse than `min_time`.

## Predictor (`predictor-8a9cee4.jsonl`)

| Case | Train time | Predict | q0.9 coverage | q0.99 coverage |
|------|-----------:|--------:|--------------:|---------------:|
| 100 samples, default (as TrainingClient) | 2.42 s | 0.40 ms | 0.878 | 0.956 |
| 100 samples, augmentation off | 1.33 s | 0.29 ms | 0.806 | 0.828 |
| 2000 samples, default | 5.71 s | 0.30 ms | 0.944 | 0.994 |
| 2000 samples, augmentation off | 2.13 s | 0.30 ms | 0.906 | 0.984 |

The 100-sample fit runs on the Scheduler event loop today; the regression
test measured a 2.6 s stall.

## Drain probe (`probe_drain-8a9cee4.json`)

Drain reports `pending_tasks: 0, can_remove: true` with 5 tasks queued, and
after remove those 5 stay `pending` forever.
