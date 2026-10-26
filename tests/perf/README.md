# Performance baselines and guardrails

Phase 0 of the production readiness plan. These scripts record how
SwarmPilot performs today so that later changes can be held to a budget.
None of them run as part of `pytest`; each one runs with a single command
and prints its metrics as JSON.

| Script | Plan item | Needs | What it measures |
|--------|-----------|-------|------------------|
| `bench_proxy.py` | 0.1 | CPU only | Latency and throughput the Scheduler proxy adds over calling the engine directly, at 1, 64 and 512 concurrent requests, with and without streaming. Uses `fake_engine.py`, which answers instantly. |
| `bench_engine.py` | 0.2 | A running engine and Scheduler | Throughput, TTFT and TPOT (LLM) or end-to-end latency (diffusion) at P50 and P99, direct and through SwarmPilot. Works with any OpenAI-compatible engine. |
| `sim_routing.py` | 0.3 | CPU only | Routing simulation with the real strategy classes: 8 replicas, Poisson arrivals, 80% load. Reports P50/P95/P99 and the largest share of traffic one replica got. |
| `bench_predictor.py` | 0.3 | CPU only | Quantile predictor training time, prediction time and empirical coverage. |
| `../integration/probe_drain.py` | 0.3 | CPU only | What happens to queued tasks when an instance is drained and removed. |
| `fake_engine.py` | | | OpenAI-style fake engine (chat, completions, SSE streaming, images). `GET /_stats` reports requests per path and peak concurrency. |

`tests/conformance/` checks dispatch semantics (pass-through, streaming,
per-instance in-flight limit, lightweight routes, timeouts) over HTTP only,
so the same tests can later run against the Rust gateway. The proxy
benchmark is black-box in the same way: it only needs URLs.

The five known bugs also have failing tests in
`tests/regression/test_known_bugs.py`, marked `xfail(strict=True)`. The
Scheduler exposes event-loop lag at `GET /v1/metrics/event_loop`.

## Running

```bash
uv run python tests/perf/bench_proxy.py --out tests/perf/baselines/proxy-$(git rev-parse --short HEAD).json
uv run python tests/perf/sim_routing.py 2000 0.8
uv run python tests/perf/bench_predictor.py
uv run python tests/integration/probe_drain.py
```

### Engine benchmark (0.2)

Start the engine and a Scheduler yourself, register the engine as one
instance, then point the script at both. Run once per engine and GPU type.

```bash
# 1. Engine (example: vLLM on one GPU)
vllm serve Qwen/Qwen3-8B --port 8100

# 2. Scheduler in front of it
SCHEDULER_MODEL_ID=Qwen/Qwen3-8B sscheduler start --port 8000
curl -X POST localhost:8000/v1/instance/register -H 'content-type: application/json' \
  -d '{"instance_id":"e0","model_id":"Qwen/Qwen3-8B","endpoint":"http://localhost:8100",
       "platform_info":{"software_name":"vllm","software_version":"0.11.0","hardware_name":"H100"}}'

# 3. Benchmark: streaming direct only (TTFT/TPOT), then non-streaming direct vs proxy
uv run python tests/perf/bench_engine.py --model Qwen/Qwen3-8B --engine vllm \
  --direct http://localhost:8100 --concurrency 1 16 64 \
  --out tests/perf/baselines/engine-vllm-H100-stream.json
uv run python tests/perf/bench_engine.py --model Qwen/Qwen3-8B --engine vllm \
  --direct http://localhost:8100 --proxy http://localhost:8000 --no-stream \
  --concurrency 1 16 64 --out tests/perf/baselines/engine-vllm-H100.json
```

For SGLang, start `python -m sglang.launch_server --model Qwen/Qwen3-8B --port 8100`
and pass `--engine sglang`. For a diffusion server that serves
`/v1/images/generations`, pass `--workload diffusion-image`.

At the baseline commit the proxy does not support `stream: true` (the worker
calls `response.json()` on the SSE body), so streaming is only measured
directly until plan item 1.3 lands.

## Budgets

Later phases may not regress a baseline beyond these limits (from the plan's
Verification table):

- Proxy overhead: added P99 at most 0.5 ms per request.
- Engine throughput through SwarmPilot within 3% of direct calls.
- After an engine upgrade: throughput at least 97% of the old version, TTFT
  and TPOT P99 at most 105%.
- Routing simulation: default strategy P95 at least 13% better than the
  baseline and no worse than `min_time`.

Baseline results live in `baselines/`, one JSON file per run, named by
benchmark and commit.
