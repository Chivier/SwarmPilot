# SDK Example: Multi-GPU Multi Model — Design Spec

**Parent spec:** [SDK Example Scripts](2026-03-13-sdk-examples-design.md)

## Purpose

Demonstrate the full SwarmPilot lifecycle for a multi-model deployment on 4 GPUs: optimizer-driven placement, per-model data collection via round-robin, independent predictor training for each model, and strategy switching to probabilistic scheduling.

## File

```
examples/sdk/multi_gpu_multi_model.py
```

## Prerequisites

- 4 GPUs available, each with sufficient VRAM (~8-10 GB)
- Planner service running at `http://localhost:8002`
- Scheduler service(s) — automatically spawned per model by the Planner
- A PyLet cluster for deployment
- `--scheduler-url` is **not required** — scheduler URLs are obtained dynamically via `schedulers()` after deployment

## `multi_gpu_multi_model.py` — Multi-Model GPU Partitioning with Per-Model Predictors

**Goal:** Show the full lifecycle for a multi-model deployment on 4 GPUs: optimizer-driven placement, per-model data collection via round-robin, independent predictor training for each model, and strategy switching to probabilistic scheduling.

**Hardware:** 4 GPUs available.

**Scenario:** Deploy two models:
- `Qwen/Qwen3-8B-VL-FP8`: 1 GPU per replica, 2 replicas (uses 2 GPUs)
- `Qwen/Qwen2.5-7B`: 1 GPU per replica, 2 replicas (uses 2 GPUs)

The optimizer plans placement across the 4 GPUs. Each model gets its own scheduler for independent load balancing. After deployment, collect training data for each model, train per-model predictors, and switch each scheduler to probabilistic.

### Flow

1. Create `SwarmPilotClient` with planner URL
2. **Phase 1 — Register models with GPU requirements:**
   a. `register("Qwen/Qwen3-8B-VL-FP8", gpu=1, replicas=2)` — 2 replicas, 1 GPU each
   b. `register("Qwen/Qwen2.5-7B", gpu=1, replicas=2)` — 2 replicas, 1 GPU each
   c. Print: registered models with GPU budget (total: 4 GPUs)
3. **Phase 2 — Deploy with optimizer:**
   a. `deploy()` → `DeploymentResult` — optimizer plans GPU placement
   b. `result.wait_ready(timeout=600)` — wait for all instances
   c. Iterate `result["model_name"]` — show per-model groups, instances, endpoints
   d. `schedulers()` → `dict[str, str]` — get per-model scheduler URLs
4. **Phase 3 — Set round-robin & collect training data for each model:**
   a. For each model's scheduler URL:
      - `POST /v1/strategy/set {"strategy_name": "round_robin"}` via `httpx`
      - Send 20 inference requests through that scheduler's proxy (`POST /v1/chat/completions`) with varying prompt lengths and `max_tokens`
      - Worker threads on each scheduler automatically collect `execution_time_ms`
   b. Print: per-model request counts and latencies
5. **Phase 4 — Train predictors & switch strategy for each model (SDK):**
   a. For each model, create a `SwarmPilotClient` pointing to that model's scheduler URL:
      - `train(model_id, prediction_type="expect_error")` → `TrainResult`
      - Print: samples trained, strategy auto-switched to `probabilistic`
6. **Phase 5 — Verify per-model predictor status (SDK):**
   a. For each model:
      - `predictor_status(model_id)` → `ModelStatus`
      - `predict(model_id, features={...})` → `PredictResult`
      - Print: predicted runtimes for each model (different models → different predictions)
7. **Phase 6 — Verify prediction-driven scheduling:**
   a. Send 10 requests to each model's scheduler (now using probabilistic)
   b. Print: latencies under prediction-driven routing
8. **Cleanup:**
   a. `terminate(all=True)` — tear down everything

### SDK Coverage

`register()`, `deploy()`, `DeploymentResult`, `DeploymentResult.__getitem__()`, `DeploymentResult.wait_ready()`, `instances()`, `ClusterState`, `schedulers()`, `train()`, `TrainResult`, `predictor_status()`, `ModelStatus`, `predict()`, `PredictResult`, `terminate(all=True)`

### Scheduler Interaction (via httpx, per-model scheduler)

- `POST /v1/strategy/set` — set round-robin on each model's scheduler
- `POST /v1/chat/completions` — transparent proxy through each model's scheduler

### Notes

- Each model gets its own scheduler, so strategy setting, data collection, and predictor training are **independent per model**.
- The `schedulers()` SDK call returns a mapping like `{"Qwen/Qwen3-8B-VL-FP8": "http://...:8010", "Qwen/Qwen2.5-7B": "http://...:8011"}`. Each scheduler URL is used for both transparent proxy requests and predictor SDK calls.
- Predictor training and prediction for each model require creating a `SwarmPilotClient` with that model's scheduler URL as `scheduler_url`. Use `async with SwarmPilotClient(planner_url, scheduler_url=url) as model_sp:` to properly manage per-model client lifecycle.
- Different models will produce different predicted runtimes based on their actual inference characteristics.
- The scheduler is a transparent proxy: `POST /v1/chat/completions` to a model's scheduler URL is routed to that model's backend instances.
- Execution times are collected automatically by each scheduler's worker threads.

### Expected Output

```
=== Phase 1: Register models ===
Registering Qwen/Qwen3-8B-VL-FP8 (1 GPU/replica, 2 replicas → 2 GPUs)...
Registering Qwen/Qwen2.5-7B (1 GPU/replica, 2 replicas → 2 GPUs)...
Total GPU budget: 4 GPUs

=== Phase 2: Deploy with optimizer ===
Deploying 2 models across 4 GPUs...
Deployment ready.

Model: Qwen/Qwen3-8B-VL-FP8
  Instances: 2
  Endpoints: ['http://10.0.0.1:8080', 'http://10.0.0.2:8080']
  Scheduler: http://scheduler-a:8010

Model: Qwen/Qwen2.5-7B
  Instances: 2
  Endpoints: ['http://10.0.0.3:8080', 'http://10.0.0.4:8080']
  Scheduler: http://scheduler-b:8011

=== Phase 3: Collect training data (round-robin, 20 requests/model) ===
[Qwen3-8B-VL-FP8] Strategy set to round_robin.
  Request 01/20 → 298.3 ms ... Request 20/20 → 312.1 ms
  All 20 requests completed.

[Qwen2.5-7B] Strategy set to round_robin.
  Request 01/20 → 245.7 ms ... Request 20/20 → 267.4 ms
  All 20 requests completed.

=== Phase 4: Train predictors ===
[Qwen3-8B-VL-FP8] Trained on 20 samples. Strategy → probabilistic
[Qwen2.5-7B] Trained on 20 samples. Strategy → probabilistic

=== Phase 5: Per-model predictor status & predictions ===
[Qwen3-8B-VL-FP8]
  Samples: 20, Prediction types: ['expect_error']
  Predicted runtime (max_tokens=256): 845.2 ms +/-72.3 ms

[Qwen2.5-7B]
  Samples: 20, Prediction types: ['expect_error']
  Predicted runtime (max_tokens=256): 623.8 ms +/-54.1 ms

=== Phase 6: Prediction-driven scheduling ===
Sending 10 requests to each model (probabilistic strategy)...
[Qwen3-8B-VL-FP8] Avg latency: 318.4 ms
[Qwen2.5-7B] Avg latency: 258.9 ms

=== Cleanup ===
Terminating all...
Done.
```

## Conventions

Follows the shared conventions defined in the [parent spec](2026-03-13-sdk-examples-design.md#shared-conventions): `argparse` CLI args, `async with SwarmPilotClient(...)`, `try/finally` cleanup, `from swarmpilot.errors import SwarmPilotError`.
