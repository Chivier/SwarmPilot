# SDK Example: Multi-GPU Single Model — Design Spec

**Parent spec:** [SDK Example Scripts](2026-03-13-sdk-examples-design.md)

## Purpose

Demonstrate the full SwarmPilot lifecycle on a 4-GPU machine with a single model: deploy 4 independent instances (1 GPU each), collect training data via round-robin, train the predictor, switch to probabilistic scheduling, and verify prediction-driven load balancing.

## File

```
examples/sdk/multi_gpu_single_model.py
```

## Prerequisites

- 4 GPUs available, each with sufficient VRAM for Qwen3-8B-VL-FP8 (~8-10 GB)
- Planner service running at `http://localhost:8002`
- Scheduler service running at `http://localhost:8000`
- A PyLet cluster for deployment
- `--scheduler-url` is **required** (used for strategy setting, transparent proxy, and predictor operations)

## `multi_gpu_single_model.py` — 4-Instance Single Model with Predictor

**Goal:** Show the full lifecycle on a 4-GPU machine with a single model: deploy 4 independent instances (1 GPU each), collect training data via round-robin, train the predictor, switch to probabilistic scheduling, and verify prediction-driven load balancing across all 4 instances.

**Hardware:** 4 GPUs available. Each GPU runs one independent vLLM instance — no tensor parallelism.

**Scenario:** Deploy `Qwen/Qwen3-8B-VL-FP8` with `gpu=1, replicas=4` — 4 independent instances, each on its own GPU. Use round-robin to collect execution times across all 4 instances, then train the predictor and switch to probabilistic scheduling.

### Flow

1. Create `SwarmPilotClient` with planner and scheduler URLs
2. **Phase 1 — Deploy 4 independent instances (1 GPU each):**
   a. `serve("Qwen/Qwen3-8B-VL-FP8", name="qwen3-x4", gpu=1, replicas=4)` → `InstanceGroup`
   b. `group.wait_ready(timeout=600)` — wait for all 4 vLLM instances to load
   c. Print: 4 instances, 1 GPU each, 4/4 GPUs used, all endpoints
3. **Phase 2 — Set round-robin & collect training data:**
   a. `POST /v1/strategy/set {"strategy_name": "round_robin"}` via `httpx` to scheduler
   b. Send 40 inference requests through the scheduler proxy (`POST /v1/chat/completions`) with varying prompt lengths and `max_tokens` (64/128/256)
   c. Requests are distributed across 4 instances by round-robin; worker threads automatically collect `execution_time_ms`
   d. Print: per-request latency, showing round-robin distribution across 4 instances
4. **Phase 3 — Train predictor & switch strategy (SDK):**
   a. `train("Qwen/Qwen3-8B-VL-FP8", prediction_type="expect_error")` → `TrainResult`
   b. Print: samples trained, strategy auto-switched to `probabilistic`
5. **Phase 4 — Check predictor status (SDK):**
   a. `predictor_status("Qwen/Qwen3-8B-VL-FP8")` → `ModelStatus`
   b. Print: samples collected, prediction types
6. **Phase 5 — Verify prediction-driven scheduling:**
   a. Send 20 more requests through the scheduler proxy (now using probabilistic strategy)
   b. `predict("Qwen/Qwen3-8B-VL-FP8", features={...})` → `PredictResult`
   c. Print: predicted vs actual latencies, showing the scheduler now routes based on predicted completion time rather than round-robin
7. **Phase 6 — Instance-level lifecycle demo:**
   a. `group.instances[3].terminate()` — terminate a single instance directly (demonstrates `Instance.terminate()`)
   b. Print: removed instance name and remaining instance count
8. **Cleanup:**
   a. `terminate(all=True)` — terminate remaining instances

### SDK Coverage

`serve()` (with `replicas=4`), `InstanceGroup.wait_ready()`, `InstanceGroup.endpoints`, `InstanceGroup.instances`, `Instance.terminate()`, `instances()`, `ClusterState`, `train()`, `TrainResult`, `predictor_status()`, `ModelStatus`, `predict()`, `PredictResult`, `terminate(all=True)`

### Scheduler Interaction (via httpx)

- `POST /v1/strategy/set` — set round-robin for data collection
- `POST /v1/chat/completions` — transparent proxy to backend instances

### Notes

- Each instance is fully independent — no tensor parallelism. 1 model, 4 GPUs, 4 instances.
- Round-robin cycles through all 4 instances, collecting diverse execution time samples.
- After training, the scheduler switches to probabilistic — routing decisions are based on predicted completion times, which accounts for per-instance queue depth.
- The scheduler is a transparent proxy: `POST /v1/chat/completions` to the scheduler is routed to a backend vLLM instance and the raw response is returned.
- Execution times are collected automatically by the scheduler's worker threads.

### Expected Output

```
=== Phase 1: Deploy 4 instances ===
Serving Qwen/Qwen3-8B-VL-FP8 as 'qwen3-x4' (1 GPU/instance, 4 replicas)...
Instance group 'qwen3-x4' ready.
  Instance 0: qwen3-x4-0 @ http://10.0.0.1:8080 (GPU 0)
  Instance 1: qwen3-x4-1 @ http://10.0.0.2:8080 (GPU 1)
  Instance 2: qwen3-x4-2 @ http://10.0.0.3:8080 (GPU 2)
  Instance 3: qwen3-x4-3 @ http://10.0.0.4:8080 (GPU 3)
  GPUs used: 4/4

=== Phase 2: Collect training data (round-robin, 40 requests) ===
Strategy set to: round_robin
Request 01/40 → inst-0, 285.1 ms  (max_tokens=64)
Request 02/40 → inst-1, 291.3 ms  (max_tokens=64)
Request 03/40 → inst-2, 278.9 ms  (max_tokens=64)
Request 04/40 → inst-3, 295.7 ms  (max_tokens=64)
Request 05/40 → inst-0, 523.7 ms  (max_tokens=128)
...
Request 40/40 → inst-3, 487.2 ms  (max_tokens=128)
All 40 requests completed. Execution times collected.

=== Phase 3: Train predictor ===
Training predictor for Qwen/Qwen3-8B-VL-FP8 (expect_error)...
  Trained on 40 samples.
  Strategy auto-switched to: probabilistic

=== Phase 4: Predictor status ===
  Model: Qwen/Qwen3-8B-VL-FP8
  Samples collected: 40
  Prediction types: ['expect_error']

=== Phase 5: Prediction-driven scheduling ===
Sending 20 requests (now using probabilistic strategy)...
  All 20 requests completed. Avg latency: 298.4 ms

Predicted runtime for max_tokens=256:
  Expected: 845.2 ms, Error margin: +/-72.3 ms

=== Phase 6: Instance-level lifecycle ===
Terminating single instance qwen3-x4-3...
Remaining instances: 3

=== Cleanup ===
Terminating all...
Done.
```

## Conventions

Follows the shared conventions defined in the [parent spec](2026-03-13-sdk-examples-design.md#shared-conventions): `argparse` CLI args, `async with SwarmPilotClient(...)`, `try/finally` cleanup, `from swarmpilot.errors import SwarmPilotError`.
