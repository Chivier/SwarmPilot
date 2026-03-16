# SDK Example: Predictor Workflow — Design Spec

**Parent spec:** [SDK Example Scripts](2026-03-13-sdk-examples-design.md)

## Purpose

Demonstrate the complete predictor lifecycle from cold start on a real GPU cluster: deploy a model with round-robin scheduling, send real inference requests through the scheduler's transparent proxy to collect execution time data, train the predictor, then use it for predictions.

## File

```
examples/sdk/predictor_workflow.py
```

## Prerequisites

- 1 GPU with sufficient VRAM for Qwen3-8B-VL-FP8 (~8-10 GB)
- Planner service running at `http://localhost:8002`
- Scheduler service running at `http://localhost:8000`
- A PyLet cluster for deployment
- `--scheduler-url` is **required** for this example

## `predictor_workflow.py` — Training & Prediction (Full Lifecycle)

**Goal:** Show the complete predictor lifecycle from cold start on a real GPU cluster: deploy a real model with round-robin scheduling, send real inference requests through the scheduler's transparent proxy to collect execution time data, train the predictor, then use it for predictions.

**Model:** `Qwen/Qwen3-8B-VL-FP8` running on local GPUs.

### Key Concept — Transparent Proxy

The scheduler acts as a transparent proxy. Clients send standard OpenAI-compatible requests (e.g., `POST /v1/chat/completions`) directly to the scheduler URL. The scheduler routes the request to a backend instance via its scheduling algorithm, waits for the backend's response, and returns it unmodified. The worker thread automatically measures `execution_time_ms` for each request and feeds it into the predictor's training buffer. No `/v1/task/submit` is involved.

### Why Round-Robin First

The predictor needs completed task execution times as training data. On a fresh cluster there is no data, so training would fail. The example bootstraps data by:
1. Deploying the model via the SDK with the Planner
2. Setting the scheduler to `round_robin` (no prediction needed, simple load balancing)
3. Sending real OpenAI-compatible inference requests through the scheduler's transparent proxy — the worker threads automatically measure `execution_time_ms` and feed it to the training buffer
4. Training the predictor on the collected real execution times
5. Using the trained predictor for future runtime predictions

### Flow

1. Create `SwarmPilotClient` with both planner and scheduler URLs
2. **Phase 1 — Deploy model & set round-robin:**
   a. `serve("Qwen/Qwen3-8B-VL-FP8", name="qwen3-vl", gpu=1, replicas=1)` → `InstanceGroup`
   b. `group.wait_ready(timeout=600)` — wait for vLLM to finish loading weights
   c. `POST /v1/strategy/set {"strategy_name": "round_robin"}` via `httpx` to scheduler — set round-robin
   d. Print: model deployed, strategy set
3. **Phase 2 — Generate training data (real inference via transparent proxy):**
   a. Define a set of 30 prompts with varying lengths (short, medium, long) and varying `max_tokens` (64/128/256) to produce diverse execution times
   b. Send each as a standard OpenAI-compatible request through the scheduler's transparent proxy:
      ```
      POST http://<scheduler_url>/v1/chat/completions
      {
          "model": "Qwen/Qwen3-8B-VL-FP8",
          "messages": [
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": "<prompt>"}
          ],
          "temperature": 0.7,
          "max_tokens": 128
      }
      ```
   c. Each request is synchronous — the scheduler proxy blocks until the backend responds, then returns the raw vLLM response. The worker thread measures wall-clock `execution_time_ms` automatically.
   d. Print: per-request response time as measured client-side
4. **Phase 3 — Train predictor (SDK):**
   a. `train("Qwen/Qwen3-8B-VL-FP8", prediction_type="expect_error")` → `TrainResult`
   b. Print training result (samples trained, strategy — should auto-switch to `probabilistic` if >= 10 samples)
5. **Phase 4 — Check status (SDK):**
   a. `predictor_status("Qwen/Qwen3-8B-VL-FP8")` → `ModelStatus`
   b. Print status (samples collected, prediction types, preprocessors)
6. **Phase 5 — Predict (SDK):**
   a. `predict("Qwen/Qwen3-8B-VL-FP8", features={"prompt_length": 100, "max_tokens": 256})` → `PredictResult`
   b. Print prediction (expected runtime, error margin)
7. **Cleanup:**
   a. `group.terminate()` — shut down the model instance

### SDK Coverage

`serve()`, `InstanceGroup.wait_ready()`, `InstanceGroup.terminate()`, `train()`, `TrainResult`, `predictor_status()`, `ModelStatus`, `predict()`, `PredictResult`

### Scheduler Interaction (via httpx)

- `POST /v1/strategy/set` — set scheduling strategy (scheduler internal endpoint)
- `POST /v1/chat/completions` — standard OpenAI-compatible request, transparently proxied to backend instance

### Notes

- The scheduler is a **transparent proxy**: `POST /v1/chat/completions` to the scheduler is routed to a backend vLLM instance and the raw response is returned. No task IDs, no polling — just standard HTTP request/response.
- Execution times are collected **automatically** by the scheduler's worker threads. The worker measures wall-clock time for the HTTP call to the backend and feeds it into the training buffer.
- Requests are sent sequentially with a small delay between them to avoid overwhelming a single replica.

### Expected Output

```
=== Phase 1: Deploy model & set round-robin ===
Serving Qwen/Qwen3-8B-VL-FP8 as 'qwen3-vl' (1 GPU, 1 replica)...
Model ready at: ['http://10.0.0.1:8080']
Scheduler strategy set to: round_robin

=== Phase 2: Generate training data (30 inference requests via proxy) ===
Request 01/30: "Hello"                    (max_tokens=64)  → 187.3 ms
Request 02/30: "Explain quantum physics"  (max_tokens=128) → 423.1 ms
Request 03/30: "Write a short essay..."   (max_tokens=256) → 891.7 ms
...
Request 30/30: "Summarize the history..." (max_tokens=128) → 512.4 ms
All 30 requests completed. Execution times collected by scheduler.

=== Phase 3: Train predictor ===
Training predictor for Qwen/Qwen3-8B-VL-FP8 (expect_error)...
  Trained on 30 samples.
  Strategy auto-switched to: probabilistic

=== Phase 4: Check predictor status ===
Predictor status:
  Model: Qwen/Qwen3-8B-VL-FP8
  Samples collected: 30
  Prediction types: ['expect_error']

=== Phase 5: Predict runtime ===
Predicting for prompt_length=100, max_tokens=256...
  Expected runtime: 845.2 ms
  Error margin: +/-72.3 ms

=== Cleanup ===
Terminating qwen3-vl...
Done.
```

## Conventions

Follows the shared conventions defined in the [parent spec](2026-03-13-sdk-examples-design.md#shared-conventions): `argparse` CLI args, `async with SwarmPilotClient(...)`, `try/finally` cleanup, `from swarmpilot.errors import SwarmPilotError`.
