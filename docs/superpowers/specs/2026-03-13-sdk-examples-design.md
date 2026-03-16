# SDK Example Scripts — Design Spec

## Purpose

Provide 4 self-contained async Python examples that demonstrate the core `SwarmPilotClient` SDK API. These replace the need for users to piece together curl commands and shell scripts — each example shows a complete workflow end-to-end.

**Related specs (advanced examples):**
- [Predictor Workflow](2026-03-13-sdk-example-predictor-workflow.md) — deploy → proxy inference → train predictor → predict (1 GPU)
- [Multi-GPU Single Model](2026-03-13-sdk-example-multi-gpu-single-model.md) — 4 instances on 4 GPUs with predictor lifecycle
- [Multi-GPU Multi Model](2026-03-13-sdk-example-multi-gpu-multi-model.md) — 2 models across 4 GPUs with per-model predictors

## Target Audience

Developers integrating SwarmPilot into Python applications. Assumes familiarity with `async/await` but no prior SwarmPilot knowledge.

## File Structure

```
examples/sdk/
├── README.md                    # Overview, prerequisites, how to run
├── quickstart.py                # Example 1: Basic serve → scale → terminate (1 GPU)
├── multi_model_deploy.py        # Example 2: Serve multiple models independently
├── optimized_deploy.py          # Example 3: Register → optimize → deploy → inspect
├── vllm_wrapper.py              # Custom vLLM wrapper script (deployed by run())
└── custom_vllm_wrapper.py       # Example 4: Deploy wrapper as managed workload → test
```

See related specs for additional examples:
- `predictor_workflow.py` — defined in [separate spec](2026-03-13-sdk-example-predictor-workflow.md)
- `multi_gpu_single_model.py` — defined in [separate spec](2026-03-13-sdk-example-multi-gpu-single-model.md)
- `multi_gpu_multi_model.py` — defined in [separate spec](2026-03-13-sdk-example-multi-gpu-multi-model.md)

## Prerequisites

All examples in this spec require:
- SwarmPilot installed (`uv sync`)
- Planner service running at `http://localhost:8002` (default)
- A PyLet cluster for actual deployment
- Local GPU(s) with sufficient VRAM for the target model

**Default model:** `Qwen/Qwen3-8B-VL-FP8` (FP8 quantized, ~8-10 GB VRAM).

| Example | GPUs required | Models |
|---------|---------------|--------|
| 1. quickstart | 1 (min), 2 (for scale) | Qwen3-8B-VL-FP8 |
| 2. multi_model_deploy | 3+ | Qwen3-8B-VL-FP8, Qwen2.5-7B |
| 3. optimized_deploy | 3+ | Qwen3-8B-VL-FP8, Qwen2.5-7B |
| 4. custom_vllm_wrapper | 1 | Qwen3-8B-VL-FP8 |

Each example accepts `--planner-url` and `--scheduler-url` CLI arguments to override defaults.

## Example Specifications

### 1. `quickstart.py` — Basic Serve & Scale

**Goal:** Minimal "hello world" showing the most common SDK workflow.

**Flow:**
1. Create `SwarmPilotClient` with planner URL
2. `serve("Qwen/Qwen3-8B-VL-FP8", name="qwen3-vl", gpu=1, replicas=1)` → `InstanceGroup`
3. `group.wait_ready(timeout=600)` — block until healthy (sync, blocks the event loop; 600s timeout per instance for model loading)
4. Print `group.endpoints` — show available endpoints
5. Iterate `group.instances` — print each `Instance` name, status, endpoint
6. `group.scale(2)` — scale to 2 replicas (sync method on `InstanceGroup`). Note: the async alternative `await sp.scale(model, replicas)` achieves the same via `SwarmPilotClient`.
7. Wait for new instance: `group.instances[1].wait_ready(timeout=600)` — verify the newly added instance individually (demonstrates `Instance.wait_ready()`)
8. Print updated endpoint list
9. `group.terminate()` — cleanup (sync method)

**SDK coverage:** `SwarmPilotClient`, `serve()` (with `name=`), `InstanceGroup.wait_ready()`, `InstanceGroup.endpoints`, `InstanceGroup.instances`, `Instance` (iterate and print), `Instance.wait_ready()`, `InstanceGroup.scale()`, `InstanceGroup.terminate()`. Also notes `SwarmPilotClient.scale()` as async alternative.

**Error handling:** `try/finally` ensuring `terminate()` runs on failure. Catch `SwarmPilotError` subclasses (imported from `swarmpilot.errors`).

**Output:**
```
Serving Qwen/Qwen3-8B-VL-FP8 as 'qwen3-vl' with 1 replica...
Instance group 'qwen3-vl' ready.
Endpoints: ['http://10.0.0.1:8080']
Instances:
  qwen3-vl-0: active @ http://10.0.0.1:8080
Scaling to 2 replicas...
Waiting for new instance qwen3-vl-1...
Instance qwen3-vl-1 ready.
Endpoints: ['http://10.0.0.1:8080', 'http://10.0.0.2:8080']
Terminating...
Done.
```

### 2. `multi_model_deploy.py` — Multi-Model Deployment via Serve

**Goal:** Show how to deploy multiple models independently using `serve()` for each, then inspect the resulting cluster state. This is the simplest multi-model pattern — each `serve()` call deploys one model immediately without global optimization.

**Flow:**
1. Create `SwarmPilotClient`
2. `serve("Qwen/Qwen3-8B-VL-FP8", name="qwen3-vl", gpu=1, replicas=2)` → `InstanceGroup` — deploy first model
3. `group_a.wait_ready(timeout=600)` — wait for first model
4. Print: first model endpoints and instances
5. `serve("Qwen/Qwen2.5-7B", name="qwen25", gpu=1, replicas=1)` → `InstanceGroup` — deploy second model
6. `group_b.wait_ready(timeout=600)` — wait for second model
7. Print: second model endpoints and instances
8. `instances()` → `ClusterState` — show full cluster state with both models
9. `schedulers()` — show model-to-scheduler mapping
10. `terminate(all=True)` — cleanup everything

**SDK coverage:** `serve()` (multiple calls), `InstanceGroup.wait_ready()`, `InstanceGroup.endpoints`, `InstanceGroup.instances`, `instances()`, `ClusterState`, `schedulers()`, `terminate(all=True)`

**Notes:**
- Each `serve()` call deploys immediately and independently — no global GPU placement optimization.
- Requires enough local GPUs for both models. Adjust model names and GPU counts to match available hardware.
- Compare with Example 3 (`optimized_deploy.py`) which uses the optimizer for coordinated placement.

**Output:**
```
=== Deploy model A ===
Serving Qwen/Qwen3-8B-VL-FP8 as 'qwen3-vl' (1 GPU, 2 replicas)...
Model 'qwen3-vl' ready.
  Endpoints: ['http://10.0.0.1:8080', 'http://10.0.0.2:8080']
  Instances: qwen3-vl-0 (active), qwen3-vl-1 (active)

=== Deploy model B ===
Serving Qwen/Qwen2.5-7B as 'qwen25' (1 GPU, 1 replica)...
Model 'qwen25' ready.
  Endpoints: ['http://10.0.0.3:8080']
  Instances: qwen25-0 (active)

=== Cluster state ===
Instances: 3, Processes: 0, Groups: 2
Schedulers: {'Qwen/Qwen3-8B-VL-FP8': 'http://...', 'Qwen/Qwen2.5-7B': 'http://...'}

=== Cleanup ===
Terminating all...
Done.
```

### 3. `optimized_deploy.py` — Optimizer-Driven Deployment

**Goal:** Show the `register()` + `deploy()` pattern for coordinated multi-model GPU placement. Unlike `serve()` which deploys each model immediately, this pattern collects all model requirements first, then the Planner's SwarmOptimizer (simulated annealing + integer programming) computes optimal GPU assignments across all models before deploying.

**When to use this pattern:**
- Multiple models competing for limited GPU resources
- Need optimal GPU partitioning (memory, compute balance)
- Want the Planner to decide placement rather than manual assignment

**Flow:**
1. Create `SwarmPilotClient`
2. `register("Qwen/Qwen3-8B-VL-FP8", gpu=1, replicas=2)` — register first model's requirements
3. `register("Qwen/Qwen2.5-7B", gpu=1, replicas=1)` — register second model's requirements
4. Print: registered models (can verify via planner's `GET /v1/registered`)
5. `deploy()` → `DeploymentResult` — optimizer runs GPU placement and deploys all models
6. `result.wait_ready()` — wait for all groups (600s timeout per group)
7. Print: deployment plan (optimizer output)
8. Iterate `result["model_name"]` — show per-model `InstanceGroup`, instances, endpoints
9. `instances()` → `ClusterState` — show full cluster state
10. `schedulers()` — show model-to-scheduler mapping
11. `terminate(all=True)` — cleanup everything

**SDK coverage:** `register()`, `deploy()`, `DeploymentResult`, `DeploymentResult.__getitem__()`, `DeploymentResult.wait_ready()`, `DeploymentResult.plan`, `instances()`, `ClusterState`, `schedulers()`, `terminate(all=True)`

**Notes:**
- `register()` only stores requirements — no instances are launched until `deploy()`.
- `deploy()` calls the Planner's `apply_deployment()` which uses the SwarmOptimizer to compute placement, then deploys all models in the optimal configuration.
- The `DeploymentResult.plan` dict contains the optimizer's placement decisions (which model on which GPU).
- Compare with Example 2 (`multi_model_deploy.py`) which deploys each model independently via `serve()`.

**Output:**
```
=== Register models ===
Registering Qwen/Qwen3-8B-VL-FP8 (1 GPU, 2 replicas)...
Registering Qwen/Qwen2.5-7B (1 GPU, 1 replica)...
Registered 2 models (total: 3 GPUs requested).

=== Deploy with optimizer ===
Running optimizer...
Deployment ready (status: completed).
Optimization plan: {'Qwen/Qwen3-8B-VL-FP8': [0, 1], 'Qwen/Qwen2.5-7B': [2]}

Model: Qwen/Qwen3-8B-VL-FP8
  Instances: 2
  Endpoints: ['http://10.0.0.1:8080', 'http://10.0.0.2:8080']

Model: Qwen/Qwen2.5-7B
  Instances: 1
  Endpoints: ['http://10.0.0.3:8080']

=== Cluster state ===
Instances: 3, Processes: 0, Groups: 2
Schedulers: {'Qwen/Qwen3-8B-VL-FP8': 'http://...', 'Qwen/Qwen2.5-7B': 'http://...'}

=== Cleanup ===
Terminating all...
Done.
```

### 4. `custom_vllm_wrapper.py` — Custom vLLM Wrapper as Managed Workload

**Goal:** Show how to write a custom vLLM wrapper script and deploy it as a managed workload via `sp.run()`. This pattern is useful when you need custom preprocessing, postprocessing, authentication, or additional endpoints beyond what stock vLLM provides.

**Two files:**
- `vllm_wrapper.py` — the wrapper script (deployed as a process)
- `custom_vllm_wrapper.py` — the SDK example that deploys and tests it

#### `vllm_wrapper.py` — The Wrapper Script

A lightweight FastAPI server that wraps vLLM with custom logic:

```python
"""Custom vLLM wrapper with preprocessing and health monitoring.

Wraps a vLLM backend with:
- Input validation and sanitization
- Token counting and request logging
- Custom /health endpoint with model stats
- Configurable via environment variables
"""
```

**Features:**
- Launches vLLM as a subprocess (`python -m vllm.entrypoints.openai.api_server ...`)
- Waits for vLLM's `/health` endpoint to become ready
- Exposes its own FastAPI server on `$WRAPPER_PORT` (default 8080):
  - `POST /v1/chat/completions` — validates input, forwards to vLLM, logs response stats (tokens, latency)
  - `GET /health` — returns wrapper + vLLM health status, request count, avg latency
- Configurable via env vars: `VLLM_MODEL`, `VLLM_PORT` (internal), `WRAPPER_PORT`, `VLLM_GPU_MEMORY_UTILIZATION`

**Key code patterns:**
```python
# Forward request to vLLM with preprocessing
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    # Validate & sanitize
    body.setdefault("temperature", 0.7)
    body["max_tokens"] = min(body.get("max_tokens", 512), 2048)
    # Forward to vLLM
    resp = await http_client.post(
        f"http://localhost:{VLLM_PORT}/v1/chat/completions",
        json=body,
    )
    # Log stats
    result = resp.json()
    log_request_stats(body, result, elapsed_ms)
    return JSONResponse(content=result, status_code=resp.status_code)
```

#### `custom_vllm_wrapper.py` — The Example Script

**Flow:**
1. Create `SwarmPilotClient` with planner URL
2. **Deploy the wrapper:**
   a. `run("python vllm_wrapper.py", name="custom-qwen3", gpu=1)` → `Process`
      - The Planner launches the wrapper script with 1 GPU allocated
      - The wrapper internally starts vLLM and exposes its own endpoint
   b. Print process info (name, status, endpoint)
3. **Verify deployment:**
   a. `instances()` → `ClusterState` — show the process in the cluster
   b. Print processes from cluster state
   c. Wait for wrapper's `/health` to return ready (poll with `httpx`)
4. **Send test inference request:**
   a. Send `POST /v1/chat/completions` to the wrapper's endpoint directly (not through scheduler proxy — this is a standalone process, not a scheduler-managed instance)
   b. Print the response and wrapper-added stats
5. **Check wrapper health:**
   a. `GET /health` on the wrapper endpoint
   b. Print: request count, avg latency, vLLM status
6. **Cleanup:**
   a. `process.terminate()` — stop the process (sync method, kills wrapper + vLLM subprocess)

**SDK coverage:** `run()`, `Process`, `Process.terminate()`, `instances()`, `ClusterState.processes`

**Notes:**
- The wrapper script must be on the machine where the Planner launches processes. In a local dev setup, this is the same machine.
- `sp.run()` passes the command string to the Planner, which manages the process lifecycle (start, monitor, terminate).
- The wrapper script handles its own vLLM subprocess lifecycle — when terminated, it should clean up the vLLM process via signal handling.
- Environment variables (`VLLM_MODEL`, `VLLM_PORT`, etc.) can be set by the Planner or hardcoded in the wrapper for the example.
- The wrapper uses `Qwen/Qwen3-8B-VL-FP8` as the default model.

**Output:**
```
=== Deploy custom vLLM wrapper ===
Running 'python vllm_wrapper.py' as 'custom-qwen3' (1 GPU)...
Process started:
  Name: custom-qwen3
  Status: running
  Endpoint: http://10.0.0.1:8080

Cluster state:
  Instances: 0
  Processes: 1 (custom-qwen3: running)

Waiting for wrapper to become ready...
Wrapper ready (vLLM model loaded).

=== Test inference request ===
POST /v1/chat/completions → 200 OK
Response: "Hello! How can I help you today?"
  Tokens: prompt=25, completion=9
  Latency: 312.5 ms

=== Wrapper health check ===
GET /health → 200 OK
  vLLM status: healthy
  Total requests: 1
  Avg latency: 312.5 ms

=== Cleanup ===
Terminating process custom-qwen3...
Done.
```

## Shared Conventions

### Structure
Each example follows the same template:
```python
"""<Title>

Demonstrates: <SDK methods covered>

Prerequisites:
    - Planner running at http://localhost:8002
    - [Additional requirements]

Usage:
    python quickstart.py [--planner-url URL] [--scheduler-url URL]
"""

from __future__ import annotations

import argparse
import asyncio

from swarmpilot.errors import SwarmPilotError
from swarmpilot.sdk import SwarmPilotClient


async def main(planner_url: str, scheduler_url: str | None = None) -> None:
    """<One-line description>."""
    async with SwarmPilotClient(planner_url, scheduler_url) as sp:
        # ... workflow ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--planner-url", default="http://localhost:8002")
    parser.add_argument("--scheduler-url", default=None)
    args = parser.parse_args()
    asyncio.run(main(args.planner_url, args.scheduler_url))
```

### Error handling
- `try/finally` for cleanup in examples that create resources
- Import errors from `swarmpilot.errors` (not re-exported by `swarmpilot.sdk`)
- Catch and print `SwarmPilotError` subclasses with helpful messages
- No bare `except` clauses

### Sync vs async note
Lifecycle methods on model dataclasses (`InstanceGroup.wait_ready()`, `.scale()`, `.terminate()`, `Instance.wait_ready()`, `.terminate()`, `Process.terminate()`) are **synchronous** (blocking) calls using `httpx` sync client. `SwarmPilotClient` methods are all **async**. Each example includes a brief comment noting this distinction where relevant.

### Timeout semantics
`wait_ready(timeout=600)` timeouts are **per instance** at the `Instance` level, **per instance** at the `InstanceGroup` level (iterates instances), and **per group** at the `DeploymentResult` level (iterates groups). Total wall-clock time may exceed the timeout parameter if multiple instances/groups are present.

### Printing
- Use `print()` with clear status messages at each step
- No logging framework — keep examples simple
- Show actual return values from SDK calls

### CLI arguments
- `--planner-url` (default `http://localhost:8002`)
- `--scheduler-url` (default `None`, required for predictor-related examples)
- Use `argparse` (stdlib, no extra dependency)

### Direct httpx usage
Examples involving predictor training (in separate specs) use `httpx.AsyncClient` directly for two purposes:
1. **Scheduler internal endpoint** — `POST /v1/strategy/set` to configure round-robin (not covered by SDK)
2. **Transparent proxy** — `POST /v1/chat/completions` sent to the scheduler URL, which routes to the backend vLLM instance and returns the raw response. This is how real clients use SwarmPilot — the scheduler is a drop-in proxy, not a task queue.

The scheduler worker threads automatically collect execution times for all proxied requests. Examples 1-4 in this spec use only the SDK.

## README.md

The README covers:
1. **Overview** — what the examples demonstrate
2. **Prerequisites** — services that must be running
3. **Quick start** — copy-paste command to run each example
4. **Example descriptions** — one paragraph per example with SDK methods covered
5. **Configuration** — environment variables and CLI arguments

## Success Criteria

- Each example runs end-to-end against a live SwarmPilot cluster
- Each example is self-contained (no imports between examples)
- Every public `SwarmPilotClient` method appears in at least one example (across all specs)
- Every SDK model dataclass appears in at least one example (across all specs)
- Examples follow project code conventions (Google style, 80-char lines, type hints)

## Out of Scope

- Mock servers or test fixtures (examples target real clusters)
- Jupyter notebooks
- Async generator / streaming patterns (not supported by current SDK)
- WebSocket examples (internal to scheduler, not exposed in SDK)
