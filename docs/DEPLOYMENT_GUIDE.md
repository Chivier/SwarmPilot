# Deployment Guide

How to deploy workloads onto SwarmPilot instances: direct instance startup,
Planner-managed deployment, and service teardown.

For cluster startup see [SYSTEM_WORKFLOW.md](SYSTEM_WORKFLOW.md). For API
details see [API_REFERENCE.md](API_REFERENCE.md).

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Option A: Direct Instance Deployment](#option-a-direct-instance-deployment)
  - [A1. Start an Instance Process](#a1-start-an-instance-process)
  - [A2. Register with the Scheduler](#a2-register-with-the-scheduler)
  - [A3. Verify](#a3-verify)
- [Option B: Planner-Managed Deployment](#option-b-planner-managed-deployment)
  - [B1. Manual Target State](#b1-manual-target-state)
  - [B2. Optimizer-Driven Deployment](#b2-optimizer-driven-deployment)
  - [B3. Single-Model Scaling](#b3-single-model-scaling)
  - [B4. Deployment Status](#b4-deployment-status)
- [Option C: SDK Deployment (Python)](#option-c-sdk-deployment-python)
  - [C1. Deploy a Model](#c1-deploy-a-model)
  - [C2. Run a Custom Workload](#c2-run-a-custom-workload)
  - [C3. Register + Deploy (Optimized)](#c3-register--deploy-optimized)
  - [C4. Scale, List, and Terminate](#c4-scale-list-and-terminate)
  - [C5. Error Handling](#c5-error-handling)
- [Option D: CLI Deployment (`splanner`)](#option-d-cli-deployment-splanner)
  - [D1. Deploy a Model](#d1-deploy-a-model)
  - [D2. Run a Custom Workload](#d2-run-a-custom-workload)
  - [D3. Register + Deploy (Optimized)](#d3-register--deploy-optimized)
  - [D4. List, Scale, and Terminate](#d4-list-scale-and-terminate)
- [Predictor Management](#predictor-management)
- [Legacy REST API](#legacy-rest-api)
- [Stopping Services](#stopping-services)
  - [Drain a Single Instance](#drain-a-single-instance)
  - [Remove a Single Instance](#remove-a-single-instance)
  - [Terminate All PyLet Instances](#terminate-all-pylet-instances)
  - [Full Cluster Shutdown](#full-cluster-shutdown)
- [Instance Server Contract](#instance-server-contract)
- [Backend-Specific Launch Commands](#backend-specific-launch-commands)
- [Reference: Complete Examples](#reference-complete-examples)

---

## Prerequisites

A running SwarmPilot cluster. At minimum:

- **Scheduler** running on a known port (e.g., `http://localhost:8000`)
- **Planner** running (only needed for Option B, e.g., `http://localhost:8002`)
- **PyLet cluster** running (only needed for Planner-managed deployments)

See [SYSTEM_WORKFLOW.md](SYSTEM_WORKFLOW.md) Section 1 for startup instructions.

---

## Option A: Direct Instance Deployment

Use this when you want full control over instance processes without the
Planner or PyLet. You start the instance process yourself and register it
with the Scheduler via HTTP.

### A1. Start an Instance Process

An instance is any HTTP server that implements the
[Instance Server Contract](#instance-server-contract). SwarmPilot supports
any backend — the Scheduler only cares that the instance accepts HTTP
requests at the registered endpoint.

**vLLM:**

```bash
PORT=8100 vllm serve Qwen/Qwen3-0.6B \
  --port 8100 --host 0.0.0.0
```

**SGLang:**

```bash
PORT=8100 python -m sglang.launch_server \
  --model Qwen/Qwen3-0.6B --port 8100 --host 0.0.0.0
```

**Mock server** (for testing, from `examples/llm_cluster/mock_vllm_server.py`):

```bash
MODEL_ID=llm-fast PORT=8100 python examples/llm_cluster/mock_vllm_server.py
```

Wait for the health endpoint to respond before registering:

```bash
# Poll until the instance is ready
until curl -sf http://localhost:8100/health > /dev/null 2>&1; do
  sleep 1
done
echo "Instance healthy"
```

### A2. Register with the Scheduler

Once the instance is healthy, register it with the Scheduler:

```bash
curl -X POST http://localhost:8000/v1/instance/register \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "inst-001",
    "model_id": "Qwen/Qwen3-0.6B",
    "endpoint": "http://localhost:8100",
    "platform_info": {
      "software_name": "vllm",
      "software_version": "0.4.0",
      "hardware_name": "gpu-node-1"
    }
  }'
```

**Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `instance_id` | Yes | Unique identifier for this instance |
| `model_id` | Yes | Model this instance serves |
| `endpoint` | Yes | Reachable HTTP URL (with `http://` prefix) |
| `platform_info.software_name` | Yes | Backend name (`vllm`, `sglang`, etc.) |
| `platform_info.software_version` | Yes | Backend version string |
| `platform_info.hardware_name` | Yes | Hardware identifier (e.g., `A100`, `gpu-node-1`) |

The Scheduler creates a dedicated worker queue thread for the instance and
sets its status to `INITIALIZING`. After background task redistribution
completes, the status transitions to `ACTIVE`.

**Register multiple instances** (e.g., 3 replicas on sequential ports):

```bash
for i in 1 2 3; do
  PORT=$((8099 + i))
  curl -X POST http://localhost:8000/v1/instance/register \
    -H "Content-Type: application/json" \
    -d "{
      \"instance_id\": \"inst-$(printf '%03d' $i)\",
      \"model_id\": \"Qwen/Qwen3-0.6B\",
      \"endpoint\": \"http://localhost:${PORT}\",
      \"platform_info\": {
        \"software_name\": \"vllm\",
        \"software_version\": \"0.4.0\",
        \"hardware_name\": \"gpu-node-1\"
      }
    }"
done
```

### A3. Verify

```bash
# List all registered instances
curl http://localhost:8000/v1/instance/list | python -m json.tool

# Check a specific instance
curl "http://localhost:8000/v1/instance/info?instance_id=inst-001"

# Submit a test task
curl -X POST http://localhost:8000/v1/task/submit \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test-001",
    "model_id": "Qwen/Qwen3-0.6B",
    "task_input": {"prompt": "Hello", "max_tokens": 10}
  }'
```

---

## Option B: Planner-Managed Deployment

Use this when you want the Planner to handle the instance lifecycle
(deploy, health check, register, drain, terminate) through PyLet. The
Planner requires `PYLET_ENABLED=true` and a running PyLet cluster.

### B1. Manual Target State

Specify the exact number of instances per model. The Planner reconciles
current state toward the target by adding or removing instances.

```bash
curl -X POST http://localhost:8002/v1/deploy_manually \
  -H "Content-Type: application/json" \
  -d '{
    "target_state": {
      "Qwen/Qwen3-0.6B": 3,
      "meta-llama/Llama-3-8B": 2
    }
  }'
```

**What happens internally:**

1. Planner computes diff: current state vs target state
2. **Removes excess instances first** (selects oldest instances, FIFO)
   - Drain → deregister from Scheduler → cancel via PyLet
3. **Deploys new instances** via `pylet.submit()`
   - PyLet runs the launch command for the configured backend
   - Planner waits for instance to reach RUNNING state
   - Planner polls `/health` until 200 OK
   - Planner registers instance with the appropriate Scheduler
4. Returns deployment result

**Request fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `target_state` | Yes | `{model_id: instance_count}` |
| `scheduler_url` | No | Override scheduler URL (default: from registry) |

**Response:**

```json
{
  "status": "success",
  "current_state": {"Qwen/Qwen3-0.6B": 3, "meta-llama/Llama-3-8B": 2},
  "actions_taken": {
    "added": {"Qwen/Qwen3-0.6B": 3, "meta-llama/Llama-3-8B": 2},
    "removed": {}
  }
}
```

### B2. Optimizer-Driven Deployment

Let the Planner compute the optimal model-to-machine allocation, then
deploy the result. Uses integer programming (PuLP) or simulated annealing.

```bash
curl -X POST http://localhost:8002/v1/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "M": 10,
    "N": 3,
    "B": [
      [5.0, 3.0, 1.0],
      [5.0, 3.0, 1.0],
      [5.0, 3.0, 1.0],
      [5.0, 3.0, 1.0],
      [5.0, 3.0, 1.0],
      [5.0, 3.0, 1.0],
      [5.0, 3.0, 1.0],
      [5.0, 3.0, 1.0],
      [5.0, 3.0, 1.0],
      [5.0, 3.0, 1.0]
    ],
    "target": [50.0, 30.0, 20.0],
    "model_ids": ["llm-fast", "llm-medium", "llm-slow"],
    "algorithm": "simulated_annealing",
    "method": "relative_error"
  }'
```

**Request fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `M` | Yes | Number of machines (rows in capacity matrix) |
| `N` | Yes | Number of models (columns in capacity matrix) |
| `B` | Yes | Capacity matrix `[M x N]` — throughput each machine achieves per model |
| `target` | Yes | Target request distribution (length N) |
| `model_ids` | Yes | Model identifiers (length N), must match registered schedulers |
| `algorithm` | No | `simulated_annealing` (default) or `integer_programming` |
| `method` | No | Objective: `relative_error` (default) or `ratio_difference` |
| `initial` | No | Current assignment array (length M, `-1` = unassigned) |
| `a` | No | Change constraint: fraction of machines that can change (0.0–1.0) |

**Response:**

```json
{
  "deployment_success": true,
  "deployment": [0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
  "service_capacity": [25.0, 9.0, 2.0],
  "score": 0.0012,
  "message": "Optimization complete"
}
```

The `deployment` array maps each machine index to a model index.

### B3. Single-Model Scaling

Scale a specific model to a target replica count:

```bash
curl -X POST http://localhost:8002/v1/scale \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "Qwen/Qwen3-0.6B",
    "target_count": 5
  }'
```

If the model currently has 2 instances, this adds 3 more. If it has 7,
this removes 2 (oldest first).

### B4. Deployment Status

Check the current state of all PyLet-managed instances:

```bash
curl http://localhost:8002/v1/status | python -m json.tool
```

Returns the PyLet service status including active instances, their states,
and model assignments.

---

## Option C: SDK Deployment (Python)

Use the `SwarmPilotClient` async client to deploy and manage models
programmatically. The client wraps the Planner REST API with typed
dataclass responses.

### C1. Deploy a Model

```python
from swarmpilot.sdk import SwarmPilotClient

async with SwarmPilotClient("http://localhost:8002") as sp:
    group = await sp.serve(
        "Qwen/Qwen2.5-7B",
        gpu=1,
        replicas=2,
        scheduler="auto",
    )
    print(group.endpoints)
```

`serve()` returns an `InstanceGroup` with access to individual
`Instance` objects and their endpoints. Set `scheduler="auto"` to
let the Planner resolve the scheduler from its registry, or pass an
explicit URL.

### C2. Run a Custom Workload

```python
async with SwarmPilotClient("http://localhost:8002") as sp:
    proc = await sp.run(
        "python train.py --epochs 10",
        name="train-job",
        gpu=2,
    )
    print(proc.endpoint)
```

### C3. Register + Deploy (Optimized)

Register model requirements first, then trigger a single optimized
deployment:

```python
async with SwarmPilotClient("http://localhost:8002") as sp:
    await sp.register("Qwen/Qwen2.5-7B", gpu=1, replicas=3)
    await sp.register("meta-llama/Llama-3-8B", gpu=2, replicas=2)
    result = await sp.deploy()
    print(result.status)
    print(result["Qwen/Qwen2.5-7B"].endpoints)
```

### C4. Scale, List, and Terminate

```python
async with SwarmPilotClient("http://localhost:8002") as sp:
    # Scale a model
    group = await sp.scale("Qwen/Qwen2.5-7B", replicas=5)

    # List all instances
    state = await sp.instances()
    for inst in state.instances:
        print(f"{inst.name}  {inst.model}  {inst.status}")

    # Get scheduler mapping
    mapping = await sp.schedulers()

    # Terminate by model
    await sp.terminate(model="Qwen/Qwen2.5-7B")

    # Terminate all
    await sp.terminate(all=True)
```

### C5. Error Handling

The SDK raises structured exceptions from `swarmpilot.errors`:

| Exception | Trigger |
|-----------|---------|
| `ModelNotDeployed` | 404 from Planner (model not found) |
| `SchedulerNotFound` | 404 from Scheduler (no scheduler for model) |
| `DeployError` | Instance enters `"failed"` status |
| `SwarmPilotTimeoutError` | Instance not ready within timeout |
| `ValueError` | 400 (bad request) |

---

## Option D: CLI Deployment (`splanner`)

The `splanner` CLI provides the same deployment operations as the
SDK. All commands accept `--planner-url` (or `PLANNER_URL` env var).

### D1. Deploy a Model

```bash
# Deploy a model with vllm (auto-generated command)
splanner serve Qwen/Qwen2.5-7B --gpu 1 --replicas 2

# Deploy with a custom command
splanner serve "vllm serve my-model" --replicas 1 --name my-deploy

# Specify scheduler
splanner serve Qwen/Qwen2.5-7B --scheduler http://localhost:8000
```

### D2. Run a Custom Workload

```bash
splanner run "python train.py" --name my-job --gpu 2
```

### D3. Register + Deploy (Optimized)

```bash
splanner register Qwen/Qwen2.5-7B --gpu 1 --replicas 3
splanner register meta-llama/Llama-3-8B --gpu 2 --replicas 2
splanner deploy
```

### D4. List, Scale, and Terminate

```bash
# List all instances
splanner ps

# Scale a model
splanner scale Qwen/Qwen2.5-7B --replicas 5

# Show scheduler mapping
splanner schedulers

# Terminate by name, model, or all
splanner terminate my-deployment
splanner terminate --model Qwen/Qwen2.5-7B
splanner terminate --all
```

---

## Predictor Management

The Scheduler exposes predictor endpoints under `/v1/predictor/`
for training, prediction, and status queries. These are used by the
SDK `train()`, `predict()`, and `predictor_status()` methods (which
require `scheduler_url` to be set on the client).

### Train the Predictor

```bash
curl -X POST http://localhost:8000/v1/predictor/train \
  -H "Content-Type: application/json" \
  -d '{"model_id": "Qwen/Qwen2.5-7B"}'
```

Or via SDK:

```python
async with SwarmPilotClient(
    "http://localhost:8002",
    scheduler_url="http://localhost:8000",
) as sp:
    result = await sp.train("Qwen/Qwen2.5-7B")
    print(result.samples_trained, result.metrics)
```

### Manual Prediction

```bash
curl -X POST http://localhost:8000/v1/predictor/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "Qwen/Qwen2.5-7B",
    "features": {"token_count": 150},
    "platform_info": {
      "software_name": "vllm",
      "software_version": "0.4.0",
      "hardware_name": "A100"
    }
  }'
```

### Predictor Status

```bash
curl http://localhost:8000/v1/predictor/status/Qwen%2FQwen2.5-7B
```

### List Trained Models

```bash
curl http://localhost:8000/v1/predictor/models
```

---

## Legacy REST API

Options A and B above (direct instance deployment and Planner-managed
deployment via `curl`) remain fully supported. The SDK and CLI are
higher-level wrappers around the same Planner and Scheduler REST
endpoints.

---

## Stopping Services

### Drain a Single Instance

Draining stops new task assignments while allowing in-flight tasks to
complete. This is the first step in graceful removal.

```bash
curl -X POST http://localhost:8000/v1/instance/drain \
  -H "Content-Type: application/json" \
  -d '{"instance_id": "inst-001"}'
```

**Response:**

```json
{
  "instance_id": "inst-001",
  "status": "draining",
  "pending_tasks": 3,
  "estimated_completion_ms": 1500.0
}
```

**Poll drain status** until safe to remove:

```bash
curl "http://localhost:8000/v1/instance/drain/status?instance_id=inst-001"
```

```json
{
  "instance_id": "inst-001",
  "pending_tasks": 0,
  "can_remove": true
}
```

### Remove a Single Instance

After draining (or immediately if you don't need graceful shutdown):

```bash
curl -X POST http://localhost:8000/v1/instance/remove \
  -H "Content-Type: application/json" \
  -d '{"instance_id": "inst-001"}'
```

This deregisters the instance from the Scheduler and stops its worker queue
thread. Note: removal is allowed even with pending tasks — callbacks from
already-dispatched tasks are still accepted.

**Full graceful removal sequence:**

```bash
INSTANCE_ID="inst-001"
SCHEDULER="http://localhost:8000"

# 1. Drain
curl -X POST ${SCHEDULER}/v1/instance/drain \
  -H "Content-Type: application/json" \
  -d "{\"instance_id\": \"${INSTANCE_ID}\"}"

# 2. Wait for drain to complete
while true; do
  STATUS=$(curl -s "${SCHEDULER}/v1/instance/drain/status?instance_id=${INSTANCE_ID}")
  CAN_REMOVE=$(echo "$STATUS" | python -c "import sys,json; print(json.load(sys.stdin).get('can_remove', False))")
  if [ "$CAN_REMOVE" = "True" ]; then
    break
  fi
  sleep 1
done

# 3. Remove from scheduler
curl -X POST ${SCHEDULER}/v1/instance/remove \
  -H "Content-Type: application/json" \
  -d "{\"instance_id\": \"${INSTANCE_ID}\"}"

# 4. Stop the instance process
kill $(cat instance_${INSTANCE_ID}.pid)
```

### Terminate All PyLet Instances

When using Planner-managed deployment, terminate all instances at once:

```bash
curl -X POST http://localhost:8002/v1/terminate-all
```

This triggers the full lifecycle for each instance:
1. Set status to `DRAINING`
2. Drain instance on Scheduler (stop new tasks)
3. Poll drain status until `can_remove` or pending tasks = 0
4. Remove instance from Scheduler
5. Cancel instance on PyLet (`pylet.cancel(delete=True)`)
6. Set status to `TERMINATED`

### Full Cluster Shutdown

The recommended shutdown order (from `examples/*/stop_cluster.sh`):

```bash
PLANNER_URL="http://localhost:8002"
LOG_DIR="/tmp/swarmpilot"

# Step 1: Terminate all PyLet-managed instances
# This cleanly drains and deregisters all instances
curl -X POST ${PLANNER_URL}/v1/terminate-all 2>/dev/null

# Step 2: Stop Scheduler(s)
for pidfile in ${LOG_DIR}/scheduler-*.pid; do
  if [ -f "$pidfile" ]; then
    PID=$(cat "$pidfile")
    kill "$PID" 2>/dev/null
    # Wait up to 5 seconds for graceful shutdown
    for i in $(seq 1 5); do
      kill -0 "$PID" 2>/dev/null || break
      sleep 1
    done
    # Force kill if still running
    kill -9 "$PID" 2>/dev/null
    rm -f "$pidfile"
  fi
done

# Step 3: Stop Planner
if [ -f "${LOG_DIR}/planner.pid" ]; then
  PID=$(cat "${LOG_DIR}/planner.pid")
  kill "$PID" 2>/dev/null
  sleep 2
  kill -9 "$PID" 2>/dev/null
  rm -f "${LOG_DIR}/planner.pid"
fi

# Step 4: Stop Predictor (if running standalone)
if [ -f "${LOG_DIR}/predictor.pid" ]; then
  PID=$(cat "${LOG_DIR}/predictor.pid")
  kill "$PID" 2>/dev/null
  sleep 2
  kill -9 "$PID" 2>/dev/null
  rm -f "${LOG_DIR}/predictor.pid"
fi
```

**Shutdown order matters:**
1. **Instances first** — drain and deregister from Schedulers
2. **Schedulers second** — no more tasks to route
3. **Planner third** — no more deployments to manage
4. **Predictor last** — no more predictions needed

Note: the PyLet cluster itself is managed separately and is NOT stopped
by the SwarmPilot shutdown scripts.

---

## Instance Server Contract

Any HTTP server can serve as a SwarmPilot instance as long as it meets
the following requirements:

### Required Endpoint

**Health check** — `GET /health`

Must return HTTP 200 when the instance is ready to accept tasks. The
response body is not validated; any 200 response suffices. The Planner
polls this endpoint during deployment to confirm readiness.

### Task Execution

The Scheduler's `WorkerQueueThread` sends tasks to the instance as HTTP
requests. The request format depends on how the task was submitted:

- **Task submit** (`/v1/task/submit`): The worker sends
  `POST {endpoint}/{path}` where `path` comes from `metadata["path"]`
  (default: `/v1/completions`). The body is `task_input` as JSON.

- **Transparent proxy** (`/{path:path}`): The worker forwards the
  original request path, method, headers, and body to the instance.

### Optional: Self-Registration

Instances deployed via PyLet can self-register with the Scheduler on
startup. This is the pattern used by all example servers:

```python
# On startup (FastAPI lifespan)
async def register():
    scheduler_url = os.environ.get("SCHEDULER_URL")
    if not scheduler_url:
        return
    payload = {
        "instance_id": os.environ["INSTANCE_ID"],
        "model_id": os.environ["MODEL_ID"],
        "endpoint": f"http://{hostname}:{port}",
        "platform_info": {
            "software_name": "vllm",
            "software_version": "0.4.0",
            "hardware_name": "gpu-node-1",
        },
    }
    httpx.post(f"{scheduler_url}/v1/instance/register", json=payload)
```

```python
# On shutdown (SIGTERM handler)
async def deregister():
    scheduler_url = os.environ.get("SCHEDULER_URL")
    if not scheduler_url:
        return
    payload = {"instance_id": os.environ["INSTANCE_ID"]}
    httpx.post(f"{scheduler_url}/v1/instance/remove", json=payload)
```

When the Planner manages deployment, the Planner handles registration
(not the instance itself). Set `SCHEDULER_URL=""` to disable
self-registration in this case.

### Environment Variables Set by PyLet

When PyLet launches an instance, it injects these environment variables:

| Variable | Description |
|----------|-------------|
| `PORT` | Auto-allocated port number |
| `INSTANCE_ID` | Unique instance identifier |
| `MODEL_ID` | Model identifier |
| `SCHEDULER_URL` | Scheduler URL for self-registration |

The launch command template can reference `$PORT` which PyLet replaces
at runtime. `MODEL_ID` and `SCHEDULER_URL` are passed as environment
variables via the PyLet submit call.

---

## Backend-Specific Launch Commands

SwarmPilot's PyLet client uses these command templates
(`swarmpilot/planner/pylet/client.py`):

| Backend | Launch Command |
|---------|---------------|
| `vllm` | `vllm serve {model_id} --port $PORT --host 0.0.0.0` |
| `sglang` | `python -m sglang.launch_server --model {model_id} --port $PORT --host 0.0.0.0` |
| custom | Any command via `deploy_command` parameter |

`{model_id}` is substituted by the Planner. `$PORT` is substituted by
PyLet with an auto-allocated port.

**Custom commands** can be passed via the deployment scripts or the
`deploy_model.sh` pattern:

```bash
# From examples/llm_cluster/deploy_model.sh
DEPLOY_COMMAND="MODEL_ID={model_id} .venv/bin/python mock_vllm_server.py"
```

The `{model_id}` placeholder is replaced per instance with the assigned
model's identifier.

---

## Reference: Complete Examples

The `examples/` directory contains four end-to-end cluster examples:

### `examples/llm_cluster/` — Optimizer-driven multi-model

Deploys 3 model types (llm-fast, llm-medium, llm-slow) using a mock
vLLM server with realistic latency distributions.

```bash
# 1. Start cluster (Predictor + Planner + 3 Schedulers)
bash examples/llm_cluster/start_cluster.sh

# 2. Deploy instances via optimizer
bash examples/llm_cluster/deploy_model.sh

# 3. Generate workload
python examples/llm_cluster/generate_workload.py

# 4. Stop everything
bash examples/llm_cluster/stop_cluster.sh
```

### `examples/mock_llm_cluster/` — Multi-model with 2 schedulers

Similar to `llm_cluster` but with 2 models (llm-7b, llm-32b) and
optimizer-driven deployment.

```bash
bash examples/mock_llm_cluster/start_cluster.sh
bash examples/mock_llm_cluster/deploy_models.sh
python examples/mock_llm_cluster/generate_workload.py
bash examples/mock_llm_cluster/stop_cluster.sh
```

### `examples/multi_scheduler/` — Manual target state deployment

Deploys 3 sleep models using `deploy_manually` (no optimizer). Each model
gets a dedicated Scheduler.

```bash
bash examples/multi_scheduler/start_cluster.sh
bash examples/multi_scheduler/deploy_model.sh    # uses /v1/deploy_manually
python examples/multi_scheduler/generate_workload.py
bash examples/multi_scheduler/stop_cluster.sh
```

### `examples/pylet_benchmark/` — Direct deployment (no Planner)

Starts instance processes directly and registers them manually with the
Scheduler. No Planner or PyLet involved.

```bash
bash examples/pylet_benchmark/start_cluster.sh
bash examples/pylet_benchmark/deploy_model.sh    # starts processes + curl register
python examples/pylet_benchmark/generate_workload.py
bash examples/pylet_benchmark/stop_cluster.sh
```

### `examples/planner/` — Standalone optimizer

Demonstrates the Planner's optimization API without deployment:

```bash
# Simple optimization: 4 machines, 3 models
python examples/planner/0.generate_simple_plan.py

# Larger optimization: 30 machines, 2 models
python examples/planner/1.generate_plan_for_initial.py
```
