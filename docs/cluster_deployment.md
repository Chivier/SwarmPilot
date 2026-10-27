# Cluster Deployment

Production deployment guide for SwarmPilot: starting services, deploying models via PyLet, scaling, and cluster management.

## Architecture

```
┌──────────────────┐     ┌─────────────┐
│    Scheduler     │◀────│   Planner   │
│     (8000)       │     │   (8002)    │
│  + embedded      │     │  optimizer  │
│    predictor     │     └──────┬──────┘
└──────┬───────────┘            │
       │                        │ deploys via
       │ routes requests        ▼
       │                 ┌─────────────┐
       ▼                 │   PyLet     │
┌─────────────┐          │  Cluster    │
│  Instances  │◀─────────│  (local or  │
│  (dynamic)  │          │   remote)   │
└─────────────┘          └─────────────┘
```

- **Scheduler** routes tasks to instances and embeds the Predictor as a library (direct Python calls, no separate service)
- **Planner** runs optimization and deploys/scales via PyLet
- **PyLet** provisions, drains, and terminates instances on the cluster

> The Predictor runs in library mode embedded in the Scheduler by default. No standalone Predictor service is needed.

---

## Starting Core Services

The startup pattern follows the predictor training playground: Planner with local PyLet first, then Scheduler registering with Planner.

### Local PyLet Mode (Development / Single-Node)

```bash
# Step 1: Start Planner with local PyLet cluster
splanner start --port 8002

# Step 2: Start Scheduler (registers with Planner, model assigned on first deploy)
PLANNER_REGISTRATION_URL=http://localhost:8002 \
SCHEDULER_SELF_URL=http://localhost:8000 \
sscheduler start --port 8000
```

When `PYLET_LOCAL_MODE=true`, `PYLET_ENABLED` is automatically set and `PYLET_HEAD_URL` defaults to `http://localhost:{PYLET_LOCAL_PORT}`. The Planner starts a local PyLet head + worker as subprocesses.

The Scheduler starts **without** `SCHEDULER_MODEL_ID` -- the Planner assigns the model dynamically on the first `serve()` call via `/v1/model/reassign`.

> **One-click start:** `bash examples/6.predictor_training/start_qwen_cluster.sh` handles the full startup sequence including a temporary dummy health server needed for PyLet initialization.

### External PyLet Cluster (Production)

```bash
# Step 1: Start Planner pointing to external PyLet
PYLET_ENABLED=true \
  PYLET_HEAD_URL=http://your-pylet-head:8000 \
  PYLET_BACKEND=vllm \
  PYLET_GPU_COUNT=4 \
  SCHEDULER_URL=http://localhost:8000 \
  splanner start --port 8002

# Step 2: Start Scheduler
SCHEDULING_STRATEGY=round_robin \
  PROXY_ENABLED=true \
  PROXY_TIMEOUT=600.0 \
  PLANNER_REGISTRATION_URL=http://localhost:8002 \
  SCHEDULER_SELF_URL=http://localhost:8000 \
  sscheduler start --port 8000
```

### Verify

```bash
curl http://localhost:8002/v1/health      # Planner
curl http://localhost:8000/v1/health      # Scheduler
curl http://localhost:8002/v1/schedulers  # Registered schedulers
```

---

## Deployment Methods

### SDK / CLI Deployment

Deploy models using the SwarmPilot SDK or `splanner` CLI:

```python
from swarmpilot.sdk import SwarmPilotClient

async with SwarmPilotClient("http://localhost:8002") as sp:
    group = await sp.serve("Qwen/Qwen3-8B-VL-Instruct", gpu=4, replicas=2)
    await group.wait_ready(timeout=600)
    print(group.endpoints)
```

```bash
splanner serve "Qwen/Qwen3-8B-VL-Instruct" --gpu 4 --replicas 2
```

See [SDK Usage](sdk_usage.md) for the full API reference.

### Planner REST API Deployment

**Manual target state** -- specify exact instance counts per model:

```bash
curl -X POST http://localhost:8002/v1/pylet/deploy_manually \
  -H "Content-Type: application/json" \
  -d '{
    "target_state": {
      "Qwen/Qwen3-8B-VL-Instruct": 3,
      "meta-llama/Llama-3-8B": 2
    }
  }'
```

**Optimizer-driven deployment** -- let the Planner compute optimal allocation:

```bash
curl -X POST http://localhost:8002/v1/pylet/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "M": 4,
    "N": 2,
    "B": [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    "target": [60.0, 40.0],
    "a": 0.5,
    "model_ids": ["Qwen/Qwen3-8B-VL-Instruct", "meta-llama/Llama-3-8B"],
    "wait_for_ready": true
  }'
```

**Scale a single model:**

```bash
curl -X POST http://localhost:8002/v1/pylet/scale \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "Qwen/Qwen3-8B-VL-Instruct",
    "target_count": 4,
    "wait_for_ready": true
  }'
```

**Check deployment status:**

```bash
curl http://localhost:8002/v1/pylet/status
splanner ps
```

### Online API Endpoints (Cloud Providers)

Deploy against cloud inference APIs (Together AI, Fireworks AI, OpenAI, etc.) without running any local GPU instances. The Scheduler registers remote endpoints as virtual instances and load-balances across them.

**Step 1: Create a YAML config** listing your providers:

```yaml
# online_endpoints.yaml
endpoints:
  - name: "together-qwen3-8b"
    base_url: "https://api.together.xyz"
    api_key_env: "TOGETHER_API_KEY"       # reads key from this env var
    model_id: "Qwen/Qwen3-8B"
    concurrency_limit: 10                 # max concurrent requests to this provider

  - name: "fireworks-qwen3-8b"
    base_url: "https://api.fireworks.ai/inference"
    api_key_env: "FIREWORKS_API_KEY"
    model_id: "Qwen/Qwen3-8B"
    concurrency_limit: 5
```

**Step 2: Set API keys and start the Scheduler:**

```bash
export TOGETHER_API_KEY="your-key"
export FIREWORKS_API_KEY="your-key"

SCHEDULER_MODEL_ID="Qwen/Qwen3-8B" \
  ONLINE_ENDPOINTS_CONFIG="./online_endpoints.yaml" \
  sscheduler start --port 8000
```

The Scheduler auto-registers all endpoints at startup. No Planner needed.

**Step 3: Send requests** — same as local instances:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'
```

The Scheduler picks a provider, injects the `Authorization: Bearer` header, and forwards the request. The client never sees which provider served the response.

#### How `concurrency_limit` Works

Each endpoint's `concurrency_limit` creates that many **virtual instances**, each with its own serial task queue. For example, `concurrency_limit: 10` creates 10 virtual instances (`online-together-qwen3-8b-0` through `online-together-qwen3-8b-9`), each processing one request at a time — bounding total concurrent requests to that provider at 10.

The scheduling algorithms see all virtual instances and route tasks using the same strategies as local instances. The Predictor learns a single latency model per provider (all virtual instances share the same `platform_info`), and differentiation between virtual instances comes from their queue state.

```
concurrency_limit: 3  →  3 virtual instances, each with serial queue
                          ┌─── online-provider-0 ─── [task] ──→ Provider API
                          ├─── online-provider-1 ─── [task] ──→ Provider API
                          └─── online-provider-2 ─── [task] ──→ Provider API
                          Max 3 concurrent requests to this provider
```

Traffic distribution is proportional to `concurrency_limit`. If Provider A has `concurrency_limit: 5` and Provider B has `concurrency_limit: 15`, Provider B receives ~75% of traffic (15 out of 20 total virtual instances).

#### YAML Config Reference

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `name` | Yes | — | Unique endpoint name. Becomes part of instance IDs: `online-{name}-{slot}` |
| `base_url` | Yes | — | Provider API base URL |
| `api_key_env` | Yes | — | Environment variable name holding the API key |
| `model_id` | Yes | — | Model identifier for scheduling (must match across endpoints sharing a scheduler) |
| `concurrency_limit` | No | `10` | Number of virtual instances (concurrent request slots) |

#### Multi-Model with Online APIs

Each model needs its own Scheduler. Start multiple schedulers with separate YAML configs:

```bash
# Scheduler A — Qwen3-8B across 3 providers
SCHEDULER_MODEL_ID="Qwen/Qwen3-8B" \
  ONLINE_ENDPOINTS_CONFIG="./online_qwen3_8b.yaml" \
  sscheduler start --port 8010

# Scheduler B — GPT-4o across 2 providers
SCHEDULER_MODEL_ID="gpt-4o" \
  ONLINE_ENDPOINTS_CONFIG="./online_gpt4o.yaml" \
  sscheduler start --port 8020
```

See `examples/7.online_api/` for a full 3-model x 3-provider working example, and `examples/8.virtual_concurrency/` for concurrency testing.

### Direct Instance Deployment

For full manual control without the Planner. Start an instance process yourself, then register it with the Scheduler.

```bash
# Start a vLLM instance
vllm serve Qwen/Qwen3-8B-VL-Instruct --port 8100 --host 0.0.0.0

# Wait for health, then register
until curl -sf http://localhost:8100/health > /dev/null 2>&1; do sleep 1; done

curl -X POST http://localhost:8000/v1/instance/register \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "inst-001",
    "model_id": "Qwen/Qwen3-8B-VL-Instruct",
    "endpoint": "http://localhost:8100",
    "platform_info": {
      "software_name": "vllm",
      "software_version": "0.8.0",
      "hardware_name": "NVIDIA RTX A6000"
    }
  }'
```

---

## Multi-Scheduler Setup

In production, each model has its own Scheduler. The Planner maintains a registry mapping `model_id -> scheduler_url`.

**How it works:**
1. Each Scheduler sets `PLANNER_REGISTRATION_URL` and `SCHEDULER_SELF_URL`
2. On startup, the Scheduler registers itself with the Planner
3. The Planner routes PyLet-deployed instances to the correct Scheduler

```bash
# Scheduler A (port 8000)
PLANNER_REGISTRATION_URL=http://localhost:8002 \
  SCHEDULER_SELF_URL=http://localhost:8000 \
  sscheduler start --port 8000

# Scheduler B (port 8010)
PLANNER_REGISTRATION_URL=http://localhost:8002 \
  SCHEDULER_SELF_URL=http://localhost:8010 \
  sscheduler start --port 8010
```

**Verify registration:**

```bash
curl http://localhost:8002/v1/scheduler/list
```

> **Dynamic assignment:** `SCHEDULER_MODEL_ID` is optional. When a `serve()` request arrives for a model with no scheduler, the Planner finds an idle scheduler (zero instances) and calls `POST /v1/model/reassign` to bind it. When all instances of a model are terminated, the scheduler becomes idle and can be reassigned.

---

## LLM Backend Configuration

### vLLM (Default)

```bash
PYLET_BACKEND=vllm \
  PYLET_GPU_COUNT=4 \
  # ... other Planner vars
  splanner start --port 8002
```

Launch command: `vllm serve {model_id} --port $PORT --host 0.0.0.0`

### SGLang

```bash
PYLET_BACKEND=sglang \
  PYLET_GPU_COUNT=4 \
  # ... other Planner vars
  splanner start --port 8002
```

Launch command: `python -m sglang.launch_server --model {model_id} --port $PORT --host 0.0.0.0`

### Custom Command

```bash
PYLET_CUSTOM_COMMAND="python my_model_server.py" \
  # ... other Planner vars
  splanner start --port 8002
```

`$PORT` is replaced with an auto-allocated port by PyLet.

---

## Mixed Hardware

PyLet workers report GPU counts but not GPU types, so tell the Planner
which hardware each worker has. Every instance is then registered with
the hardware of the worker it landed on, which keeps predictor models for
different GPUs apart, and a model can be pinned to one kind of hardware.

```bash
# Patterns match the PyLet worker id first, then the worker's host.
PLANNER_HARDWARE_MAP='{"gpu-h100-*": "H100-SXM-80GB", "10.0.2.*": "A100-40GB"}' \
  PLANNER_HARDWARE_NAME=gpu \
  splanner start --port 8002

splanner serve meta-llama/Llama-3.1-70B --gpu 4 --replicas 2 --hardware 'H100*'
```

| Variable | Default | Meaning |
|---|---|---|
| `PLANNER_HARDWARE_MAP` | empty | JSON object, or the path of a JSON/YAML file, mapping worker id or host patterns to hardware names (first match wins) |
| `PLANNER_HARDWARE_NAME` | `gpu` | Hardware name for workers no pattern matches |

`--hardware` (`hardware` in `POST /v1/serve` and `SwarmPilotClient.serve`)
is a case-insensitive pattern. It is kept for the model, so later scaling,
deploys and replacements stay on that hardware too; set it per model in
`PYLET_MODEL_CONFIGS` with `"hardware": "H100*"` as well. Replicas that do
not fit on free GPUs of that hardware are reported as failed rather than
queued. In sim mode the map comes from the simulator's node list.

## Engine Failures

When an engine crashes, its requests are not lost:

- **Scheduler:** the first refused connection marks the instance
  `unhealthy`. No new request is routed to it, and every request waiting
  for it (none had reached the engine) is scheduled again on the model's
  other instances. A request already running on the engine when it died
  gets a dropped connection or a 503; nothing has reached the client yet
  (a stream that has started cannot be taken back), so it too runs again
  on another instance, at most twice. The same holds for an instance the
  Planner removes after its drain times out with requests still running.
  The Scheduler checks unhealthy instances every
  `SCHEDULER_HEALTH_CHECK_INTERVAL` seconds (default 1) and routes to them
  again when they answer; one that stays unreachable for
  `SCHEDULER_UNHEALTHY_REMOVE_AFTER` seconds (default 120, 0 = never) is
  removed. Online API endpoints are never removed.
- **Planner:** every `PLANNER_INSTANCE_WATCH_INTERVAL` seconds (default 2,
  0 disables) it looks for active instances that PyLet reports `FAILED`,
  `COMPLETED` or `CANCELLED`, or no longer knows. It removes them from
  their Scheduler, deletes them in PyLet and, unless
  `PLANNER_REPLACE_FAILED=false`, deploys replacements, never beyond the
  replica count last asked for with serve, scale or deploy.

---

## Stopping Services

### Terminate Instances

```bash
# Via CLI
splanner terminate --all

# Via REST API
curl -X POST http://localhost:8002/v1/pylet/terminate-all
```

### Drain a Single Instance

```bash
# Drain (stop accepting new tasks, finish in-flight)
curl -X POST http://localhost:8000/v1/instance/drain \
  -H "Content-Type: application/json" \
  -d '{"instance_id": "inst-001"}'

# Poll until safe to remove
curl "http://localhost:8000/v1/instance/drain/status?instance_id=inst-001"

# Remove from scheduler
curl -X POST http://localhost:8000/v1/instance/remove \
  -H "Content-Type: application/json" \
  -d '{"instance_id": "inst-001"}'
```

### Full Cluster Shutdown

Shutdown order:
1. **Instances first** -- terminate all managed instances
2. **Scheduler second** -- no more tasks to route
3. **Planner last** -- auto-stops local PyLet cluster on exit

```bash
# 1. Terminate instances
splanner terminate --all

# 2. Stop Scheduler (SIGTERM)
# 3. Stop Planner (auto-stops local PyLet if PYLET_LOCAL_MODE)
```

> **One-click stop:** `bash examples/6.predictor_training/stop_qwen_cluster.sh`

---

## Instance Server Contract

Any HTTP server can serve as a SwarmPilot instance if it implements:

**Required:** `GET /health` -- must return HTTP 200 when ready.

**Task execution:** The Scheduler sends tasks as HTTP requests:
- Via task submit: `POST {endpoint}/{path}` where `path` comes from `metadata["path"]` (default: `/v1/completions`)
- Via transparent proxy: forwards the original request path, method, headers, and body to the instance

**PyLet environment variables** injected at launch:

| Variable | Description |
|----------|-------------|
| `PORT` | Auto-allocated port number |
| `INSTANCE_ID` | Unique instance identifier |
| `MODEL_ID` | Model identifier |
| `SCHEDULER_URL` | Scheduler URL for self-registration |

---

## Monitoring

```bash
curl http://localhost:8000/v1/health      # Scheduler (+ embedded predictor)
curl http://localhost:8002/v1/health      # Planner
curl http://localhost:8002/v1/pylet/status      # PyLet status
curl http://localhost:8002/v1/info        # Planner info + registered schedulers
```

**Real-time results:** Connect to `ws://localhost:8000/v1/task/get_result` for streaming task results.

**Deployment timeline:**

```bash
curl http://localhost:8002/v1/timeline
curl -X POST http://localhost:8002/v1/timeline/clear
```

### Autoscaling agent workloads

With `AUTO_OPTIMIZE_ENABLED=true`, each Scheduler reports its arrival rate,
a q0.9 estimate of its outstanding work (from the queue sketches) and its
outstanding task count every `SCHEDULER_AUTO_REPORT` seconds (with
`PLANNER_URL` set). The default `tail` scaler gives each model a share of the machines proportional to the
instances it needs: its arrival rate times its execution time, plus its
q0.9 backlog drained over one interval.

By default the loop only re-divides the instances that exist, so a model
grows only by taking machines from another and free GPUs stay free. With
`AUTO_OPTIMIZE_ELASTIC=true` it also sizes the pool: the tail scaler's need
is in tasks in flight, and the pool wants one instance per
`AUTO_OPTIMIZE_TARGET_CONCURRENCY` of them. It grows at once into free
GPUs (up to `AUTO_OPTIMIZE_MAX_INSTANCES`) and shrinks only after
`AUTO_OPTIMIZE_SCALE_DOWN_ROUNDS` rounds in a row that want fewer, by at
most the change factor per round, from the models with the most surplus
(fresh reports of no traffic count as wanting the minimum).
Set the target near the concurrency at which your engines' latency starts
to climb (for 7-8B models on one A100 or L40S under vLLM continuous
batching, roughly 50 to 100).

On mixed hardware, consider `AUTO_OPTIMIZE_HARDWARE_AWARE=true`. Without
it every machine counts the same, the deployment service removes the
oldest instances of a model that shrinks, and new instances go wherever
PyLet puts them, so a model can be left on one slow GPU while a fast one
sits idle. With it the loop learns how fast each kind of hardware is from
the execution times the Schedulers report (fitting `log t = model +
hardware`, the fastest kind counting 1) and:

- weighs each machine by that speed, so it chooses which instances move,
  removes exactly those and starts their replacements on the workers they
  freed (two models can trade machines);
- sizes an elastic pool in units of the fastest hardware
  (`AUTO_OPTIMIZE_TARGET_CONCURRENCY` tasks in flight each), grows into
  the fastest free GPUs first and shrinks by the slowest instances;
- when the pool cannot carry all the need, serves each model's arrivals
  (`rate * s`) before anyone's backlog, so one model's backlog cannot take
  the machines another needs to keep up;
- leaves a model at `min_replicas` its fastest instances.

Models can take different GPU counts and kinds of hardware
(`PYLET_MODEL_CONFIGS`, e.g. `{"meta-llama/Llama-3.1-70B-Instruct":
{"gpu_count": 4, "hardware": "[ah]100*"}}`). The loop knows each model's
shape: a machine changes model only to one with the same GPU count that
runs on its hardware, and a growing pool gives each model short of
capacity a free place that holds one of its instances: the slowest kind
of hardware that covers the shortfall on its own (else the fastest), on
the worker with the fewest free GPUs that fit, so large nodes stay whole
for multi-GPU models. Without `AUTO_OPTIMIZE_HARDWARE_AWARE` the loop
re-divides counts only among models of the same shape. Replicas placed
without a named worker (serve, scale, replacements) also go best fit:
PyLet itself takes the first worker that fits, so 1-GPU replicas would
fill an 8-GPU node and leave no room for a 4-GPU model.

Hardware names come from `PLANNER_HARDWARE_MAP` (or the simulator's
node list); the speeds are in the loop's log line.

Each move costs a drain (up to `PYLET_DRAIN_TIMEOUT`) and a cold start,
so every moved machine must improve the objective by
`AUTO_OPTIMIZE_CHANGE_THRESHOLD` on its own, an instance that started
serving less than four intervals ago is not moved, each model keeps its
`AUTO_OPTIMIZE_MIN_REPLICAS` on machines that do not move in that round,
and the removals of one round drain side by side.

When one model's tasks call another (an agent's LLM step calling a tool or
a diffusion model), mark those calls with the calling model so the Planner
can size the callee from the caller's traffic instead of waiting for calls
to pile up:

```bash
# Proxy path
curl http://scheduler-tool:8000/v1/images/generations \
  -H "X-SwarmPilot-Parent: Qwen/Qwen3-8B" -d '{"prompt": "..."}'

# Task API
curl -X POST http://scheduler-tool:8000/v1/task/submit -d '{
  "task_id": "t1", "model_id": "sdxl", "task_input": {...},
  "metadata": {"parent_model": "Qwen/Qwen3-8B"}}'
```

`GET /v1/demand` on the Planner shows the rates, tail work and estimated
call ratios it plans from. In the agent-workload simulation
(`tests/perf/agent_scaling.py`) the tail scaler missed the per-task SLO on
7.0% of tasks against 40.4% for the `count` scaler with short agent steps,
and 1.7% against 13.5% with long ones. Schedulers older than this release
send no arrival rate; while any model lacks one the Planner uses `count`.
`tests/integration/autoscale_e2e.py` checks the whole loop on a local PyLet
cluster with fake LLM and diffusion engines.

---

## Environment Variables

### Scheduler

Source: `swarmpilot/scheduler/config.py`

| Variable | Default | Description |
|----------|---------|-------------|
| `SCHEDULER_HOST` | `0.0.0.0` | Bind address |
| `SCHEDULER_PORT` | `8000` | Bind port |
| `SCHEDULER_ENABLE_CORS` | `true` | Enable CORS middleware |
| `SCHEDULING_STRATEGY` | `adaptive_bootstrap` | Strategy: `adaptive_bootstrap`, `min_time`, `probabilistic`, `round_robin`, `random`, `po2`, `severless` |
| `SCHEDULING_PROBABILISTIC_QUANTILE` | `0.9` | Target quantile for probabilistic strategy |
| `TRAINING_ENABLE_AUTO` | `true` | Online adaptation: train cold platforms, retrain on drift |
| `TRAINING_RETRAIN_MODE` | `drift` | `drift` retrains a platform with a model only when it drifts; `batch` retrains every full batch |
| `DRIFT_WINDOW` | `200` | Scored tasks per drift window |
| `DRIFT_MIN_SAMPLES` | `50` | Scored tasks before a group can drift |
| `DRIFT_LOSS_RATIO` | `2.0` | Drift when q0.9 pinball loss exceeds this times its baseline |
| `DRIFT_MISS_RATIO` | `3.0` | Drift when the q0.9 miss rate exceeds this times 0.1 (and is significant) |
| `DRIFT_RECOVERY_SAMPLES` | `100` | Post-drift samples collected before the retrain |
| `TRAINING_BATCH_SIZE` | `100` | Batch size for training data collection |
| `TRAINING_FREQUENCY` | `3600` | Training frequency in seconds |
| `TRAINING_MIN_SAMPLES` | `10` | Minimum samples before training |
| `TRAINING_PREDICTION_TYPES` | `quantile` | Prediction types to train (comma-separated) |
| `PREDICTOR_STORAGE_DIR` | `models` | Model file storage directory |
| `PREDICTOR_CACHE_MAX_SIZE` | `100` | Max models in memory cache |
| `PREPROCESSOR_CONFIG_FILE` | _(empty)_ | Path to preprocessor rules JSON |
| `PREPROCESSOR_STRICT` | `true` | Fail if preprocessor model unavailable |
| `SCHEDULER_LOGURU_LEVEL` | `INFO` | Log level |
| `SCHEDULER_LOG_DIR` | `logs` | Log file directory |
| `PLANNER_URL` | _(empty)_ | Planner URL for throughput reporting |
| `SCHEDULER_AUTO_REPORT` | `0` | Auto-report interval (seconds, 0 = disabled) |
| `PROXY_ENABLED` | `true` | Enable transparent proxy router |
| `PROXY_TIMEOUT` | `300.0` | Proxy request timeout (seconds) |
| `WORKER_HTTP_TIMEOUT` | `300.0` | Worker queue HTTP timeout (seconds) |
| `PLANNER_REGISTRATION_URL` | _(empty)_ | Planner URL for multi-scheduler registration |
| `SCHEDULER_MODEL_ID` | _(empty)_ | Initial model ID (optional, Planner assigns dynamically) |
| `SCHEDULER_SELF_URL` | _(empty)_ | Advertised scheduler URL |
| `ONLINE_ENDPOINTS_CONFIG` | _(empty)_ | Path to online endpoints YAML config file |

### Planner

Source: `swarmpilot/planner/config.py`

| Variable | Default | Description |
|----------|---------|-------------|
| `PLANNER_HOST` | `0.0.0.0` | Bind address |
| `PLANNER_PORT` | `8000` | Bind port |
| `SCHEDULER_URL` | _(none)_ | Default scheduler URL |
| `INSTANCE_TIMEOUT` | `30` | Instance operation timeout (seconds) |
| `INSTANCE_MAX_RETRIES` | `3` | Max retries for instance operations |
| `INSTANCE_RETRY_DELAY` | `1.0` | Delay between retries (seconds) |
| `AUTO_OPTIMIZE_ENABLED` | `false` | Re-plan the deployment from Scheduler demand reports (needs PyLet) |
| `AUTO_OPTIMIZE_INTERVAL` | `60.0` | Optimization interval (seconds) |
| `AUTO_OPTIMIZE_CHANGE_THRESHOLD` | `0.05` | Minimum objective improvement before a round changes anything |
| `AUTO_OPTIMIZE_CHANGE_FACTOR` | `0.3` | Share of machines that may change model per round (at least one) |
| `AUTO_OPTIMIZE_MIN_REPLICAS` | `1` | Replicas a deployed model keeps even with no demand |
| `AUTO_OPTIMIZE_ALGORITHM` | `integer_programming` | Optimizer: `integer_programming` or `simulated_annealing` |
| `AUTO_OPTIMIZE_ELASTIC` | `false` | Also grow the pool into free GPUs and shrink it when idle (tail scaler only) |
| `AUTO_OPTIMIZE_TARGET_CONCURRENCY` | `64` | Tasks in flight per instance an elastic pool aims for |
| `AUTO_OPTIMIZE_MIN_INSTANCES` | `1` | Smallest elastic pool |
| `AUTO_OPTIMIZE_MAX_INSTANCES` | `0` | Largest elastic pool; 0 = as many as the free GPUs allow |
| `AUTO_OPTIMIZE_SCALE_DOWN_ROUNDS` | `3` | Rounds in a row that must want fewer instances before the pool shrinks |
| `AUTO_OPTIMIZE_HARDWARE_AWARE` | `false` | Weigh machines by hardware speed learned from execution times; choose which instances move and where (tail scaler only) |
| `AUTO_OPTIMIZE_SCALER` | `tail` | `tail` sizes models by arrival rate, q0.9 backlog and calls between models; `count` by outstanding tasks only (see [Autoscaling agent workloads](#autoscaling-agent-workloads)) |
| `PYLET_ENABLED` | `false` | Enable PyLet cluster integration |
| `PYLET_HEAD_URL` | _(none)_ | PyLet head node URL (required when `PYLET_ENABLED=true`) |
| `PYLET_BACKEND` | `vllm` | Backend engine: `vllm` or `sglang` |
| `PYLET_GPU_COUNT` | `1` | GPUs per instance |
| `PYLET_CPU_COUNT` | `1` | CPUs per instance |
| `PYLET_DEPLOY_TIMEOUT` | `300.0` | Deployment timeout (seconds) |
| `PYLET_DRAIN_TIMEOUT` | `30.0` | Drain timeout (seconds) |
| `PYLET_CUSTOM_COMMAND` | _(none)_ | Custom command template (overrides backend) |
| `PYLET_REUSE_CLUSTER` | `false` | Reuse existing PyLet cluster |
| `PYLET_LOCAL_MODE` | `false` | Start local PyLet cluster as subprocesses |
| `PYLET_LOCAL_PORT` | `5100` | Local PyLet head port |
| `PYLET_LOCAL_NUM_WORKERS` | `1` | Number of local workers |
| `PYLET_LOCAL_CPU_PER_WORKER` | `8` | CPUs per local worker |
| `PYLET_LOCAL_GPU_PER_WORKER` | `4` | GPUs per local worker |
| `PYLET_LOCAL_WORKER_PORT_START` | `5300` | Starting port for local workers |
| `PYLET_LOCAL_WORKER_PORT_GAP` | `200` | Port gap between local workers |
| `PYLET_LOCAL_MEMORY_PER_WORKER` | `65536` | Memory per local worker (MB) |

---

## Example Clusters

| Directory | Description |
|-----------|-------------|
| `examples/6.predictor_training/` | Full cluster with Planner + local PyLet + predictor training |
| `examples/5.multi_model_planner/` | Multi-model with Planner optimizer and PyLet |
| `examples/3.multi_model_direct/` | Multi-model with direct registration (no Planner) |
| `examples/7.online_api/` | Online API multi-provider (3 models x 3 cloud providers) |
| `examples/8.virtual_concurrency/` | Virtual concurrency testing (concurrency 1/2/3 per provider) |
