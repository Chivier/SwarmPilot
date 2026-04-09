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
| `TRAINING_ENABLE_AUTO` | `false` | Enable automatic predictor retraining |
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
| `AUTO_OPTIMIZE_ENABLED` | `false` | Enable periodic auto-optimization |
| `AUTO_OPTIMIZE_INTERVAL` | `60.0` | Optimization interval (seconds) |
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
