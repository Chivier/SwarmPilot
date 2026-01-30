# Deployment

Production deployment guide using PyLet for automatic instance management.

## Architecture with PyLet

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Predictor  │────▶│  Scheduler  │◀────│   Planner   │
│   (8001)    │     │   (8000)    │     │   (8002)    │
└─────────────┘     └──────┬──────┘     └──────┬──────┘
                           │                   │
                           │ registers         │ deploys via
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Instances  │◀────│   PyLet     │
                    │  (dynamic)  │     │  Cluster    │
                    └─────────────┘     └─────────────┘
```

- **Predictor** provides runtime estimates to the Scheduler
- **Scheduler** routes tasks to registered instances
- **Planner** runs optimization and deploys/scales via PyLet
- **PyLet** provisions, drains, and terminates instances on the cluster

## Prerequisites

- A running PyLet cluster with a reachable head node
- GPU resources for LLM deployments
- Python >= 3.11
- Install with PyLet extra: `pip install swarmpilot[pylet]`

---

## Starting Core Services

```bash
# Terminal 1: Predictor
spredictor start --port 8001

# Terminal 2: Scheduler
PREDICTOR_URL=http://localhost:8001 \
  sscheduler start --port 8000

# Terminal 3: Planner with PyLet
PYLET_ENABLED=true \
  PYLET_HEAD_URL=http://your-pylet-head:8000 \
  SCHEDULER_URL=http://localhost:8000 \
  splanner start --port 8002
```

---

## Deploying Models via PyLet

### Optimizer + Deploy

The `/v1/deploy` endpoint runs the optimization algorithm to compute the optimal instance allocation, then deploys via PyLet:

```bash
curl -X POST http://localhost:8002/v1/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "M": 2,
    "N": 1,
    "B": [[1.0], [1.0]],
    "target": [1.0],
    "a": 0.5,
    "model_ids": ["sleep_model"],
    "wait_for_ready": true
  }'
```

### Manual Deploy

Deploy to an explicit target state without running the optimizer:

```bash
curl -X POST http://localhost:8002/v1/deploy_manually \
  -H "Content-Type: application/json" \
  -d '{
    "target_state": {"sleep_model": 2},
    "wait_for_ready": true
  }'
```

### Scale a Single Model

```bash
curl -X POST http://localhost:8002/v1/scale \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "sleep_model",
    "target_count": 4,
    "wait_for_ready": true
  }'
```

### Check Deployment Status

```bash
curl http://localhost:8002/v1/status
```

### Terminate All Instances

```bash
curl -X POST "http://localhost:8002/v1/terminate-all?wait_for_drain=true"
```

---

## Multi-Scheduler Setup

In production, each model type can have its own Scheduler. The Planner maintains a registry mapping `model_id -> scheduler_url`.

### How It Works

1. Each Scheduler sets `PLANNER_REGISTRATION_URL`, `SCHEDULER_MODEL_ID`, and `SCHEDULER_SELF_URL`
2. On startup, the Scheduler registers itself with the Planner
3. The Planner routes PyLet-deployed instances to the correct Scheduler
4. On shutdown, the Scheduler deregisters

### Example: Two Schedulers

```bash
# Scheduler A: handles "model-a"
PLANNER_REGISTRATION_URL=http://localhost:8002 \
  SCHEDULER_MODEL_ID=model-a \
  SCHEDULER_SELF_URL=http://localhost:8000 \
  sscheduler start --port 8000

# Scheduler B: handles "model-b"
PLANNER_REGISTRATION_URL=http://localhost:8002 \
  SCHEDULER_MODEL_ID=model-b \
  SCHEDULER_SELF_URL=http://localhost:8010 \
  sscheduler start --port 8010
```

### Verify Registration

```bash
curl http://localhost:8002/v1/scheduler/list
```

---

## LLM Deployment

### Using vLLM

```bash
PYLET_ENABLED=true \
  PYLET_HEAD_URL=http://your-pylet-head:8000 \
  PYLET_BACKEND=vllm \
  PYLET_GPU_COUNT=1 \
  SCHEDULER_URL=http://localhost:8000 \
  splanner start --port 8002
```

Then deploy:

```bash
curl -X POST http://localhost:8002/v1/deploy_manually \
  -H "Content-Type: application/json" \
  -d '{
    "target_state": {"meta-llama/Llama-3.1-8B-Instruct": 2},
    "wait_for_ready": true
  }'
```

### Using SGLang

```bash
PYLET_BACKEND=sglang \
  PYLET_GPU_COUNT=1 \
  # ... other vars same as above
  splanner start --port 8002
```

### Using a Custom Command

For testing or non-standard backends:

```bash
PYLET_CUSTOM_COMMAND="python my_model_server.py" \
  # ... other vars
  splanner start --port 8002
```

The `$PORT` placeholder in the command will be replaced with an auto-allocated port.

---

## Monitoring

### Health Endpoints

```bash
curl http://localhost:8001/health         # Predictor
curl http://localhost:8000/v1/health      # Scheduler
curl http://localhost:8002/v1/health      # Planner
curl http://localhost:8002/v1/status      # PyLet status
curl http://localhost:8002/v1/info        # Planner info + registered schedulers
```

### Real-Time Results (WebSocket)

Connect to `ws://localhost:8000/v1/task/get_result` to stream task results as they complete.

### Deployment Timeline

```bash
# Get timeline of deployment events
curl http://localhost:8002/v1/timeline

# Clear timeline (before a new experiment)
curl -X POST http://localhost:8002/v1/timeline/clear
```

---

## PyLet Environment Variables

See [Configuration](CONFIGURATION.md#pylet) for the full list. Key variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PYLET_ENABLED` | Yes | `false` | Enable PyLet integration |
| `PYLET_HEAD_URL` | Yes* | - | PyLet head node URL |
| `PYLET_BACKEND` | No | `vllm` | Backend: `vllm` or `sglang` |
| `PYLET_GPU_COUNT` | No | `1` | GPUs per instance |
| `PYLET_CPU_COUNT` | No | `1` | CPUs per instance |
| `PYLET_DEPLOY_TIMEOUT` | No | `300.0` | Deploy timeout (seconds) |
| `PYLET_DRAIN_TIMEOUT` | No | `30.0` | Drain timeout (seconds) |
| `PYLET_CUSTOM_COMMAND` | No | - | Custom command (overrides backend) |
| `PYLET_REUSE_CLUSTER` | No | `false` | Reuse existing cluster |

*Required when `PYLET_ENABLED=true`.

---

## Example Clusters

See `examples/` for ready-to-run configurations:

| Directory | Description |
|-----------|-------------|
| `examples/mock_llm_cluster/` | Local test cluster with mock predictor |
| `examples/llm_cluster/` | Real LLM cluster with vLLM/SGLang |
| `examples/multi_scheduler/` | Multi-scheduler setup with planner |
| `examples/pylet_benchmark/` | PyLet benchmarking with sleep models |
