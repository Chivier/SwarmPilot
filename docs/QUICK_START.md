# SwarmPilot Quick Start

Get SwarmPilot running in **5 minutes** with a local test cluster. No Docker or external services required.

## Prerequisites

| Requirement | Version | Installation |
|-------------|---------|--------------|
| Python | >= 3.11 | [python.org](https://python.org) |
| uv | Latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

---

## Installation

```bash
# Using pip
pip install swarmpilot

# Using uv (recommended for development)
git clone <repo-url> swarmpilot-refresh
cd swarmpilot-refresh
uv sync
```

This installs one package with three CLI tools: `sscheduler`, `spredictor`, `splanner`.

---

## Option A: One-Click Start (Recommended)

```bash
cd swarmpilot-refresh
./scripts/quick_start.sh
```

This starts:
- **Mock Predictor** on port 8001 (no ML model training required)
- **Scheduler** on port 8000
- **2 Sleep Model instances** on ports 8300-8301

To stop: `./scripts/quick_stop.sh`

---

## Option B: Manual Start (3 Terminals)

For understanding the architecture or customization.

### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Mock     │────▶│  Scheduler  │◀────│  Sleep      │
│  Predictor  │     │   (8000)    │     │  Models     │
│   (8001)    │     └─────────────┘     │  (8300+)    │
└─────────────┘                         └─────────────┘
```

- **Mock Predictor**: Returns sleep_time as predicted runtime (no ML training needed)
- **Scheduler**: Routes tasks to available instances based on predictions
- **Instances**: Execute tasks (sleep models for testing)

### Step 1: Install Dependencies

```bash
cd swarmpilot-refresh
uv sync
```

### Step 2: Start Services (3 Terminals)

**Terminal 1: Mock Predictor**
```bash
PREDICTOR_PORT=8001 uv run python examples/mock_llm_cluster/mock_predictor_server.py
```

**Terminal 2: Scheduler**
```bash
PREDICTOR_URL=http://localhost:8001 sscheduler start --port 8000
```

**Terminal 3: Sleep Model Instance**
```bash
PORT=8300 \
  MODEL_ID=sleep_model \
  SCHEDULER_URL=http://localhost:8000 \
  uv run python examples/pylet_benchmark/pylet_sleep_model.py
```

> **Tip**: Start additional instances on different ports (8301, 8302, etc.) for parallel task execution.

---

## Verify It Works

### Submit a Test Task

```bash
curl -X POST http://localhost:8000/task/submit \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test-001",
    "model_id": "sleep_model",
    "task_input": {"sleep_time": 2},
    "metadata": {}
  }'
```

Expected response:
```json
{"success": true, "message": "Task queued...", "task": {"task_id": "test-001", "status": "pending"}}
```

### Check Task Status

```bash
curl "http://localhost:8000/task/info?task_id=test-001"
```

Expected response (after ~2 seconds):
```json
{
  "task": {
    "task_id": "test-001",
    "status": "completed",
    "result": {
      "sleep_time": 2.0,
      "actual_sleep_time": 2.003,
      "model_id": "sleep_model",
      "instance_id": "sleep_model-001",
      "message": "Slept for 2.003 seconds"
    },
    "execution_time_ms": 2003
  }
}
```

### Health Checks

```bash
# All services
curl http://localhost:8001/health  # Predictor
curl http://localhost:8000/health  # Scheduler
curl http://localhost:8300/health  # Sleep Model
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Connection refused` on 8001 | Start Predictor first |
| `Connection refused` on 8000 | Start Scheduler with `PREDICTOR_URL` set |
| Task stuck in `pending` | Ensure sleep model is registered (check `SCHEDULER_URL`) |
| Port already in use | Run `./scripts/quick_stop.sh` or change port numbers |

### View Logs

One-click start logs: `/tmp/swarmpilot_quickstart/`

Manual start: Check the terminal output for each service.

---

## Next Steps

Once you've verified the basic setup works:

1. **Scale up**: Start more sleep model instances for parallel execution
2. **Use PyLet**: For production deployments with automatic instance management
3. **Deploy LLMs**: Replace sleep models with vLLM/SGLang instances

See the sections below for production deployments.

---

## Advanced: Production with PyLet

For production deployments, use **PyLet** to manage instances automatically.

### Prerequisites

- PyLet cluster running (see [PyLet documentation](https://github.com/your-org/pylet))
- GPU resources (for LLM deployments)
- Install with PyLet extra: `pip install swarmpilot[pylet]`

### Architecture with PyLet

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

### Start Core Services

```bash
# Terminal 1: Predictor
spredictor start --port 8001

# Terminal 2: Scheduler
PREDICTOR_URL=http://localhost:8001 sscheduler start --port 8000

# Terminal 3: Planner with PyLet
PYLET_ENABLED=true \
  PYLET_HEAD_URL=http://your-pylet-head:8000 \
  SCHEDULER_URL=http://localhost:8000 \
  splanner start --port 8002
```

### Deploy Instances via PyLet

The `/deploy` endpoint runs the optimization algorithm to compute optimal instance allocation,
then deploys the result via PyLet.

```bash
# Deploy with planner optimization (2 instances, 1 model type)
curl -X POST http://localhost:8002/deploy \
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

# Or deploy manually with explicit target state
curl -X POST http://localhost:8002/deploy_manually \
  -H "Content-Type: application/json" \
  -d '{
    "target_state": {"sleep_model": 2},
    "wait_for_ready": true
  }'

# Check deployment status
curl http://localhost:8002/status
```

### Deploy LLM Instances

```bash
# Configure Planner for vLLM
export PYLET_BACKEND=vllm
export PYLET_GPU_COUNT=1

# Deploy 2 Llama instances with planner optimization
curl -X POST http://localhost:8002/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "M": 2,
    "N": 1,
    "B": [[1.0], [1.0]],
    "target": [1.0],
    "a": 0.5,
    "model_ids": ["meta-llama/Llama-3.1-8B-Instruct"],
    "wait_for_ready": true
  }'
```

### PyLet Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PYLET_ENABLED` | Yes | `false` | Enable PyLet integration |
| `PYLET_HEAD_URL` | Yes* | - | PyLet head node URL |
| `PYLET_BACKEND` | No | `vllm` | Backend: `vllm` or `sglang` |
| `PYLET_GPU_COUNT` | No | `1` | GPUs per instance |
| `PYLET_CPU_COUNT` | No | `1` | CPUs per instance |
| `PYLET_CUSTOM_COMMAND` | No | - | Custom command (overrides backend) |

*Required when `PYLET_ENABLED=true`

---

## Reference

- [PyLet Integration Guide](../planner/docs/2.PYLET_INTEGRATION.md)
- [Scheduler Usage Examples](../scheduler/docs/9.USAGE_EXAMPLES.md)
- [API Reference](../planner/docs/1.API_REFERENCE.md)
