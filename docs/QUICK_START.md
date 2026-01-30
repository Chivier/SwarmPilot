# Quick Start

Get SwarmPilot running locally with a test cluster. No Docker or external services required.

## Prerequisites

| Requirement | Version | Installation |
|-------------|---------|--------------|
| Python | >= 3.11 | [python.org](https://python.org) |
| uv | Latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

## Installation

```bash
# Using pip
pip install swarmpilot

# Using uv (development)
git clone <repo-url> swarmpilot-refresh
cd swarmpilot-refresh
uv sync
```

This installs three CLI tools: `sscheduler`, `spredictor`, `splanner`.

---

## Option A: One-Click Start

```bash
cd swarmpilot-refresh
./scripts/quick_start.sh
```

This starts:
- **Mock Predictor** on port 8001
- **Scheduler** on port 8000
- **2 Sleep Model instances** on ports 8300-8301

To stop: `./scripts/quick_stop.sh`

---

## Option B: Manual Start (3 Terminals)

### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Mock     │────▶│  Scheduler  │◀────│  Sleep      │
│  Predictor  │     │   (8000)    │     │  Models     │
│   (8001)    │     └─────────────┘     │  (8300+)    │
└─────────────┘                         └─────────────┘
```

- **Mock Predictor**: Returns `sleep_time` as predicted runtime (no ML training needed)
- **Scheduler**: Routes tasks to instances based on predictions
- **Instances**: Execute tasks (sleep models for testing)

### Step 1: Install Dependencies

```bash
cd swarmpilot-refresh
uv sync
```

### Step 2: Start Services

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

> Start additional instances on different ports (8301, 8302, etc.) for parallel task execution.

---

## Verify It Works

### Submit a Test Task

```bash
curl -X POST http://localhost:8000/v1/task/submit \
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
curl "http://localhost:8000/v1/task/info?task_id=test-001"
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
curl http://localhost:8001/health          # Predictor (no /v1 prefix)
curl http://localhost:8000/v1/health       # Scheduler
curl http://localhost:8300/health          # Sleep Model
```

> **Note:** The Scheduler and Planner APIs use a `/v1/` prefix. The Predictor API does **not**.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Connection refused` on 8001 | Start Predictor first |
| `Connection refused` on 8000 | Start Scheduler with `PREDICTOR_URL` set |
| Task stuck in `pending` | Ensure sleep model is registered (check `SCHEDULER_URL`) |
| Port already in use | Run `./scripts/quick_stop.sh` or change port numbers |
| `404 Not Found` | Ensure you're using `/v1/` prefix for Scheduler endpoints |

### View Logs

One-click start logs: `/tmp/swarmpilot_quickstart/`

Manual start: Check the terminal output for each service.

---

## Next Steps

1. **Scale up** -- Start more sleep model instances for parallel execution
2. **Explore the API** -- See [API Reference](API_REFERENCE.md) for all endpoints
3. **Deploy to production** -- See [Deployment](DEPLOYMENT.md) for PyLet-based production clusters
4. **Understand the design** -- See [Architecture](ARCHITECTURE.md) for system internals
