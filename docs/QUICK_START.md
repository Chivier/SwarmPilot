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
bash examples/single_model/start_cluster.sh
```

This starts:
- **Scheduler** on port 8000 (with embedded predictor via `PREDICTOR_MODE=library`)
- **Mock vLLM instances** on ports 8100+

To stop: `bash examples/single_model/stop_cluster.sh`

---

## Option B: Manual Start (2 Terminals)

### Architecture

```
┌─────────────┐     ┌─────────────┐
│  Scheduler  │◀────│  Mock vLLM  │
│   (8000)    │     │  Instance   │
│  + Predictor│     │  (8100+)    │
└─────────────┘     └─────────────┘
```

- **Scheduler**: Routes tasks to instances, with embedded predictor (`PREDICTOR_MODE=library`)
- **Instances**: Mock vLLM servers that simulate inference latency

### Step 1: Install Dependencies

```bash
cd swarmpilot-refresh
uv sync
```

### Step 2: Start Services

**Terminal 1: Scheduler (with embedded predictor)**
```bash
PREDICTOR_MODE=library sscheduler start --port 8000
```

**Terminal 2: Mock vLLM Instance**
```bash
MODEL_ID=sleep_model PORT=8100 \
  uv run python examples/single_model/mock_vllm_server.py
```

> Start additional instances on different ports (8101, 8102, etc.) for parallel task execution.
> Each instance self-registers with the Scheduler on startup.

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
curl http://localhost:8000/v1/health       # Scheduler
curl http://localhost:8100/health          # Mock vLLM Instance
```

> **Note:** The Scheduler and Planner APIs use a `/v1/` prefix. Instance health endpoints have no prefix.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Connection refused` on 8000 | Start Scheduler first |
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
