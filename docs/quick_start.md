# Quick Start

Get SwarmPilot running with a real cluster: Planner manages instance lifecycle via PyLet, Scheduler routes inference requests.

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | >= 3.11 | [python.org](https://python.org) |
| uv | Latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| GPU | NVIDIA | Required for vLLM inference |
| pylet | Latest | Installed automatically if missing |

## Installation

```bash
git clone <repo-url> swarmpilot-refresh
cd swarmpilot-refresh
uv sync
```

This installs three CLI tools: `sscheduler`, `spredictor`, `splanner`.

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Scheduler  │◀────│   Planner   │────▶│   PyLet     │
│   (8000)    │     │   (8002)    │     │  (local)    │
│  proxy +    │     │  optimizer  │     │  cluster    │
│  predictor  │     └─────────────┘     └──────┬──────┘
└──────┬──────┘                                │
       │ routes requests                       │ deploys
       ▼                                       ▼
┌─────────────┐                         ┌─────────────┐
│   Client    │                         │  vLLM       │
│             │                         │  Instances  │
└─────────────┘                         └─────────────┘
```

- **Planner** deploys and manages vLLM instances via a local PyLet cluster
- **Scheduler** routes inference requests to instances via transparent proxy
- **PyLet** handles instance provisioning, health checks, and termination

---

## Step 1: Start the Cluster

The cluster requires three components started in order: the Planner (with local PyLet), and the Scheduler. The Scheduler starts without a fixed model -- the Planner assigns it dynamically on the first `serve()` call.

### 1.1 Prepare

Create a log directory and ensure PyLet is installed:

```bash
mkdir -p /tmp/qwen_cluster

# Install pylet if not already present
uv run python -c "import pylet" 2>/dev/null || uv pip install pylet
```

### 1.2 Start the Planner (with Local PyLet Cluster)

Launch the Planner in local PyLet mode. Adjust GPU/CPU counts to match your hardware:

```bash
PYLET_ENABLED="true" \
PYLET_LOCAL_MODE="true" \
PYLET_BACKEND="vllm" \
PYLET_GPU_COUNT="4" \
PYLET_DEPLOY_TIMEOUT="600" \
  uv run splanner start --port 8002
```

### 1.3 Start the Scheduler

Start the Scheduler with round-robin strategy and register it with the Planner:

```bash
PROXY_ENABLED="true" \
PROXY_TIMEOUT="600.0" \
PLANNER_REGISTRATION_URL="http://localhost:8002" \
SCHEDULER_SELF_URL="http://localhost:8000" \
  uv run sscheduler start --port 8000
```

### 1.4 Verify

```bash
curl http://localhost:8002/v1/health      # Planner
curl http://localhost:8000/v1/health      # Scheduler
curl http://localhost:8002/v1/schedulers  # Registered schedulers
```

---

## Step 2: Deploy a Model

Use the SwarmPilot SDK to deploy vLLM instances:

```python
import asyncio
from swarmpilot.sdk import SwarmPilotClient

async def main():
    async with SwarmPilotClient("http://localhost:8002") as sp:
        group = await sp.serve(
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            gpu=4,
            replicas=1,
        )
        await group.wait_ready(timeout=600)
        print(f"Instances ready: {group.endpoints}")

asyncio.run(main())
```

Or via CLI:

```bash
splanner serve "Qwen/Qwen3-Next-80B-A3B-Instruct" --gpu 4 --replicas 1
```

The Planner:
1. Finds an idle Scheduler and assigns the model via `/v1/model/reassign`
2. Launches vLLM instances on the PyLet cluster
3. Waits for health checks to pass
4. Registers instances with the Scheduler

### Verify Deployment

```bash
# List instances
splanner ps

# Check scheduler registration
curl http://localhost:8002/v1/schedulers
```

---

## Step 3: Send Inference Requests

The Scheduler acts as a transparent proxy -- send OpenAI-compatible requests directly:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "messages": [{"role": "user", "content": "What is SwarmPilot?"}],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

The Scheduler routes the request to a vLLM instance based on the current scheduling strategy (round_robin by default).

---

## Step 4: Stop the Cluster

### 4.1 Terminate Managed Instances

```bash
uv run splanner terminate --all --planner-url http://localhost:8002
```

### 4.2 Stop the Scheduler

```bash
kill $(cat /tmp/qwen_cluster/scheduler.pid) 2>/dev/null
```

### 4.3 Stop the Planner

Stopping the Planner also auto-stops the local PyLet cluster (head + workers):

```bash
kill $(cat /tmp/qwen_cluster/planner.pid) 2>/dev/null
```

### 4.4 Clean Up

Stop the dummy health server if still running, and verify all processes are stopped:

```bash
kill $(cat /tmp/qwen_cluster/dummy_health.pid) 2>/dev/null
# Verify no lingering processes
lsof -i:8000 -i:8002 -i:5100 -i:9999 2>/dev/null || echo "All ports free"
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Port already in use` | Run Step 5 to stop the cluster first |
| `pylet not found` | Run `uv pip install pylet` manually |
| Planner fails to start | Check `PYLET_LOCAL_GPU_PER_WORKER` matches available GPUs |
| Instance stuck deploying | Check `/tmp/qwen_cluster/planner.log` for PyLet errors |
| Scheduler proxy returns 503 | Instances not yet ready -- wait or check `splanner ps` |

---

## Next Steps

1. **Deploy a production cluster** -- See [Cluster Deployment](cluster_deployment.md) for multi-scheduler setups and external PyLet clusters
2. **Use the Python SDK** -- See [SDK Usage](sdk_usage.md) for the full programmatic API
3. **Train the predictor** -- See [Predictor](predictor.md) for MLP architecture and training details
