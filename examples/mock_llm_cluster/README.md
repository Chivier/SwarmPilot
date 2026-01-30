# Mock LLM Cluster Example

This example demonstrates PyLet-managed deployment of mock LLM models using the SwarmPilot Planner's optimizer. It simulates a realistic scenario with two models (7B and 32B) with different latency profiles and a 1:5 traffic ratio.

## Overview

```
┌─────────────┐
│    Mock     │  (load prediction)
│  Predictor  │
│   (:8001)   │
└─────────────┘

┌─────────────────────────────────────────────────┐
│              Planner (:8002)                     │
│  - Scheduler registration                       │
│  - PyLet instance deployment                    │
│  - Optimizer for instance allocation            │
└───────────┬─────────────────────┬───────────────┘
            │                     │
            ▼                     ▼
      ┌───────────┐        ┌───────────┐
      │ Scheduler │        │ Scheduler │
      │  llm-7b   │        │  llm-32b  │
      │  (:8010)  │        │  (:8020)  │
      └─────┬─────┘        └─────┬─────┘
            │                     │
            ▼                     ▼
      ┌───────────┐        ┌───────────┐
      │   PyLet   │        │   PyLet   │
      │ Instances │        │ Instances │
      │  (7B)     │        │  (32B)    │
      └───────────┘        └───────────┘
```

## Model Configuration

| Model | Mean Latency | Throughput | Distribution |
|-------|--------------|------------|--------------|
| `llm-7b` | ~200ms | ~5 req/s | Exponential |
| `llm-32b` | ~1000ms | ~1 req/s | Gamma |

## Traffic Pattern

- **QPS Ratio**: 1:5 (32B receives 5x more traffic than 7B)
- 7B: ~16.7% of incoming requests
- 32B: ~83.3% of incoming requests

## Quick Start

### Prerequisites

1. PyLet cluster running:
   ```bash
   ./scripts/start_pylet_test_cluster.sh
   ```

2. Python dependencies installed:
   ```bash
   uv sync
   ```

### Run the Example

```bash
# 1. Start services (Predictor, Planner, 2 Schedulers)
./examples/mock_llm_cluster/start_cluster.sh

# 2. Deploy models using optimizer
./examples/mock_llm_cluster/deploy_models.sh

# 3. Generate traffic
python examples/mock_llm_cluster/generate_workload.py

# 4. Stop cluster when done
./examples/mock_llm_cluster/stop_cluster.sh
```

## Scripts

### start_cluster.sh

Starts the SwarmPilot services in 4 steps:

| Step | Service | Port |
|------|---------|------|
| 1 | Mock Predictor | 8001 |
| 2 | Planner (PyLet-enabled) | 8002 |
| 3 | Scheduler (llm-7b) | 8010 |
| 4 | Scheduler (llm-32b) | 8020 |

Each scheduler self-registers with the planner on startup. The scheduler is started
via `src.cli start` (not `uvicorn src.api:app`).

Environment variables:
```bash
PREDICTOR_PORT=8001
SCHEDULER_7B_PORT=8010
SCHEDULER_32B_PORT=8020
PLANNER_PORT=8002
PYLET_HEAD_PORT=5100
```

### deploy_models.sh

Uses the Planner's `/v1/deploy` endpoint to calculate and deploy optimal instance allocation.

```bash
# Deploy default 16 instances
./examples/mock_llm_cluster/deploy_models.sh

# Deploy custom number of instances
./examples/mock_llm_cluster/deploy_models.sh 8
```

The optimizer considers:
- Traffic ratio (1:5)
- Model throughput (7B: 5 req/s, 32B: 1 req/s)
- Total available instances

### generate_workload.py

Discovers per-model schedulers via the planner and generates tasks with the 1:5 QPS ratio.

```bash
# Default: 120 tasks at 6 QPS
python examples/mock_llm_cluster/generate_workload.py

# Custom configuration
python examples/mock_llm_cluster/generate_workload.py \
    --total-tasks 300 \
    --target-qps 10 \
    --planner-url http://localhost:8002
```

Output includes:
- Per-model latency statistics
- Traffic distribution analysis
- Completion rates

### stop_cluster.sh

Terminates all PyLet instances and stops services in 6 steps:

1. Terminate PyLet instances via planner `/v1/terminate-all`
2. Stop Scheduler (llm-7b)
3. Stop Scheduler (llm-32b)
4. Stop Planner
5. Stop legacy Scheduler (if any)
6. Stop Predictor

## API Endpoints

### Planner APIs

| Endpoint | Description |
|----------|-------------|
| `GET /v1/health` | Planner health check |
| `GET /v1/status` | Get PyLet service status |
| `GET /v1/scheduler/list` | List registered schedulers |
| `POST /v1/deploy` | Run optimizer and deploy result |
| `POST /v1/deploy_manually` | Deploy instances to explicit target state |
| `POST /v1/terminate-all` | Terminate all instances |

### Scheduler APIs

| Endpoint | Description |
|----------|-------------|
| `POST /v1/task/submit` | Submit a task for processing |
| `GET /v1/task/info?task_id=X` | Get task status |
| `GET /v1/instance/list` | List registered instances |
| `GET /v1/health` | Scheduler health check |

## Understanding the Optimizer

The optimizer (via `/v1/deploy`) uses:

1. **Target Distribution**: Normalized QPS targets `[16.67, 83.33]` for 1:5 ratio
2. **Capacity Matrix (B)**: Each instance can serve either model
   - 7B: 5 req/s (200ms latency)
   - 32B: 1 req/s (1000ms latency)
3. **Algorithm**: Simulated annealing to find optimal allocation

Example response:
```json
{
  "deployment": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  "score": 0.0012,
  "service_capacity": [10.0, 14.0],
  "deployment_success": true,
  "added_count": 16
}
```

This means: 2 instances for 7B (indices 0,1) and 14 instances for 32B.

## Customization

### Adding More Models

Edit `mock_llm_server.py` to add latency distributions:

```python
MODEL_LATENCY_DISTRIBUTIONS = {
    "llm-7b": LatencyDistribution(mean_ms=200.0, distribution="exponential"),
    "llm-32b": LatencyDistribution(mean_ms=1000.0, distribution="gamma", shape=2.0),
    "llm-70b": LatencyDistribution(mean_ms=3000.0, distribution="gamma", shape=3.0),
}
```

### Changing Traffic Ratios

Modify `deploy_models.sh` target distribution and `generate_workload.py` task queue logic.

## Troubleshooting

### Services Won't Start
- Check if ports are in use: `lsof -i:8010` or `lsof -i:8020`
- Check logs: `tail -f /tmp/mock_llm_cluster/*.log`

### Deployment Fails
- Verify PyLet cluster: `curl http://localhost:5100/workers`
- Check Planner logs for PyLet errors
- Verify schedulers are registered: `curl http://localhost:8002/v1/scheduler/list`

### Tasks Not Completing
- Check instance health via each scheduler:
  ```bash
  curl http://localhost:8010/v1/instance/list | python3 -m json.tool
  curl http://localhost:8020/v1/instance/list | python3 -m json.tool
  ```
- Verify task routing: `curl "http://localhost:8010/v1/task/info?task_id=task-0001"`
