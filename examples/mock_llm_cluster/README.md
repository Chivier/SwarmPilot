# Mock LLM Cluster Example

This example demonstrates PyLet-managed deployment of mock LLM models using the SwarmPilot Planner's optimizer. It simulates a realistic scenario with two models (7B and 32B) with different latency profiles and a 1:5 traffic ratio.

## Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Mock     │────▶│  Scheduler  │◀────│   Planner   │
│  Predictor  │     │   (8000)    │     │   (8002)    │
│   (8001)    │     └──────┬──────┘     └──────┬──────┘
└─────────────┘            │                   │
                           │ task dispatch     │ PyLet deploy
                           ▼                   ▼
              ┌────────────────────────────────────────┐
              │           16 Mock LLM Instances        │
              │  ┌─────────┐ ┌─────────┐               │
              │  │ 7B Mock │ │ 32B Mock│               │
              │  │ ~200ms  │ │ ~1000ms │               │
              │  └─────────┘ └─────────┘               │
              └────────────────────────────────────────┘
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
# 1. Start services (Predictor, Scheduler, Planner)
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

Starts the SwarmPilot services:
- **Mock Predictor** (port 8001): Simulates load prediction
- **Scheduler** (port 8000): Routes tasks to instances
- **Planner** (port 8002): Manages PyLet deployments with optimizer

Environment variables:
```bash
PREDICTOR_PORT=8001
SCHEDULER_PORT=8000
PLANNER_PORT=8002
PYLET_HEAD_PORT=5100
```

### deploy_models.sh

Uses the Planner's `/pylet/optimize` endpoint to calculate and deploy optimal instance allocation.

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

Generates tasks with the 1:5 QPS ratio and collects statistics.

```bash
# Default: 120 tasks at 6 QPS
python examples/mock_llm_cluster/generate_workload.py

# Custom configuration
python examples/mock_llm_cluster/generate_workload.py \
    --total-tasks 300 \
    --target-qps 10 \
    --scheduler-url http://localhost:8000
```

Output includes:
- Per-model latency statistics
- Traffic distribution analysis
- Completion rates

### stop_cluster.sh

Terminates all PyLet instances and stops services.

## API Endpoints

### Planner APIs

| Endpoint | Description |
|----------|-------------|
| `GET /pylet/status` | Get PyLet service status |
| `POST /pylet/deploy` | Deploy instances to target state |
| `POST /pylet/optimize` | Run optimizer and deploy result |
| `POST /pylet/terminate-all` | Terminate all instances |

### Scheduler APIs

| Endpoint | Description |
|----------|-------------|
| `POST /task/submit` | Submit a task for processing |
| `GET /task/info?task_id=X` | Get task status |
| `GET /instance/list` | List registered instances |

## Understanding the Optimizer

The optimizer (`/pylet/optimize`) uses:

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
- Check if ports are in use: `lsof -i:8000`
- Check logs: `tail -f /tmp/mock_llm_cluster/*.log`

### Deployment Fails
- Verify PyLet cluster: `curl http://localhost:5100/workers`
- Check Planner logs for PyLet errors

### Tasks Not Completing
- Check instance health: `curl http://localhost:8000/instance/list`
- Verify task routing: `curl http://localhost:8000/task/info?task_id=task-0001`
