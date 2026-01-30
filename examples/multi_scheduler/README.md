# Multi-Scheduler Example (Sleep Models)

This example demonstrates SwarmPilot's **multi-scheduler architecture** where each model gets its own scheduler, coordinated through a central planner. Tasks are submitted to model-specific schedulers based on their target model_id.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Planner (PyLet)                    │
│  - Orchestrates instance deployment                 │
│  - Manages scheduler registration                   │
│  - Coordinates PyLet cluster                        │
└─────────────────────────────────────────────────────┘
              ↓           ↓           ↓
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │ Scheduler A │ │ Scheduler B │ │ Scheduler C │
    │  (sleep_a)  │ │  (sleep_b)  │ │  (sleep_c)  │
    │   :8010     │ │   :8011     │ │   :8012     │
    └─────────────┘ └─────────────┘ └─────────────┘
         ↓               ↓               ↓
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   PyLet     │ │   PyLet     │ │   PyLet     │
    │  Instances  │ │  Instances  │ │  Instances  │
    │ (sleep_a)   │ │ (sleep_b)   │ │ (sleep_c)   │
    └─────────────┘ └─────────────┘ └─────────────┘
```

## Key Features

- **Per-Model Schedulers**: Each sleep model (a, b, c) has a dedicated scheduler instance
- **Planner Coordination**: Central planner manages scheduler registration and instance deployment
- **Equal Distribution**: Workload generator distributes tasks equally across models (1:1:1 ratio)
- **Sleep Models**: Uses lightweight sleep models instead of LLMs for testing
- **Dummy Health Server**: Solves chicken-and-egg problem during planner initialization

## The Dummy Health Server Pattern

The planner's PyLet service requires a scheduler health check during initialization, but real schedulers aren't available yet. The solution:

1. Start a dummy HTTP server on :8001 that responds to `/health`
2. Planner initializes against the dummy server
3. Stop the dummy server
4. Start real per-model schedulers (they register with planner)

This pattern is implemented in `start_cluster.sh` steps [1-2].

## Quick Start

### Prerequisites

- Python 3.11+
- `uv` package manager
- PyLet cluster running: `./scripts/start_pylet_test_cluster.sh`

### Step 1: Start Services

```bash
./examples/multi_scheduler/start_cluster.sh
```

This will:
- Start dummy health server (:8001)
- Start planner (:8003)
- Stop dummy health server
- Start 3 per-model schedulers (:8010, :8011, :8012)
- Verify all are healthy and registered

### Step 2: Deploy Instances

```bash
./examples/multi_scheduler/deploy_model.sh 4
```

Deploys 4 instances per model (12 total) via planner `/v1/deploy_manually`.

### Step 3: Generate Workload

```bash
python examples/multi_scheduler/generate_workload.py --total-tasks 120 --target-qps 6
```

Generates 120 tasks across 3 models at 6 QPS total.

### Step 4: Stop Services

```bash
./examples/multi_scheduler/stop_cluster.sh
```

Terminates PyLet instances and stops all services.

## Script Descriptions

### `start_cluster.sh`

Starts the complete multi-scheduler setup:

| Step | Action | Port |
|------|--------|------|
| 1 | Start dummy health server | 8001 |
| 2 | Start planner with PyLet | 8003 |
| - | Stop dummy health server | - |
| 3 | Start scheduler A (sleep_model_a) | 8010 |
| 4 | Start scheduler B (sleep_model_b) | 8011 |
| 5 | Start scheduler C (sleep_model_c) | 8012 |

**Environment Variables:**
- `PYLET_HEAD_PORT`: PyLet head node port (default: 5100)
- `PLANNER_PORT`: Planner port (default: 8003)
- `SCHEDULER_A_PORT`, `SCHEDULER_B_PORT`, `SCHEDULER_C_PORT`: Scheduler ports

**Log Directory:** `/tmp/multi_scheduler/`

### `deploy_model.sh [instances_per_model]`

Deploys instances using planner's `/v1/deploy_manually` endpoint.

**Parameters:**
- `instances_per_model`: Instances to deploy per model (default: 4)

**Example:**
```bash
./examples/multi_scheduler/deploy_model.sh 8    # 8 instances per model = 24 total
```

### `generate_workload.py`

Generates synthetic workload across all models.

**Options:**
```
--total-tasks       Total tasks to submit (default: 120)
--target-qps        Target QPS (default: 6)
--duration          Duration in seconds (default: 20)
--sleep-time-min    Min sleep time (default: 0.1s)
--sleep-time-max    Max sleep time (default: 1.0s)
--planner-url       Planner URL (default: http://localhost:8003)
```

**Example:**
```bash
python examples/multi_scheduler/generate_workload.py \
  --total-tasks 300 \
  --target-qps 10 \
  --sleep-time-min 0.2 \
  --sleep-time-max 0.5
```

The generator will:
1. Query planner for scheduler discovery
2. Verify each scheduler is healthy
3. Submit tasks with equal distribution (1:1:1 across models)
4. Poll for task completion
5. Print per-model statistics

### `stop_cluster.sh`

Stops all services in reverse order:

1. Terminate PyLet instances via planner `/v1/terminate-all`
2. Stop scheduler A
3. Stop scheduler B
4. Stop scheduler C
5. Stop planner

## Configuration

### Scheduler Registration

Schedulers register with the planner using environment variables:

```bash
SCHEDULER_MODEL_ID="sleep_model_a"
PLANNER_REGISTRATION_URL="http://localhost:8003"
SCHEDULER_SELF_URL="http://localhost:8010"
PREDICTOR_MODE="library"  # Use library mode (no external service)
```

### Planner PyLet Configuration

```bash
PYLET_ENABLED=true
PYLET_HEAD_URL="http://localhost:5100"
PYLET_CUSTOM_COMMAND="MODEL_ID={model_id} python pylet_sleep_model.py"
PYLET_GPU_COUNT=0
PYLET_CPU_COUNT=1
```

## Sleep Model Details

The sleep model (`examples/multi_scheduler/pylet_sleep_model.py`) is a lightweight FastAPI service that:

- Reads port from `$PORT` environment variable (set by PyLet)
- Registers with its scheduler on startup via `SCHEDULER_URL`
- Handles `/inference` requests by sleeping for specified duration
- Deregisters on graceful shutdown (SIGTERM/SIGINT)

**Endpoints:**
- `POST /inference` - Sleep and return result
- `GET /health` - Health check
- `GET /stats` - Instance statistics

## Troubleshooting

### Scheduler not registering with planner

Check scheduler logs:
```bash
tail -f /tmp/multi_scheduler/scheduler-a.log
tail -f /tmp/multi_scheduler/scheduler-b.log
tail -f /tmp/multi_scheduler/scheduler-c.log
```

Verify planner is healthy:
```bash
curl http://localhost:8003/v1/health
```

### PyLet instances not starting

Check planner logs for PyLet errors:
```bash
tail -f /tmp/multi_scheduler/planner.log
```

Verify PyLet cluster is running:
```bash
curl http://localhost:5100/workers
```

### Deploy fails with "no schedulers registered"

Ensure all 3 schedulers have registered:
```bash
curl http://localhost:8003/v1/scheduler/list | python -m json.tool
```

### Tasks not completing

Check scheduler logs for errors, verify instances are registered:
```bash
curl http://localhost:8010/v1/instance/list | python -m json.tool
curl http://localhost:8011/v1/instance/list | python -m json.tool
curl http://localhost:8012/v1/instance/list | python -m json.tool
```

## Related Examples

- **Mock LLM Cluster** (`examples/mock_llm_cluster/`): Multi-scheduler with mock LLM models (llm-7b, llm-32b)
- **LLM Cluster** (`examples/llm_cluster/`): Single-scheduler with 3 LLM models and optimizer
- **PyLet Benchmark** (`examples/pylet_benchmark/`): Direct registration, no planner

## References

- [Scheduler Configuration](../../scheduler/README.md)
- [Planner Documentation](../../planner/README.md)
- [PyLet Integration Guide](../../docs/pylet_integration.md)
