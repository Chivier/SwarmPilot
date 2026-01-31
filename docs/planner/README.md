# Planner Service

Model deployment optimization service for SwarmPilot. Computes optimal model-to-instance assignments and optionally deploys them via PyLet cluster management.

## Quick Start

```bash
cd planner
uv sync
uv run splanner start
curl http://localhost:8000/v1/health
```

## API Quick Reference

### Core

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/health` | Health check |
| GET | `/v1/info` | Service info and capabilities |
| POST | `/v1/plan` | Compute optimal deployment (no execution) |
| POST | `/v1/instance/register` | Register an available instance |

### Scheduler Registry

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/scheduler/register` | Register a scheduler for a model |
| POST | `/v1/scheduler/deregister` | Deregister a scheduler |
| GET | `/v1/scheduler/list` | List all registered schedulers |
| GET | `/v1/scheduler/{model_id}` | Get scheduler for a model |

### PyLet Deployment

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/status` | PyLet service status |
| POST | `/v1/deploy_manually` | Deploy to manual target state |
| POST | `/v1/deploy` | Optimize then deploy via PyLet |
| POST | `/v1/scale` | Scale a specific model |
| POST | `/v1/migrate` | Migrate an instance (cancel-and-resubmit) |
| POST | `/v1/optimize` | Optimize and deploy (simplified input) |
| POST | `/v1/terminate-all` | Terminate all instances |

### Scheduler Compatibility Stubs

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/instance/drain` | Stub: drain acknowledgment |
| GET | `/v1/instance/drain/status` | Stub: drain status |
| POST | `/v1/instance/remove` | Stub: removal acknowledgment |
| POST | `/v1/task/resubmit` | Stub: resubmit acknowledgment |

### Timeline

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/timeline` | Get instance count timeline |
| POST | `/v1/timeline/clear` | Clear the timeline |

## Core Workflow: Plan

```bash
curl -X POST http://localhost:8000/v1/plan \
  -H "Content-Type: application/json" \
  -d '{
    "M": 4, "N": 3,
    "B": [[10,5,0],[8,6,4],[0,10,8],[6,0,12]],
    "initial": [0,1,2,2],
    "a": 0.5,
    "target": [20,30,25]
  }'
```

```json
{
  "deployment": [0, 1, 1, 2],
  "score": 0.023,
  "stats": {"algorithm": "simulated_annealing", "iterations": 5000, "final_score": 0.023},
  "service_capacity": [10.0, 16.0, 12.0],
  "changes_count": 1
}
```

## Core Workflow: PyLet Deploy

```bash
# Optimize and deploy to real instances
curl -X POST http://localhost:8000/v1/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "M": 4, "N": 3,
    "B": [[10,5,0],[8,6,4],[0,10,8],[6,0,12]],
    "a": 0.5, "target": [20,30,25],
    "model_ids": ["model-a","model-b","model-c"],
    "wait_for_ready": true
  }'

# Deploy to manually specified state
curl -X POST http://localhost:8000/v1/deploy_manually \
  -H "Content-Type: application/json" \
  -d '{"target_state": {"model-a": 2, "model-b": 1}}'
```

## Key Data Models

### PlannerInput (POST /v1/plan)

| Field | Type | Required | Default |
|---|---|---|---|
| `M` | int | yes | |
| `N` | int | yes | |
| `B` | float[][] | yes | |
| `initial` | int[] \| null | no | null |
| `a` | float | yes | |
| `target` | float[] | yes | |
| `algorithm` | string | no | `"simulated_annealing"` |
| `objective_method` | string | no | `"relative_error"` |

### PlannerOutput

| Field | Type |
|---|---|
| `deployment` | int[] |
| `score` | float |
| `stats` | object |
| `service_capacity` | float[] |
| `changes_count` | int |

### PyLetDeploymentInput (POST /v1/deploy_manually)

| Field | Type | Required | Default |
|---|---|---|---|
| `target_state` | object | yes | |
| `wait_for_ready` | bool | no | `true` |
| `register_with_scheduler` | bool | no | `true` |

### PyLetOptimizeInput (POST /v1/optimize)

| Field | Type | Required | Default |
|---|---|---|---|
| `target` | float[] | yes | |
| `model_ids` | string[] | yes | |
| `B` | float[][] | yes | |
| `a` | float | no | `0.3` |
| `objective_method` | string | no | `"ratio_difference"` |
| `algorithm` | string | no | `"simulated_annealing"` |
| `wait_for_ready` | bool | no | `true` |

## Algorithms

| Aspect | Simulated Annealing | Integer Programming |
|---|---|---|
| Optimality | Approximate | Exact (relative_error only) |
| Objective support | All 3 | Only relative_error |
| Scalability | Linear | Exponential worst-case |

## Configuration

| Variable | Default |
|---|---|
| `PLANNER_HOST` | `0.0.0.0` |
| `PLANNER_PORT` | `8000` |
| `SCHEDULER_URL` | *(none)* |
| `INSTANCE_TIMEOUT` | `30` |
| `INSTANCE_MAX_RETRIES` | `3` |
| `INSTANCE_RETRY_DELAY` | `1.0` |
| `AUTO_OPTIMIZE_ENABLED` | `false` |
| `AUTO_OPTIMIZE_INTERVAL` | `60.0` |
| `PYLET_ENABLED` | `false` |
| `PYLET_HEAD_URL` | *(none)* |
| `PYLET_BACKEND` | `vllm` |
| `PYLET_GPU_COUNT` | `1` |
| `PYLET_CPU_COUNT` | `1` |
| `PYLET_DEPLOY_TIMEOUT` | `300.0` |
| `PYLET_DRAIN_TIMEOUT` | `30.0` |
| `PYLET_CUSTOM_COMMAND` | *(none)* |
| `PYLET_REUSE_CLUSTER` | `false` |

## Common Patterns

```bash
# 1. Health check
curl http://localhost:8000/v1/health

# 2. Compute a plan
curl -X POST http://localhost:8000/v1/plan -H "Content-Type: application/json" -d '{"M":4,"N":3,"B":[[10,5,0],[8,6,4],[0,10,8],[6,0,12]],"initial":[0,1,2,2],"a":0.5,"target":[20,30,25]}'

# 3. Scale a model
curl -X POST http://localhost:8000/v1/scale -H "Content-Type: application/json" -d '{"model_id":"model-a","target_count":3}'

# 4. Check PyLet status
curl http://localhost:8000/v1/status

# 5. Terminate all instances
curl -X POST http://localhost:8000/v1/terminate-all
```

## Error Handling

| Status | Meaning |
|---|---|
| `200` | Success |
| `400` | Invalid input |
| `404` | Resource not found |
| `422` | Validation error |
| `500` | Internal error |
| `503` | PyLet unavailable |

## Detailed Documentation

- [Introduction](docs/1.introduction.md) — Architecture, tech stack, module structure
- [Quick Start](docs/2.quick_start.md) — Installation, configuration, first run
- [API Specification](docs/3.api_specification.md) — Full endpoint schemas and data models
- [Algorithm Details](docs/4.algorithm_details.md) — Math formulas, pseudocode, PyLet internals
