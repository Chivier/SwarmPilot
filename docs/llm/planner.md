# Planner Service - LLM Reference

> Single-file reference for the Planner service. For detailed documentation, see [docs/](../).

## Overview

| Property | Value |
|----------|-------|
| **Service** | Deployment Optimizer |
| **Version** | 1.0.0 |
| **Port** | 8002 (typical) |
| **API Prefix** | `/v1/` |
| **Framework** | FastAPI |
| **Entry Point** | `swarmpilot/planner/cli.py` |
| **Main API** | `swarmpilot/planner/api.py` |
| **CLI** | `splanner` |

**Purpose:** Computes optimal model deployment strategies using mathematical optimization (Simulated Annealing or Integer Programming), and deploys via PyLet.

---

## File Structure

```
swarmpilot/planner/
├── api.py                    # FastAPI endpoints (entry point)
├── pylet_api.py              # PyLet integration endpoints (router, mounted at /v1)
├── cli.py                    # CLI entry point (splanner)
├── config.py                 # Configuration
├── models.py                 # Pydantic models
├── core/
│   └── swarm_optimizer.py    # SA and IP optimizers
├── pylet/                    # PyLet integration
│   ├── client.py             # PyLet API client
│   ├── instance_manager.py   # Instance lifecycle
│   ├── deployment_executor.py # Deployment execution
│   ├── deployment_service.py  # High-level service
│   ├── migration_executor.py  # Instance migration
│   └── scheduler_client.py    # Scheduler client
├── scheduler_registry.py     # Model -> Scheduler URL mapping
├── available_instance_store.py
├── instance_timeline_tracker.py
└── services/
    └── model_validation.py   # Model validation for deploy
```

---

## API Endpoints

**All endpoints use the `/v1/` prefix.**

### Core Planning

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/plan` | POST | Compute optimal deployment plan (dry run) |
| `/v1/health` | GET | Health check |
| `/v1/info` | GET | Service info + PyLet status |
| `/v1/timeline` | GET | Get deployment timeline |
| `/v1/timeline/clear` | POST | Clear deployment timeline |

### Scheduler Registry

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/scheduler/register` | POST | Register a scheduler for a model |
| `/v1/scheduler/deregister` | POST | Deregister a scheduler |
| `/v1/scheduler/list` | GET | List all registered schedulers |
| `/v1/scheduler/{model_id}` | GET | Get scheduler for a model |

### Instance Management (Scheduler-Compatible)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/instance/register` | POST | Register instance to available store |
| `/v1/instance/drain` | POST | Drain instance (dummy, always succeeds) |
| `/v1/instance/drain/status` | GET | Drain status (dummy, always can_remove) |
| `/v1/instance/remove` | POST | Remove instance (dummy, always succeeds) |
| `/v1/task/resubmit` | POST | Task resubmit (dummy, no-op) |

### PyLet Deployment

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/status` | GET | PyLet service status |
| `/v1/deploy` | POST | Run optimizer and deploy result |
| `/v1/deploy_manually` | POST | Deploy to manual target state |
| `/v1/scale` | POST | Scale specific model |
| `/v1/migrate` | POST | Migrate instance |
| `/v1/optimize` | POST | Optimize and deploy (simplified) |
| `/v1/terminate-all` | POST | Terminate all instances |

---

## Key Request/Response Schemas

### Plan Request
```json
// POST /v1/plan
{
  "M": 4,                    // Number of instances
  "N": 3,                    // Number of model types
  "B": [[10, 5, 0], ...],    // Batch capacity matrix [M×N]
  "initial": [0, 1, 2, 2],   // Initial deployment
  "a": 0.5,                  // Change constraint (0 < a ≤ 1)
  "target": [20, 30, 25],    // Target distribution
  "algorithm": "simulated_annealing",
  "objective_method": "relative_error"
}
```

### Plan Response
```json
{
  "deployment": [0, 1, 1, 2],
  "score": 0.0667,
  "stats": {"algorithm": "...", "iterations": 5000},
  "service_capacity": [10.0, 16.0, 12.0],
  "changes_count": 1
}
```

### PyLet Deploy (with Optimizer)
```json
// POST /v1/deploy - runs optimizer then deploys
{
  "M": 4,                    // Number of instances
  "N": 2,                    // Number of model types
  "B": [[10, 8], [10, 8], [10, 8], [10, 8]],  // Capacity matrix
  "target": [0.6, 0.4],      // Target distribution (normalized)
  "a": 0.5,                  // Change constraint
  "model_ids": ["model-a", "model-b"],  // Model ID mapping
  "algorithm": "simulated_annealing",
  "wait_for_ready": true
}
```

### PyLet Deploy Manually
```json
// POST /v1/deploy_manually - deploy to explicit target state
{
  "target_state": {"model-a": 2, "model-b": 1},
  "wait_for_ready": true
}
```

### Scheduler Register
```json
// POST /v1/scheduler/register
{
  "model_id": "sleep_model",
  "scheduler_url": "http://localhost:8000"
}
```

---

## Optimization Algorithms

| Algorithm | Best For | Key Parameters |
|-----------|----------|----------------|
| `simulated_annealing` | Large problems, fast | Temperature schedule |
| `integer_programming` | Optimal solution | Time limit |

### Objective Methods

| Method | Description |
|--------|-------------|
| `relative_error` | Relative deviation (balanced) |
| `ratio_difference` | Ratio mismatch (proportional) |
| `weighted_squared` | Squared errors (penalize large) |

---

## PyLet Instance States

```
DEPLOYING → WAITING_HEALTH → REGISTERING → ACTIVE
                                              ↓
                                          DRAINING
                                              ↓
                                         TERMINATING
                                              ↓
                                          TERMINATED
```

**States:** `deploying`, `waiting_health`, `registering`, `active`, `draining`, `terminating`, `terminated`, `failed`

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PLANNER_HOST` | `0.0.0.0` | Bind host |
| `PLANNER_PORT` | `8000` | Bind port |
| `SCHEDULER_URL` | _(none)_ | Default scheduler URL |
| `PYLET_ENABLED` | `false` | Enable PyLet |
| `PYLET_HEAD_URL` | _(none)_ | PyLet head URL (required when enabled) |
| `PYLET_BACKEND` | `vllm` | Default backend (`vllm` or `sglang`) |
| `PYLET_GPU_COUNT` | `1` | GPUs per instance |
| `PYLET_CPU_COUNT` | `1` | CPUs per instance |
| `PYLET_DEPLOY_TIMEOUT` | `300.0` | Deploy timeout (seconds) |
| `PYLET_DRAIN_TIMEOUT` | `30.0` | Drain timeout (seconds) |
| `PYLET_CUSTOM_COMMAND` | _(none)_ | Custom command (overrides backend) |
| `PYLET_REUSE_CLUSTER` | `false` | Reuse existing cluster |

See [CONFIGURATION.md](../CONFIGURATION.md) for the full list.

---

**Version:** 1.0.0 | **Updated:** 2026-01-30
