# Planner Service - LLM Reference

> Single-file reference for the Planner service. For detailed documentation, see `planner/README_FOR_LLM.md`.

## Overview

| Property | Value |
|----------|-------|
| **Service** | Deployment Optimizer |
| **Version** | 0.1.0 |
| **Port** | 8000 (default) |
| **Framework** | FastAPI |
| **Entry Point** | `planner/src/api.py` |
| **Main API** | `planner/src/api.py` |

**Purpose:** Computes optimal model deployment strategies using mathematical optimization (Simulated Annealing or Integer Programming).

---

## File Structure

```
planner/src/
├── api.py                    # FastAPI endpoints (entry point)
├── pylet_api.py              # PyLet integration endpoints
├── config.py                 # Configuration
├── core/
│   └── swarm_optimizer.py    # SA and IP optimizers
├── models/
│   ├── planner.py            # Planner I/O models
│   ├── pylet.py              # PyLet models
│   ├── instance.py           # Instance models
│   └── scheduler_compat.py   # Scheduler compatibility
└── pylet/                    # PyLet integration
    ├── client.py             # PyLet API client
    ├── instance_manager.py   # Instance lifecycle
    ├── deployment_executor.py # Deployment execution
    ├── deployment_service.py  # High-level service
    ├── migration_executor.py  # Instance migration
    └── scheduler_client.py    # Scheduler client
```

---

## API Endpoints

### Planning

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/plan` | POST | Compute optimal deployment plan |
| `/health` | GET | Health check |
| `/info` | GET | Service info |
| `/timeline` | GET | Get deployment timeline |
| `/timeline/clear` | POST | Clear deployment timeline |

### PyLet Integration

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | PyLet service status |
| `/deploy` | POST | Run optimizer and deploy result |
| `/deploy_manually` | POST | Deploy to manual target state |
| `/scale` | POST | Scale specific model |
| `/migrate` | POST | Migrate instance |
| `/optimize` | POST | Optimize and deploy (simplified) |
| `/terminate-all` | POST | Terminate all instances |

### Scheduler Compatibility

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/instance/register` | POST | Register instance |
| `/instance/drain` | POST | Drain instance |
| `/instance/remove` | POST | Remove instance |

---

## Key Request/Response Schemas

### Plan Request
```json
// POST /plan
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

### PyLet Deploy (with Planner Algorithm)
```json
// POST /deploy - runs optimizer then deploys
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
// POST /deploy_manually - deploy to explicit target state
{
  "target_state": {"model-a": 2, "model-b": 1},
  "wait_for_ready": true
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
| `PYLET_ENABLED` | `false` | Enable PyLet |
| `PYLET_HEAD_URL` | - | PyLet head URL |
| `PYLET_BACKEND` | `vllm` | Default backend |
| `SCHEDULER_URL` | `http://localhost:8100` | Scheduler URL |

---

**Version:** 0.1.0 | **Updated:** 2026-01-16
