# Scheduler Service - LLM Reference

> Single-file reference for the Scheduler service. For detailed documentation, see [docs/](../).

## Overview

| Property | Value |
|----------|-------|
| **Service** | Task Scheduler |
| **Version** | 1.0.0 |
| **Port** | 8000 (default) |
| **API Prefix** | `/v1/` |
| **Framework** | FastAPI |
| **Entry Point** | `swarmpilot/scheduler/cli.py` |
| **Main API** | `swarmpilot/scheduler/api.py` |
| **CLI** | `sscheduler` |

**Purpose:** Intelligent task scheduling service that distributes tasks across compute instances using ML-based runtime predictions.

---

## File Structure

```
swarmpilot/scheduler/
├── api.py                    # FastAPI endpoints
├── cli.py                    # CLI entry point (sscheduler)
├── config.py                 # Configuration (env vars)
├── models.py                 # Pydantic models
├── algorithms/               # 7 scheduling strategies
│   ├── base.py              # Abstract base class
│   ├── factory.py           # Strategy factory
│   ├── adaptive_bootstrap.py # Default strategy
│   ├── min_expected_time.py # Greedy shortest queue
│   ├── probabilistic.py     # Monte Carlo quantile-based
│   ├── round_robin.py       # Cyclic distribution
│   ├── random.py            # Uniform random
│   ├── power_of_two.py      # Pick best of 2 random
│   └── serverless.py        # Serverless scaling
├── registry/                 # State management
│   ├── task_registry.py     # Task state
│   └── instance_registry.py # Instance state
├── services/                 # Background services
│   ├── worker_queue_manager.py  # Per-worker coordination
│   ├── worker_queue_thread.py   # Worker execution thread
│   ├── websocket_manager.py     # WebSocket connections
│   ├── planner_registrar.py     # Planner registration
│   └── task_result_callback.py  # Result handling
├── clients/                  # External clients
│   ├── predictor_library_client.py  # Predictor client
│   └── training_client.py          # Training HTTP client
└── utils/                    # Utilities
```

---

## API Endpoints

**All endpoints use the `/v1/` prefix.**

### Instance Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/instance/register` | POST | Register compute instance |
| `/v1/instance/remove` | POST | Remove instance |
| `/v1/instance/list` | GET | List all instances |
| `/v1/instance/info` | GET | Get instance details |
| `/v1/instance/drain` | POST | Start draining instance |
| `/v1/instance/drain/status` | GET | Check drain status |
| `/v1/instance/redeploy/start` | POST | Start instance redeploy |
| `/v1/instance/redeploy/complete` | POST | Complete instance redeploy |

### Task Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/task/submit` | POST | Submit task for execution |
| `/v1/task/list` | GET | List tasks with filters |
| `/v1/task/info` | GET | Get task details |
| `/v1/task/clear` | POST | Clear all tasks |
| `/v1/task/resubmit` | POST | Resubmit failed task |
| `/v1/task/update_metadata` | POST | Update task metadata |
| `/v1/task/repredict` | POST | Re-run prediction for task |
| `/v1/task/schedule_info` | GET | Get scheduling info for task |

### Callback & WebSocket

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/callback/task_result` | POST | Instance result callback |
| `/v1/task/get_result` | WebSocket | Real-time results |

### Strategy & Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/strategy/get` | GET | Get current strategy |
| `/v1/strategy/set` | POST | Change strategy |
| `/v1/health` | GET | Health check |

---

## Key Request/Response Schemas

### Task Submit
```json
// POST /v1/task/submit
{
  "task_id": "task-001",
  "model_id": "llama-7b",
  "task_input": {"prompt": "...", "max_tokens": 100},
  "metadata": {"user_id": "..."}
}
```

### Instance Register
```json
// POST /v1/instance/register
{
  "instance_id": "worker-001",
  "model_id": "llama-7b",
  "endpoint": "http://worker:8080",
  "platform_info": {"gpu_type": "A100"}
}
```

### Task Result Callback
```json
// POST /v1/callback/task_result
{
  "task_id": "task-001",
  "status": "completed",
  "result": {"generated_text": "..."},
  "execution_time_ms": 234.56
}
```

---

## Scheduling Strategies

| Strategy | Key | File | Predictor Required | Use Case |
|----------|-----|------|-------------------|----------|
| Adaptive Bootstrap | `adaptive_bootstrap` | `adaptive_bootstrap.py` | Yes | **Default.** Bootstrapped prediction intervals |
| Min Expected Time | `min_time` | `min_expected_time.py` | Yes | Minimize avg latency |
| Probabilistic | `probabilistic` | `probabilistic.py` | Yes | Minimize tail latency (SLA) |
| Round Robin | `round_robin` | `round_robin.py` | No | Equal distribution, testing |
| Random | `random` | `random.py` | No | Baseline |
| Power of Two | `po2` | `power_of_two.py` | Yes | Large scale balance |
| Serverless | `severless` | `serverless.py` | Yes | Serverless workloads |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SCHEDULER_HOST` | `0.0.0.0` | Bind host |
| `SCHEDULER_PORT` | `8000` | Bind port |
| `SCHEDULER_ENABLE_CORS` | `true` | Enable CORS |
| `SCHEDULING_STRATEGY` | `adaptive_bootstrap` | Default strategy |
| `SCHEDULING_PROBABILISTIC_QUANTILE` | `0.9` | Quantile for probabilistic strategy |
| `TRAINING_ENABLE_AUTO` | `false` | Auto-training |
| `PREDICTOR_STORAGE_DIR` | `models` | Model storage (library mode) |
| `PREDICTOR_CACHE_MAX_SIZE` | `100` | Model cache size |
| `PROXY_ENABLED` | `true` | Transparent proxy |
| `WORKER_HTTP_TIMEOUT` | `300.0` | Worker HTTP timeout (s) |

See [CONFIGURATION.md](../CONFIGURATION.md) for the full list.

---

## Status Enums

**TaskStatus:** `pending`, `running`, `completed`, `failed`

**InstanceStatus:** `initializing`, `active`, `draining`, `removing`, `redeploying`

---

**Version:** 1.0.0 | **Updated:** 2026-01-30
