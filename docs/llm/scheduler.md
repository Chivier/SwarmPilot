# Scheduler Service - LLM Reference

> Single-file reference for the Scheduler service. For detailed documentation, see `scheduler/README_FOR_LLM.md`.

## Overview

| Property | Value |
|----------|-------|
| **Service** | Task Scheduler |
| **Version** | 0.1.0 |
| **Port** | 8000 (default) |
| **Framework** | FastAPI |
| **Entry Point** | `scheduler/src/cli.py` |
| **Main API** | `scheduler/src/api.py` |

**Purpose:** Intelligent task scheduling service that distributes tasks across compute instances using ML-based runtime predictions.

---

## File Structure

```
scheduler/src/
├── api.py                    # FastAPI endpoints (3140 lines)
├── cli.py                    # CLI entry point
├── config.py                 # Configuration
├── model.py                  # Pydantic models
├── algorithms/               # 8 scheduling strategies
│   ├── base.py              # Abstract base class
│   ├── factory.py           # Strategy factory
│   ├── min_expected_time.py # Greedy shortest queue
│   ├── probabilistic.py     # Monte Carlo quantile-based
│   ├── round_robin.py       # Cyclic distribution
│   └── ...                  # Additional strategies
├── registry/                 # State management
│   ├── task_registry.py     # Task state
│   └── instance_registry.py # Instance state
├── services/                 # Background services
│   ├── background_scheduler.py  # Non-blocking scheduling
│   ├── central_queue.py         # FIFO task queue
│   ├── worker_queue_manager.py  # Per-worker coordination
│   ├── worker_queue_thread.py   # Worker execution thread
│   └── ...
├── clients/                  # External clients
│   ├── predictor_client.py  # Predictor WebSocket client
│   └── training_client.py   # Training HTTP client
└── utils/                    # Utilities
```

---

## API Endpoints

### Instance Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/instance/register` | POST | Register compute instance |
| `/instance/remove` | POST | Remove instance |
| `/instance/list` | GET | List all instances |
| `/instance/info` | GET | Get instance details |
| `/instance/drain` | POST | Start draining instance |
| `/instance/drain/status` | GET | Check drain status |
| `/instance/redeploy/start` | POST | Start instance redeploy |
| `/instance/redeploy/finish` | POST | Finish instance redeploy |

### Task Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/task/submit` | POST | Submit task for execution |
| `/task/list` | GET | List tasks with filters |
| `/task/info` | GET | Get task details |
| `/task/clear` | POST | Clear all tasks |
| `/task/resubmit` | POST | Resubmit failed task |
| `/task/update_metadata` | POST | Update task metadata |
| `/task/repredict` | POST | Re-run prediction for task |
| `/task/schedule_info` | GET | Get scheduling info for task |

### Callback & WebSocket

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/callback/task_result` | POST | Instance result callback |
| `/task/get_result` | WebSocket | Real-time results |

### Strategy & Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/strategy/get` | GET | Get current strategy |
| `/strategy/set` | POST | Change strategy |
| `/health` | GET | Health check |

---

## Key Request/Response Schemas

### Task Submit
```json
// POST /task/submit
{
  "task_id": "task-001",
  "model_id": "llama-7b",
  "task_input": {"prompt": "...", "max_tokens": 100},
  "metadata": {"user_id": "..."}
}
```

### Instance Register
```json
// POST /instance/register
{
  "instance_id": "worker-001",
  "model_id": "llama-7b",
  "endpoint": "http://worker:8080",
  "platform_info": {"gpu_type": "A100"}
}
```

### Task Result Callback
```json
// POST /callback/task_result
{
  "task_id": "task-001",
  "status": "completed",
  "result": {"generated_text": "..."},
  "execution_time_ms": 234.56
}
```

---

## Scheduling Strategies

| Strategy | File | Predictor Required | Use Case |
|----------|------|-------------------|----------|
| `round_robin` | `round_robin.py` | No | Equal distribution, testing |
| `random` | `random.py` | No | Baseline |
| `min_time` | `min_expected_time.py` | Yes | Minimize avg latency |
| `probabilistic` | `probabilistic.py` | Yes | Minimize tail latency (SLA) |
| `power_of_two` | `power_of_two.py` | Yes | Large scale balance |
| `serverless` | `serverless.py` | Yes | Serverless workloads |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SCHEDULER_HOST` | `0.0.0.0` | Bind host |
| `SCHEDULER_PORT` | `8000` | Bind port |
| `SCHEDULING_STRATEGY` | `probabilistic` | Default strategy |
| `PREDICTOR_URL` | `http://localhost:8001` | Predictor service |

---

## Status Enums

**TaskStatus:** `pending`, `running`, `completed`, `failed`

**InstanceStatus:** `initializing`, `active`, `draining`, `removing`, `redeploying`

---

**Version:** 0.1.0 | **Updated:** 2026-01-16
