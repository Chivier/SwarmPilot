# API Reference

Complete endpoint reference for all three SwarmPilot services.

> **Prefix rules:** Scheduler and Planner endpoints use `/v1/`. Predictor endpoints have **no** prefix.

---

## Scheduler API (port 8000)

All endpoints use the `/v1/` prefix.

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/health` | Health check with instance/task counts |

```bash
curl http://localhost:8000/v1/health
```

### Instance Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/instance/register` | Register a compute instance |
| POST | `/v1/instance/remove` | Remove an instance |
| GET | `/v1/instance/list` | List all registered instances |
| GET | `/v1/instance/info` | Get details for one instance |
| POST | `/v1/instance/drain` | Start draining an instance (stop new tasks) |
| GET | `/v1/instance/drain/status` | Check drain progress |
| POST | `/v1/instance/redeploy/start` | Begin instance redeployment |
| POST | `/v1/instance/redeploy/complete` | Complete instance redeployment |

#### Register Instance

```bash
curl -X POST http://localhost:8000/v1/instance/register \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "sleep_model-001",
    "model_id": "sleep_model",
    "endpoint": "http://localhost:8300",
    "platform_info": {"software_name": "custom", "hardware_name": "cpu"}
  }'
```

#### List Instances

```bash
curl http://localhost:8000/v1/instance/list
```

#### Drain Instance

```bash
curl -X POST http://localhost:8000/v1/instance/drain \
  -H "Content-Type: application/json" \
  -d '{"instance_id": "sleep_model-001"}'
```

### Task Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/task/submit` | Submit a task for scheduling |
| GET | `/v1/task/list` | List tasks with optional status filter |
| GET | `/v1/task/info` | Get task details by task_id |
| POST | `/v1/task/clear` | Clear all tasks |
| POST | `/v1/task/resubmit` | Resubmit a failed/completed task |
| POST | `/v1/task/update_metadata` | Update task metadata |
| POST | `/v1/task/repredict` | Re-run prediction for a task |
| GET | `/v1/task/schedule_info` | Get scheduling decision details |

#### Submit Task

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

Response:
```json
{
  "success": true,
  "message": "Task queued successfully",
  "task": {
    "task_id": "test-001",
    "status": "pending"
  }
}
```

#### Get Task Info

```bash
curl "http://localhost:8000/v1/task/info?task_id=test-001"
```

#### List Tasks

```bash
# All tasks
curl http://localhost:8000/v1/task/list

# Filter by status
curl "http://localhost:8000/v1/task/list?status=completed"
```

### Callback

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/callback/task_result` | Receive task result from an instance |

This endpoint is called by compute instances, not by clients directly.

### Strategy

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/strategy/get` | Get current scheduling strategy |
| POST | `/v1/strategy/set` | Change the scheduling strategy |

```bash
# Get current strategy
curl http://localhost:8000/v1/strategy/get

# Change strategy
curl -X POST http://localhost:8000/v1/strategy/set \
  -H "Content-Type: application/json" \
  -d '{"strategy_name": "round_robin"}'
```

Available strategies: `adaptive_bootstrap`, `min_time`, `probabilistic`, `round_robin`, `random`, `po2`, `severless`.

### WebSocket

| Protocol | Endpoint | Description |
|----------|----------|-------------|
| WS | `/v1/task/get_result` | Stream task results in real-time |

Connect via WebSocket to receive task results as they complete. Supports filtering by task_id.

---

## Predictor API (port 8001)

All endpoints have **no** prefix (mounted at root).

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with storage status |

```bash
curl http://localhost:8001/health
```

### Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Get runtime prediction for a task |

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "sleep_model",
    "platform_info": {
      "software_name": "custom",
      "software_version": "1.0",
      "hardware_name": "cpu"
    },
    "prediction_type": "expect_error",
    "features": {"sleep_time": 2.0}
  }'
```

Response (expect_error):
```json
{
  "model_id": "sleep_model",
  "platform_info": {"software_name": "custom", "software_version": "1.0", "hardware_name": "cpu"},
  "prediction_type": "expect_error",
  "result": {
    "expected_runtime_ms": 2000.0,
    "error_margin_ms": 50.0
  }
}
```

Response (quantile):
```json
{
  "prediction_type": "quantile",
  "result": {
    "quantiles": {"0.5": 1950.0, "0.9": 2100.0, "0.95": 2200.0}
  }
}
```

### Training

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/train` | Train or update a predictor model |

```bash
curl -X POST http://localhost:8001/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "sleep_model",
    "platform_info": {
      "software_name": "custom",
      "software_version": "1.0",
      "hardware_name": "cpu"
    },
    "prediction_type": "expect_error",
    "features_list": [
      {"sleep_time": 1.0, "runtime_ms": 1005},
      {"sleep_time": 2.0, "runtime_ms": 2003}
    ]
  }'
```

Requires at least 10 samples.

### Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/list` | List all trained models with metadata |

```bash
curl http://localhost:8001/list
```

### Cache

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/cache/stats` | Get model cache hit/miss statistics |
| POST | `/cache/clear` | Clear in-memory model cache |

```bash
curl http://localhost:8001/cache/stats
curl -X POST http://localhost:8001/cache/clear
```

### WebSocket

| Protocol | Endpoint | Description |
|----------|----------|-------------|
| WS | `/ws/predict` | Real-time predictions over WebSocket |

Send `PredictionRequest` JSON, receive `PredictionResponse` JSON. Connection stays open for multiple requests.

---

## Planner API (port 8002)

Core endpoints use the `/v1/` prefix. PyLet endpoints are mounted under `/v1/` via router.

### Core

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/health` | Health check |
| GET | `/v1/info` | Service info, PyLet status, registered schedulers |
| POST | `/v1/plan` | Compute optimal deployment plan (dry run) |
| GET | `/v1/timeline` | Get instance deployment timeline |
| POST | `/v1/timeline/clear` | Clear timeline data |

#### Plan (Dry Run)

```bash
curl -X POST http://localhost:8002/v1/plan \
  -H "Content-Type: application/json" \
  -d '{
    "M": 4,
    "N": 2,
    "B": [[10, 5], [10, 5], [8, 6], [8, 6]],
    "initial": [0, 0, 1, 1],
    "a": 0.5,
    "target": [60, 40],
    "algorithm": "simulated_annealing",
    "objective_method": "relative_error"
  }'
```

Response:
```json
{
  "deployment": [0, 0, 0, 1],
  "score": 0.0667,
  "service_capacity": [30, 5],
  "changes_count": 1,
  "stats": {}
}
```

### Scheduler Registry

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/scheduler/register` | Register a scheduler for a model |
| POST | `/v1/scheduler/deregister` | Deregister a scheduler |
| GET | `/v1/scheduler/list` | List all registered schedulers |
| GET | `/v1/scheduler/{model_id}` | Get scheduler for a model |

```bash
# Register
curl -X POST http://localhost:8002/v1/scheduler/register \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "sleep_model",
    "scheduler_url": "http://localhost:8000"
  }'

# List
curl http://localhost:8002/v1/scheduler/list
```

### Instance Management (Scheduler-Compatible)

These dummy endpoints allow PyLet-managed instances to complete their registration/deregistration lifecycle when registered to the Planner instead of a Scheduler.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/instance/register` | Register instance to available store |
| POST | `/v1/instance/drain` | Acknowledge drain (always succeeds) |
| GET | `/v1/instance/drain/status` | Drain status (always `can_remove=true`) |
| POST | `/v1/instance/remove` | Remove instance (always succeeds) |
| POST | `/v1/task/resubmit` | Task resubmit (no-op) |

### PyLet Deployment

These endpoints require `PYLET_ENABLED=true`. All are under the `/v1/` prefix.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/status` | PyLet service status and active instances |
| POST | `/v1/deploy` | Run optimizer, then deploy result via PyLet |
| POST | `/v1/deploy_manually` | Deploy to a manually specified target state |
| POST | `/v1/scale` | Scale a specific model to target count |
| POST | `/v1/migrate` | Migrate an instance to a different model |
| POST | `/v1/optimize` | Optimize and deploy (simplified input) |
| POST | `/v1/terminate-all` | Terminate all PyLet-managed instances |

#### Deploy (Optimizer + PyLet)

```bash
curl -X POST http://localhost:8002/v1/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "M": 2,
    "N": 1,
    "B": [[1.0], [1.0]],
    "target": [1.0],
    "a": 0.5,
    "model_ids": ["sleep_model"],
    "wait_for_ready": true
  }'
```

#### Deploy Manually

```bash
curl -X POST http://localhost:8002/v1/deploy_manually \
  -H "Content-Type: application/json" \
  -d '{
    "target_state": {"sleep_model": 2},
    "wait_for_ready": true
  }'
```

#### Scale Model

```bash
curl -X POST http://localhost:8002/v1/scale \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "sleep_model",
    "target_count": 3,
    "wait_for_ready": true
  }'
```

#### Check Status

```bash
curl http://localhost:8002/v1/status
```

#### Terminate All

```bash
curl -X POST "http://localhost:8002/v1/terminate-all?wait_for_drain=true"
```
