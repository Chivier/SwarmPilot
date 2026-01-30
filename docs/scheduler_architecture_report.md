# Scheduler Architecture Analysis Report

> Generated from source code analysis of `scheduler/src/` on 2026-01-29.
> Task: [PYLET-027]

---

## 1. Project Overview

SwarmPilot is a **microservices monorepo** (managed via `uv` workspaces) with four core services:

| Service | Default Port | Purpose |
|---------|-------------|---------|
| **Scheduler** | 8000 | Task orchestration, instance management, intelligent routing (includes in-process predictor) |
| **Planner** | 8002 | Deployment optimization via linear programming (ILP) |
| **Instance** | 8300+ | Task execution nodes wrapping model containers |

> **Note:** The Predictor is embedded in the Scheduler as an in-process library (`PredictorClient`). There is no separate Predictor HTTP service.

The Scheduler uses **in-process library imports** for prediction (no HTTP to a separate Predictor service). Scheduler ↔ Planner and Scheduler ↔ Instance communication is **HTTP/REST**. Real-time task result delivery uses **WebSocket**.

---

## 2. Scheduler Internal Architecture

### 2.1 Directory Layout

```
scheduler/src/
├── api.py                         # FastAPI app — main entry point (~3100 lines)
├── cli.py                         # CLI via Typer (`sscheduler start`)
├── config.py                      # 8 dataclass configs loaded from env vars (230 lines)
├── model.py                       # Re-exports from models/ (backward compat)
├── instance_sync.py               # Declarative instance sync logic (323 lines)
│
├── algorithms/                    # Scheduling strategies (pluggable)
│   ├── base.py                    # Abstract SchedulingStrategy (343 lines)
│   ├── factory.py                 # get_strategy() factory (72 lines)
│   ├── min_expected_time.py       # Minimize (queue_wait + predicted_runtime)
│   ├── min_expected_time_dt.py    # Decision tree variant
│   ├── min_expected_time_lr.py    # Linear regression variant
│   ├── probabilistic.py           # Monte Carlo sampling on quantiles
│   ├── power_of_two.py            # Random pick 2, select faster
│   ├── round_robin.py             # Simple rotation
│   ├── random.py                  # Random baseline
│   └── serverless.py              # Serverless-optimized variant
│
├── clients/                       # In-process predictor and training clients
│   ├── __init__.py                # Client exports
│   ├── models.py                  # Shared data models (Prediction, TrainingSample)
│   ├── predictor_library_client.py  # PredictorClient (in-process prediction)
│   ├── _predictor_lib.py          # Predictor library internals
│   └── training_library_client.py # TrainingClient (in-process training)
│
├── models/                        # Pydantic data models
│   ├── __init__.py                # Re-exports
│   ├── core.py                    # Instance, InstanceStats, InstanceQueueBase
│   ├── queue.py                   # InstanceQueueExpectError, InstanceQueueProbabilistic
│   ├── requests.py                # API request schemas
│   ├── responses.py               # API response schemas
│   ├── status.py                  # TaskStatus, InstanceStatus, StrategyType enums
│   └── websocket.py               # WebSocket message models
│
├── registry/                      # Thread-safe state management
│   ├── __init__.py
│   ├── instance_registry.py       # Instance lifecycle + queue info (525 lines)
│   └── task_registry.py           # Task lifecycle tracking (394 lines)
│
├── services/                      # Core business logic
│   ├── worker_queue_manager.py    # Coordinates all per-worker threads (318 lines)
│   ├── worker_queue_thread.py     # Per-worker FIFO queue + HTTP dispatch
│   ├── task_result_callback.py    # Bridges worker threads → asyncio loop (324 lines)
│   ├── websocket_manager.py       # WebSocket subscription management (214 lines)
│   ├── planner_registrar.py       # Register/deregister with planner (164 lines)
│   └── shutdown_handler.py        # Graceful shutdown orchestration (170 lines)
│
├── proxy/                         # Transparent proxy router
│   ├── __init__.py
│   └── router.py                  # Catch-all proxy to workers
│
└── utils/
    ├── logger.py                  # Loguru setup
    ├── http_error_logger.py       # Structured HTTP error logging
    ├── planner_reporter.py        # Periodic task-count reporting to planner (240 lines)
    └── throughput_tracker.py      # Sliding-window throughput metrics
```

**Total estimated source:** ~9,000 lines across ~35 files.

### 2.2 Core Internal Components

#### Registries (State Layer)

- **`InstanceRegistry`** (`registry/instance_registry.py`, 525 lines) — async, thread-safe registry (using `asyncio.Lock`) tracking all worker instances. Manages status transitions: `INITIALIZING` → `ACTIVE` → `DRAINING` → `REMOVING` (and `REDEPLOYING`). Stores per-instance queue info (`InstanceQueueExpectError` or `InstanceQueueProbabilistic` depending on `queue_info_type`), and `InstanceStats` (pending/completed/failed counts). Provides optimized batch `get_all_queue_info()` with single-lock acquisition.

- **`TaskRegistry`** (`registry/task_registry.py`, 394 lines) — tracks the full task lifecycle via `TaskRecord` objects. Fields: `task_id`, `model_id`, `task_input`, `metadata`, `assigned_instance`, `status`, `result`, `error`, predicted values (`predicted_time_ms`, `predicted_error_margin_ms`, `predicted_quantiles`), and timestamps (`submitted_at`, `started_at`, `completed_at`). Supports filtering by `status`, `model_id`, `instance_id` and pagination. `execution_time_ms` is a computed property that prefers actual recorded time over calculated difference.

#### Worker Queue System (Execution Layer)

- **`WorkerQueueManager`** (`services/worker_queue_manager.py`, 318 lines) — central coordinator using `threading.Lock` (not asyncio). Creates/destroys `WorkerQueueThread` instances per worker. Key methods: `register_worker()`, `deregister_worker()` (returns pending tasks for rescheduling), `enqueue_task()`, `get_queue_depth()`, `get_all_queue_depths()`, `get_estimated_wait_times()`, `shutdown()` (stops all workers and returns dropped tasks).

- **`WorkerQueueThread`** (`services/worker_queue_thread.py`) — runs in a **dedicated OS thread** (not blocking the asyncio loop). Maintains a FIFO `Queue` of `QueuedTask` objects. Dispatches tasks via synchronous HTTP POST (using `httpx`) to the worker's endpoint. Reports results via a thread-safe callback closure. Supports configurable `max_retries` and `retry_delay` for transient connection errors.

- **`TaskResultCallback`** (`services/task_result_callback.py`, 324 lines) — bridges worker threads back into the asyncio event loop using `asyncio.run_coroutine_threadsafe()`. Maintains a `_futures` dict mapping `task_id → asyncio.Future` for proxy mode. The `handle_result()` coroutine: updates `TaskRegistry` status, updates `InstanceRegistry` stats, records throughput, sends training samples (if auto-training enabled), broadcasts results to WebSocket subscribers, and resolves proxy futures.

#### Scheduling Strategies (Decision Layer)

- **Abstract base:** `SchedulingStrategy` (`algorithms/base.py`, 343 lines) — defines a template method `schedule_task()` that orchestrates:
  1. `get_predictions()` — fetch predictions from `PredictorClient` (groups instances by `platform_info` for batching)
  2. `collect_queue_info()` — single-lock batch retrieval from `InstanceRegistry`
  3. `select_instance()` — abstract method implemented by each strategy
  4. `update_queue()` — abstract method to update queue state post-scheduling
  5. Returns `ScheduleResult(selected_instance_id, selected_prediction)`

- **6 concrete strategies** selected via `factory.py::get_strategy()`:

| Strategy Name | Class | Approach |
|--------------|-------|----------|
| `min_time` | `MinimumExpectedTimeStrategy` | Minimize (queue_wait + predicted_runtime) |
| `probabilistic` | `ProbabilisticSchedulingStrategy` | Monte Carlo sampling on quantiles (default) |
| `round_robin` | `RoundRobinStrategy` | Simple rotation |
| `random` | `RandomStrategy` | Random baseline |
| `po2` | `PowerOfTwoStrategy` | Random pick 2, select faster |
| `serverless` | `MinimumExpectedTimeServerlessStrategy` | Serverless-optimized variant |

Strategy is configurable at runtime via `POST /strategy/set`.

#### Proxy Router

- **`ProxyRouter`** (`proxy/router.py`) — mounted as a catch-all route (`/{path:path}`) after all explicit routes. Provides transparent reverse proxy so clients can use the scheduler as a single entrypoint to any model instance.
- Flow: intercept request → create task in `TaskRegistry` → register `asyncio.Future` → run scheduling strategy → enqueue to `WorkerQueueManager` → await `Future` with timeout → return worker's response transparently.
- Configurable via `PROXY_ENABLED` and `PROXY_TIMEOUT` (default 300s).

### 2.3 Internal Data Flow (Task Lifecycle)

```
1. Client → POST /task/submit → api.py
2. api.py → TaskRegistry.create_task() → TaskRecord(status=PENDING)
3. api.py → SchedulingStrategy.schedule_task()
   └── Strategy → PredictorClient.predict() → predictions for all active instances
   └── Strategy → InstanceRegistry.get_all_queue_info() → queue state
   └── Strategy → select_instance() → best instance ID
4. api.py → WorkerQueueManager.enqueue_task(instance_id, QueuedTask)
   └── WorkerQueueThread.put(QueuedTask) → internal FIFO queue
5. WorkerQueueThread → HTTP POST to instance endpoint
   └── TaskRegistry.update_status(RUNNING)
6. Instance → POST /callback/task_result → api.py
   └── TaskResultCallback.handle_result()
       ├── TaskRegistry.update_status(COMPLETED/FAILED)
       ├── InstanceRegistry increment_completed()/increment_failed()
       ├── ThroughputTracker.record()
       ├── TrainingClient.add_sample() (if auto-training enabled)
       └── WebSocketManager.broadcast_task_result(task_id, result)
7. Client ← WebSocket /task/get_result or GET /task/info
```

---

## 3. Dependency Relations

### 3.1 Dependency Diagram

```
                    ┌─────────────────────────────┐
                    │         PLANNER              │
                    │        (port 8002)           │
                    │                              │
                    │  /scheduler/register         │
                    │  /scheduler/deregister       │
                    │  /submit_target              │
                    │  /submit_throughput           │
                    └──────┬───────────┬───────────┘
                           │           │
              registration │           │ instance management
              + reporting  │           │ (planner → scheduler)
                           ▼           ▼
                    ┌─────────────────────────────┐    ┌──────────────┐
                    │        SCHEDULER             │───►│  INSTANCES   │
                    │       (port 8000)            │    │ (port 8300+) │
                    │                              │    │              │
                    │  Algorithms ←→ Predictor     │    │ HTTP dispatch│
                    │    (in-process library)       │    │ Task callback│
                    │  Registries ←→ Planner       │    │              │
                    │  WorkerQueues ←→ Instances   │    │              │
                    └─────────────────────────────┘    └──────────────┘
```

### 3.2 Predictor Integration (In-Process Library)

| Aspect | Detail |
|--------|--------|
| **Purpose** | Get runtime predictions to make intelligent scheduling decisions |
| **Mode** | In-process library import via `PredictorClient` (`predictor_library_client.py`) |
| **Transport** | Direct Python method calls — no network overhead |
| **Config** | `PREDICTOR_STORAGE_DIR` (model files), `PREDICTOR_CACHE_MAX_SIZE` (in-memory model cache) |
| **Failure mode** | `ValueError` for model-not-found or invalid features; standard Python exceptions propagate directly |
| **Data sent** | `model_id`, `platform_info`, `prediction_type`, `features`, `quantiles` (optional) |
| **Data received** | `Prediction(instance_id, predicted_time_ms, confidence, quantiles, error_margin_ms)` |
| **Batching** | Groups instances by `platform_info` — one prediction call per unique platform |

**Training feedback loop** (in-process):

| Aspect | Detail |
|--------|--------|
| **Purpose** | Send actual execution times back for model retraining |
| **Mechanism** | Direct in-process training via `TrainingClient` (`training_library_client.py`) |
| **Client** | `TrainingClient` with internal sample buffer, shares `ModelStorage`/`ModelCache` with `PredictorClient` |
| **Trigger** | Buffer reaches `TRAINING_BATCH_SIZE` (default 100) |
| **Data sent** | `TrainingSample(model_id, platform_info, features, actual_runtime_ms, timestamp)` |
| **Config** | `TRAINING_ENABLE_AUTO` (false), `TRAINING_BATCH_SIZE` (100), `TRAINING_FREQUENCY` (3600s), `TRAINING_MIN_SAMPLES` (10), `TRAINING_PREDICTION_TYPES` ("expect_error,quantile") |

### 3.3 Scheduler → Planner (Optional Dependency)

Two distinct interaction patterns:

**A. Registration (startup/shutdown lifecycle):**

| Aspect | Detail |
|--------|--------|
| **Purpose** | Register this scheduler as handler for a specific `model_id` |
| **Component** | `PlannerRegistrar` service (`services/planner_registrar.py`) |
| **On startup** | `POST {planner_url}/scheduler/register` with `{model_id, scheduler_url}` |
| **On shutdown** | `POST {planner_url}/scheduler/deregister` with `{model_id}` |
| **Failure behavior** | **Fail-hard**: raises `RuntimeError` if registration fails after retries |
| **Config** | `PLANNER_REGISTRATION_URL`, `SCHEDULER_MODEL_ID`, `SCHEDULER_SELF_URL` |
| **Enabled when** | All three config values are non-empty (`PlannerRegistrationConfig.enabled` property) |
| **Retry logic** | Up to `PLANNER_REGISTRATION_MAX_RETRIES` (3) with exponential backoff starting at `PLANNER_REGISTRATION_RETRY_DELAY` (5.0s) |

**B. Periodic reporting (background task):**

| Aspect | Detail |
|--------|--------|
| **Purpose** | Report uncompleted task count so planner can make scaling decisions |
| **Component** | `PlannerReporter` utility (`utils/planner_reporter.py`) |
| **Target endpoint** | `POST {planner_url}/submit_target` with `{model_id, value}` |
| **Value reported** | `float(pending_tasks + running_tasks)` (total uncompleted) |
| **Throughput** | Also reports `POST {planner_url}/submit_throughput` with `{instance_url, avg_execution_time}` per instance with recent data |
| **Interval** | `SCHEDULER_AUTO_REPORT` seconds (0 disables) |
| **Failure behavior** | Log warning, continue running — does not crash |
| **Model ID** | Set lazily when first instance registers (`set_model_id()`) |

### 3.4 Planner → Scheduler (Reverse Dependency)

| Aspect | Detail |
|--------|--------|
| **Purpose** | Planner manages instance fleet by telling scheduler which instances to use |
| **Mechanism** | Declarative instance sync: planner sends complete target instance list |
| **Handler** | `instance_sync.py::handle_instance_sync()` |
| **Behavior** | Computes diff (current vs target), removes stale instances first, then adds new ones |
| **Task safety** | Pending tasks from removed instances are automatically rescheduled via the scheduling algorithm |
| **Note** | The sync logic is implemented but the `/instance/sync` endpoint is **not yet wired** in `api.py`. Planner can alternatively use the individual endpoints: `POST /instance/register`, `POST /instance/remove`, `POST /instance/drain` |

**Instance management endpoints used by planner:**

```
POST /instance/register    → Register a new worker instance
POST /instance/remove      → Remove a worker instance
POST /instance/drain       → Mark instance as draining (stop accepting new tasks)
GET  /instance/drain/status → Check drain progress
POST /instance/redeploy/start    → Start instance redeployment
POST /instance/redeploy/complete → Complete redeployment, return to ACTIVE
```

### 3.5 Scheduler → Instances (Required Dependency)

| Aspect | Detail |
|--------|--------|
| **Purpose** | Dispatch tasks to worker instances for execution |
| **Mechanism** | `WorkerQueueThread` sends synchronous HTTP POST to instance endpoint |
| **Direction** | Scheduler pushes tasks; instances send results back |
| **Task dispatch** | HTTP POST with task payload to instance's `endpoint` URL |
| **Result delivery** | Instance calls `POST /callback/task_result` on scheduler |
| **Callback payload** | `TaskResultCallbackRequest` containing task result data |
| **Retry** | `max_retries` (3) with `retry_delay` (1.0s) for transient connection errors |
| **HTTP timeout** | `WORKER_HTTP_TIMEOUT` (300.0s) |

### 3.6 Instances → Scheduler (Reverse Dependency)

| Aspect | Detail |
|--------|--------|
| **Purpose** | Register as available worker; report task completion |
| **Registration** | `POST /instance/register` with `InstanceRegisterRequest` (instance_id, model_id, endpoint, platform_info) |
| **Task results** | `POST /callback/task_result` with completion data |
| **Deregistration** | `POST /instance/remove` (graceful) or detected via health checks |

---

## 4. Inter-Component Communication Protocols

### 4.1 Scheduler API Endpoints (Complete Listing)

All endpoints defined in `api.py` with their response models:

**Instance Management** (used by Planner and Instances):

| Method | Path | Response Model | Purpose |
|--------|------|---------------|---------|
| `POST` | `/instance/register` | `InstanceRegisterResponse` | Register a worker instance |
| `POST` | `/instance/remove` | `InstanceRemoveResponse` | Deregister an instance |
| `POST` | `/instance/drain` | `InstanceDrainResponse` | Stop accepting new tasks on instance |
| `GET` | `/instance/drain/status` | `InstanceDrainStatusResponse` | Check drain progress |
| `POST` | `/instance/redeploy/start` | `InstanceRedeployResponse` | Begin instance redeployment |
| `POST` | `/instance/redeploy/complete` | `InstanceRegisterResponse` | Complete redeployment |
| `GET` | `/instance/list` | `InstanceListResponse` | List all registered instances |
| `GET` | `/instance/info` | `InstanceInfoResponse` | Get instance details |

**Task Management** (used by Clients and Proxy):

| Method | Path | Response Model | Purpose |
|--------|------|---------------|---------|
| `POST` | `/task/submit` | `TaskSubmitResponse` | Submit task for scheduling |
| `POST` | `/task/resubmit` | `TaskResubmitResponse` | Resubmit a completed/failed task |
| `GET` | `/task/list` | `TaskListResponse` | List tasks with status filters |
| `GET` | `/task/info` | `TaskDetailResponse` | Get task details + result |
| `POST` | `/task/clear` | `TaskClearResponse` | Clear task records |
| `POST` | `/task/update_metadata` | `TaskUpdateMetadataResponse` | Update task metadata |
| `POST` | `/task/repredict` | `TaskRepredictResponse` | Re-run predictions for all pending tasks |
| `GET` | `/task/schedule_info` | `TaskScheduleInfoResponse` | Get scheduling decision info |

**Callbacks** (used by Instances):

| Method | Path | Response Model | Purpose |
|--------|------|---------------|---------|
| `POST` | `/callback/task_result` | `TaskResultCallbackResponse` | Report task completion/failure |

**Strategy** (runtime configuration):

| Method | Path | Response Model | Purpose |
|--------|------|---------------|---------|
| `GET` | `/strategy/get` | `StrategyGetResponse` | Get current scheduling strategy |
| `POST` | `/strategy/set` | `StrategySetResponse` | Switch strategy at runtime |

**Health & WebSocket**:

| Method | Path | Response Model | Purpose |
|--------|------|---------------|---------|
| `GET` | `/health` | `HealthResponse` | Health check with stats |
| `WS` | `/task/get_result` | — | Real-time result streaming |

**Proxy** (when `PROXY_ENABLED=true`):

| Method | Path | Purpose |
|--------|------|---------|
| `*` | `/{path:path}` | Catch-all transparent proxy to instances |

### 4.2 Status Enums

**`InstanceStatus`** (5 states):
```
INITIALIZING → ACTIVE → DRAINING → REMOVING
                  ↓
              REDEPLOYING → ACTIVE (via redeploy/complete)
```

**`TaskStatus`** (4 states):
```
PENDING → RUNNING → COMPLETED
                  → FAILED
```

**`StrategyType`**: `MIN_TIME`, `PROBABILISTIC`, `ROUND_ROBIN`, `RANDOM`, `POWEROFTWO`, `SERVERLESS`

**`WSMessageType`**: `SUBSCRIBE`, `UNSUBSCRIBE`, `RESULT`, `ERROR`, `ACK`, `PING`, `PONG`

### 4.3 Shared Data Structures

These structures are used across component boundaries:

```python
# platform_info — shared between Scheduler, Predictor, and Instance
platform_info = {
    "software_name": str,    # e.g. "vllm", "sglang"
    "software_version": str, # e.g. "0.6.0"
    "hardware_name": str     # e.g. "gpu", "cpu"
}

# Instance registration payload (Instance → Scheduler)
# Model: InstanceRegisterRequest
{
    "instance_id": str,
    "model_id": str,
    "endpoint": str,          # e.g. "http://worker-1:8300"
    "platform_info": dict
}

# Task submission (Client → Scheduler)
# Model: TaskSubmitRequest
{
    "task_id": str,
    "model_id": str,
    "task_input": dict,
    "metadata": dict           # Used as prediction features
}

# Task result callback (Instance → Scheduler)
# Model: TaskResultCallbackRequest
{
    "task_id": str,
    "instance_id": str,
    "status": "completed" | "failed",
    "result": dict | None,
    "error": str | None,
    "execution_time_ms": float
}

# Instance sync (Planner → Scheduler)
# Model: InstanceSyncRequest (in instance_sync.py)
{
    "instances": [
        {
            "instance_id": str,
            "endpoint": str,
            "model_id": str
        }
    ]
}
```

---

## 5. Configuration Environment Variables

All configuration is loaded via 8 dataclass configs in `config.py` (230 lines):

### Server Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `SCHEDULER_HOST` | `0.0.0.0` | Server bind host |
| `SCHEDULER_PORT` | `8000` | Server bind port |
| `SCHEDULER_ENABLE_CORS` | `true` | Enable CORS headers |

### Predictor Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `PREDICTOR_STORAGE_DIR` | `models` | Model storage directory |
| `PREDICTOR_CACHE_MAX_SIZE` | `100` | Maximum number of models cached in memory |

### Scheduling Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `SCHEDULING_STRATEGY` | `probabilistic` | Default algorithm (see strategy table) |
| `SCHEDULING_PROBABILISTIC_QUANTILE` | `0.9` | Target quantile for probabilistic strategy |

### Training Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `TRAINING_ENABLE_AUTO` | `false` | Enable automatic retraining feedback |
| `TRAINING_BATCH_SIZE` | `100` | Samples before triggering training |
| `TRAINING_FREQUENCY` | `3600` | Minimum seconds between training runs |
| `TRAINING_MIN_SAMPLES` | `10` | Minimum samples required |
| `TRAINING_PREDICTION_TYPES` | `expect_error,quantile` | Types to train |

### Planner Registration Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `PLANNER_REGISTRATION_URL` | _(empty)_ | Planner URL for registration (empty disables) |
| `SCHEDULER_MODEL_ID` | _(empty)_ | Model this scheduler handles |
| `SCHEDULER_SELF_URL` | _(empty)_ | Advertised scheduler URL |
| `PLANNER_REGISTRATION_TIMEOUT` | `10.0` | Registration HTTP timeout |
| `PLANNER_REGISTRATION_MAX_RETRIES` | `3` | Max registration retry attempts |
| `PLANNER_REGISTRATION_RETRY_DELAY` | `5.0` | Registration retry delay |

### Planner Report Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `PLANNER_URL` | _(empty)_ | Planner URL for periodic reporting (empty disables) |
| `SCHEDULER_AUTO_REPORT` | `0` | Report interval in seconds (0 disables) |
| `PLANNER_REPORT_TIMEOUT` | `5.0` | Report HTTP timeout |

### Proxy Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `PROXY_ENABLED` | `true` | Enable transparent proxy router |
| `PROXY_TIMEOUT` | `300.0` | Proxy request timeout (seconds) |
| `WORKER_HTTP_TIMEOUT` | `300.0` | Worker dispatch HTTP timeout |

### Logging Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `SCHEDULER_LOGURU_LEVEL` | `INFO` | Log level |
| `SCHEDULER_LOG_DIR` | `logs` | Log output directory |
| `SCHEDULER_ENABLE_JSON_LOGS` | `false` | Enable JSON-formatted logs |

---

## 6. Failure Modes and Resilience

| Dependency | Required? | On Failure | Recovery |
|------------|-----------|------------|----------|
| **Predictor** (prediction-based strategies) | Yes (in-process) | `ValueError` for missing model or invalid features; exceptions propagate directly | No network retries needed — errors are immediate in-process |
| **Planner registration** | Optional | **Fail-hard** — raises `RuntimeError`, scheduler exits | Must restart scheduler |
| **Planner reporting** | Optional | Logs warning, continues running | Planner uses stale data until next report |
| **Instances** (dispatch) | Yes | Tasks queue up in `WorkerQueueThread` with retry (`max_retries=3`) | Register new instances; tasks remain queued |
| **Instance callback** | Yes | Task stays in RUNNING state indefinitely | Manual resubmit via `POST /task/resubmit` |
| **WebSocket** | No | Dead connections detected and removed on next broadcast | Client reconnects |

---

## 7. Multi-Scheduler Architecture

Enabled by PYLET-024, this allows **per-model scheduler instances**:

```
                     ┌─────────┐
                     │ Planner │
                     │ Registry│
                     └────┬────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
   │ Scheduler A │ │ Scheduler B │ │ Scheduler C │
   │ model=gpt-4 │ │ model=llama │ │ model=ocr   │
   └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
          │               │               │
     ┌────┴────┐     ┌────┴────┐     ┌────┴────┐
     │Instance │     │Instance │     │Instance │
     │ pool A  │     │ pool B  │     │ pool C  │
     └─────────┘     └─────────┘     └─────────┘
```

- Each scheduler sets `SCHEDULER_MODEL_ID` to its assigned model
- On startup, `PlannerRegistrar` calls `POST {planner}/scheduler/register` with `{model_id, scheduler_url}`
- Planner maintains `SchedulerRegistry` mapping `model_id → scheduler_url`
- Planner routes instance management operations to the correct scheduler
- On shutdown, scheduler deregisters via `POST {planner}/scheduler/deregister`
- Enables independent scaling per model type

---

## 8. Threading Model

The scheduler uses a **hybrid async/threaded** architecture:

| Component | Concurrency Model | Reason |
|-----------|-------------------|--------|
| FastAPI endpoints | `asyncio` (async/await) | Non-blocking HTTP handling |
| `InstanceRegistry` | `asyncio.Lock` | Shared by async endpoints |
| `TaskRegistry` | `asyncio.Lock` | Shared by async endpoints |
| `WorkerQueueManager` | `threading.Lock` | Coordinates OS threads |
| `WorkerQueueThread` | `threading.Thread` + `queue.Queue` | Blocking HTTP dispatch without blocking event loop |
| `TaskResultCallback` | Bridge: `run_coroutine_threadsafe()` | Thread → event loop communication |
| `PlannerReporter` | `asyncio.Task` (background) | Non-blocking periodic reports |
| `WebSocketManager` | `asyncio` (async/await) | Non-blocking WebSocket I/O |
| `PredictorClient` | Sync calls from `async` methods | In-process library — no network I/O |

The key design decision is that **task dispatch to workers happens in OS threads** (`WorkerQueueThread`) because HTTP calls to instances can block for up to 300 seconds (the worker timeout). Using OS threads prevents these long-running calls from blocking the asyncio event loop that serves the FastAPI endpoints. Prediction calls are in-process (library mode) and do not involve network I/O.

---

## 9. Notes

### Instance Sync Gap

The `instance_sync.py` module (323 lines) contains complete logic for declarative instance synchronization (`handle_instance_sync()`), including diff computation, addition/removal, and task rescheduling. However, this function is **not exposed as a FastAPI endpoint** in `api.py`. The planner can achieve the same result by using the individual instance management endpoints (`/instance/register`, `/instance/remove`).

### Proxy Router Mounting

The proxy router (`proxy/router.py`) is mounted as a catch-all `/{path:path}` route. Because FastAPI evaluates routes in definition order, it must be defined **after** all explicit endpoints to avoid intercepting API calls.

### Prediction Batching

The base `SchedulingStrategy` groups instances by `platform_info` before calling the predictor. This means one predictor API call per unique platform configuration rather than one per instance, reducing prediction latency when many instances share the same hardware/software stack.
