# README FOR LLM: Scheduler Service

**Version:** 0.1.0
**Python Requirement:** >=3.11
**Primary Entry Point:** `src/cli.py` (sscheduler CLI)
**Main API Definition:** `src/api.py` (3140 lines)

---

## SYSTEM OVERVIEW

### Purpose
An intelligent task scheduling service that distributes computational tasks across multiple compute instances using ML-based runtime predictions and configurable scheduling strategies.

### Core Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        Scheduler Service                         │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  FastAPI     │  │  Scheduling      │  │  WebSocket       │  │
│  │  Endpoints   │  │  Algorithms (8)  │  │  Manager         │  │
│  └──────┬───────┘  └──────┬───────────┘  └──────────────────┘  │
│         │                 │                                      │
│  ┌──────▼─────────────────▼───────┐  ┌──────────────────────┐  │
│  │     Registry Layer             │  │  Services Layer      │  │
│  │  (TaskRegistry, InstanceReg)   │  │  (Background, Queue) │  │
│  └────────────────────────────────┘  └──────────────────────┘  │
│         │                                      │                │
│  ┌──────▼──────────────┐            ┌─────────▼─────────────┐  │
│  │  Clients Layer      │            │  Worker Queue Manager │  │
│  │  (Predictor, Train) │            │  (Per-instance queues)│  │
│  └─────────────────────┘            └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │                                      │
         ▼                                      ▼
┌──────────────────┐                  ┌──────────────────┐
│ Compute Instance │◄─── Callback ────┤ Predictor Service│
│  (HTTP endpoint) │                  │  (ML predictions)│
└──────────────────┘                  └──────────────────┘
```

### Directory Structure

```
scheduler/
├── src/
│   ├── api.py                          # FastAPI app, all endpoints (3140 lines)
│   ├── cli.py                          # Typer CLI (sscheduler command)
│   ├── config.py                       # Environment-based configuration
│   ├── model.py                        # Main Pydantic data models
│   ├── instance_sync.py                # Instance synchronization
│   │
│   ├── algorithms/                     # Scheduling strategies (8 algorithms)
│   │   ├── __init__.py                 # Exports all strategies
│   │   ├── base.py                     # Abstract SchedulingStrategy class
│   │   ├── factory.py                  # get_strategy() factory function
│   │   ├── min_expected_time.py        # MinimumExpectedTimeStrategy
│   │   ├── min_expected_time_dt.py     # Decision tree-based prediction
│   │   ├── min_expected_time_lr.py     # Linear regression-based prediction
│   │   ├── power_of_two.py             # Power of Two Choices strategy
│   │   ├── probabilistic.py            # Monte Carlo probabilistic strategy
│   │   ├── random.py                   # Random selection strategy
│   │   ├── round_robin.py              # Round robin strategy
│   │   ├── serverless.py               # Serverless-optimized strategy
│   │   └── queue_state_adapter.py      # Queue state query helpers
│   │
│   ├── registry/                       # Thread-safe state management
│   │   ├── __init__.py
│   │   ├── task_registry.py            # Task state management
│   │   └── instance_registry.py        # Instance & queue management
│   │
│   ├── services/                       # Background services
│   │   ├── __init__.py
│   │   ├── background_scheduler.py     # Non-blocking task scheduling
│   │   ├── central_queue.py            # FIFO task queue
│   │   ├── worker_queue_manager.py     # Per-worker queue coordination
│   │   ├── worker_queue_thread.py      # Individual worker queue thread
│   │   ├── task_dispatcher.py          # Async HTTP dispatch
│   │   ├── task_result_callback.py     # Callback handling
│   │   ├── websocket_manager.py        # WebSocket connection manager
│   │   └── shutdown_handler.py         # Graceful shutdown
│   │
│   ├── clients/                        # External service clients
│   │   ├── __init__.py
│   │   ├── predictor_client.py         # WebSocket client for predictor
│   │   └── training_client.py          # HTTP client for training service
│   │
│   ├── utils/                          # Utilities
│   │   ├── __init__.py
│   │   ├── logger.py                   # Loguru configuration
│   │   ├── http_error_logger.py        # HTTP error logging
│   │   ├── throughput_tracker.py       # Throughput metrics
│   │   └── planner_reporter.py         # Planner integration
│   │
│   └── models/                         # Extended Pydantic models
│       ├── __init__.py
│       ├── core.py                     # Core model types
│       ├── queue.py                    # Queue-related models
│       ├── requests.py                 # Request schemas
│       ├── responses.py                # Response schemas
│       ├── status.py                   # Status enums
│       └── websocket.py                # WebSocket message schemas
│
├── tests/                              # Test suite
├── docs/                               # Documentation
├── pyproject.toml                      # Project metadata
└── uv.lock                             # Dependency lock file
```

### Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| **API Layer** | `src/api.py` | FastAPI endpoints for all operations |
| **Task Registry** | `src/registry/task_registry.py` | Thread-safe task state management |
| **Instance Registry** | `src/registry/instance_registry.py` | Thread-safe instance & queue management |
| **Scheduling Algorithms** | `src/algorithms/` | 8 different scheduling algorithms |
| **Background Scheduler** | `src/services/background_scheduler.py` | Non-blocking task scheduling |
| **Central Queue** | `src/services/central_queue.py` | FIFO task queue management |
| **Worker Queue Manager** | `src/services/worker_queue_manager.py` | Per-instance queue coordination |
| **Task Dispatcher** | `src/services/task_dispatcher.py` | Async task execution on instances |
| **Predictor Client** | `src/clients/predictor_client.py` | WebSocket client for ML predictions |
| **Training Client** | `src/clients/training_client.py` | HTTP client for training data collection |
| **WebSocket Manager** | `src/services/websocket_manager.py` | Real-time result notifications |
| **Configuration** | `src/config.py` | Environment-based configuration |

---

## QUICK START

### Installation

```bash
cd /path/to/scheduler
uv sync
uv pip install -e .
```

### Basic Startup

```bash
# Default configuration (0.0.0.0:8000)
uv run sscheduler start

# Custom host and port
uv run sscheduler start --host 127.0.0.1 --port 9000

# With configuration file
uv run sscheduler start --config config.toml

# Check version
uv run sscheduler version
```

---

## ENVIRONMENT VARIABLES

All environment variables are defined in `src/config.py` and loaded via Pydantic settings.

### Predictor Service Configuration
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PREDICTOR_URL` | str | `"http://localhost:8001"` | Base URL of the predictor service |
| `PREDICTOR_TIMEOUT` | float | `5.0` | HTTP request timeout in seconds |
| `PREDICTOR_MAX_RETRIES` | int | `3` | Maximum retry attempts for failed requests |
| `PREDICTOR_RETRY_DELAY` | float | `1.0` | Initial retry delay in seconds |
| `PREDICTOR_CACHE_TTL` | int | `300` | Prediction cache TTL in seconds |
| `PREDICTOR_ENABLE_CACHE` | bool | `true` | Enable prediction caching |

### Scheduling Configuration
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SCHEDULING_STRATEGY` | str | `"probabilistic"` | Default strategy name |
| `SCHEDULING_PROBABILISTIC_QUANTILE` | float | `0.9` | Target quantile for probabilistic strategy |

Available strategies: `min_time`, `probabilistic`, `round_robin`, `random`, `power_of_two`, `serverless`

### Training Configuration
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TRAINING_ENABLE_AUTO` | bool | `false` | Enable automatic training data collection |
| `TRAINING_BATCH_SIZE` | int | `100` | Batch size before auto-flush |
| `TRAINING_FREQUENCY` | int | `3600` | Training frequency in seconds |
| `TRAINING_MIN_SAMPLES` | int | `10` | Minimum samples required |

### Server Configuration
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SCHEDULER_HOST` | str | `"0.0.0.0"` | Server bind host |
| `SCHEDULER_PORT` | int | `8000` | Server bind port |
| `SCHEDULER_LOG_LEVEL` | str | `"INFO"` | Log level |
| `SCHEDULER_LOG_DIR` | str | `"logs"` | Log directory |

---

## API ENDPOINTS

All endpoints are defined in `src/api.py` as a FastAPI application.

### Instance Management

#### POST /instance/register
Register a compute instance to make it available for task scheduling.

**Request:**
```json
{
  "instance_id": "worker-001",
  "model_id": "llama-7b",
  "endpoint": "http://worker-001:8080",
  "platform_info": {
    "gpu_type": "A100",
    "gpu_count": 1,
    "memory_gb": 80
  }
}
```

#### POST /instance/remove
Deregister an instance from the scheduler.

#### GET /instance/list
List all registered instances, optionally filtered by model_id.

#### GET /instance/info
Get detailed information about a specific instance including queue state.

#### POST /instance/drain
Signal an instance to start draining (preparing for removal).

#### GET /instance/drain/status
Check if an instance can be safely removed.

#### POST /instance/redeploy
Redeploy an instance with a different model.

---

### Task Management

#### POST /task/submit
Submit a new task for execution. The scheduler selects the optimal instance using the configured scheduling strategy.

**Request:**
```json
{
  "task_id": "task-001",
  "model_id": "llama-7b",
  "task_input": {
    "prompt": "Explain quantum computing",
    "max_tokens": 100
  },
  "metadata": {"user_id": "user-123"}
}
```

#### GET /task/list
List tasks with filtering by status, model_id, instance_id with pagination.

#### GET /task/info
Get detailed information about a specific task.

#### POST /task/clear
Clear all tasks from the registry.

#### POST /task/resubmit
Resubmit a failed or cancelled task.

#### POST /task/update_metadata
Update task metadata.

---

### Callback Endpoints

#### POST /callback/task_result
Callback endpoint for compute instances to report task completion.

**Request:**
```json
{
  "task_id": "task-001",
  "status": "completed",
  "result": {"generated_text": "..."},
  "execution_time_ms": 234.56
}
```

---

### WebSocket Endpoints

#### WS /task/get_result
Real-time task result notifications via WebSocket.

**Subscribe:**
```json
{"type": "subscribe", "task_ids": ["task-001", "task-002"]}
```

**Result notification:**
```json
{
  "type": "result",
  "task_id": "task-001",
  "status": "completed",
  "result": {"generated_text": "..."},
  "execution_time_ms": 234.56
}
```

---

### Strategy Management

#### GET /strategy/get
Get information about the currently active scheduling strategy.

#### POST /strategy/set
Switch the scheduling strategy.

**Request:**
```json
{
  "strategy_name": "probabilistic",
  "target_quantile": 0.95
}
```

---

### Health Check

#### GET /health
Service health check with comprehensive statistics.

---

## SCHEDULING ALGORITHMS

All strategies are in `src/algorithms/` and inherit from `SchedulingStrategy` base class in `src/algorithms/base.py`.

### Available Strategies

| Strategy | File | Description |
|----------|------|-------------|
| **round_robin** | `round_robin.py` | Simple cyclic distribution |
| **random** | `random.py` | Random instance selection |
| **min_time** | `min_expected_time.py` | Greedy shortest queue selection |
| **probabilistic** | `probabilistic.py` | Monte Carlo quantile-based scheduling |
| **power_of_two** | `power_of_two.py` | Power of two choices algorithm |
| **serverless** | `serverless.py` | Optimized for serverless workloads |

### Strategy Comparison

| Feature | Round Robin | Min Time | Probabilistic | Power of Two |
|---------|-------------|----------|---------------|--------------|
| **Predictor Required** | No | Yes | Yes | Yes |
| **Optimization Goal** | Equal distribution | Minimize avg latency | Minimize tail latency | Balance & efficiency |
| **Complexity** | O(1) | O(n) | O(n) | O(1) |
| **Best For** | Testing | Heterogeneous | SLA-based | Large scale |

---

## SERVICES LAYER

### BackgroundScheduler (`src/services/background_scheduler.py`)

Handles CPU-intensive scheduling operations in the background, allowing API endpoints to return immediately.

**Key features:**
- Non-blocking task submission
- Configurable concurrent scheduling limit (default: 50)
- Backpressure management with HIGH_WATER_MARK and LOW_WATER_MARK

### CentralTaskQueue (`src/services/central_queue.py`)

FIFO task queue for task dispatch with event-driven processing.

**Key features:**
- FIFO ordering by enqueue time
- Event-driven dispatch
- Parallel dispatch with configurable concurrency

### WorkerQueueManager (`src/services/worker_queue_manager.py`)

Central coordinator for all WorkerQueueThread instances.

**Key features:**
- Creates/destroys threads when workers register/deregister
- Routes tasks to correct worker threads
- Provides queue depth for scheduling decisions
- Handles task redistribution on worker removal

### WorkerQueueThread (`src/services/worker_queue_thread.py`)

Individual per-worker queue thread for task dispatch.

**Key features:**
- Dedicated thread per worker
- FIFO task queue
- HTTP POST dispatch with retries
- Callback invocation on completion

---

## DATA MODELS

All Pydantic models are defined in `src/model.py` and `src/models/`.

### Core Enumerations

**TaskStatus** (`src/model.py`)
```python
class TaskStatus(str, Enum):
    PENDING = "pending"      # Submitted, not yet dispatched
    RUNNING = "running"      # Currently executing on instance
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"        # Execution failed
```

**InstanceStatus** (`src/model.py`)
```python
class InstanceStatus(str, Enum):
    ACTIVE = "active"      # Registered and available
    INACTIVE = "inactive"  # Registered but unavailable
    ERROR = "error"        # Error state
    DRAINING = "draining"  # Draining, no new tasks
```

### Key Models

- **Task**: Complete task information with status, timing, result
- **Instance**: Instance registration and statistics
- **PredictionResponse**: ML prediction result with quantiles
- **QueueInfo**: Instance queue state information

---

## DEVELOPMENT GUIDE

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_scheduler.py
```

### Profiling Endpoints

All API endpoints support profiling via query parameters:

```bash
curl "http://localhost:8000/task/submit?profile=true&profile_format=speedscope"
```

---

## COMPONENT INTERACTIONS

### External Dependencies

#### Predictor Service
- **Client:** `src/clients/predictor_client.py`
- **WebSocket:** `/ws/predict` for runtime predictions
- **HTTP:** `POST /train` for training data submission

#### Compute Instances
- **Client:** `src/services/task_dispatcher.py`
- **Endpoint:** `POST <instance_endpoint>/task/submit`
- **Callback:** `POST /callback/task_result`

### Task Submission Flow

```
1. API Endpoint (src/api.py)
   ↓
2. BackgroundScheduler.schedule_task_background()
   ↓
3. SchedulingStrategy.schedule() (src/algorithms/)
   ├─→ PredictorClient.predict() (src/clients/predictor_client.py)
   └─→ Select optimal instance
   ↓
4. TaskRegistry.add_task() (src/registry/task_registry.py)
   ↓
5. WorkerQueueManager.enqueue_task() (src/services/worker_queue_manager.py)
   ↓
6. WorkerQueueThread dispatches to instance
   ↓
7. Instance → POST /callback/task_result
   ↓
8. TaskResultCallback updates registries
```

---

## APPENDIX: QUICK REFERENCE

### Status Values
- **Task:** pending, running, completed, failed
- **Instance:** initializing, active, draining, removing, redeploying

### Strategy Names
- `round_robin`, `random`, `min_time`, `probabilistic`
- `power_of_two`, `serverless`

### Default Ports
- **Scheduler:** 8000
- **Predictor:** 8001

### Important File References

| Component | File |
|-----------|------|
| API endpoints | `src/api.py` |
| Scheduling strategies | `src/algorithms/*.py` |
| Task registry | `src/registry/task_registry.py` |
| Instance registry | `src/registry/instance_registry.py` |
| Background scheduler | `src/services/background_scheduler.py` |
| Central queue | `src/services/central_queue.py` |
| Worker queue manager | `src/services/worker_queue_manager.py` |
| Predictor client | `src/clients/predictor_client.py` |
| Configuration | `src/config.py` |
| CLI entry | `src/cli.py` |

---

**Document Version:** 2.0
**Last Updated:** 2026-01-16
**Scheduler Version:** 0.1.0
