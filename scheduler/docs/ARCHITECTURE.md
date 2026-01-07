# Scheduler Architecture

## Overview

The Scheduler is a FastAPI-based service that manages task distribution across model instances. It receives inference requests, selects optimal instances using prediction-based scheduling algorithms, and tracks task execution.

```
                                    ┌─────────────────────────────────────────┐
                                    │              Planner                     │
                                    │  (Manages instances via PyLet)          │
                                    └────────────────┬────────────────────────┘
                                                     │
                           POST /instance/sync       │  Reports throughput
                           (declarative instance list)│  via PlannerReporter
                                                     │
                                                     ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                                  SCHEDULER                                      │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                            FastAPI Application                            │  │
│  │                                (api.py)                                   │  │
│  │                                                                           │  │
│  │  Endpoints:                                                               │  │
│  │  - POST /task/submit     → Submit task for scheduling                    │  │
│  │  - POST /instance/sync   → Sync instances (declarative)                  │  │
│  │  - GET /task/info        → Get task status                               │  │
│  │  - WS /ws/{task_id}      → Real-time result delivery                    │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                       │                                         │
│                                       ▼                                         │
│  ┌──────────────────────┐    ┌──────────────────────┐    ┌─────────────────┐  │
│  │  Scheduling Strategy  │    │   WorkerQueueManager  │    │ InstanceRegistry │  │
│  │   (algorithms/)       │◄───│     (services/)       │───►│   (registry/)    │  │
│  │                       │    │                       │    │                  │  │
│  │  - MinExpectedTime    │    │  Coordinates per-     │    │  Tracks active   │  │
│  │  - PowerOfTwo         │    │  worker queue threads │    │  instances       │  │
│  │  - Probabilistic      │    │                       │    │                  │  │
│  │  - RoundRobin         │    └───────────┬───────────┘    └─────────────────┘  │
│  │  - Random             │                │                                      │
│  └──────────────────────┘                │                                      │
│                                          ▼                                      │
│                           ┌──────────────────────────┐                          │
│                           │   WorkerQueueThread(s)   │                          │
│                           │      (services/)         │                          │
│                           │                          │                          │
│                           │  Per-worker priority     │                          │
│                           │  queue with FIFO order   │                          │
│                           └──────────────┬───────────┘                          │
└──────────────────────────────────────────┼──────────────────────────────────────┘
                                           │
                                           │  HTTP requests
                                           ▼
                              ┌────────────────────────┐
                              │   Model Instances      │
                              │   (PyLet-managed)      │
                              │                        │
                              │  - Instance 1 (:9000)  │
                              │  - Instance 2 (:9001)  │
                              │  - Instance N (:900N)  │
                              └────────────────────────┘
```

## Directory Structure

```
scheduler/src/
├── api.py                    # FastAPI application and endpoints
├── config.py                 # Configuration settings
├── model.py                  # Re-exports from models/
├── instance_sync.py          # Instance sync logic for declarative API
│
├── algorithms/               # Scheduling strategies
│   ├── base.py               # SchedulingStrategy base class
│   ├── factory.py            # get_strategy() factory function
│   ├── queue_state_adapter.py # Converts queue state for algorithms
│   ├── min_expected_time.py  # Minimum expected time strategy
│   ├── serverless.py         # Serverless-optimized strategy
│   ├── power_of_two.py       # Power of two choices
│   ├── probabilistic.py      # Probabilistic scheduling
│   ├── round_robin.py        # Simple round-robin
│   └── random.py             # Random selection
│
├── clients/                  # External service clients
│   ├── predictor_client.py   # Predictor service communication
│   └── training_client.py    # Training data collection
│
├── models/                   # Data models
│   ├── core.py               # Instance, Task models
│   ├── queue.py              # Queue state models
│   ├── requests.py           # API request models
│   ├── responses.py          # API response models
│   ├── status.py             # TaskStatus enum
│   └── websocket.py          # WebSocket message models
│
├── registry/                 # State management
│   ├── instance_registry.py  # Instance state tracking
│   └── task_registry.py      # Task state tracking
│
├── services/                 # Background services
│   ├── background_scheduler.py   # Background task processing
│   ├── central_queue.py          # Central task queue (legacy)
│   ├── task_dispatcher.py        # Task dispatch logic
│   ├── websocket_manager.py      # WebSocket connections
│   ├── worker_queue_manager.py   # Manages worker queue threads
│   ├── worker_queue_thread.py    # Per-worker task queue
│   ├── task_result_callback.py   # Handles task completion
│   └── shutdown_handler.py       # Graceful shutdown logic
│
└── utils/                    # Utilities
    ├── logger.py             # Logging configuration
    ├── http_error_logger.py  # HTTP error logging
    ├── planner_reporter.py   # Throughput reporting
    └── throughput_tracker.py # Instance throughput tracking
```

## Core Components

### 1. Scheduling Strategies (`algorithms/`)

All strategies inherit from `SchedulingStrategy` base class:

```python
class SchedulingStrategy(ABC):
    def __init__(self, predictor_client, instance_registry):
        self.predictor_client = predictor_client
        self.instance_registry = instance_registry
        self._worker_queue_manager = None  # Set via set_worker_queue_manager()

    @abstractmethod
    async def schedule_task(
        self,
        model_id: str,
        metadata: dict,
        queue_states: list[InstanceQueueBase],
    ) -> ScheduleResult:
        """Select an instance for task execution."""
        pass
```

**Available Strategies:**

| Strategy | Description |
|----------|-------------|
| `MinimumExpectedTimeStrategy` | Selects instance with minimum expected completion time |
| `MinimumExpectedTimeServerlessStrategy` | Optimized for serverless environments |
| `PowerOfTwoStrategy` | Randomly picks 2 instances, selects the faster one |
| `ProbabilisticSchedulingStrategy` | Uses quantile sampling for probabilistic selection |
| `RoundRobinStrategy` | Simple round-robin across instances |
| `RandomStrategy` | Random instance selection (baseline) |

### 2. Worker Queue System (`services/`)

The scheduler uses a per-worker queue architecture for task management:

```
WorkerQueueManager
    │
    ├── WorkerQueueThread (worker-1)
    │       └── PriorityQueue (FIFO by enqueue_time)
    │
    ├── WorkerQueueThread (worker-2)
    │       └── PriorityQueue (FIFO by enqueue_time)
    │
    └── WorkerQueueThread (worker-N)
            └── PriorityQueue (FIFO by enqueue_time)
```

**Key Classes:**

- `WorkerQueueManager`: Coordinates all worker queue threads
- `WorkerQueueThread`: Per-worker thread that processes tasks from its queue
- `QueuedTask`: Dataclass representing a queued task

### 3. Instance Sync (`instance_sync.py`)

The Planner manages instances declaratively via `POST /instance/sync`:

```python
# Request: Send complete target instance list
{
    "instances": [
        {"instance_id": "w1", "endpoint": "http://...", "model_id": "..."},
        {"instance_id": "w2", "endpoint": "http://...", "model_id": "..."}
    ]
}

# Response: What changed
{
    "success": true,
    "added": ["w2"],      # Newly registered
    "removed": ["w3"],    # Removed (was active, not in target)
    "rescheduled": 5      # Tasks rescheduled from removed instances
}
```

### 4. Graceful Shutdown (`services/shutdown_handler.py`)

Handles instance removal and scheduler shutdown:

- **Instance Removal**: Tasks from removed instances are rescheduled to remaining workers
- **Scheduler Shutdown**: Tasks are dropped (no workers available for rescheduling)

## Request Flow

### Task Submission

```
1. Client → POST /task/submit
       │
       ▼
2. API validates request, creates TaskRecord in TaskRegistry
       │
       ▼
3. SchedulingStrategy.schedule_task() selects optimal instance
       │
       ▼
4. WorkerQueueManager.enqueue_task() adds to selected worker's queue
       │
       ▼
5. WorkerQueueThread picks up task, sends HTTP request to instance
       │
       ▼
6. TaskResultCallback processes result, updates TaskRegistry
       │
       ▼
7. Client receives result via WebSocket or polling GET /task/info
```

### Instance Sync

```
1. Planner → POST /instance/sync with target instance list
       │
       ▼
2. Scheduler computes diff: to_add, to_remove
       │
       ▼
3. For each instance to remove:
   a. Get pending tasks from WorkerQueueThread
   b. Stop the thread
   c. Reschedule tasks to remaining workers
   d. Remove from InstanceRegistry
       │
       ▼
4. For each instance to add:
   a. Register in InstanceRegistry
   b. Create WorkerQueueThread
       │
       ▼
5. Return summary: added, removed, rescheduled count
```

## Interaction with Planner

The Scheduler and Planner communicate in two ways:

### 1. Planner → Scheduler: Instance Management

The Planner uses PyLet to manage model instances and syncs them with the Scheduler:

```
POST /instance/sync
{
    "instances": [
        {
            "instance_id": "gpu-0-model-A",
            "endpoint": "http://10.0.0.1:9000",
            "model_id": "llama-7b"
        }
    ]
}
```

### 2. Scheduler → Planner: Throughput Reporting

The `PlannerReporter` periodically reports metrics to the Planner:

```python
class PlannerReporter:
    async def _report_loop(self):
        while self._running:
            # Collect metrics
            pending_count = await self.task_registry.count_by_status(PENDING)
            running_count = await self.task_registry.count_by_status(RUNNING)
            throughput = self.throughput_tracker.get_throughput()

            # Report to planner
            await self._send_report({
                "uncompleted_tasks": pending_count + running_count,
                "throughput": throughput
            })

            await asyncio.sleep(self.report_interval)
```

## Interaction with Instances

### Task Execution

The `WorkerQueueThread` sends tasks to instances via HTTP:

```python
async def _execute_task(self, task: QueuedTask) -> TaskResult:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{self.worker_endpoint}/v1/completions",
            json=task.task_input,
            timeout=self.http_timeout
        )
        return TaskResult(
            task_id=task.task_id,
            status="completed",
            result=response.json(),
            execution_time_ms=elapsed_ms
        )
```

### Instance Health

Instances are expected to provide a health endpoint:

```
GET /health
→ {"status": "healthy", "model_id": "...", "queue_depth": N}
```

## Configuration

Key configuration options in `config.py`:

| Setting | Description | Default |
|---------|-------------|---------|
| `MODEL_ID` | Model identifier | Required |
| `PREDICTOR_URL` | Predictor service URL | Required |
| `PLANNER_URL` | Planner service URL | Optional |
| `SCHEDULING_STRATEGY` | Algorithm name | `min_expected_time` |
| `MAX_CONCURRENT` | Max concurrent tasks | 50 |
| `HTTP_TIMEOUT` | Task execution timeout | 30.0s |

## Error Handling

### Network Errors

- `WorkerQueueThread` retries transient connection errors (configurable retries)
- Failed tasks trigger callback with `status="failed"`

### Instance Removal

- Pending tasks are rescheduled to other workers
- If no workers available, tasks are marked as FAILED

### Scheduler Shutdown

- Graceful shutdown waits for in-progress tasks (with timeout)
- Pending tasks are dropped and counted in shutdown result

## Testing

Tests are organized by component:

```
tests/
├── conftest.py                     # Shared fixtures
├── test_scheduler.py               # Scheduling strategy tests
├── test_worker_queue_thread.py     # Worker queue tests
├── test_worker_queue_manager.py    # Manager tests
├── test_instance_sync_api.py       # Instance sync tests
├── test_shutdown_handler.py        # Shutdown tests
├── test_queue_state_adapter.py     # Adapter tests
└── integration/
    └── test_phase3_integration.py  # End-to-end tests
```

Run tests:
```bash
uv run pytest tests/ -v
```

## Module Import Guidelines

Always use canonical import paths:

```python
# Correct
from src.algorithms import get_strategy, SchedulingStrategy
from src.services.worker_queue_manager import WorkerQueueManager
from src.registry.instance_registry import InstanceRegistry
from src.clients.predictor_client import PredictorClient

# Avoid (legacy paths removed)
# from src.scheduler import get_strategy  # No longer exists
# from src.instance_registry import ...   # No longer exists
```

## See Also

- [API Reference](1.API_REFERENCE.md) - Complete endpoint documentation
- [Scheduling Strategies](8.SCHEDULING_STRATEGIES.md) - Strategy details
- [WebSocket API](7.WEBSOCKET_API.md) - Real-time result delivery
