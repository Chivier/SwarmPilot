# Scheduler Services Layer

This document describes the services layer in the scheduler module, which handles background processing, task dispatching, and result handling.

## Overview

The services layer provides non-blocking task processing through a layered architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Services Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  API Layer (src/api.py)                                         │
│       │                                                          │
│       ▼                                                          │
│  ┌────────────────────┐    ┌───────────────────────────┐       │
│  │ BackgroundScheduler│    │     CentralTaskQueue      │       │
│  │  (Non-blocking)    │    │   (FIFO, event-driven)    │       │
│  └─────────┬──────────┘    └─────────────┬─────────────┘       │
│            │                             │                       │
│            └──────────┬──────────────────┘                       │
│                       ▼                                          │
│            ┌──────────────────────┐                             │
│            │ WorkerQueueManager   │                             │
│            │ (Central coordinator)│                             │
│            └─────────┬────────────┘                             │
│                      │                                          │
│       ┌──────────────┼──────────────┐                          │
│       ▼              ▼              ▼                          │
│  ┌────────┐    ┌────────┐    ┌────────┐                        │
│  │Thread 1│    │Thread 2│    │Thread N│  WorkerQueueThread     │
│  │Worker-A│    │Worker-B│    │Worker-N│  (Per-worker threads)  │
│  └────┬───┘    └────┬───┘    └────┬───┘                        │
│       │             │             │                             │
│       └─────────────┼─────────────┘                             │
│                     ▼                                           │
│            ┌──────────────────────┐                             │
│            │ TaskResultCallback   │                             │
│            │ (Thread→EventLoop)   │                             │
│            └──────────────────────┘                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. BackgroundScheduler

**File:** `src/services/background_scheduler.py`

Handles CPU-intensive scheduling operations in the background, allowing API endpoints to return immediately.

#### Purpose
When a task is submitted via the API:
1. API creates task record immediately and returns
2. BackgroundScheduler processes scheduling asynchronously:
   - Get available instances
   - Call predictor for time estimates
   - Select optimal instance
   - Update queue info
   - Assign instance to task
   - Dispatch task

This prevents scenarios where 500+ workflows would block the API for extended periods due to serial prediction requests.

#### Key Features
- **Non-blocking task submission**: API returns immediately after recording the task
- **Configurable concurrency**: `max_concurrent_scheduling` parameter (default: 50)
- **Backpressure management**: Uses HIGH_WATER_MARK (5) and LOW_WATER_MARK (3) constants

#### Usage
```python
scheduler = BackgroundScheduler(
    scheduling_strategy=strategy,
    task_registry=task_registry,
    instance_registry=instance_registry,
    task_dispatcher=task_dispatcher,
    max_concurrent_scheduling=50,
)

# Non-blocking task scheduling
scheduler.schedule_task_background(
    task_id="task-001",
    model_id="llama-7b",
    task_input={"prompt": "..."},
    metadata={},
)
```

---

### 2. CentralTaskQueue

**File:** `src/services/central_queue.py`

FIFO task queue for task dispatch with event-driven processing.

#### Purpose
Manages the central queue of tasks waiting to be dispatched to workers. Provides event-driven dispatch that wakes up when tasks are enqueued or when capacity becomes available.

#### Key Features
- **FIFO ordering**: Tasks dispatched in enqueue order
- **Event-driven dispatch**: Dispatcher wakes on task enqueue or capacity changes
- **Parallel dispatch**: Configurable concurrency via `max_concurrent_dispatch`
- **Generation tracking**: Handles task queue clearing gracefully

#### Data Structures
```python
@dataclass
class QueuedTask:
    task_id: str
    model_id: str
    task_input: dict
    metadata: dict
    enqueue_time: float
    generation: int  # Tracks which clear cycle this task belongs to
```

#### Usage
```python
queue = CentralTaskQueue(
    task_registry=task_registry,
    instance_registry=instance_registry,
    max_concurrent_dispatch=50,
)

# Start the dispatcher background task
await queue.start()

# Enqueue a task
queue.enqueue(task_id, model_id, task_input, metadata)

# Shutdown gracefully
pending_tasks = await queue.shutdown()
```

---

### 3. WorkerQueueManager

**File:** `src/services/worker_queue_manager.py`

Central coordinator for all WorkerQueueThread instances.

#### Purpose
Implements PYLET-017: Worker Queue Manager. This class is the central coordinator that:
1. Creates threads when workers register
2. Destroys threads when workers deregister
3. Routes tasks to the correct worker thread
4. Provides queue depth information for scheduling decisions
5. Handles task redistribution on worker removal

#### Key Features
- **Thread management**: Automatic lifecycle management of worker threads
- **Task routing**: Routes tasks to correct worker based on assignment
- **Queue depth queries**: Provides `get_all_queue_depths()` for scheduling decisions
- **Thread safety**: Uses `threading.Lock` for worker dictionary access

#### Usage
```python
manager = WorkerQueueManager(
    callback=task_result_callback.create_thread_callback(loop),
    http_timeout=300.0,
)

# Register a worker (creates dedicated thread)
manager.register_worker("worker-1", "http://localhost:8001", "gpt-4")

# Enqueue task for a worker
manager.enqueue_task("worker-1", queued_task)

# Query queue state for scheduling
depths = manager.get_all_queue_depths()
# Returns: {"worker-1": 3, "worker-2": 1, ...}

# Deregister worker (stops thread, returns pending tasks)
pending = manager.deregister_worker("worker-1")

# Full shutdown
all_pending = manager.shutdown()
```

---

### 4. WorkerQueueThread

**File:** `src/services/worker_queue_thread.py`

Dedicated thread for processing tasks on a specific worker.

#### Purpose
Implements PYLET-015: Worker Queue Thread. Each registered worker gets its own thread that:
1. Maintains a FIFO task queue
2. Executes tasks synchronously against the worker's model API
3. Handles results via library-based callbacks

The thread runs independently of the main asyncio event loop, avoiding blocking during model inference calls.

#### Key Features
- **FIFO task queue**: Tasks processed in order of arrival
- **Sequential execution**: One task at a time per worker
- **Retry logic**: Retries for transient connection errors
- **Library-based callbacks**: Results passed via callback function

#### Data Structures
```python
@dataclass
class QueuedTask:
    task_id: str
    model_id: str
    task_input: dict
    metadata: dict
    enqueue_time: float
    predicted_time_ms: float | None = None

@dataclass
class TaskResult:
    task_id: str
    worker_id: str
    status: Literal["completed", "failed"]
    result: dict | None = None
    error: str | None = None
    execution_time_ms: float = 0.0
```

#### Usage
```python
def handle_result(result: TaskResult):
    print(f"Task {result.task_id}: {result.status}")

thread = WorkerQueueThread(
    worker_id="worker-1",
    worker_endpoint="http://localhost:8001",
    model_id="gpt-4",
    callback=handle_result,
    http_timeout=300.0,
)

# Start the thread
thread.start()

# Enqueue tasks
thread.enqueue(QueuedTask(...))

# Get queue depth
depth = thread.queue_depth()

# Stop gracefully
pending = thread.stop()
```

---

### 5. TaskResultCallback

**File:** `src/services/task_result_callback.py`

Callback handler for task results from worker threads.

#### Purpose
Implements PYLET-016: Library-Based Callback Mechanism. Bridges worker threads (synchronous) with the main asyncio event loop (async registries and WebSocket).

#### Key Responsibilities
1. Receive results from worker threads
2. Update TaskRegistry with status/result/error
3. Update InstanceRegistry with statistics
4. Record throughput for planner reporting
5. Broadcast results to WebSocket subscribers

#### Thread Safety
Uses `asyncio.run_coroutine_threadsafe()` to safely schedule async operations from worker threads.

#### Usage
```python
callback_handler = TaskResultCallback(
    task_registry=task_registry,
    instance_registry=instance_registry,
    websocket_manager=websocket_manager,
    throughput_tracker=throughput_tracker,
)

# Get thread-safe callback for worker threads
loop = asyncio.get_event_loop()
thread_callback = callback_handler.create_thread_callback(loop)

# Pass to WorkerQueueThread
worker_thread = WorkerQueueThread(
    worker_id="worker-1",
    callback=thread_callback,
    ...
)
```

---

### 6. TaskDispatcher

**File:** `src/services/task_dispatcher.py`

Async dispatcher for executing tasks on instances.

#### Purpose
Manages HTTP communication with compute instances for task execution. Handles retries, timeouts, and result processing.

#### Key Features
- **Connection pooling**: Configures httpx with keepalive management
- **Retry logic**: Exponential backoff for transient errors
- **Fire-and-forget**: Non-blocking dispatch with background result handling
- **Timeout management**: Configurable per-task timeout

#### Configuration
```python
DEFAULT_DISPATCH_RETRIES = 3
DEFAULT_DISPATCH_RETRY_DELAY = 0.1  # 100ms initial delay
```

#### Usage
```python
dispatcher = TaskDispatcher(
    task_registry=task_registry,
    instance_registry=instance_registry,
    websocket_manager=websocket_manager,
    training_client=training_client,
    timeout=60.0,
    callback_base_url="http://localhost:8000",
    dispatch_retries=3,
    dispatch_retry_delay=0.1,
    throughput_tracker=throughput_tracker,
)

# Dispatch a task (fire-and-forget)
await dispatcher.dispatch_task(
    task_id="task-001",
    instance_id="worker-001",
    task_input={"prompt": "..."},
    metadata={},
    predicted_time_ms=100.0,
)
```

---

### 7. WebSocketManager (ConnectionManager)

**File:** `src/services/websocket_manager.py`

Manages WebSocket connections for real-time task result notifications.

#### Purpose
Maintains active WebSocket connections and handles broadcasting task results to subscribed clients.

#### Key Features
- **Connection tracking**: Tracks active WebSocket connections
- **Subscription management**: Clients subscribe to specific task IDs
- **Result broadcasting**: Sends results to all subscribers of a task
- **Ping/pong keepalive**: Maintains connection health

---

### 8. ShutdownHandler

**File:** `src/services/shutdown_handler.py`

Manages graceful shutdown of all services.

#### Purpose
Coordinates clean shutdown of background schedulers, worker threads, and active connections.

---

## Task Flow

### 1. Task Submission
```
POST /task/submit
    │
    ▼
BackgroundScheduler.schedule_task_background()
    │
    ▼
SchedulingStrategy.schedule()  ───► PredictorClient.predict()
    │
    ▼
Select optimal instance
    │
    ▼
WorkerQueueManager.enqueue_task(instance_id, task)
```

### 2. Task Execution
```
WorkerQueueManager
    │
    ▼
WorkerQueueThread (dedicated thread for instance)
    │
    ▼
HTTP POST to instance endpoint
    │
    ▼
Wait for response
    │
    ▼
TaskResult created
```

### 3. Result Handling
```
WorkerQueueThread
    │
    ▼
callback(TaskResult)
    │
    ▼
TaskResultCallback.create_thread_callback()
    │
    ▼
asyncio.run_coroutine_threadsafe()
    │
    ▼
TaskResultCallback.handle_result()
    │
    ├──► TaskRegistry.update_task_status()
    ├──► InstanceRegistry.update_stats()
    ├──► ThroughputTracker.record()
    └──► WebSocketManager.broadcast()
```

---

## Related Files

| File | Description |
|------|-------------|
| `src/services/__init__.py` | Package exports |
| `src/services/background_scheduler.py` | Non-blocking task scheduling |
| `src/services/central_queue.py` | FIFO task queue |
| `src/services/worker_queue_manager.py` | Worker thread coordinator |
| `src/services/worker_queue_thread.py` | Per-worker execution thread |
| `src/services/task_result_callback.py` | Thread-to-eventloop bridge |
| `src/services/task_dispatcher.py` | HTTP task dispatcher |
| `src/services/websocket_manager.py` | WebSocket connection manager |
| `src/services/shutdown_handler.py` | Graceful shutdown handling |

---

**Last Updated:** 2026-01-16
