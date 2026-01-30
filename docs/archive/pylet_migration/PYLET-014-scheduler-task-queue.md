# PYLET-014: Scheduler-Side Task Queue Design

## Status: DESIGN COMPLETE

## Objective

Design and document the scheduler-side task queue architecture that replaces instance-side task queuing. This enables centralized queue management with per-worker task threads.

## Key Design Constraints

Based on requirements review:

1. **Single Model per Scheduler**: Each scheduler handles only one model type (LLM, image gen, embeddings, etc.). Planner manages instance information via scheduler API.
2. **Reuse Existing Algorithms**: No new scheduling algorithms - use existing `min_expected_time`, `probabilistic`, `round_robin`, etc.
3. **Unified Transparent Proxy**: ALL requests go through the Transparent Proxy. Non-management requests go to Dispatcher, which routes to worker queues.
4. **Planner-Controlled Instance Management**: Planner updates scheduler's instance list via API (not self-registration).

## Background

### Current Architecture (Instance-Side Queue)

```
┌─────────────┐     ┌─────────────────────┐     ┌─────────────┐
│  Scheduler  │────▶│      Instance       │────▶│    Model    │
│             │     │  ┌───────────────┐  │     │   (vLLM)    │
│ POST /task/ │     │  │  Task Queue   │  │     │             │
│   submit    │◀────│  │  (FIFO)       │  │◀────│             │
│             │     │  └───────────────┘  │     │             │
│  (callback) │     │  + Callback logic   │     │             │
└─────────────┘     └─────────────────────┘     └─────────────┘
```

### Target Architecture (Scheduler-Side Queue)

**Unified Proxy Architecture**: ALL requests go through the Transparent Proxy. Non-management requests are routed to the Dispatcher, which queues and load-balances them to workers.

```
                    ┌─────────────────────────────────────────────────────┐
                    │              PLANNER                                 │
                    │  ┌─────────────────────────────────────────────┐    │
                    │  │ Instance Management via Scheduler API        │    │
                    │  │ POST /instance/register                      │    │
                    │  │ POST /instance/remove                        │    │
                    │  │ POST /instance/drain                         │    │
                    │  └─────────────────────────────────────────────┘    │
                    └──────────────────────┬──────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    SCHEDULER (Single Model Type: any)                     │
│                    (could be LLM, image gen, embeddings, etc.)            │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │                    Transparent Proxy (Entry Point)                 │   │
│  │                    ALL requests enter here                         │   │
│  │  ┌─────────────────────────────────────────────────────────────┐  │   │
│  │  │  Route by Path:                                              │  │   │
│  │  │  • /instance/*, /health, /info → Management API              │  │   │
│  │  │  • ALL other paths → Dispatcher (queued & load-balanced)     │  │   │
│  │  └─────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────┬─────────────────────────────────────┘   │
│                                │                                          │
│                    ┌───────────┴───────────┐                             │
│                    │                       │                             │
│                    ▼                       ▼                             │
│           ┌───────────────┐     ┌─────────────────────────────────┐     │
│           │ Management    │     │         Dispatcher               │     │
│           │ API           │     │ (ALL non-management requests)    │     │
│           │               │     │                                  │     │
│           │ /instance/*   │     │ ┌──────────────────────────────┐ │     │
│           │ /health       │     │ │  Scheduling Strategy          │ │     │
│           │ /info         │     │ │  (existing: min_time, etc.)   │ │     │
│           └───────────────┘     │ └─────────────┬────────────────┘ │     │
│                                 └───────────────┼────────────────────┘     │
│                                                 │                          │
│              ┌──────────────────────────────────┼──────────────────┐      │
│              │                                  │                  │      │
│              ▼                                  ▼                  ▼      │
│       ┌─────────────┐                    ┌─────────────┐    ┌─────────────┐
│       │Worker Queue │                    │Worker Queue │    │Worker Queue │
│       │  Thread 1   │                    │  Thread 2   │    │  Thread N   │
│       │ ┌─────────┐ │                    │ ┌─────────┐ │    │ ┌─────────┐ │
│       │ │  FIFO   │ │                    │ │  FIFO   │ │    │ │  FIFO   │ │
│       │ └─────────┘ │                    │ └─────────┘ │    │ └─────────┘ │
│       └──────┬──────┘                    └──────┬──────┘    └──────┬──────┘
│              │                                  │                  │      │
└──────────────┼──────────────────────────────────┼──────────────────┼──────┘
               │                                  │                  │
               ▼                                  ▼                  ▼
        ┌─────────────┐                    ┌─────────────┐    ┌─────────────┐
        │PyLet Worker │                    │PyLet Worker │    │PyLet Worker │
        │  (model_x)  │                    │  (model_x)  │    │  (model_x)  │
        │  Any API    │                    │  Any API    │    │  Any API    │
        └─────────────┘                    └─────────────┘    └─────────────┘
```

## Architecture Components

### 1. Single Model Constraint

Each scheduler instance handles exactly one model. This is enforced via:

```python
@dataclass
class SchedulerConfig:
    # Model ID this scheduler handles (set on first registration or via env)
    model_id: str = os.getenv("SCHEDULER_MODEL_ID", "")

    # If True, reject registrations for different model_id
    enforce_single_model: bool = True
```

**Validation on Registration:**

```python
@app.post("/instance/register")
async def register_instance(request: InstanceRegisterRequest):
    # Verify model_id matches scheduler's configured model
    if config.model_id and request.model_id != config.model_id:
        raise HTTPException(
            status_code=400,
            detail=f"Model mismatch: scheduler handles '{config.model_id}', "
                   f"got '{request.model_id}'"
        )

    # Set model_id on first registration if not configured
    if not config.model_id:
        config.model_id = request.model_id

    # ... existing registration logic ...
```

### 2. Planner-Managed Instance API (Instance List Submission)

The planner controls instance lifecycle by submitting the **complete target instance list** to the scheduler. The scheduler computes the diff and handles additions/removals internally.

#### Primary API: Instance List Submission

| Endpoint | Purpose | Description |
|----------|---------|-------------|
| `POST /instance/sync` | Submit target instance list | Scheduler computes diff, adds/removes instances |
| `GET /instance/list` | Get current instance list | For state synchronization |

**Instance List Submission Flow:**

```python
@dataclass
class InstanceInfo:
    instance_id: str
    endpoint: str  # "host:port"
    model_id: str

@dataclass
class InstanceSyncRequest:
    instances: list[InstanceInfo]  # Target instance list

@dataclass
class InstanceSyncResponse:
    success: bool
    added: list[str]      # Instance IDs added
    removed: list[str]    # Instance IDs removed
    rescheduled: int      # Tasks rescheduled from removed instances
    message: str


@app.post("/instance/sync")
async def sync_instances(request: InstanceSyncRequest) -> InstanceSyncResponse:
    """Submit target instance list. Scheduler computes diff and handles changes.

    This is the primary API for planner to manage instances. The planner:
    1. Cancels PyLet instances before calling this
    2. Sends the new target list
    3. Scheduler handles the diff internally

    For removed instances, unfinished tasks are rescheduled to other queues.
    Queues don't exit on network errors - tasks stay with retry flag.
    """
    # 1. Validate all instances have correct model_id
    for inst in request.instances:
        if config.model_id and inst.model_id != config.model_id:
            raise HTTPException(
                status_code=400,
                detail=f"Model mismatch for {inst.instance_id}: "
                       f"expected '{config.model_id}', got '{inst.model_id}'"
            )

    # 2. Compute diff: what to add, what to remove
    current_ids = set(instance_registry.get_all_ids())
    target_ids = {inst.instance_id for inst in request.instances}

    to_add = target_ids - current_ids
    to_remove = current_ids - target_ids

    added_list = []
    removed_list = []
    total_rescheduled = 0

    # 3. Remove instances first (reschedule their tasks)
    for instance_id in to_remove:
        rescheduled = await _handle_instance_removal(instance_id)
        removed_list.append(instance_id)
        total_rescheduled += rescheduled

    # 4. Add new instances (create queues)
    target_by_id = {inst.instance_id: inst for inst in request.instances}
    for instance_id in to_add:
        inst = target_by_id[instance_id]
        await _handle_instance_addition(inst)
        added_list.append(instance_id)

    return InstanceSyncResponse(
        success=True,
        added=added_list,
        removed=removed_list,
        rescheduled=total_rescheduled,
        message=f"Synced: +{len(added_list)} -{len(removed_list)}, "
                f"{total_rescheduled} tasks rescheduled"
    )


async def _handle_instance_addition(inst: InstanceInfo) -> None:
    """Add a new instance: register and create priority queue."""
    # Register in instance registry
    await instance_registry.add(InstanceRecord(
        instance_id=inst.instance_id,
        model_id=inst.model_id,
        endpoint=inst.endpoint,
        status=InstanceStatus.ACTIVE,
    ))

    # Create priority queue for this worker
    worker_queue_manager.register_worker(
        worker_id=inst.instance_id,
        worker_endpoint=inst.endpoint,
        model_id=inst.model_id,
    )


async def _handle_instance_removal(instance_id: str) -> int:
    """Remove an instance: reschedule unfinished tasks to other queues.

    Returns: Number of tasks rescheduled
    """
    # 1. Mark as draining (no new tasks)
    await instance_registry.start_draining(instance_id)

    # 2. Get unfinished tasks from this worker's queue
    unfinished_tasks = worker_queue_manager.get_unfinished_tasks(instance_id)

    # 3. Deregister worker (removes queue)
    worker_queue_manager.deregister_worker(instance_id)

    # 4. Reschedule unfinished tasks to other queues
    rescheduled_count = 0
    for task in unfinished_tasks:
        # Re-run scheduling algorithm for this task
        success = await _reschedule_task(task)
        if success:
            rescheduled_count += 1

    # 5. Remove from registry
    await instance_registry.remove(instance_id)

    return rescheduled_count


async def _reschedule_task(task: QueuedTask) -> bool:
    """Reschedule a task to another available queue.

    Re-runs the scheduling algorithm to select a new worker.
    Task retains its original enqueue_time for priority ordering.
    """
    available_instances = await instance_registry.get_active_instances(task.model_id)

    if not available_instances:
        # No workers available - task stays in limbo (or central queue)
        logger.warning(f"No workers to reschedule task {task.task_id}")
        return False

    # Re-run scheduling algorithm
    schedule_result = await scheduling_strategy.schedule_task(
        model_id=task.model_id,
        metadata=task.metadata,
        available_instances=available_instances,
    )

    if not schedule_result.selected_instance_id:
        return False

    # Enqueue to new worker's priority queue
    # Task keeps original enqueue_time for priority ordering
    worker_queue_manager.enqueue_task(
        schedule_result.selected_instance_id,
        task,  # Retains original enqueue_time
    )

    return True
```

#### Legacy API (for backward compatibility)

| Endpoint | Purpose | Status |
|----------|---------|--------|
| `POST /instance/register` | Add single worker | Deprecated, use `/instance/sync` |
| `POST /instance/remove` | Remove single worker | Deprecated, use `/instance/sync` |
| `POST /instance/drain` | Drain before removal | Still useful for graceful single-worker drain |

### 3. WorkerQueueThread (Priority Queue)

A dedicated thread for each registered worker that:
- Maintains a **priority queue** ordered by `enqueue_time` (earlier = higher priority)
- Processes tasks sequentially (no parallelism within a worker)
- Makes synchronous HTTP calls to the worker's model API
- Invokes callbacks via `asyncio.run_coroutine_threadsafe()`
- **Resilient to network errors**: Queue doesn't exit on error; tasks stay with retry flag

**Priority Queue Implementation:**

```python
import heapq
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

@dataclass(order=True)
class PrioritizedTask:
    """Task wrapper for priority queue ordering."""
    priority: float  # enqueue_time (lower = higher priority)
    task: Any = field(compare=False)  # QueuedTask

class PriorityTaskQueue:
    """Thread-safe priority queue ordered by enqueue time.

    Tasks that arrived earlier have higher priority (earlier = front of queue).
    """
    def __init__(self):
        self._heap: list[PrioritizedTask] = []
        self._lock = Lock()

    def put(self, task: QueuedTask) -> None:
        """Add task to queue. Priority = enqueue_time (earlier = higher)."""
        with self._lock:
            heapq.heappush(
                self._heap,
                PrioritizedTask(priority=task.enqueue_time, task=task)
            )

    def get(self, timeout: float | None = None) -> QueuedTask | None:
        """Get highest priority (earliest) task."""
        with self._lock:
            if not self._heap:
                return None
            return heapq.heappop(self._heap).task

    def peek(self) -> QueuedTask | None:
        """Look at highest priority task without removing."""
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0].task

    def get_all(self) -> list[QueuedTask]:
        """Get all tasks (for rescheduling on removal)."""
        with self._lock:
            tasks = [pt.task for pt in self._heap]
            self._heap.clear()
            return tasks

    def qsize(self) -> int:
        with self._lock:
            return len(self._heap)
```

**Network Error Handling (Resilient Queue):**

```python
class WorkerQueueThread(threading.Thread):
    """Worker thread with resilient error handling.

    Key behavior:
    - Queue does NOT exit on network errors
    - Tasks stay in queue with retry flag on dispatch error
    - Planner cancels PyLet instance before sending new instance list
    - Scheduler rescheduled tasks from removed instance to others
    """

    def __init__(self, ...):
        self._queue = PriorityTaskQueue()
        self._running = True
        self._max_retries = 3
        self._retry_delay_base = 1.0  # seconds

    def _execute_task(self, task: QueuedTask) -> None:
        """Execute task with retry on network error."""
        retries = task.metadata.get("_retry_count", 0)

        try:
            # Attempt to dispatch to worker
            response = self._client.post(
                f"{self._endpoint}/{task.metadata['path']}",
                json=task.task_input,
                timeout=self._http_timeout,
            )
            response.raise_for_status()

            # Success - invoke callback
            self._callback(
                task.task_id,
                TaskStatus.COMPLETED,
                response.json(),
                None,
            )

        except httpx.NetworkError as e:
            # Network error - keep task in queue with retry flag
            # Queue does NOT exit - Planner will send new instance list
            retries += 1
            task.metadata["_retry_count"] = retries
            task.metadata["_last_error"] = str(e)

            if retries < self._max_retries:
                # Re-queue for retry (keeps original enqueue_time for priority)
                logger.warning(
                    f"Network error for task {task.task_id}, "
                    f"retry {retries}/{self._max_retries}: {e}"
                )
                time.sleep(self._retry_delay_base * retries)
                self._queue.put(task)  # Re-queue with same priority
            else:
                # Max retries reached - mark as failed
                logger.error(
                    f"Task {task.task_id} failed after {retries} retries: {e}"
                )
                self._callback(
                    task.task_id,
                    TaskStatus.FAILED,
                    None,
                    f"Network error after {retries} retries: {e}",
                )

        except httpx.HTTPStatusError as e:
            # HTTP error (4xx/5xx) - mark as failed
            self._callback(
                task.task_id,
                TaskStatus.FAILED,
                None,
                f"HTTP error {e.response.status_code}: {e.response.text}",
            )
```

See [PYLET-015](PYLET-015-worker-queue-thread.md) for full implementation details.

### 4. TaskResultCallback

A callback handler that runs in the main event loop:
- Updates task registry (status, result, error)
- Updates instance registry (statistics)
- Records throughput for planner reporting
- Broadcasts results to WebSocket subscribers

See [PYLET-016](PYLET-016-callback-mechanism.md) for implementation details.

### 5. WorkerQueueManager

Manages all `WorkerQueueThread` instances:
- Creates threads when workers register
- Destroys threads when workers deregister
- Provides queue depth information for scheduling
- Handles task redistribution on worker removal

See [PYLET-017](PYLET-017-worker-queue-manager.md) for implementation details.

### 6. Reuse Existing Scheduling Algorithms

**No new algorithms.** Use existing strategies with queue depth from `WorkerQueueManager`:

```python
# Existing strategies (no changes to algorithm logic)
# - MinimumExpectedTimeStrategy
# - ProbabilisticSchedulingStrategy
# - RoundRobinStrategy
# - RandomStrategy
# - PowerOfTwoStrategy

# Queue depth is already tracked in InstanceRegistry
# Strategies already consider queue state via InstanceQueueExpectError/InstanceQueueProbabilistic
```

The key change is that queue state is now maintained **scheduler-side** instead of **instance-side**, but the scheduling algorithm interface remains the same.

### 7. Unified Transparent Proxy

**ALL requests** enter through the Transparent Proxy. Non-management requests go to the Dispatcher for queuing and load-balancing:

```python
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def transparent_proxy(request: Request, path: str):
    """Unified entry point for all requests.

    Routes requests based on path:
    1. Management endpoints → handled by scheduler directly
    2. ALL other endpoints → Dispatcher → Worker Queue (queued & load-balanced)

    This unified architecture:
    - Simplifies client interface (call standard model endpoints)
    - Handles queuing transparently
    - Works with ANY model type (LLM, image gen, embeddings, etc.)
    """
    # 1. Management endpoints - handled by scheduler directly
    management_prefixes = [
        "instance/", "health", "info", "docs", "openapi"
    ]
    if any(path.startswith(prefix) for prefix in management_prefixes):
        # Let FastAPI handle these via normal routes
        raise HTTPException(status_code=404, detail="Not found")

    # Get available workers
    instances = await instance_registry.list_active()
    if not instances:
        raise HTTPException(
            status_code=503,
            detail="No workers available"
        )

    # 2. ALL other requests → Dispatcher → Worker Queue
    return await _dispatch_request(request, path, instances)


async def _dispatch_request(
    request: Request,
    path: str,
    instances: list,
) -> Response:
    """Route request through Dispatcher to Worker Queue.

    ALL non-management requests are queued and load-balanced.
    This works for any model type (LLM, image gen, embeddings, etc.).
    """
    # Get request body (may be empty for GET requests)
    body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.json()
        except:
            body = await request.body()

    # Create task and get task_id
    task_id = str(uuid.uuid4())
    task_record = TaskRecord(
        task_id=task_id,
        model_id=config.model_id,
        task_input=body,
        metadata={
            "path": path,
            "method": request.method,
            "headers": dict(request.headers),
        },
        status=TaskStatus.PENDING,
    )
    await task_registry.add(task_record)

    # Select worker using existing scheduling strategy
    schedule_result = await scheduling_strategy.schedule_task(
        model_id=config.model_id,
        metadata=task_record.metadata,
        available_instances=instances,
    )

    if not schedule_result.selected_instance_id:
        # No worker available - return error (or queue if central queue enabled)
        raise HTTPException(
            status_code=503,
            detail="All workers busy"
        )

    # Enqueue to worker's queue thread
    queued_task = QueuedTask(
        task_id=task_id,
        model_id=config.model_id,
        task_input=body,
        metadata={
            "path": path,
            "method": request.method,
            "headers": dict(request.headers),
        },
        enqueue_time=time.time(),
    )
    worker_queue_manager.enqueue_task(
        schedule_result.selected_instance_id,
        queued_task,
    )

    # Wait for result (synchronous from client perspective)
    result = await task_registry.wait_for_result(task_id, timeout=300.0)

    return JSONResponse(content=result)
```

**Request Routing:**

| Path Pattern | Routing | Description |
|--------------|---------|-------------|
| `/instance/*`, `/health`, `/info` | Management API | Scheduler handles directly |
| **ALL other paths** | Dispatcher → Queue | Queued & load-balanced to workers |

**Benefits of Unified Proxy:**
- **Simple client interface**: Clients call standard model endpoints (e.g., `/v1/completions`, `/generate`, etc.)
- **Transparent queuing**: All requests are automatically queued and load-balanced
- **Model-agnostic**: Works with ANY model type (LLM, image generation, embeddings, audio, etc.)
- **Single model per scheduler**: Each scheduler handles one model type, verified on registration

## Data Flow

### Unified Request Flow

All requests enter through the Transparent Proxy:

```
ANY /{path}
    │
    ├── Transparent Proxy (Entry Point)
    │
    ├── 1. Check path type:
    │       │
    │       ├── /instance/*, /health, /info, /docs
    │       │       └── Management API (scheduler handles)
    │       │
    │       ├── /v1/completions, /v1/chat/*, /generate
    │       │       └── Queued Request → Dispatcher
    │       │
    │       └── Other paths (/v1/models, /health on worker, etc.)
    │               └── Direct Forward (round-robin)
    │
    ▼
```

### Queued Request Flow (LLM Inference)

```
POST /v1/completions (via Proxy)
    │
    ├── 1. Parse request body
    │
    ├── 2. Create TaskRecord (status=PENDING)
    │
    ├── 3. Select worker via existing SchedulingStrategy
    │       ├── Get predictions for each worker
    │       └── Use existing algorithm (min_time, probabilistic, etc.)
    │
    ├── 4. If no worker available:
    │       └── Return 503 (or queue to CentralQueue)
    │
    ├── 5. Enqueue to WorkerQueueThread
    │       ├── TaskRecord.assigned_instance = worker_id
    │       ├── TaskRecord.status = QUEUED
    │       └── WorkerQueueThread.enqueue(task)
    │
    ├── 6. Wait for result (async)
    │       └── TaskRegistry.wait_for_result(task_id)
    │
    └── 7. Return response to client
            └── Same format as direct worker call
```

### Direct Forward Flow (Non-Queued)

```
GET /v1/models (via Proxy)
    │
    ├── 1. Select worker (round-robin by path hash)
    │
    ├── 2. Forward request to worker
    │       └── httpx.request(method, url, headers, body)
    │
    └── 3. Return response directly
            └── No queuing, no task tracking
```

## API Changes

### External API (Simplified via Unified Proxy)

With the unified proxy, clients call standard model endpoints directly:

| Endpoint | Routing | Description |
|----------|---------|-------------|
| `POST /v1/completions` | Queued | LLM text completion (queued, load-balanced) |
| `POST /v1/chat/completions` | Queued | Chat completion (queued, load-balanced) |
| `GET /v1/models` | Direct | List available models |
| `GET /health` | Direct | Worker health check |
| `POST /embeddings` | Direct | Generate embeddings |
| `ANY /{other}` | Direct | Arbitrary model endpoints |

**Legacy Task API** (optional, for backward compatibility):
- `POST /task/submit` - Submit a task with explicit task_id
- `GET /task/info` - Get task information by task_id
- `WS /task/get_result` - Subscribe to task results

### Instance Management API (Enhanced Validation)

Existing endpoints with added model verification:

| Endpoint | Enhancement |
|----------|-------------|
| `POST /instance/register` | Verify `model_id` matches scheduler config |
| `POST /instance/remove` | No change |
| `POST /instance/drain` | No change |
| `GET /instance/list` | No change |

### Unified Transparent Proxy (Entry Point)

| Endpoint | Purpose |
|----------|---------|
| `ANY /{path}` | Routes ALL requests based on path pattern |

## Thread Safety

### Critical Sections

1. **WorkerQueueThread._queue**: Thread-safe `queue.Queue`
2. **WorkerQueueManager._workers**: Protected by `threading.Lock`
3. **TaskRecord updates**: Via `asyncio.run_coroutine_threadsafe()`
4. **InstanceRegistry updates**: Via `asyncio.Lock` (in event loop)

### Thread Interaction

```
Main Event Loop (asyncio)
    │
    ├── POST /task/submit
    │       └── Enqueue task to WorkerQueueThread
    │
    └── TaskResultCallback.handle_result()
            └── Called via run_coroutine_threadsafe()

WorkerQueueThread (dedicated thread per worker)
    │
    ├── Dequeue task
    ├── Execute HTTP to worker
    └── Invoke callback
            └── asyncio.run_coroutine_threadsafe(handle_result, loop)
```

## Configuration

```python
@dataclass
class SchedulerConfig:
    # Model this scheduler handles (empty = set on first registration)
    model_id: str = os.getenv("SCHEDULER_MODEL_ID", "")

    # Enforce single model constraint
    enforce_single_model: bool = True

    # Verify worker endpoints on registration
    verify_endpoints: bool = os.getenv("SCHEDULER_VERIFY_ENDPOINTS", "false").lower() == "true"

    # Enable transparent proxy for non-management requests
    enable_proxy: bool = os.getenv("SCHEDULER_ENABLE_PROXY", "true").lower() == "true"
```

## Error Handling

### Worker Thread Errors

| Error | Handling |
|-------|----------|
| HTTP Timeout | Mark task FAILED, invoke callback |
| HTTP Error (4xx/5xx) | Mark task FAILED, invoke callback |
| Connection Error | Retry with backoff, then FAILED |
| Thread Crash | Log error, restart thread, reschedule pending tasks |

### Model Mismatch Errors

| Error | HTTP Code | Message |
|-------|-----------|---------|
| Wrong model_id on register | 400 | "Model mismatch: expected X, got Y" |
| Endpoint unreachable | 400 | "Worker endpoint unreachable" |

## Migration Path

### Phase 3.1: Core Implementation

1. PYLET-015: Implement `WorkerQueueThread`
2. PYLET-016: Implement `TaskResultCallback`
3. PYLET-017: Implement `WorkerQueueManager`

### Phase 3.2: Integration

4. PYLET-018: Update instance registration with model validation (reuse existing algorithms)
5. PYLET-019: Update API integration
6. PYLET-020: Add transparent proxy, implement graceful shutdown

### Phase 3.3: Testing

7. PYLET-021: Integration tests

## Testing Strategy

### Unit Tests

- `WorkerQueueThread`: Queue operations, task execution, callback invocation
- `TaskResultCallback`: Registry updates, WebSocket notifications
- `WorkerQueueManager`: Worker registration/deregistration, thread lifecycle
- Model validation: Correct model accepted, wrong model rejected

### Integration Tests

- Full task submission flow (submit -> queue -> execute -> callback)
- Worker registration/deregistration with task redistribution
- Transparent proxy forwarding
- Graceful shutdown with pending tasks

## Dependencies

- Python `threading` module (standard library)
- Python `queue` module (standard library)
- `httpx` (synchronous client for worker threads)
- Existing: `TaskRegistry`, `InstanceRegistry`, `ConnectionManager`
- Existing: Scheduling strategies (no modifications)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Thread leaks | Explicit thread join on deregistration |
| Deadlocks | No nested locks, clear lock ordering |
| Memory growth | Bounded task history, cleanup on completion |
| Worker unavailable | Health checks, automatic retry |
| Wrong model registered | Validate model_id on registration |

## References

- [PyLet Migration Overview](../pylet_migration.md)
- [Current Instance Task Queue](../../instance/src/task_queue.py)
- [Current Task Dispatcher](../../scheduler/src/services/task_dispatcher.py)
- [Current Central Queue](../../scheduler/src/services/central_queue.py)
- [Current Scheduling Algorithms](../../scheduler/src/algorithms/)
