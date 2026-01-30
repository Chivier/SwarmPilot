# PYLET-019: API Integration

## Status: DONE

## Objective

Implement the **Unified Transparent Proxy** architecture where ALL requests enter through a single proxy endpoint. ALL non-management requests are routed through the Dispatcher to Worker Queues. Instance management uses a **declarative sync API** where the Planner submits the complete target instance list.

## Prerequisites

- PYLET-015: `WorkerQueueThread` implemented
- PYLET-016: `TaskResultCallback` implemented
- PYLET-017: `WorkerQueueManager` implemented
- PYLET-018: Queue-aware scheduling implemented

## Design

### Unified Proxy Architecture

ALL requests enter through the Transparent Proxy, which routes them:

| Path Pattern | Routing | Description |
|--------------|---------|-------------|
| `/instance/sync`, `/health`, `/info` | Management API | Scheduler handles directly |
| ALL other paths | Dispatcher → Queue | Queued & load-balanced to workers |

**Key Design Point**: The scheduler is model-agnostic. It handles exactly ONE model type (configured at startup), which could be LLM, image generation, embeddings, or any other model type. ALL non-management requests go through the Dispatcher regardless of model type.

### Changes Overview

1. **Unified Proxy**: Single entry point routes ALL requests by path
2. **Instance Sync API**: `POST /instance/sync` - Planner submits target instance list, scheduler computes diff
3. **Instance Addition**: Create `WorkerQueueThread` for new instances
4. **Instance Removal**: Stop thread and reschedule unfinished tasks to other queues
5. **Dispatcher**: Routes ALL non-management requests to Worker Queues

### Unified Transparent Proxy Implementation

```python
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def transparent_proxy(request: Request, path: str):
    """Unified entry point for ALL requests.

    Routes requests based on path:
    1. Management endpoints → handled by scheduler directly
    2. ALL other endpoints → Dispatcher → Worker Queue (queued & load-balanced)

    The scheduler is model-agnostic and handles exactly ONE model type
    (configured at startup). ALL non-management requests go through
    the Dispatcher regardless of the specific endpoint path.
    """
    # 1. Management endpoints - let FastAPI handle via normal routes
    management_prefixes = ["instance/", "health", "info", "docs", "openapi"]
    if any(path.startswith(prefix) for prefix in management_prefixes):
        raise HTTPException(status_code=404, detail="Not found")

    # Get available workers
    instances = await instance_registry.list_active()
    if not instances:
        raise HTTPException(status_code=503, detail="No workers available")

    # 2. ALL other requests → Dispatcher → Worker Queue
    return await _dispatch_request(request, path, instances)


async def _dispatch_request(
    request: Request,
    path: str,
    instances: list,
) -> Response:
    """Route request through Dispatcher to Worker Queue.

    ALL non-management requests are queued and load-balanced.
    This applies to any model type (LLM, image gen, embeddings, etc.).
    """
    # Parse body if present
    body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.json()
        except:
            body = await request.body()

    # Create task
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
        raise HTTPException(status_code=503, detail="All workers busy")

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

### Instance Sync API

The Planner uses `POST /instance/sync` to submit the complete target instance list. The scheduler computes the diff and handles additions/removals internally.

```python
@dataclass
class InstanceInfo:
    instance_id: str
    endpoint: str
    model_id: str

@dataclass
class InstanceSyncRequest:
    instances: list[InstanceInfo]  # Complete target list

@dataclass
class InstanceSyncResponse:
    success: bool
    added: list[str]      # Instance IDs that were added
    removed: list[str]    # Instance IDs that were removed
    rescheduled: int      # Number of tasks rescheduled from removed instances


@app.post("/instance/sync")
async def sync_instances(request: InstanceSyncRequest) -> InstanceSyncResponse:
    """Submit target instance list. Scheduler computes diff and handles changes.

    The Planner should:
    1. Cancel PyLet instances BEFORE calling this endpoint
    2. Send the complete target instance list
    3. Scheduler handles task rescheduling for removed instances
    """
    # 1. Validate all instances have correct model_id
    for inst in request.instances:
        if inst.model_id != config.model_id:
            raise HTTPException(
                status_code=400,
                detail=f"Model mismatch: {inst.model_id} != {config.model_id}",
            )

    # 2. Compute diff: what to add, what to remove
    current_ids = set(instance_registry.get_all_ids())
    target_ids = {inst.instance_id for inst in request.instances}

    to_add = target_ids - current_ids
    to_remove = current_ids - target_ids

    response = InstanceSyncResponse(
        success=True,
        added=[],
        removed=[],
        rescheduled=0,
    )

    # 3. Remove instances first (reschedule their tasks)
    for instance_id in to_remove:
        rescheduled = await _handle_instance_removal(instance_id)
        response.removed.append(instance_id)
        response.rescheduled += rescheduled

    # 4. Add new instances (create queues)
    target_map = {inst.instance_id: inst for inst in request.instances}
    for instance_id in to_add:
        inst = target_map[instance_id]
        await _handle_instance_addition(inst)
        response.added.append(instance_id)

    return response


async def _handle_instance_addition(inst: InstanceInfo) -> None:
    """Add a new instance: register and create queue thread."""
    # Register in instance registry
    await instance_registry.register(
        instance_id=inst.instance_id,
        endpoint=inst.endpoint,
        model_id=inst.model_id,
    )

    # Create worker queue thread (priority queue ordered by enqueue_time)
    worker_queue_manager.register_worker(
        worker_id=inst.instance_id,
        worker_endpoint=inst.endpoint,
        model_id=inst.model_id,
    )

    logger.info(f"Added instance {inst.instance_id} at {inst.endpoint}")


async def _handle_instance_removal(instance_id: str) -> int:
    """Remove an instance: stop queue and reschedule unfinished tasks.

    Returns:
        Number of tasks rescheduled
    """
    # 1. Get all unfinished tasks from this worker's queue
    unfinished_tasks = worker_queue_manager.get_unfinished_tasks(instance_id)

    # 2. Stop worker queue thread (don't wait for current task)
    worker_queue_manager.deregister_worker(instance_id, stop_timeout=5.0)

    # 3. Reschedule each task to other workers using scheduling algorithm
    rescheduled = 0
    for task in unfinished_tasks:
        success = await _reschedule_task(task, exclude_instance=instance_id)
        if success:
            rescheduled += 1
        else:
            # No other workers available - mark task as failed
            logger.warning(f"Cannot reschedule task {task.task_id}: no workers")
            await task_registry.update_status(
                task.task_id,
                TaskStatus.FAILED,
                error="No workers available after instance removal",
            )

    # 4. Remove from registry
    await instance_registry.remove(instance_id)

    logger.info(
        f"Removed instance {instance_id}, rescheduled {rescheduled} tasks"
    )
    return rescheduled


async def _reschedule_task(task: QueuedTask, exclude_instance: str) -> bool:
    """Reschedule a task to another worker using existing scheduling algorithm.

    Returns:
        True if successfully rescheduled, False if no workers available
    """
    # Get available instances (excluding the removed one)
    available = await instance_registry.get_active_instances(task.model_id)
    available = [i for i in available if i.instance_id != exclude_instance]

    if not available:
        return False

    # Use existing scheduling algorithm to select new worker
    schedule_result = await scheduling_strategy.schedule_task(
        model_id=task.model_id,
        metadata=task.metadata,
        available_instances=available,
    )

    if not schedule_result.selected_instance_id:
        return False

    # Enqueue to new worker (keeps original enqueue_time for priority)
    worker_queue_manager.enqueue_task(
        schedule_result.selected_instance_id,
        task,  # Task keeps original enqueue_time
    )

    # Update task record with new assignment
    await task_registry.update_assignment(
        task.task_id,
        schedule_result.selected_instance_id,
    )

    logger.info(
        f"Rescheduled task {task.task_id} to {schedule_result.selected_instance_id}"
    )
    return True
```

### Task Submission Changes

The task submission flow changes from:
1. Create task record → 2. Enqueue to central queue → 3. Central queue dispatches via HTTP

To:
1. Create task record → 2. Select worker → 3. Enqueue to worker's thread

```python
@app.post("/task/submit")
async def submit_task(request: TaskSubmitRequest):
    # 1. Validate task doesn't exist
    existing = await task_registry.get(request.task_id)
    if existing:
        raise HTTPException(status_code=409, detail="Task already exists")

    # 2. Create task record
    task_record = TaskRecord(
        task_id=request.task_id,
        model_id=request.model_id,
        task_input=request.task_input,
        metadata=request.metadata,
        status=TaskStatus.PENDING,
    )
    await task_registry.add(task_record)

    # 3. Get available instances for this model
    available_instances = await instance_registry.get_active_instances(
        request.model_id
    )

    if not available_instances:
        # No workers available - queue in central queue
        position = await central_queue.enqueue(
            task_id=request.task_id,
            model_id=request.model_id,
            task_input=request.task_input,
            metadata=request.metadata,
        )
        return TaskSubmitResponse(
            success=True,
            message=f"Task queued at position {position}, waiting for worker",
            task=task_record.to_response(),
        )

    # 4. Select worker using scheduling strategy
    schedule_result = await scheduling_strategy.schedule_task(
        model_id=request.model_id,
        metadata=request.metadata,
        available_instances=available_instances,
    )

    if not schedule_result.selected_instance_id:
        # Scheduling failed - queue in central queue
        position = await central_queue.enqueue(
            task_id=request.task_id,
            model_id=request.model_id,
            task_input=request.task_input,
            metadata=request.metadata,
        )
        return TaskSubmitResponse(
            success=True,
            message=f"Task queued at position {position}",
            task=task_record.to_response(),
        )

    # 5. Update task record with assignment
    task_record.assigned_instance = schedule_result.selected_instance_id
    task_record.status = TaskStatus.QUEUED

    if schedule_result.selected_prediction:
        task_record.predicted_time_ms = (
            schedule_result.selected_prediction.predicted_time_ms
        )
        task_record.predicted_error_margin_ms = (
            schedule_result.selected_prediction.error_margin_ms
        )

    # 6. Enqueue to worker's thread
    queued_task = QueuedTask(
        task_id=request.task_id,
        model_id=request.model_id,
        task_input=request.task_input,
        metadata=request.metadata,
        enqueue_time=time.time(),
        predicted_time_ms=task_record.predicted_time_ms,
    )

    queue_position = worker_queue_manager.enqueue_task(
        schedule_result.selected_instance_id,
        queued_task,
    )

    # 7. Update instance pending count
    await instance_registry.increment_pending(
        schedule_result.selected_instance_id
    )

    return TaskSubmitResponse(
        success=True,
        message=f"Task assigned to {schedule_result.selected_instance_id}, "
                f"queue position {queue_position}",
        task=task_record.to_response(),
    )
```

### Central Queue Changes

The `CentralTaskQueue` now only handles tasks waiting for workers:

```python
class CentralTaskQueue:
    """Queue for tasks waiting for available workers.

    After Phase 3, this queue only holds tasks for models that have
    no registered workers. When a worker registers, pending tasks
    are dispatched to worker queue threads.
    """

    async def notify_worker_registered(self, model_id: str) -> None:
        """Called when a new worker registers for a model.

        Dispatches any pending tasks for this model to available workers.
        """
        async with self._queue_lock:
            # Find tasks for this model
            tasks_to_dispatch = []
            remaining_tasks = []

            for task in self._queue:
                if task.model_id == model_id:
                    tasks_to_dispatch.append(task)
                else:
                    remaining_tasks.append(task)

            self._queue = deque(remaining_tasks)

        # Dispatch to workers
        for task in tasks_to_dispatch:
            # Re-submit through task submission flow
            await self._resubmit_task(task)
```

### Initialization Changes

```python
# In api.py lifespan or startup

# Create callback handler
task_result_callback = TaskResultCallback(
    task_registry=task_registry,
    instance_registry=instance_registry,
    websocket_manager=websocket_manager,
    throughput_tracker=throughput_tracker,
)

# Get main event loop
loop = asyncio.get_event_loop()

# Create worker queue manager
worker_queue_manager = WorkerQueueManager(
    callback=task_result_callback.create_thread_callback(loop),
    http_timeout=config.task_timeout,
)

# Configure scheduling strategy
scheduling_strategy.set_worker_queue_manager(worker_queue_manager)
```

## Implementation Steps

1. [x] Create `src/instance_sync.py` module with core logic
2. [x] Implement `InstanceInfo` and `InstanceSyncRequest` dataclasses
3. [x] Implement `handle_instance_sync()` function
4. [x] Implement `handle_instance_addition()` function
5. [x] Implement `handle_instance_removal()` function
6. [x] Implement `reschedule_task()` function
7. [x] Write unit tests (15 tests)

## Testing

### Unit Tests

```python
def test_sync_adds_new_instances():
    """Test sync creates queue threads for new instances."""

def test_sync_removes_instances():
    """Test sync stops queue threads for removed instances."""

def test_sync_reschedules_unfinished_tasks():
    """Test removed instance tasks are rescheduled to other queues."""

def test_sync_validates_model_id():
    """Test sync rejects instances with wrong model_id."""

def test_reschedule_preserves_enqueue_time():
    """Test rescheduled tasks keep original enqueue_time for priority."""

def test_reschedule_uses_scheduling_algorithm():
    """Test rescheduling uses existing scheduling strategy."""
```

### Integration Tests

```python
def test_full_task_flow():
    """Test submit → queue → execute → callback → result."""

def test_task_rescheduling_on_instance_removal():
    """Test tasks are rescheduled when instance is removed via sync."""

def test_multiple_workers_load_balance():
    """Test tasks distributed across workers."""

def test_priority_queue_ordering():
    """Test tasks processed in enqueue_time order (earlier = first)."""

def test_network_error_retry():
    """Test tasks stay in queue with retry on network errors."""
```

## Acceptance Criteria

- [x] `handle_instance_sync()` computes diff between current and target instances
- [x] Removed instance tasks are rescheduled using existing scheduling algorithm
- [x] Rescheduled tasks preserve original `enqueue_time` for priority ordering
- [x] Model ID validation ensures all instances match config
- [x] Removals happen before additions for clean state
- [x] Unit tests pass (15 tests)

## References

- [PYLET-014: Design Overview](PYLET-014-scheduler-task-queue.md)
- [Current API](../../scheduler/src/api.py)
