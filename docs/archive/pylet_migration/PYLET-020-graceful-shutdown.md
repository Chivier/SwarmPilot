# PYLET-020: Graceful Shutdown

## Status: DONE

## Objective

Implement graceful shutdown handling for the scheduler-side task queue system, ensuring tasks are properly handled during instance removal (via `POST /instance/sync`) and scheduler shutdown.

## Prerequisites

- PYLET-017: `WorkerQueueManager` implemented
- PYLET-019: API integration complete (including `POST /instance/sync`)

## Design

### Shutdown Scenarios

1. **Instance Removal via Sync**: Planner sends new instance list via `POST /instance/sync`, scheduler removes instances not in new list
2. **Model Scale-Down**: Multiple instances removed in single sync call
3. **Scheduler Shutdown**: Entire scheduler shutting down
4. **Network Error Recovery**: Worker becomes temporarily unavailable (task stays in queue with retry)

### Key Design Points

| Aspect | Behavior |
|--------|----------|
| **Instance Removal** | Triggered by `POST /instance/sync` - Planner cancels PyLet instance first, then sends new list |
| **Task Rescheduling** | Unfinished tasks from removed instances are rescheduled to other workers using existing scheduling algorithm |
| **Priority Preservation** | Rescheduled tasks keep original `enqueue_time` for correct priority ordering |
| **Network Error** | Tasks stay in queue with retry flag; queue does NOT exit on network errors |
| **No Central Queue** | Tasks either reschedule to other workers or fail (Planner manages all instances) |

### Transparent Proxy Considerations

The scheduler supports a transparent proxy for non-management requests (see [PYLET-014](PYLET-014-scheduler-task-queue.md)). During shutdown:

| Scenario | Behavior |
|----------|----------|
| Instance removed via sync | Tasks rescheduled to remaining workers |
| No workers available | Proxy returns 503 "No workers available" |
| Scheduler shutdown | Proxy requests immediately rejected |

**Key point**: The transparent proxy is stateless and doesn't queue requests - it forwards immediately. Task-based requests go through `WorkerQueueThread` (priority queue) and are properly rescheduled.

### Instance Removal Flow (via /instance/sync)

```
1. POST /instance/sync received with new target list
   │
   └── Scheduler computes diff: to_add, to_remove
   │
   ▼
2. For each instance to remove:
   │
   ├── Get all unfinished tasks (queued + running)
   └── Stop worker queue thread (short timeout)
   │
   ▼
3. Reschedule unfinished tasks
   │
   ├── For each task: run scheduling algorithm
   ├── Enqueue to selected worker (keeps original enqueue_time)
   └── If no workers available → mark task FAILED
   │
   ▼
4. Remove from registry
   │
   ▼
5. For each instance to add:
   │
   ├── Register in instance registry
   └── Create priority queue thread
```

### Network Error Handling in Queue

```
1. Task dispatch fails with NetworkError
   │
   ├── Increment retry count in task metadata
   └── Check if retries < max_retries
   │
   ▼
2. If retries available:
   │
   ├── Wait (backoff: base_delay * retry_count)
   └── Re-enqueue task (same enqueue_time = same priority)
   │
   ▼
3. If max retries exceeded:
   │
   └── Invoke callback with FAILED status
```

### Implementation

```python
# In scheduler/src/services/shutdown_handler.py

class InstanceRemovalHandler:
    """Handles instance removal and task rescheduling.

    Used by POST /instance/sync when instances are removed from target list.
    Also used during scheduler shutdown.
    """

    def __init__(
        self,
        worker_queue_manager: WorkerQueueManager,
        instance_registry: InstanceRegistry,
        scheduling_strategy: SchedulingStrategy,
        task_registry: TaskRegistry,
    ):
        self.worker_queue_manager = worker_queue_manager
        self.instance_registry = instance_registry
        self.scheduling_strategy = scheduling_strategy
        self.task_registry = task_registry

    async def remove_instance(
        self,
        instance_id: str,
        stop_timeout: float = 5.0,
    ) -> dict:
        """Remove an instance and reschedule its unfinished tasks.

        Called by POST /instance/sync when an instance is not in target list.
        Planner should cancel PyLet instance BEFORE calling this.

        Args:
            instance_id: Instance to remove
            stop_timeout: Max time to wait for thread to stop

        Returns:
            Summary: tasks_rescheduled, tasks_failed
        """
        logger.info(f"Removing instance {instance_id}")

        result = {
            "instance_id": instance_id,
            "tasks_rescheduled": 0,
            "tasks_failed": 0,
        }

        # 1. Get all unfinished tasks from this worker's queue
        unfinished_tasks = self.worker_queue_manager.get_unfinished_tasks(
            instance_id
        )

        # 2. Stop worker queue thread (don't wait for current task)
        self.worker_queue_manager.deregister_worker(
            instance_id,
            stop_timeout=stop_timeout,
        )

        # 3. Reschedule each task to other workers
        for task in unfinished_tasks:
            success = await self._reschedule_task(
                task,
                exclude_instance=instance_id,
            )
            if success:
                result["tasks_rescheduled"] += 1
            else:
                result["tasks_failed"] += 1

        # 4. Remove from registry
        await self.instance_registry.remove(instance_id)

        logger.info(
            f"Instance {instance_id} removed: "
            f"{result['tasks_rescheduled']} rescheduled, "
            f"{result['tasks_failed']} failed"
        )

        return result

    async def _reschedule_task(
        self,
        task: QueuedTask,
        exclude_instance: str,
    ) -> bool:
        """Reschedule a task to another worker using existing scheduling algorithm.

        Args:
            task: Task to reschedule (keeps original enqueue_time for priority)
            exclude_instance: Instance to exclude from selection

        Returns:
            True if rescheduled, False if failed (no workers available)
        """
        # Get available instances (excluding the removed one)
        available = await self.instance_registry.get_active_instances(
            task.model_id
        )
        available = [i for i in available if i.instance_id != exclude_instance]

        if not available:
            # No other workers - mark task as failed
            await self.task_registry.update_status(
                task.task_id,
                TaskStatus.FAILED,
                error="No workers available after instance removal",
            )
            logger.warning(
                f"Task {task.task_id} failed: no available workers"
            )
            return False

        # Use existing scheduling algorithm to select new worker
        try:
            schedule_result = await self.scheduling_strategy.schedule_task(
                model_id=task.model_id,
                metadata=task.metadata,
                available_instances=available,
            )

            if schedule_result.selected_instance_id:
                # Enqueue to new worker (task keeps original enqueue_time)
                self.worker_queue_manager.enqueue_task(
                    schedule_result.selected_instance_id,
                    task,  # Original enqueue_time preserved for priority
                )

                # Update task record with new assignment
                await self.task_registry.update_assignment(
                    task.task_id,
                    schedule_result.selected_instance_id,
                )

                logger.info(
                    f"Task {task.task_id} rescheduled to "
                    f"{schedule_result.selected_instance_id}"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to reschedule task {task.task_id}: {e}")

        # Scheduling failed - mark task as failed
        await self.task_registry.update_status(
            task.task_id,
            TaskStatus.FAILED,
            error="Scheduling failed during rescheduling",
        )
        return False

    async def shutdown_all(
        self,
        timeout: float = 60.0,
    ) -> dict:
        """Gracefully shutdown all workers (scheduler shutdown only).

        Note: This is different from instance removal via /instance/sync.
        During scheduler shutdown, tasks are NOT rescheduled (no workers left).

        Args:
            timeout: Total timeout for shutdown

        Returns:
            Summary: workers_stopped, tasks_dropped
        """
        logger.info("Starting graceful shutdown of all workers...")

        result = {
            "workers_stopped": 0,
            "tasks_dropped": 0,
            "timeout_occurred": False,
        }

        start = time.time()
        worker_ids = self.worker_queue_manager.get_worker_ids()

        for worker_id in worker_ids:
            remaining_time = timeout - (time.time() - start)
            if remaining_time <= 0:
                result["timeout_occurred"] = True
                logger.warning("Shutdown timeout, forcing remaining workers")
                break

            per_worker_timeout = remaining_time / max(
                len(worker_ids) - result["workers_stopped"],
                1,
            )

            try:
                # Get unfinished tasks count before stopping
                unfinished = self.worker_queue_manager.get_unfinished_tasks(
                    worker_id
                )
                result["tasks_dropped"] += len(unfinished)

                # Stop worker queue thread
                self.worker_queue_manager.deregister_worker(
                    worker_id,
                    stop_timeout=per_worker_timeout,
                )

                # Remove from registry
                await self.instance_registry.remove(worker_id)
                result["workers_stopped"] += 1

            except Exception as e:
                logger.error(f"Error stopping worker {worker_id}: {e}")

        logger.info(
            f"Shutdown complete: {result['workers_stopped']} workers stopped, "
            f"{result['tasks_dropped']} tasks dropped"
        )

        return result
```

### Integration with FastAPI Lifespan

```python
# In api.py

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting scheduler...")
    # ... initialization ...

    # Create instance removal handler
    global instance_removal_handler
    instance_removal_handler = InstanceRemovalHandler(
        worker_queue_manager=worker_queue_manager,
        instance_registry=instance_registry,
        scheduling_strategy=scheduling_strategy,
        task_registry=task_registry,
    )

    yield

    # Shutdown
    logger.info("Shutting down scheduler...")
    await instance_removal_handler.shutdown_all(timeout=60.0)
    logger.info("Scheduler shutdown complete")
```

### Integration with /instance/sync

```python
# In api.py - POST /instance/sync

async def _handle_instance_removal(instance_id: str) -> int:
    """Remove instance via InstanceRemovalHandler."""
    result = await instance_removal_handler.remove_instance(
        instance_id,
        stop_timeout=5.0,
    )
    return result["tasks_rescheduled"]
```

## Implementation Steps

1. [x] Create `src/services/shutdown_handler.py`
2. [x] Implement `ShutdownHandler` class
3. [x] Implement `ShutdownResult` dataclass
4. [x] Implement `shutdown_all()` method with timeout distribution
5. [x] Add optional on_shutdown_complete callback
6. [x] Export from `services/__init__.py`
7. [x] Write unit tests (9 tests)

Note: Task rescheduling during instance removal is handled by PYLET-019 (`instance_sync.py`)

## Testing

### Unit Tests

```python
def test_remove_instance_reschedules_tasks():
    """Test instance removal reschedules unfinished tasks."""

def test_reschedule_preserves_enqueue_time():
    """Test rescheduled tasks keep original enqueue_time."""

def test_reschedule_uses_scheduling_algorithm():
    """Test rescheduling uses existing scheduling strategy."""

def test_reschedule_fails_when_no_workers():
    """Test task marked FAILED when no workers available."""

def test_shutdown_all_stops_all_workers():
    """Test shutdown stops all workers."""

def test_shutdown_timeout_forces_stop():
    """Test timeout triggers forced stop."""
```

### Integration Tests

```python
def test_instance_sync_removal_reschedules():
    """Test /instance/sync removal triggers task rescheduling."""

def test_priority_preserved_after_reschedule():
    """Test rescheduled tasks maintain priority order."""

def test_shutdown_during_load():
    """Test shutdown under task load drops tasks correctly."""

def test_network_error_retry_in_queue():
    """Test network errors trigger retry, not immediate failure."""
```

## Acceptance Criteria

- [x] Instance removal (via sync) reschedules unfinished tasks (PYLET-019)
- [x] Scheduler shutdown drops tasks (no rescheduling)
- [x] Timeout is distributed across workers proportionally
- [x] On_shutdown_complete callback invoked when provided
- [x] Handles registry errors gracefully
- [x] Unit tests pass (9 tests)

Note: Network retry logic is in WorkerQueueThread (PYLET-015)

## References

- [PYLET-014: Design Overview](PYLET-014-scheduler-task-queue.md)
- [PYLET-017: Worker Queue Manager](PYLET-017-worker-queue-manager.md)
- [PYLET-019: API Integration](PYLET-019-api-integration.md)
