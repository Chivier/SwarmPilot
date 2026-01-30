# PYLET-018: Queue-Aware Scheduling Integration

## Status: DONE

## Objective

Integrate the scheduler-side queue depth information into existing scheduling strategies. **This task does NOT add new algorithms** - it only updates existing strategies to receive queue state from `WorkerQueueManager`.

## Key Constraint: Reuse Existing Algorithms

Per design requirements, we **do NOT create new scheduling algorithms**. The existing strategies already handle queue-based scheduling:

| Existing Strategy | Already Handles |
|-------------------|-----------------|
| `MinimumExpectedTimeStrategy` | Queue expected time + error margin |
| `ProbabilisticSchedulingStrategy` | Quantile-based queue distribution |
| `RoundRobinStrategy` | Simple rotation |
| `RandomStrategy` | Random selection |
| `PowerOfTwoStrategy` | Sample-based selection |

These algorithms remain **unchanged**. The only modification is providing them with **scheduler-side queue state** instead of instance-side queue state.

## Prerequisites

- PYLET-014: Design review complete
- PYLET-017: `WorkerQueueManager` implemented

## Design

### Problem Statement

After Phase 3, queue state moves from instance-side to scheduler-side:

| Aspect | Before (Instance-Side) | After (Scheduler-Side) |
|--------|------------------------|------------------------|
| Queue location | Each instance has local queue | Scheduler maintains per-worker queues |
| Queue state source | Instance reports via callback | `WorkerQueueManager.get_queue_depth()` |
| Algorithm logic | **Same** | **Same** |

### Solution: Queue State Adapter

Create a simple adapter that provides queue state in the format existing algorithms expect:

```python
# Existing algorithms expect InstanceQueueExpectError:
@dataclass
class InstanceQueueExpectError:
    expected_time_ms: float
    error_margin_ms: float

# New: Create this from WorkerQueueManager data
def get_queue_info_from_manager(
    worker_queue_manager: WorkerQueueManager,
    instance_id: str,
    avg_exec_time_ms: float,
) -> InstanceQueueExpectError:
    """Convert scheduler-side queue state to algorithm format.

    Args:
        worker_queue_manager: Manager with queue state
        instance_id: Instance to query
        avg_exec_time_ms: Average execution time per task

    Returns:
        Queue info in format existing algorithms expect
    """
    thread = worker_queue_manager.get_worker(instance_id)
    if thread is None:
        return InstanceQueueExpectError(
            expected_time_ms=0.0,
            error_margin_ms=0.0,
        )

    wait_time = thread.get_estimated_wait_time(avg_exec_time_ms)

    return InstanceQueueExpectError(
        expected_time_ms=wait_time,
        error_margin_ms=avg_exec_time_ms * 0.2,  # 20% uncertainty
    )
```

### Integration Point

The change happens at the **call site**, not in the algorithms:

```python
# In Central Dispatcher (task_dispatcher.py)

async def dispatch_task(task: TaskRecord) -> str:
    """Dispatch task to worker using existing scheduling strategy."""

    # Get predictions (unchanged)
    predictions = await prediction_service.get_predictions(
        task.model_id,
        task.metadata,
    )

    # Get queue info FROM SCHEDULER-SIDE (changed)
    # Before: came from instance callbacks
    # After: comes from WorkerQueueManager
    queue_info = {}
    for instance_id in predictions.keys():
        queue_info[instance_id] = get_queue_info_from_manager(
            worker_queue_manager,
            instance_id,
            avg_exec_time_ms=predictions[instance_id].predicted_time_ms,
        )

    # Use EXISTING algorithm (unchanged)
    selected = scheduling_strategy.select_instance(
        predictions=predictions,
        queue_info=queue_info,  # Now from scheduler-side
    )

    return selected
```

### What Changes vs. What Stays Same

| Component | Changes? | Description |
|-----------|----------|-------------|
| `MinimumExpectedTimeStrategy` | **No** | Algorithm unchanged |
| `ProbabilisticSchedulingStrategy` | **No** | Algorithm unchanged |
| `RoundRobinStrategy` | **No** | Algorithm unchanged |
| `RandomStrategy` | **No** | Algorithm unchanged |
| `PowerOfTwoStrategy` | **No** | Algorithm unchanged |
| Queue info source | **Yes** | From `WorkerQueueManager` instead of instance callbacks |
| Task dispatcher | **Yes** | Calls `WorkerQueueManager.get_queue_depth()` |

### Helper Methods on Base Class

Add helper methods to `SchedulingStrategy` base class for convenience:

```python
# In src/algorithms/base.py

class SchedulingStrategy(ABC):
    """Base class for scheduling strategies."""

    def __init__(self):
        self._worker_queue_manager: WorkerQueueManager | None = None

    def set_worker_queue_manager(
        self,
        manager: "WorkerQueueManager",
    ) -> None:
        """Set the worker queue manager for queue state queries.

        This allows strategies to access scheduler-side queue depth
        if needed. Most strategies will receive queue_info as a
        parameter and don't need to call this directly.

        Args:
            manager: WorkerQueueManager instance
        """
        self._worker_queue_manager = manager

    def get_scheduler_queue_depth(self, instance_id: str) -> int:
        """Get scheduler-side queue depth for an instance.

        Args:
            instance_id: Instance to query

        Returns:
            Queue depth (0 if manager not set or instance not found)
        """
        if self._worker_queue_manager is None:
            return 0
        return self._worker_queue_manager.get_queue_depth(instance_id)
```

## Implementation Steps

1. [x] Add `set_worker_queue_manager()` to `SchedulingStrategy` base class
2. [x] Add `get_scheduler_queue_depth()` helper method
3. [x] Create `get_queue_info_from_manager()` adapter function
4. [x] Create `get_all_queue_info_from_manager()` batch adapter
5. [x] Update exports in `algorithms/__init__.py`
6. [x] Write tests to verify queue state flows correctly (13 tests)

## Testing

### Unit Tests

```python
def test_queue_info_adapter():
    """Test adapter converts manager state to algorithm format."""

def test_strategy_receives_scheduler_queue_state():
    """Test scheduling uses scheduler-side queue depth."""

def test_graceful_degradation_without_queue_manager():
    """Test strategies work when manager not set."""
```

### Integration Tests

```python
def test_load_balancing_with_scheduler_queue():
    """Test tasks are balanced using scheduler queue state."""

def test_scheduling_prefers_shorter_queues():
    """Test workers with shorter queues are preferred."""
```

## Acceptance Criteria

- [x] Existing algorithms remain **unchanged**
- [x] Queue state comes from `WorkerQueueManager`
- [x] Adapter function converts queue state to expected format
- [x] Strategies can access queue manager via helper methods
- [x] Graceful degradation when manager not set
- [x] Unit tests pass (13 tests)

## References

- [PYLET-014: Design Overview](PYLET-014-scheduler-task-queue.md)
- [PYLET-017: Worker Queue Manager](PYLET-017-worker-queue-manager.md)
- [Current MinExpectedTime Strategy](../../scheduler/src/algorithms/min_expected_time.py)
- [Current Probabilistic Strategy](../../scheduler/src/algorithms/probabilistic.py)
