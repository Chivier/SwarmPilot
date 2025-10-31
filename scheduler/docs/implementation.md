# Implementation Details

This document describes the implementation details of the scheduler service, particularly focusing on queue management and task lifecycle.

## Queue Update Implementation

The scheduler implements dynamic queue information updates at two critical points:

1. **Task Submission** - When a task is assigned to an instance
2. **Task Completion** - When an instance finishes executing a task

### Task Prediction Storage

To enable accurate queue updates on task completion, prediction information is stored with each task record:

```python
class TaskRecord:
    # ... other fields ...

    # Prediction information (for queue updates on task completion)
    predicted_time_ms: Optional[float]
    predicted_error_margin_ms: Optional[float]
    predicted_quantiles: Optional[Dict[float, float]]
```

These fields are populated during task submission (`src/api.py:409-424`) from the predictor service response.

### Queue Update on Task Submission

When a task is submitted and assigned to an instance, the queue is updated based on the scheduling strategy:

#### Shortest Queue Strategy

**Location**: `src/api.py:433-458`

**Formula**:
```
new_expected = current_expected + predicted_time_ms
new_error = sqrt(current_error² + predicted_error²)
```

**Implementation**:
```python
new_expected = current_queue.expected_time_ms + task_expected
new_error = math.sqrt(
    current_queue.error_margin_ms ** 2 + task_error ** 2
)
```

#### Probabilistic Strategy

**Location**: `src/api.py:460-513`

**Method**: Monte Carlo sampling (1000 samples)

**Implementation**:
1. Sample from current queue distribution
2. Sample from task prediction distribution
3. Add samples: `total_samples = queue_samples + task_samples`
4. Compute new quantiles from total samples

### Queue Update on Task Completion

When a task completes, the queue is updated to reflect the actual execution time.

**Location**: `src/task_dispatcher.py:174-182, 240-335`

#### Shortest Queue Strategy

**Formula**:
```
new_expected = current_expected - predicted_time_ms + actual_time_ms
error remains unchanged
```

**Implementation** (`src/task_dispatcher.py:266-278`):
```python
new_expected = current_queue.expected_time_ms - predicted_time_ms + actual_time_ms
new_expected = max(0.0, new_expected)  # Ensure non-negative

updated_queue = InstanceQueueExpectError(
    instance_id=instance_id,
    expected_time_ms=new_expected,
    error_margin_ms=current_queue.error_margin_ms,  # Keep unchanged
)
```

**Rationale**:
- The expected time is adjusted by removing the predicted contribution and adding the actual time
- The error margin represents the uncertainty in the queue state and is not updated based on individual task completions

#### Probabilistic Strategy

**Method**: Monte Carlo sampling (1000 samples)

**Implementation** (`src/task_dispatcher.py:280-335`):
1. Sample from current queue distribution
2. Sample from predicted task distribution
3. Compute updated queue: `updated_samples = queue_samples - task_samples + actual_time_ms`
4. Ensure non-negative values
5. Compute new quantiles from updated samples

**Fallback**: If quantile predictions are unavailable, use simple arithmetic:
```python
updated_values = [
    max(0.0, current_queue.values[i] - predicted_time_ms + actual_time_ms)
    for i in range(len(current_queue.quantiles))
]
```

## Task Lifecycle

### 1. Task Submission
- Client submits task via `/task/submit`
- Scheduler fetches predictions from predictor service
- Scheduling strategy selects best instance
- Task record created with prediction information
- Queue updated (add predicted task)
- Task dispatched to instance asynchronously

### 2. Task Execution
- Instance receives task via `/task/submit` endpoint
- Instance executes task
- Instance sends result via callback to `/callback/task_result`

### 3. Task Completion
- Scheduler receives callback with actual execution time
- Task status updated to COMPLETED or FAILED
- Queue updated (remove predicted task, add actual time)
- Training data collected (if enabled)
- WebSocket subscribers notified

## Code Structure

### Key Files

- `src/task_registry.py` - Task record storage and management
- `src/instance_registry.py` - Instance and queue information storage
- `src/task_dispatcher.py` - Task dispatching and completion handling
- `src/api.py` - API endpoints and request handling
- `src/scheduler.py` - Scheduling strategy implementations

### Key Methods

- `TaskRegistry.create_task()` - Create task with prediction info
- `TaskDispatcher.handle_task_result()` - Handle task completion
- `TaskDispatcher._update_queue_on_completion()` - Update queue on completion
- `InstanceRegistry.update_queue_info()` - Thread-safe queue update

## Testing

All implementations maintain backward compatibility. The `task_registry` test suite (35 tests) passes completely with the new prediction fields.

Key test coverage:
- Task creation with and without prediction info
- Queue updates on task submission
- Queue updates on task completion
- Concurrent operations safety
