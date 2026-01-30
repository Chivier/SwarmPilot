# PYLET-021: Phase 3 Integration Tests

## Status: DONE

## Objective

Comprehensive integration testing for the scheduler-side task queue system, verifying end-to-end functionality and edge cases.

## Prerequisites

- All Phase 3 implementation tasks complete (PYLET-015 through PYLET-020)

## Test Categories

### 1. Basic Task Flow

| Test | Description |
|------|-------------|
| `test_submit_execute_complete` | Full task lifecycle: submit → queue → execute → result |
| `test_submit_to_specific_worker` | Task routed to correct worker |
| `test_task_result_via_websocket` | Result notification via WebSocket |
| `test_task_status_transitions` | Status changes: PENDING → QUEUED → RUNNING → COMPLETED |

### 2. Queue Behavior

| Test | Description |
|------|-------------|
| `test_fifo_ordering` | Tasks processed in FIFO order |
| `test_queue_depth_tracking` | Queue depth accurately reported |
| `test_estimated_wait_time` | Wait time estimates reasonable |
| `test_queue_under_load` | Queue handles high task volume |

### 3. Multi-Worker Scenarios

| Test | Description |
|------|-------------|
| `test_load_balancing` | Tasks distributed across workers |
| `test_queue_aware_scheduling` | Shorter queues preferred |
| `test_worker_isolation` | Slow worker doesn't affect others |
| `test_same_model_multiple_workers` | Tasks balanced for same model |

### 4. Worker Registration/Deregistration

| Test | Description |
|------|-------------|
| `test_register_creates_queue` | Registration creates worker thread |
| `test_deregister_redistributes` | Deregistration redistributes tasks |
| `test_deregister_during_execution` | Drain waits for current task |
| `test_pending_tasks_on_new_worker` | Pending tasks dispatched on registration |

### 5. Error Handling

| Test | Description |
|------|-------------|
| `test_worker_timeout` | Task fails on worker timeout |
| `test_worker_error_response` | Task fails on worker HTTP error |
| `test_worker_connection_retry` | Transient errors trigger retry |
| `test_task_failure_notification` | Failed tasks notify subscribers |

### 6. Graceful Shutdown

| Test | Description |
|------|-------------|
| `test_drain_completes_task` | Drain waits for current task |
| `test_drain_timeout` | Drain respects timeout |
| `test_shutdown_all_workers` | Scheduler shutdown stops all |
| `test_shutdown_redistributes_tasks` | Shutdown preserves pending tasks |

### 7. Edge Cases

| Test | Description |
|------|-------------|
| `test_no_workers_queues_task` | Task queued when no workers |
| `test_duplicate_task_rejected` | Duplicate task_id rejected |
| `test_unknown_model_id` | Graceful handling of unknown model |
| `test_concurrent_submits` | Concurrent submissions handled |
| `test_rapid_register_deregister` | Rapid worker churn |

## Test Implementation

### Fixtures

```python
# tests/integration/test_phase3.py

import pytest
import asyncio
import httpx
from unittest.mock import Mock, AsyncMock

@pytest.fixture
async def scheduler_app():
    """Create test scheduler application."""
    from src.api import create_app
    app = await create_app()
    yield app
    # Cleanup

@pytest.fixture
async def mock_worker():
    """Create mock worker that responds to requests."""
    class MockWorker:
        def __init__(self, port: int):
            self.port = port
            self.endpoint = f"http://localhost:{port}"
            self.requests = []
            self.response_delay = 0.1

        async def handle_request(self, request):
            self.requests.append(request)
            await asyncio.sleep(self.response_delay)
            return {"id": "test", "choices": [{"text": "response"}]}

    # Start mock server
    worker = MockWorker(port=9999)
    # ... server setup ...
    yield worker
    # Cleanup

@pytest.fixture
async def client(scheduler_app):
    """HTTP client for scheduler API."""
    async with httpx.AsyncClient(
        app=scheduler_app,
        base_url="http://test",
    ) as client:
        yield client
```

### Basic Task Flow Tests

```python
@pytest.mark.asyncio
async def test_submit_execute_complete(client, mock_worker):
    """Test complete task lifecycle."""
    # Register worker
    await client.post("/instance/register", json={
        "instance_id": "worker-1",
        "endpoint": mock_worker.endpoint,
        "model_id": "test-model",
    })

    # Submit task
    response = await client.post("/task/submit", json={
        "task_id": "task-1",
        "model_id": "test-model",
        "task_input": {"prompt": "test"},
        "metadata": {},
    })
    assert response.status_code == 200
    assert response.json()["success"]

    # Wait for completion
    await asyncio.sleep(0.5)

    # Check task completed
    response = await client.get("/task/info", params={"task_id": "task-1"})
    assert response.json()["task"]["status"] == "COMPLETED"
```

### Queue Behavior Tests

```python
@pytest.mark.asyncio
async def test_fifo_ordering(client, mock_worker):
    """Test tasks processed in FIFO order."""
    mock_worker.response_delay = 0.2

    # Register worker
    await client.post("/instance/register", json={
        "instance_id": "worker-1",
        "endpoint": mock_worker.endpoint,
        "model_id": "test-model",
    })

    # Submit multiple tasks rapidly
    task_ids = [f"task-{i}" for i in range(5)]
    for task_id in task_ids:
        await client.post("/task/submit", json={
            "task_id": task_id,
            "model_id": "test-model",
            "task_input": {"prompt": "test"},
            "metadata": {},
        })

    # Wait for all to complete
    await asyncio.sleep(2.0)

    # Verify order from mock worker requests
    request_order = [r["task_id"] for r in mock_worker.requests]
    assert request_order == task_ids
```

### Worker Deregistration Tests

```python
@pytest.mark.asyncio
async def test_deregister_redistributes(client, mock_worker):
    """Test deregistration redistributes pending tasks."""
    mock_worker.response_delay = 5.0  # Long delay

    # Register worker
    await client.post("/instance/register", json={
        "instance_id": "worker-1",
        "endpoint": mock_worker.endpoint,
        "model_id": "test-model",
    })

    # Submit multiple tasks
    for i in range(5):
        await client.post("/task/submit", json={
            "task_id": f"task-{i}",
            "model_id": "test-model",
            "task_input": {"prompt": "test"},
            "metadata": {},
        })

    # Deregister worker (force, don't wait)
    response = await client.post("/instance/drain", json={
        "instance_id": "worker-1",
        "force": True,
    })

    # Check tasks are back in central queue or pending
    response = await client.get("/queue/info")
    assert response.json()["total_size"] > 0
```

### Graceful Shutdown Tests

```python
@pytest.mark.asyncio
async def test_drain_completes_task(client, mock_worker):
    """Test drain waits for current task to complete."""
    mock_worker.response_delay = 1.0

    # Register and submit
    await client.post("/instance/register", json={
        "instance_id": "worker-1",
        "endpoint": mock_worker.endpoint,
        "model_id": "test-model",
    })

    await client.post("/task/submit", json={
        "task_id": "task-1",
        "model_id": "test-model",
        "task_input": {"prompt": "test"},
        "metadata": {},
    })

    # Wait for task to start
    await asyncio.sleep(0.1)

    # Initiate drain (should wait)
    response = await client.post("/instance/drain", json={
        "instance_id": "worker-1",
        "drain_timeout": 10.0,
    })

    # Task should be completed
    response = await client.get("/task/info", params={"task_id": "task-1"})
    assert response.json()["task"]["status"] == "COMPLETED"
```

## Performance Tests

### Throughput Test

```python
@pytest.mark.asyncio
async def test_throughput_under_load():
    """Test scheduler handles high task volume."""
    # Register multiple workers
    # Submit 1000 tasks
    # Measure completion rate
    # Assert throughput meets target
```

### Latency Test

```python
@pytest.mark.asyncio
async def test_latency_distribution():
    """Test task latency under various loads."""
    # Submit tasks at varying rates
    # Measure queue wait time
    # Assert P99 latency within bounds
```

## Implementation Steps

1. [x] Set up test infrastructure and fixtures
2. [x] Implement basic task flow tests (2 tests)
3. [x] Implement queue behavior tests (2 tests)
4. [x] Implement queue state adapter tests (2 tests)
5. [x] Implement registration/deregistration tests (3 tests)
6. [x] Implement instance sync integration tests (2 tests)
7. [x] Implement graceful shutdown tests (2 tests)
8. [x] Implement task result callback tests (2 tests)
9. [x] Implement multi-worker scenario tests (2 tests)

Total: 17 integration tests

## Acceptance Criteria

- [x] All basic task flow tests pass
- [x] Queue behavior correctly verified
- [x] Multi-worker scenarios work correctly
- [x] Worker lifecycle properly tested
- [x] Error handling verified
- [x] Graceful shutdown tested
- [x] Edge cases covered
- [x] Test coverage > 80%

## References

- [PYLET-014: Design Overview](PYLET-014-scheduler-task-queue.md)
- [Current Scheduler Tests](../../scheduler/tests/)
