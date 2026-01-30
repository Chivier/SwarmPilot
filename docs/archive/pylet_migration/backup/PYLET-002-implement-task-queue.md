# PYLET-002: Implement Task Queue

## Objective

Implement a request-level task queue for the SWorker wrapper that receives tasks from the SwarmPilot scheduler, processes them sequentially against the wrapped model service, and sends results back via callback.

## Prerequisites

- [PYLET-001](PYLET-001-create-sworker-wrapper.md) completed
- Understanding of current TaskQueue implementation in Instance service

## Background

The task queue is the core component that differentiates SwarmPilot from direct PyLet usage:
- **PyLet**: Direct endpoint exposure (`$PORT` → model)
- **SwarmPilot**: Request-level scheduling (scheduler → task queue → model → callback)

The task queue maintains FIFO ordering with priority support and handles callbacks to the scheduler.

## Files to Create/Modify

```
sworker-wrapper/src/
├── task_queue.py     # NEW: Task queue implementation
├── models.py         # NEW: Data models
├── api.py            # MODIFY: Add task endpoints
└── model_client.py   # NEW: Internal model service client
```

## Implementation Steps

### Step 1: Create Data Models

Create `sworker-wrapper/src/models.py`:

```python
"""Data models for SWorker wrapper."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task status enumeration."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    """A task in the queue."""

    task_id: str
    model_id: str
    task_input: dict[str, Any]
    callback_url: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    # Timing
    enqueue_time: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    submitted_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Status
    status: TaskStatus = TaskStatus.QUEUED

    # Result
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None

    def mark_started(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now(timezone.utc).isoformat()

    def mark_completed(self, result: dict[str, Any]) -> None:
        """Mark task as completed with result."""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def mark_failed(self, error: str) -> None:
        """Mark task as failed with error."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = datetime.now(timezone.utc).isoformat()


class TaskSubmission(BaseModel):
    """Request to submit a task."""

    task_id: str
    model_id: str
    task_input: dict[str, Any]
    callback_url: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    enqueue_time: Optional[float] = None


class TaskResponse(BaseModel):
    """Response after task submission."""

    task_id: str
    status: TaskStatus
    queue_position: int


class TaskResult(BaseModel):
    """Task result for callback."""

    task_id: str
    status: str  # "completed" or "failed"
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None
    instance_id: Optional[str] = None


class QueueStats(BaseModel):
    """Queue statistics."""

    total: int
    queued: int
    running: int
    completed: int
    failed: int
```

### Step 2: Create Model Client

Create `sworker-wrapper/src/model_client.py`:

```python
"""Client for internal model service communication."""

import asyncio
from typing import Any, Optional

import httpx
from loguru import logger

from src.config import get_config


class ModelClient:
    """HTTP client for wrapped model service."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 300.0):
        """Initialize model client.

        Args:
            base_url: Model service URL. If None, uses config.model_port.
            timeout: Request timeout in seconds.
        """
        if base_url is None:
            config = get_config()
            base_url = f"http://localhost:{config.model_port}"
        self.base_url = base_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def invoke(self, task_input: dict[str, Any]) -> dict[str, Any]:
        """Invoke model with task input.

        This method should be customized based on the model service API.
        Common patterns:
        - vLLM: POST /v1/completions
        - OpenAI-compatible: POST /v1/chat/completions
        - Custom: Depends on model service

        Args:
            task_input: Task input dictionary.

        Returns:
            Model response dictionary.

        Raises:
            httpx.HTTPError: On HTTP errors.
        """
        client = await self._get_client()

        # Determine endpoint based on task_input structure
        if "messages" in task_input:
            # OpenAI chat format
            endpoint = "/v1/chat/completions"
        elif "prompt" in task_input:
            # Completion format
            endpoint = "/v1/completions"
        else:
            # Generic inference endpoint
            endpoint = "/inference"

        logger.debug(f"Invoking model at {self.base_url}{endpoint}")

        response = await client.post(endpoint, json=task_input)
        response.raise_for_status()

        return response.json()

    async def health_check(self) -> bool:
        """Check if model service is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Model health check failed: {e}")
            return False

    async def wait_ready(
        self,
        timeout: float = 300.0,
        interval: float = 5.0,
    ) -> bool:
        """Wait for model service to become ready.

        Args:
            timeout: Maximum time to wait in seconds.
            interval: Check interval in seconds.

        Returns:
            True if ready within timeout, False otherwise.
        """
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            if await self.health_check():
                logger.info("Model service is ready")
                return True
            await asyncio.sleep(interval)

        logger.error(f"Model service not ready after {timeout}s")
        return False


# Global model client instance
_model_client: Optional[ModelClient] = None


def get_model_client() -> ModelClient:
    """Get the global model client instance."""
    global _model_client
    if _model_client is None:
        _model_client = ModelClient()
    return _model_client


async def close_model_client() -> None:
    """Close the global model client."""
    global _model_client
    if _model_client is not None:
        await _model_client.close()
        _model_client = None
```

### Step 3: Implement Task Queue

Create `sworker-wrapper/src/task_queue.py`:

```python
"""Task queue with FIFO processing and callback support."""

import asyncio
import heapq
import time
import traceback
from collections import deque
from typing import Any, Optional

import httpx
from loguru import logger

from src.config import get_config
from src.model_client import get_model_client
from src.models import QueueStats, Task, TaskStatus


class TaskQueue:
    """Manages task queue with priority-based processing.

    Tasks are processed based on enqueue_time (earliest first) using a min-heap.
    """

    def __init__(self):
        """Initialize the task queue."""
        self.tasks: dict[str, Task] = {}  # All tasks by task_id
        self.queue: list[tuple[float, str]] = []  # (enqueue_time, task_id)
        self._insertion_order: deque = deque()  # Track for LIFO fetch
        self._queue_lock: asyncio.Lock = asyncio.Lock()

        self.current_task_id: Optional[str] = None
        self.is_processing = False
        self._processing_task: Optional[asyncio.Task] = None
        self._shutdown_requested = False

    async def submit_task(
        self,
        task: Task,
        enqueue_time: Optional[float] = None,
    ) -> int:
        """Submit a new task to the queue.

        Args:
            task: Task object to submit.
            enqueue_time: Optional priority timestamp.

        Returns:
            Position in queue (1-indexed).

        Raises:
            ValueError: If task_id already exists.
            RuntimeError: If shutdown is in progress.
        """
        if self._shutdown_requested:
            raise RuntimeError("Queue is shutting down, not accepting new tasks")

        if task.task_id in self.tasks:
            raise ValueError(f"Task with ID {task.task_id} already exists")

        # Set enqueue_time
        if enqueue_time is not None:
            task.enqueue_time = enqueue_time

        # Add to storage and priority queue
        self.tasks[task.task_id] = task

        async with self._queue_lock:
            heapq.heappush(self.queue, (task.enqueue_time, task.task_id))
            self._insertion_order.append(task.task_id)
            queue_size = len(self.queue)

        logger.info(
            f"Task {task.task_id} submitted, "
            f"enqueue_time={task.enqueue_time:.3f}, "
            f"queue_position={queue_size}"
        )

        # Start processing if not already running
        if not self.is_processing:
            self._processing_task = asyncio.create_task(self._process_queue())

        return queue_size

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    async def list_tasks(
        self,
        status_filter: Optional[TaskStatus] = None,
        limit: Optional[int] = None,
    ) -> list[Task]:
        """List all tasks with optional filtering."""
        tasks = list(self.tasks.values())

        if status_filter:
            tasks = [t for t in tasks if t.status == status_filter]

        # Sort by submission time (most recent first)
        tasks.sort(key=lambda t: t.submitted_at, reverse=True)

        if limit:
            tasks = tasks[:limit]

        return tasks

    async def get_queue_stats(self) -> QueueStats:
        """Get task queue statistics."""
        return QueueStats(
            total=len(self.tasks),
            queued=sum(1 for t in self.tasks.values() if t.status == TaskStatus.QUEUED),
            running=sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING),
            completed=sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED),
            failed=sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED),
        )

    async def _process_queue(self) -> None:
        """Process tasks from the queue sequentially."""
        self.is_processing = True
        logger.info("Task queue processing started")

        try:
            while True:
                # Get next task
                async with self._queue_lock:
                    if not self.queue:
                        break
                    enqueue_time, task_id = heapq.heappop(self.queue)

                task = self.tasks.get(task_id)
                if not task:
                    logger.error(f"Task {task_id} not found in storage")
                    continue

                self.current_task_id = task_id
                logger.info(
                    f"Processing task {task_id} "
                    f"(wait_time={time.time() - enqueue_time:.3f}s)"
                )

                try:
                    await self._execute_task(task)
                except Exception as e:
                    logger.error(f"Task {task_id} execution error: {e}")
                    task.mark_failed(str(e))

                self.current_task_id = None

        finally:
            self.is_processing = False
            logger.info("Task queue processing stopped")

    async def _execute_task(self, task: Task) -> None:
        """Execute a single task."""
        task.mark_started()
        logger.info(f"Task {task.task_id} started")

        start_time = time.time()
        execution_time_ms: Optional[float] = None

        try:
            # Get model client and invoke
            model_client = get_model_client()
            result = await model_client.invoke(task.task_input)

            execution_time_ms = (time.time() - start_time) * 1000
            task.execution_time_ms = execution_time_ms
            task.mark_completed(result)

            logger.info(
                f"Task {task.task_id} completed in {execution_time_ms:.2f}ms"
            )

            # Send success callback
            await self._send_callback(
                task_id=task.task_id,
                status="completed",
                result=result,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            task.execution_time_ms = execution_time_ms
            error_msg = str(e)
            task.mark_failed(error_msg)

            logger.error(
                f"Task {task.task_id} failed after {execution_time_ms:.2f}ms: {e}"
            )

            # Send failure callback
            await self._send_callback(
                task_id=task.task_id,
                status="failed",
                error=error_msg,
                execution_time_ms=execution_time_ms,
            )

    async def _send_callback(
        self,
        task_id: str,
        status: str,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
    ) -> bool:
        """Send task result callback to scheduler.

        Args:
            task_id: Task ID.
            status: "completed" or "failed".
            result: Task result (if completed).
            error: Error message (if failed).
            execution_time_ms: Execution time.

        Returns:
            True if callback sent successfully.
        """
        task = self.tasks.get(task_id)
        if not task or not task.callback_url:
            logger.warning(f"No callback URL for task {task_id}")
            return False

        config = get_config()

        callback_data = {
            "task_id": task_id,
            "status": status,
            "instance_id": config.instance_id,
        }
        if result is not None:
            callback_data["result"] = result
        if error is not None:
            callback_data["error"] = error
        if execution_time_ms is not None:
            callback_data["execution_time_ms"] = execution_time_ms

        # Retry callback with exponential backoff
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        task.callback_url,
                        json=callback_data,
                    )
                    response.raise_for_status()
                    logger.info(f"Callback sent for task {task_id}")
                    return True

            except Exception as e:
                logger.warning(
                    f"Callback attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))

        logger.error(f"Failed to send callback for task {task_id}")
        return False

    async def extract_pending_tasks(self) -> list[dict[str, Any]]:
        """Extract pending tasks for redistribution.

        Used during graceful shutdown to return queued tasks to scheduler.

        Returns:
            List of task dictionaries for redistribution.
        """
        extracted = []

        async with self._queue_lock:
            # Extract all queued tasks
            while self.queue:
                enqueue_time, task_id = heapq.heappop(self.queue)
                task = self.tasks.get(task_id)

                if task and task.status == TaskStatus.QUEUED:
                    extracted.append({
                        "task_id": task.task_id,
                        "model_id": task.model_id,
                        "task_input": task.task_input,
                        "enqueue_time": task.enqueue_time,
                        "callback_url": task.callback_url,
                        "metadata": task.metadata,
                    })
                    del self.tasks[task_id]

            self._insertion_order.clear()

        logger.info(f"Extracted {len(extracted)} pending tasks for redistribution")
        return extracted

    async def shutdown(self, wait_for_current: bool = True) -> None:
        """Initiate graceful shutdown.

        Args:
            wait_for_current: Wait for currently running task to complete.
        """
        self._shutdown_requested = True
        logger.info("Task queue shutdown initiated")

        if wait_for_current and self.current_task_id:
            logger.info(f"Waiting for current task {self.current_task_id}")
            # Wait for processing to complete
            while self.is_processing:
                await asyncio.sleep(0.5)

        # Cancel processing task if still running
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()

        logger.info("Task queue shutdown complete")

    async def stop_processing(self) -> None:
        """Stop queue processing immediately."""
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass


# Global task queue instance
_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """Get the global task queue instance."""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
    return _task_queue


async def shutdown_task_queue() -> None:
    """Shutdown the global task queue."""
    global _task_queue
    if _task_queue is not None:
        await _task_queue.shutdown()
        _task_queue = None
```

### Step 4: Update API with Task Endpoints

Update `sworker-wrapper/src/api.py`:

```python
"""SWorker Wrapper HTTP API."""

from fastapi import FastAPI, HTTPException
from loguru import logger

from src.config import get_config
from src.models import QueueStats, Task, TaskResponse, TaskSubmission
from src.task_queue import get_task_queue


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="SWorker Wrapper",
        description="PyLet to SwarmPilot bridge",
        version="0.1.0",
    )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        queue = get_task_queue()
        stats = await queue.get_queue_stats()
        return {
            "status": "healthy",
            "queue_stats": stats.model_dump(),
        }

    @app.get("/info")
    async def info():
        """Instance information."""
        config = get_config()
        return {
            "instance_id": config.instance_id,
            "model_id": config.model_id,
            "port": config.port,
            "model_port": config.model_port,
        }

    @app.post("/task/submit", response_model=TaskResponse)
    async def submit_task(submission: TaskSubmission):
        """Submit a task to the queue.

        Args:
            submission: Task submission request.

        Returns:
            Task response with queue position.
        """
        task = Task(
            task_id=submission.task_id,
            model_id=submission.model_id,
            task_input=submission.task_input,
            callback_url=submission.callback_url,
            metadata=submission.metadata,
        )

        queue = get_task_queue()

        try:
            queue_position = await queue.submit_task(
                task,
                enqueue_time=submission.enqueue_time,
            )
            return TaskResponse(
                task_id=task.task_id,
                status=task.status,
                queue_position=queue_position,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    @app.get("/task/{task_id}")
    async def get_task(task_id: str):
        """Get task status by ID."""
        queue = get_task_queue()
        task = await queue.get_task(task_id)

        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")

        return task.model_dump()

    @app.get("/task/list")
    async def list_tasks(limit: int = 100):
        """List all tasks."""
        queue = get_task_queue()
        tasks = await queue.list_tasks(limit=limit)
        return [t.model_dump() for t in tasks]

    @app.get("/queue/stats", response_model=QueueStats)
    async def queue_stats():
        """Get queue statistics."""
        queue = get_task_queue()
        return await queue.get_queue_stats()

    return app
```

### Step 5: Create Tests

Create `sworker-wrapper/tests/test_task_queue.py`:

```python
"""Tests for task queue module."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.models import Task, TaskStatus
from src.task_queue import TaskQueue


@pytest.fixture
def task_queue():
    """Create a fresh task queue for each test."""
    return TaskQueue()


@pytest.fixture
def sample_task():
    """Create a sample task."""
    return Task(
        task_id="test-task-001",
        model_id="test-model",
        task_input={"prompt": "Hello"},
        callback_url="http://localhost:8000/callback",
    )


class TestTaskQueue:
    """Tests for TaskQueue class."""

    @pytest.mark.asyncio
    async def test_submit_task(self, task_queue, sample_task):
        """Test task submission."""
        position = await task_queue.submit_task(sample_task)
        assert position == 1

        # Verify task is stored
        task = await task_queue.get_task("test-task-001")
        assert task is not None
        assert task.status == TaskStatus.QUEUED

    @pytest.mark.asyncio
    async def test_submit_duplicate_task(self, task_queue, sample_task):
        """Test submitting duplicate task raises error."""
        await task_queue.submit_task(sample_task)

        with pytest.raises(ValueError, match="already exists"):
            await task_queue.submit_task(sample_task)

    @pytest.mark.asyncio
    async def test_queue_stats(self, task_queue, sample_task):
        """Test queue statistics."""
        # Submit a task
        await task_queue.submit_task(sample_task)

        stats = await task_queue.get_queue_stats()
        assert stats.total == 1
        assert stats.queued == 1
        assert stats.running == 0
        assert stats.completed == 0
        assert stats.failed == 0

    @pytest.mark.asyncio
    async def test_list_tasks(self, task_queue):
        """Test listing tasks."""
        # Submit multiple tasks
        for i in range(5):
            task = Task(
                task_id=f"task-{i}",
                model_id="test-model",
                task_input={"prompt": f"Test {i}"},
            )
            await task_queue.submit_task(task)

        tasks = await task_queue.list_tasks()
        assert len(tasks) == 5

        # Test limit
        tasks = await task_queue.list_tasks(limit=3)
        assert len(tasks) == 3

    @pytest.mark.asyncio
    async def test_priority_order(self, task_queue):
        """Test tasks are processed in priority order."""
        # Submit tasks with different enqueue times
        task1 = Task(
            task_id="task-1",
            model_id="test",
            task_input={},
        )
        task2 = Task(
            task_id="task-2",
            model_id="test",
            task_input={},
        )
        task3 = Task(
            task_id="task-3",
            model_id="test",
            task_input={},
        )

        # Submit in reverse priority order
        await task_queue.submit_task(task2, enqueue_time=200.0)
        await task_queue.submit_task(task3, enqueue_time=300.0)
        await task_queue.submit_task(task1, enqueue_time=100.0)

        # Verify queue order (lowest enqueue_time first)
        assert task_queue.queue[0][0] == 100.0
        assert task_queue.queue[0][1] == "task-1"

    @pytest.mark.asyncio
    async def test_extract_pending_tasks(self, task_queue):
        """Test extracting pending tasks for redistribution."""
        # Submit tasks
        for i in range(3):
            task = Task(
                task_id=f"task-{i}",
                model_id="test-model",
                task_input={"data": i},
            )
            await task_queue.submit_task(task)

        # Stop processing to keep tasks in queue
        await task_queue.stop_processing()

        # Extract pending tasks
        extracted = await task_queue.extract_pending_tasks()

        assert len(extracted) == 3
        assert all("task_id" in t for t in extracted)

        # Verify queue is empty
        stats = await task_queue.get_queue_stats()
        assert stats.total == 0

    @pytest.mark.asyncio
    async def test_shutdown_blocks_new_tasks(self, task_queue, sample_task):
        """Test shutdown prevents new task submissions."""
        task_queue._shutdown_requested = True

        with pytest.raises(RuntimeError, match="shutting down"):
            await task_queue.submit_task(sample_task)
```

## Test Strategy

### Unit Tests

```bash
cd sworker-wrapper
uv sync
uv run pytest tests/test_task_queue.py -v
```

### Integration Testing

Test with a mock model service:

```python
# Create a simple mock server
from fastapi import FastAPI
app = FastAPI()

@app.post("/v1/completions")
async def completions(data: dict):
    return {"text": "Hello, world!"}

# Run: uvicorn mock_server:app --port 16001
```

Then test the full flow:

```bash
# Start wrapper
PORT=16000 SWORKER_COMMAND="sleep 3600" uv run sworker-wrapper &

# Submit task
curl -X POST http://localhost:16000/task/submit \
  -H "Content-Type: application/json" \
  -d '{"task_id": "test-1", "model_id": "test", "task_input": {"prompt": "Hello"}}'

# Check status
curl http://localhost:16000/task/test-1
```

## Acceptance Criteria

- [ ] Task submission adds to priority queue
- [ ] Tasks processed in FIFO order by enqueue_time
- [ ] Task status transitions work correctly
- [ ] Callbacks sent on completion/failure
- [ ] Queue statistics accurate
- [ ] extract_pending_tasks works for shutdown
- [ ] All unit tests pass

## Next Steps

After completing this task:
1. Proceed to [PYLET-003](PYLET-003-signal-handling.md) for signal handling
2. The task queue will be integrated with graceful shutdown

## Code References

- Current TaskQueue: [instance/src/task_queue.py](../../instance/src/task_queue.py)
- Current Task model: [instance/src/models.py](../../instance/src/models.py)
