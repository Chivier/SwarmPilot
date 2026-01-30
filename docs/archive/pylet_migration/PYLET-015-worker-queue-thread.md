# PYLET-015: Worker Queue Thread Implementation

## Status: DONE

## Objective

Implement the `WorkerQueueThread` class that manages a FIFO task queue for a single worker, processing tasks sequentially in a dedicated thread.

## Prerequisites

- PYLET-014: Design review complete

## Design

### Class Overview

```python
class WorkerQueueThread:
    """Dedicated thread for processing tasks on a specific worker.

    Each PyLet worker gets its own WorkerQueueThread that:
    1. Maintains a FIFO task queue
    2. Executes tasks synchronously against the worker's model API
    3. Handles results via library-based callbacks

    The thread runs independently of the main asyncio event loop,
    avoiding blocking during model inference calls.
    """
```

### Data Structures

```python
@dataclass
class QueuedTask:
    """Task waiting in worker queue."""
    task_id: str
    model_id: str
    task_input: dict[str, Any]
    metadata: dict[str, Any]
    enqueue_time: float
    predicted_time_ms: float | None = None

@dataclass
class TaskResult:
    """Result from task execution."""
    task_id: str
    worker_id: str
    status: Literal["completed", "failed"]
    result: dict[str, Any] | None = None
    error: str | None = None
    execution_time_ms: float = 0.0
```

### Implementation

```python
from queue import Queue, Empty
from threading import Thread, Event
from typing import Callable, Any
import time
import httpx
from loguru import logger


class WorkerQueueThread:
    """Dedicated thread for processing tasks on a specific worker."""

    def __init__(
        self,
        worker_id: str,
        worker_endpoint: str,
        model_id: str,
        callback: Callable[[TaskResult], None],
        http_timeout: float = 300.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize worker queue thread.

        Args:
            worker_id: Unique identifier for this worker
            worker_endpoint: HTTP endpoint for the worker's model API
            model_id: Model ID running on this worker
            callback: Function to call when task completes (thread-safe)
            http_timeout: Timeout for HTTP requests in seconds
            max_retries: Maximum retry attempts for transient errors
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.worker_id = worker_id
        self.worker_endpoint = worker_endpoint
        self.model_id = model_id
        self._callback = callback
        self._http_timeout = http_timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        # Thread-safe FIFO queue
        self._queue: Queue[QueuedTask] = Queue()

        # Thread control
        self._thread: Thread | None = None
        self._shutdown = Event()

        # Current task tracking
        self._current_task: QueuedTask | None = None
        self._current_task_started: float | None = None

        # HTTP client (created per thread for thread safety)
        self._http_client: httpx.Client | None = None

    def start(self) -> None:
        """Start the worker queue processing thread."""
        if self._thread is not None:
            raise RuntimeError(f"Worker {self.worker_id} thread already started")

        self._shutdown.clear()
        self._thread = Thread(
            target=self._process_loop,
            name=f"WorkerQueue-{self.worker_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Started worker queue thread for {self.worker_id}")

    def stop(self, timeout: float = 10.0) -> list[QueuedTask]:
        """Stop the thread and return pending tasks.

        Args:
            timeout: Maximum time to wait for thread to stop

        Returns:
            List of tasks that were still in queue
        """
        if self._thread is None:
            return []

        logger.info(f"Stopping worker queue thread for {self.worker_id}")
        self._shutdown.set()

        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning(
                f"Worker {self.worker_id} thread did not stop gracefully"
            )

        self._thread = None

        # Drain remaining tasks
        pending = []
        while not self._queue.empty():
            try:
                pending.append(self._queue.get_nowait())
            except Empty:
                break

        logger.info(
            f"Worker {self.worker_id} stopped with {len(pending)} pending tasks"
        )
        return pending

    def enqueue(self, task: QueuedTask) -> int:
        """Add a task to the worker's queue.

        Args:
            task: Task to enqueue

        Returns:
            Current queue size after enqueue
        """
        self._queue.put(task)
        size = self._queue.qsize()
        logger.debug(
            f"Task {task.task_id} enqueued to {self.worker_id}, "
            f"queue size: {size}"
        )
        return size

    def queue_size(self) -> int:
        """Get current queue size (approximate)."""
        return self._queue.qsize()

    def has_running_task(self) -> bool:
        """Check if a task is currently being executed."""
        return self._current_task is not None

    def get_current_task_id(self) -> str | None:
        """Get the ID of the currently running task."""
        return self._current_task.task_id if self._current_task else None

    def get_estimated_wait_time(self, avg_exec_time_ms: float) -> float:
        """Estimate wait time for a new task in milliseconds.

        Args:
            avg_exec_time_ms: Average execution time per task

        Returns:
            Estimated wait time in milliseconds
        """
        queue_depth = self._queue.qsize()

        # Add current task's remaining time if running
        current_remaining = 0.0
        if self._current_task and self._current_task_started:
            elapsed = (time.time() - self._current_task_started) * 1000
            predicted = self._current_task.predicted_time_ms or avg_exec_time_ms
            current_remaining = max(0, predicted - elapsed)

        return current_remaining + (queue_depth * avg_exec_time_ms)

    def _process_loop(self) -> None:
        """Main processing loop (runs in dedicated thread)."""
        logger.info(f"Worker {self.worker_id} processing loop started")

        # Create HTTP client for this thread
        self._http_client = httpx.Client(
            timeout=self._http_timeout,
            verify=False,  # Internal network
        )

        try:
            while not self._shutdown.is_set():
                try:
                    # Wait for task with timeout (allows shutdown check)
                    task = self._queue.get(timeout=1.0)
                    self._execute_task(task)
                except Empty:
                    continue
                except Exception as e:
                    logger.error(
                        f"Worker {self.worker_id} error in process loop: {e}",
                        exc_info=True,
                    )
        finally:
            self._http_client.close()
            self._http_client = None
            logger.info(f"Worker {self.worker_id} processing loop stopped")

    def _execute_task(self, task: QueuedTask) -> None:
        """Execute a single task.

        Args:
            task: Task to execute
        """
        self._current_task = task
        self._current_task_started = time.time()

        logger.info(
            f"Worker {self.worker_id} executing task {task.task_id}"
        )

        try:
            result = self._call_worker_api(task)
            execution_time_ms = (time.time() - self._current_task_started) * 1000

            logger.info(
                f"Task {task.task_id} completed in {execution_time_ms:.2f}ms"
            )

            self._callback(TaskResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                status="completed",
                result=result,
                execution_time_ms=execution_time_ms,
            ))

        except Exception as e:
            execution_time_ms = (time.time() - self._current_task_started) * 1000

            logger.error(
                f"Task {task.task_id} failed after {execution_time_ms:.2f}ms: {e}"
            )

            self._callback(TaskResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                status="failed",
                error=str(e),
                execution_time_ms=execution_time_ms,
            ))

        finally:
            self._current_task = None
            self._current_task_started = None

    def _call_worker_api(self, task: QueuedTask) -> dict[str, Any]:
        """Make HTTP call to worker's model API with retry logic.

        Args:
            task: Task containing input for the model

        Returns:
            Response from the model API

        Raises:
            httpx.HTTPError: If request fails after all retries
        """
        url = f"{self.worker_endpoint}/v1/completions"
        payload = task.task_input

        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = self._http_client.post(url, json=payload)
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                # 4xx/5xx errors - don't retry (client/server error)
                raise

            except (httpx.ConnectError, httpx.ReadError) as e:
                # Transient errors - retry with backoff
                last_error = e
                if attempt < self._max_retries - 1:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Worker {self.worker_id} connection error, "
                        f"retrying in {delay}s (attempt {attempt + 1})"
                    )
                    time.sleep(delay)
                    continue
                raise

            except httpx.TimeoutException as e:
                # Timeout - don't retry (might be model processing)
                raise

        if last_error:
            raise last_error
```

## API Reference

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `worker_id` | `str` | required | Unique identifier for this worker |
| `worker_endpoint` | `str` | required | HTTP endpoint for worker's model API |
| `model_id` | `str` | required | Model ID running on this worker |
| `callback` | `Callable` | required | Thread-safe callback for results |
| `http_timeout` | `float` | 300.0 | HTTP request timeout (seconds) |
| `max_retries` | `int` | 3 | Max retries for transient errors |
| `retry_delay` | `float` | 1.0 | Initial retry delay (seconds) |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `start()` | `None` | Start the processing thread |
| `stop(timeout)` | `list[QueuedTask]` | Stop thread, return pending tasks |
| `enqueue(task)` | `int` | Add task, return queue size |
| `queue_size()` | `int` | Get current queue depth |
| `has_running_task()` | `bool` | Check if task is executing |
| `get_current_task_id()` | `str | None` | Get current task ID |
| `get_estimated_wait_time(avg)` | `float` | Estimate queue wait time |

## Implementation Steps

1. [x] Create `src/services/worker_queue_thread.py`
2. [x] Implement `QueuedTask` and `TaskResult` dataclasses
3. [x] Implement `WorkerQueueThread` class
4. [x] Add logging and error handling
5. [x] Write unit tests

## Testing

### Unit Tests

```python
def test_enqueue_and_process():
    """Test basic enqueue and process flow."""

def test_queue_size_tracking():
    """Test queue size updates correctly."""

def test_graceful_stop():
    """Test stop returns pending tasks."""

def test_http_retry_on_transient_error():
    """Test retry logic for connection errors."""

def test_no_retry_on_timeout():
    """Test no retry on HTTP timeout."""

def test_callback_invoked_on_success():
    """Test callback receives success result."""

def test_callback_invoked_on_failure():
    """Test callback receives failure result."""
```

## Acceptance Criteria

- [x] Thread starts and processes tasks in FIFO order
- [x] HTTP calls are made synchronously to worker API
- [x] Callback is invoked with correct result/error
- [x] Transient errors trigger retry with backoff
- [x] Stop method returns pending tasks
- [x] Queue size is accurately tracked
- [x] Unit tests pass (17 tests)

## References

- [PYLET-014: Design Overview](PYLET-014-scheduler-task-queue.md)
- [Current Instance Task Queue](../../instance/src/task_queue.py)
