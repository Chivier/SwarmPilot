"""Worker queue thread for processing tasks on a specific worker.

This module implements PYLET-015: Worker Queue Thread, providing a dedicated
thread for each registered worker that manages a FIFO task queue and executes
tasks synchronously against the worker's model API.

Key features:
- FIFO task queue per worker
- Sequential task execution in a dedicated thread
- Library-based callbacks for result handling
- Retry logic for transient connection errors
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Literal

import httpx
from loguru import logger


@dataclass
class QueuedTask:
    """Task waiting in a worker queue.

    Attributes:
        task_id: Unique identifier for this task.
        model_id: ID of the model this task targets.
        task_input: Input data for the model (e.g., prompt, parameters).
        metadata: Additional task metadata (priority, path, headers, etc.).
        enqueue_time: Unix timestamp when task was enqueued.
        predicted_time_ms: Predicted execution time in milliseconds (optional).
    """

    task_id: str
    model_id: str
    task_input: dict[str, Any]
    metadata: dict[str, Any]
    enqueue_time: float
    predicted_time_ms: float | None = None


@dataclass
class TaskResult:
    """Result from task execution.

    Attributes:
        task_id: Unique identifier for the completed task.
        worker_id: ID of the worker that executed the task.
        status: Execution status ("completed" or "failed").
        result: Response from the model API (on success).
        error: Error message (on failure).
        execution_time_ms: Actual execution time in milliseconds.
    """

    task_id: str
    worker_id: str
    status: Literal["completed", "failed"]
    result: dict[str, Any] | None = None
    error: str | None = None
    execution_time_ms: float = 0.0
    http_status_code: int = 0
    response_headers: dict[str, str] = field(default_factory=dict)


class WorkerQueueThread:
    """Dedicated thread for processing tasks on a specific worker.

    Each PyLet worker gets its own WorkerQueueThread that:
    1. Maintains a FIFO task queue
    2. Executes tasks synchronously against the worker's model API
    3. Handles results via library-based callbacks

    The thread runs independently of the main asyncio event loop,
    avoiding blocking during model inference calls.

    Example:
        ```python
        def handle_result(result: TaskResult):
            print(f"Task {result.task_id}: {result.status}")

        thread = WorkerQueueThread(
            worker_id="worker-1",
            worker_endpoint="http://localhost:8001",
            model_id="gpt-4",
            callback=handle_result,
        )
        thread.start()

        task = QueuedTask(
            task_id="task-1",
            model_id="gpt-4",
            task_input={"prompt": "Hello"},
            metadata={},
            enqueue_time=time.time(),
        )
        thread.enqueue(task)

        # Later...
        pending = thread.stop()
        ```
    """

    def __init__(
        self,
        worker_id: str,
        worker_endpoint: str,
        model_id: str,
        callback: Callable[[TaskResult], None],
        http_timeout: float = 300.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        on_task_started: Callable[[str], None] | None = None,
    ):
        """Initialize worker queue thread.

        Args:
            worker_id: Unique identifier for this worker.
            worker_endpoint: HTTP endpoint for the worker's model API.
            model_id: Model ID running on this worker.
            callback: Function to call when task completes (thread-safe).
            http_timeout: Timeout for HTTP requests in seconds.
            max_retries: Maximum retry attempts for transient errors.
            retry_delay: Initial delay between retries (uses exponential backoff).
            on_task_started: Optional callback when task begins execution.
        """
        self.worker_id = worker_id
        self.worker_endpoint = worker_endpoint
        self.model_id = model_id
        self._callback = callback
        self._http_timeout = http_timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._on_task_started = on_task_started

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
        """Start the worker queue processing thread.

        Raises:
            RuntimeError: If thread is already started.
        """
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
            timeout: Maximum time to wait for thread to stop.

        Returns:
            List of tasks that were still in queue (not processed).
        """
        if self._thread is None:
            # Thread not started, just drain the queue
            pending = []
            while not self._queue.empty():
                try:
                    pending.append(self._queue.get_nowait())
                except Empty:
                    break
            return pending

        logger.info(f"Stopping worker queue thread for {self.worker_id}")
        self._shutdown.set()

        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning(f"Worker {self.worker_id} thread did not stop gracefully")

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
            task: Task to enqueue.

        Returns:
            Current queue size after enqueue.
        """
        self._queue.put(task)
        size = self._queue.qsize()
        logger.debug(
            f"Task {task.task_id} enqueued to {self.worker_id}, queue size: {size}"
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
            avg_exec_time_ms: Average execution time per task.

        Returns:
            Estimated wait time in milliseconds.
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
            if self._http_client:
                self._http_client.close()
            self._http_client = None
            logger.info(f"Worker {self.worker_id} processing loop stopped")

    def _execute_task(self, task: QueuedTask) -> None:
        """Execute a single task.

        Args:
            task: Task to execute.
        """
        self._current_task = task
        self._current_task_started = time.time()

        if self._on_task_started:
            try:
                self._on_task_started(task.task_id)
            except Exception as e:
                logger.error(f"on_task_started callback failed for {task.task_id}: {e}")

        logger.info(f"Worker {self.worker_id} executing task {task.task_id}")

        try:
            body, status_code, headers = self._call_worker_api(task)
            execution_time_ms = (time.time() - self._current_task_started) * 1000

            logger.info(f"Task {task.task_id} completed in {execution_time_ms:.2f}ms")

            self._callback(
                TaskResult(
                    task_id=task.task_id,
                    worker_id=self.worker_id,
                    status="completed",
                    result=body,
                    execution_time_ms=execution_time_ms,
                    http_status_code=status_code,
                    response_headers=headers,
                )
            )

        except httpx.HTTPStatusError as e:
            execution_time_ms = (time.time() - self._current_task_started) * 1000
            # Propagate the upstream HTTP status and body
            status_code = e.response.status_code
            try:
                error_body = e.response.json()
            except Exception:
                error_body = {"error": e.response.text}
            resp_headers = dict(e.response.headers)

            logger.warning(
                f"Task {task.task_id} upstream error {status_code} "
                f"after {execution_time_ms:.2f}ms"
            )

            self._callback(
                TaskResult(
                    task_id=task.task_id,
                    worker_id=self.worker_id,
                    status="failed",
                    result=error_body,
                    error=str(e),
                    execution_time_ms=execution_time_ms,
                    http_status_code=status_code,
                    response_headers=resp_headers,
                )
            )

        except Exception as e:
            execution_time_ms = (time.time() - self._current_task_started) * 1000

            logger.error(
                f"Task {task.task_id} failed after {execution_time_ms:.2f}ms: {e}"
            )

            self._callback(
                TaskResult(
                    task_id=task.task_id,
                    worker_id=self.worker_id,
                    status="failed",
                    error=str(e),
                    execution_time_ms=execution_time_ms,
                )
            )

        finally:
            self._current_task = None
            self._current_task_started = None

    def _call_worker_api(
        self, task: QueuedTask
    ) -> tuple[dict[str, Any], int, dict[str, str]]:
        """Make HTTP call to worker's model API with retry logic.

        Args:
            task: Task containing input for the model.

        Returns:
            Tuple of (response_body, http_status_code, response_headers).

        Raises:
            httpx.HTTPError: If request fails after all retries.
        """
        # Use path from metadata if available, otherwise default to /v1/completions
        path = task.metadata.get("path", "v1/completions")
        url = f"{self.worker_endpoint}/{path}"
        payload = task.task_input

        # Use method from metadata if available, otherwise default to POST
        method = task.metadata.get("method", "POST").upper()

        # Use request headers from metadata if available
        request_headers = task.metadata.get("headers", {})

        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = self._http_client.request(
                    method, url, json=payload, headers=request_headers
                )
                response.raise_for_status()
                return (
                    response.json(),
                    response.status_code,
                    dict(response.headers),
                )

            except httpx.HTTPStatusError:
                # 4xx/5xx errors - don't retry (client/server error)
                raise

            except (httpx.ConnectError, httpx.ReadError) as e:
                # Transient errors - retry with backoff
                last_error = e
                if attempt < self._max_retries - 1:
                    delay = self._retry_delay * (2**attempt)
                    logger.warning(
                        f"Worker {self.worker_id} connection error, "
                        f"retrying in {delay}s (attempt {attempt + 1})"
                    )
                    time.sleep(delay)
                    continue
                raise

            except httpx.TimeoutException:
                # Timeout - don't retry (might be model processing)
                raise

        if last_error:
            raise last_error

        # Should not reach here, but just in case
        raise RuntimeError("Unexpected state in _call_worker_api")
