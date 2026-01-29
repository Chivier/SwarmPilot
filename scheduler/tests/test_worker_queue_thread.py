"""Unit tests for WorkerQueueThread.

TDD tests for PYLET-015: Worker Queue Thread Implementation.
These tests define the expected behavior before implementation.
"""

import time
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.services.worker_queue_thread import (
    QueuedTask,
    TaskResult,
    WorkerQueueThread,
)

# ============================================================================
# QueuedTask Dataclass Tests
# ============================================================================


class TestQueuedTask:
    """Tests for the QueuedTask dataclass."""

    def test_creation_with_required_fields(self) -> None:
        """Test creating a QueuedTask with required fields."""
        task = QueuedTask(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "hello"},
            metadata={"priority": "high"},
            enqueue_time=1704067200.0,
        )

        assert task.task_id == "task-1"
        assert task.model_id == "test-model"
        assert task.task_input == {"prompt": "hello"}
        assert task.metadata == {"priority": "high"}
        assert task.enqueue_time == 1704067200.0
        assert task.predicted_time_ms is None

    def test_creation_with_all_fields(self) -> None:
        """Test creating a QueuedTask with all fields including optional."""
        task = QueuedTask(
            task_id="task-2",
            model_id="test-model",
            task_input={"prompt": "world"},
            metadata={},
            enqueue_time=1704067200.0,
            predicted_time_ms=150.0,
        )

        assert task.predicted_time_ms == 150.0


# ============================================================================
# TaskResult Dataclass Tests
# ============================================================================


class TestTaskResult:
    """Tests for the TaskResult dataclass."""

    def test_success_result(self) -> None:
        """Test creating a successful TaskResult."""
        result = TaskResult(
            task_id="task-1",
            worker_id="worker-1",
            status="completed",
            result={"output": "test output"},
            execution_time_ms=150.0,
        )

        assert result.task_id == "task-1"
        assert result.worker_id == "worker-1"
        assert result.status == "completed"
        assert result.result == {"output": "test output"}
        assert result.error is None
        assert result.execution_time_ms == 150.0

    def test_failure_result(self) -> None:
        """Test creating a failed TaskResult."""
        result = TaskResult(
            task_id="task-1",
            worker_id="worker-1",
            status="failed",
            error="Connection timeout",
            execution_time_ms=30000.0,
        )

        assert result.status == "failed"
        assert result.error == "Connection timeout"
        assert result.result is None


# ============================================================================
# WorkerQueueThread Tests
# ============================================================================


class TestWorkerQueueThreadBasic:
    """Basic functionality tests for WorkerQueueThread."""

    @pytest.fixture
    def mock_callback(self) -> MagicMock:
        """Create a mock callback function."""
        return MagicMock()

    @pytest.fixture
    def worker_thread(self, mock_callback: MagicMock) -> WorkerQueueThread:
        """Create a WorkerQueueThread for testing."""
        return WorkerQueueThread(
            worker_id="test-worker",
            worker_endpoint="http://localhost:8001",
            model_id="test-model",
            callback=mock_callback,
            http_timeout=5.0,
            max_retries=3,
            retry_delay=0.1,
        )

    def test_initialization(self, worker_thread: WorkerQueueThread) -> None:
        """Test WorkerQueueThread initialization."""
        assert worker_thread.worker_id == "test-worker"
        assert worker_thread.worker_endpoint == "http://localhost:8001"
        assert worker_thread.model_id == "test-model"
        assert worker_thread.queue_size() == 0
        assert not worker_thread.has_running_task()

    def test_start_and_stop_empty_queue(self, worker_thread: WorkerQueueThread) -> None:
        """Test starting and stopping with empty queue."""
        worker_thread.start()

        # Give thread time to start
        time.sleep(0.1)

        pending = worker_thread.stop(timeout=2.0)
        assert pending == []

    def test_cannot_start_twice(self, worker_thread: WorkerQueueThread) -> None:
        """Test that starting twice raises error."""
        worker_thread.start()

        try:
            with pytest.raises(RuntimeError, match="already started"):
                worker_thread.start()
        finally:
            worker_thread.stop()

    def test_enqueue_returns_queue_size(self, worker_thread: WorkerQueueThread) -> None:
        """Test that enqueue returns current queue size."""
        task1 = QueuedTask(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "hello"},
            metadata={},
            enqueue_time=time.time(),
        )
        task2 = QueuedTask(
            task_id="task-2",
            model_id="test-model",
            task_input={"prompt": "world"},
            metadata={},
            enqueue_time=time.time(),
        )

        size1 = worker_thread.enqueue(task1)
        size2 = worker_thread.enqueue(task2)

        assert size1 == 1
        assert size2 == 2
        assert worker_thread.queue_size() == 2


class TestWorkerQueueThreadExecution:
    """Tests for task execution functionality."""

    @pytest.fixture
    def results(self) -> list[TaskResult]:
        """Storage for callback results."""
        return []

    @pytest.fixture
    def callback(self, results: list[TaskResult]) -> callable:
        """Create a callback that stores results."""

        def _callback(result: TaskResult) -> None:
            results.append(result)

        return _callback

    @pytest.fixture
    def worker_thread(self, callback: callable) -> WorkerQueueThread:
        """Create a WorkerQueueThread for testing."""
        return WorkerQueueThread(
            worker_id="test-worker",
            worker_endpoint="http://localhost:8001",
            model_id="test-model",
            callback=callback,
            http_timeout=5.0,
            max_retries=3,
            retry_delay=0.1,
        )

    def test_enqueue_and_process_success(
        self,
        worker_thread: WorkerQueueThread,
        results: list[TaskResult],
    ) -> None:
        """Test basic enqueue and process flow with successful execution."""
        # Mock HTTP response
        with patch.object(worker_thread, "_call_worker_api") as mock_api:
            mock_api.return_value = (
                {"output": "test response"},
                200,
                {"content-type": "application/json"},
            )

            worker_thread.start()

            task = QueuedTask(
                task_id="task-1",
                model_id="test-model",
                task_input={"prompt": "hello"},
                metadata={},
                enqueue_time=time.time(),
            )
            worker_thread.enqueue(task)

            # Wait for processing
            time.sleep(0.5)
            worker_thread.stop()

            assert len(results) == 1
            assert results[0].task_id == "task-1"
            assert results[0].status == "completed"
            assert results[0].result == {"output": "test response"}
            mock_api.assert_called_once()

    def test_callback_invoked_on_failure(
        self,
        worker_thread: WorkerQueueThread,
        results: list[TaskResult],
    ) -> None:
        """Test callback receives failure result on error."""
        with patch.object(worker_thread, "_call_worker_api") as mock_api:
            mock_api.side_effect = httpx.HTTPStatusError(
                "Server error",
                request=MagicMock(),
                response=MagicMock(status_code=500, text="Internal error"),
            )

            worker_thread.start()

            task = QueuedTask(
                task_id="task-1",
                model_id="test-model",
                task_input={"prompt": "hello"},
                metadata={},
                enqueue_time=time.time(),
            )
            worker_thread.enqueue(task)

            # Wait for processing
            time.sleep(0.5)
            worker_thread.stop()

            assert len(results) == 1
            assert results[0].task_id == "task-1"
            assert results[0].status == "failed"
            assert results[0].error is not None

    def test_graceful_stop_returns_pending_tasks(
        self,
        worker_thread: WorkerQueueThread,
    ) -> None:
        """Test stop returns pending tasks that were not processed."""
        # Don't start the thread - tasks remain in queue

        task1 = QueuedTask(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "hello"},
            metadata={},
            enqueue_time=time.time(),
        )
        task2 = QueuedTask(
            task_id="task-2",
            model_id="test-model",
            task_input={"prompt": "world"},
            metadata={},
            enqueue_time=time.time(),
        )

        worker_thread.enqueue(task1)
        worker_thread.enqueue(task2)

        # Stop without starting - should return pending tasks
        pending = worker_thread.stop()

        assert len(pending) == 2
        assert pending[0].task_id == "task-1"
        assert pending[1].task_id == "task-2"


class TestWorkerQueueThreadRetry:
    """Tests for HTTP retry logic."""

    @pytest.fixture
    def results(self) -> list[TaskResult]:
        """Storage for callback results."""
        return []

    @pytest.fixture
    def callback(self, results: list[TaskResult]) -> callable:
        """Create a callback that stores results."""

        def _callback(result: TaskResult) -> None:
            results.append(result)

        return _callback

    @pytest.fixture
    def worker_thread(self, callback: callable) -> WorkerQueueThread:
        """Create a WorkerQueueThread with fast retry for testing."""
        return WorkerQueueThread(
            worker_id="test-worker",
            worker_endpoint="http://localhost:8001",
            model_id="test-model",
            callback=callback,
            http_timeout=1.0,
            max_retries=3,
            retry_delay=0.01,  # Fast retry for tests
        )

    def test_retry_on_connection_error(
        self,
        worker_thread: WorkerQueueThread,
        results: list[TaskResult],
    ) -> None:
        """Test retry logic for connection errors.

        We mock at the HTTP client level to test the actual retry logic
        inside _call_worker_api.
        """
        call_count = 0

        def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("Connection refused")
            # Return a mock response
            response = MagicMock()
            response.json.return_value = {"output": "success after retry"}
            response.status_code = 200
            response.headers = {"content-type": "application/json"}
            response.raise_for_status = MagicMock()
            return response

        worker_thread.start()

        # Wait for thread to create HTTP client
        time.sleep(0.2)

        # Patch the HTTP client's request method
        with patch.object(
            worker_thread._http_client, "request", side_effect=mock_request
        ):
            task = QueuedTask(
                task_id="task-1",
                model_id="test-model",
                task_input={"prompt": "hello"},
                metadata={},
                enqueue_time=time.time(),
            )
            worker_thread.enqueue(task)

            # Wait for retries (0.01 * 2^0 + 0.01 * 2^1 = 0.03s, plus processing)
            time.sleep(1.0)

        worker_thread.stop()

        assert call_count == 3
        assert len(results) == 1
        assert results[0].status == "completed"

    def test_no_retry_on_http_status_error(
        self,
        worker_thread: WorkerQueueThread,
        results: list[TaskResult],
    ) -> None:
        """Test no retry on HTTP 4xx/5xx errors."""
        call_count = 0

        def mock_api(task: QueuedTask) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            raise httpx.HTTPStatusError(
                "Bad request",
                request=MagicMock(),
                response=MagicMock(status_code=400, text="Bad request"),
            )

        with patch.object(worker_thread, "_call_worker_api", side_effect=mock_api):
            worker_thread.start()

            task = QueuedTask(
                task_id="task-1",
                model_id="test-model",
                task_input={"prompt": "hello"},
                metadata={},
                enqueue_time=time.time(),
            )
            worker_thread.enqueue(task)

            time.sleep(0.5)
            worker_thread.stop()

            # Should only call once - no retry on HTTP error
            assert call_count == 1
            assert results[0].status == "failed"

    def test_no_retry_on_timeout(
        self,
        worker_thread: WorkerQueueThread,
        results: list[TaskResult],
    ) -> None:
        """Test no retry on HTTP timeout (model may be processing)."""
        call_count = 0

        def mock_api(task: QueuedTask) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            raise httpx.TimeoutException("Request timed out")

        with patch.object(worker_thread, "_call_worker_api", side_effect=mock_api):
            worker_thread.start()

            task = QueuedTask(
                task_id="task-1",
                model_id="test-model",
                task_input={"prompt": "hello"},
                metadata={},
                enqueue_time=time.time(),
            )
            worker_thread.enqueue(task)

            time.sleep(0.5)
            worker_thread.stop()

            # Should only call once - no retry on timeout
            assert call_count == 1
            assert results[0].status == "failed"


class TestWorkerQueueThreadEstimation:
    """Tests for wait time estimation."""

    @pytest.fixture
    def worker_thread(self) -> WorkerQueueThread:
        """Create a WorkerQueueThread for testing."""
        return WorkerQueueThread(
            worker_id="test-worker",
            worker_endpoint="http://localhost:8001",
            model_id="test-model",
            callback=lambda x: None,
        )

    def test_estimated_wait_time_empty_queue(
        self, worker_thread: WorkerQueueThread
    ) -> None:
        """Test wait time estimation with empty queue."""
        wait_time = worker_thread.get_estimated_wait_time(avg_exec_time_ms=100.0)
        assert wait_time == 0.0

    def test_estimated_wait_time_with_queued_tasks(
        self, worker_thread: WorkerQueueThread
    ) -> None:
        """Test wait time estimation with tasks in queue."""
        # Enqueue 3 tasks
        for i in range(3):
            task = QueuedTask(
                task_id=f"task-{i}",
                model_id="test-model",
                task_input={"prompt": f"hello {i}"},
                metadata={},
                enqueue_time=time.time(),
            )
            worker_thread.enqueue(task)

        # With avg 100ms per task and 3 tasks, expect 300ms wait
        wait_time = worker_thread.get_estimated_wait_time(avg_exec_time_ms=100.0)
        assert wait_time == 300.0


class TestWorkerQueueThreadFIFO:
    """Tests for FIFO ordering."""

    @pytest.fixture
    def results(self) -> list[TaskResult]:
        """Storage for callback results."""
        return []

    @pytest.fixture
    def callback(self, results: list[TaskResult]) -> callable:
        """Create a callback that stores results."""

        def _callback(result: TaskResult) -> None:
            results.append(result)

        return _callback

    @pytest.fixture
    def worker_thread(self, callback: callable) -> WorkerQueueThread:
        """Create a WorkerQueueThread for testing."""
        return WorkerQueueThread(
            worker_id="test-worker",
            worker_endpoint="http://localhost:8001",
            model_id="test-model",
            callback=callback,
            http_timeout=5.0,
        )

    def test_tasks_processed_in_fifo_order(
        self,
        worker_thread: WorkerQueueThread,
        results: list[TaskResult],
    ) -> None:
        """Test that tasks are processed in FIFO order."""
        processed_order: list[str] = []

        def mock_api(task: QueuedTask) -> dict[str, Any]:
            processed_order.append(task.task_id)
            return {"task_id": task.task_id}

        with patch.object(worker_thread, "_call_worker_api", side_effect=mock_api):
            # Enqueue tasks before starting
            for i in range(5):
                task = QueuedTask(
                    task_id=f"task-{i}",
                    model_id="test-model",
                    task_input={"prompt": f"hello {i}"},
                    metadata={},
                    enqueue_time=time.time() + i,  # Later enqueue time
                )
                worker_thread.enqueue(task)

            worker_thread.start()

            # Wait for all tasks to process
            time.sleep(1.0)
            worker_thread.stop()

            # Tasks should be processed in enqueue order
            assert processed_order == [f"task-{i}" for i in range(5)]
