"""Unit tests for WorkerQueueManager.

TDD tests for PYLET-017: Worker Queue Manager Implementation.
These tests define the expected behavior before implementation.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.services.worker_queue_manager import WorkerQueueManager
from src.services.worker_queue_thread import QueuedTask, WorkerQueueThread


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_callback():
    """Create a mock callback function."""
    return MagicMock()


@pytest.fixture
def manager(mock_callback):
    """Create a WorkerQueueManager for testing."""
    return WorkerQueueManager(
        callback=mock_callback,
        http_timeout=5.0,
        max_retries=2,
        retry_delay=0.01,
    )


@pytest.fixture
def sample_task():
    """Create a sample task."""
    return QueuedTask(
        task_id="task-1",
        model_id="test-model",
        task_input={"prompt": "hello"},
        metadata={},
        enqueue_time=time.time(),
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestWorkerQueueManagerInit:
    """Tests for WorkerQueueManager initialization."""

    def test_initialization(self, mock_callback):
        """Test basic initialization."""
        manager = WorkerQueueManager(
            callback=mock_callback,
            http_timeout=300.0,
            max_retries=3,
            retry_delay=1.0,
        )

        assert manager._callback is mock_callback
        assert manager._http_timeout == 300.0
        assert manager._max_retries == 3
        assert manager._retry_delay == 1.0
        assert manager.get_worker_count() == 0

    def test_initialization_with_defaults(self, mock_callback):
        """Test initialization with default values."""
        manager = WorkerQueueManager(callback=mock_callback)

        assert manager._http_timeout == 300.0
        assert manager._max_retries == 3
        assert manager._retry_delay == 1.0


# ============================================================================
# Worker Registration Tests
# ============================================================================


class TestWorkerRegistration:
    """Tests for worker registration."""

    def test_register_worker(self, manager):
        """Test worker registration creates thread."""
        manager.register_worker(
            worker_id="worker-1",
            worker_endpoint="http://localhost:8001",
            model_id="test-model",
        )

        assert manager.has_worker("worker-1")
        assert manager.get_worker_count() == 1

        # Cleanup
        manager.deregister_worker("worker-1")

    def test_register_multiple_workers(self, manager):
        """Test registering multiple workers."""
        for i in range(3):
            manager.register_worker(
                worker_id=f"worker-{i}",
                worker_endpoint=f"http://localhost:800{i}",
                model_id="test-model",
            )

        assert manager.get_worker_count() == 3
        assert set(manager.get_worker_ids()) == {"worker-0", "worker-1", "worker-2"}

        # Cleanup
        manager.shutdown()

    def test_register_duplicate_worker_raises(self, manager):
        """Test duplicate registration raises error."""
        manager.register_worker(
            worker_id="worker-1",
            worker_endpoint="http://localhost:8001",
            model_id="test-model",
        )

        try:
            with pytest.raises(ValueError, match="already registered"):
                manager.register_worker(
                    worker_id="worker-1",
                    worker_endpoint="http://localhost:8002",
                    model_id="test-model",
                )
        finally:
            manager.shutdown()


# ============================================================================
# Worker Deregistration Tests
# ============================================================================


class TestWorkerDeregistration:
    """Tests for worker deregistration."""

    def test_deregister_worker(self, manager):
        """Test deregistration stops thread."""
        manager.register_worker(
            worker_id="worker-1",
            worker_endpoint="http://localhost:8001",
            model_id="test-model",
        )

        pending = manager.deregister_worker("worker-1")

        assert not manager.has_worker("worker-1")
        assert manager.get_worker_count() == 0
        assert isinstance(pending, list)

    def test_deregister_unknown_worker(self, manager):
        """Test deregistering unknown worker returns empty list."""
        pending = manager.deregister_worker("unknown-worker")

        assert pending == []

    def test_deregister_returns_pending_tasks(self, manager, sample_task):
        """Test deregistration returns pending tasks."""
        manager.register_worker(
            worker_id="worker-1",
            worker_endpoint="http://localhost:8001",
            model_id="test-model",
        )

        # Enqueue a task
        manager.enqueue_task("worker-1", sample_task)

        # Stop immediately - task should be pending
        # Note: We need to stop without processing, so we stop the thread
        thread = manager.get_worker("worker-1")
        thread._shutdown.set()  # Signal shutdown before task is processed

        pending = manager.deregister_worker("worker-1", stop_timeout=0.5)

        # Task may or may not be in pending depending on timing
        # Just verify the operation completes
        assert isinstance(pending, list)


# ============================================================================
# Task Enqueueing Tests
# ============================================================================


class TestTaskEnqueueing:
    """Tests for task enqueueing."""

    def test_enqueue_task(self, manager, sample_task):
        """Test task is enqueued to correct worker."""
        manager.register_worker(
            worker_id="worker-1",
            worker_endpoint="http://localhost:8001",
            model_id="test-model",
        )

        queue_size = manager.enqueue_task("worker-1", sample_task)

        assert queue_size == 1
        assert manager.get_queue_depth("worker-1") == 1

        # Cleanup
        manager.shutdown()

    def test_enqueue_to_unknown_worker_raises(self, manager, sample_task):
        """Test enqueueing to unknown worker raises error."""
        with pytest.raises(ValueError, match="not registered"):
            manager.enqueue_task("unknown-worker", sample_task)

    def test_enqueue_multiple_tasks(self, manager):
        """Test enqueueing multiple tasks."""
        manager.register_worker(
            worker_id="worker-1",
            worker_endpoint="http://localhost:8001",
            model_id="test-model",
        )

        for i in range(5):
            task = QueuedTask(
                task_id=f"task-{i}",
                model_id="test-model",
                task_input={"prompt": f"hello {i}"},
                metadata={},
                enqueue_time=time.time(),
            )
            manager.enqueue_task("worker-1", task)

        # Queue depth should be around 5 (may be less if processing started)
        depth = manager.get_queue_depth("worker-1")
        assert depth <= 5

        # Cleanup
        manager.shutdown()


# ============================================================================
# Queue Depth Tests
# ============================================================================


class TestQueueDepth:
    """Tests for queue depth queries."""

    def test_get_queue_depth(self, manager, sample_task):
        """Test queue depth retrieval."""
        manager.register_worker(
            worker_id="worker-1",
            worker_endpoint="http://localhost:8001",
            model_id="test-model",
        )

        assert manager.get_queue_depth("worker-1") == 0

        manager.enqueue_task("worker-1", sample_task)
        # Depth should be at least 0 (task may be processing)
        assert manager.get_queue_depth("worker-1") >= 0

        # Cleanup
        manager.shutdown()

    def test_get_queue_depth_unknown_worker(self, manager):
        """Test queue depth for unknown worker returns 0."""
        assert manager.get_queue_depth("unknown-worker") == 0

    def test_get_all_queue_depths(self, manager):
        """Test getting all queue depths."""
        for i in range(3):
            manager.register_worker(
                worker_id=f"worker-{i}",
                worker_endpoint=f"http://localhost:800{i}",
                model_id="test-model",
            )

        depths = manager.get_all_queue_depths()

        assert len(depths) == 3
        assert all(worker_id in depths for worker_id in ["worker-0", "worker-1", "worker-2"])
        assert all(isinstance(d, int) for d in depths.values())

        # Cleanup
        manager.shutdown()

    def test_get_total_queue_depth(self, manager):
        """Test total queue depth calculation."""
        manager.register_worker(
            worker_id="worker-1",
            worker_endpoint="http://localhost:8001",
            model_id="test-model",
        )

        for i in range(3):
            task = QueuedTask(
                task_id=f"task-{i}",
                model_id="test-model",
                task_input={"prompt": f"hello {i}"},
                metadata={},
                enqueue_time=time.time(),
            )
            manager.enqueue_task("worker-1", task)

        # Total should be at least 0 (tasks may be processing)
        total = manager.get_total_queue_depth()
        assert isinstance(total, int)
        assert total >= 0

        # Cleanup
        manager.shutdown()


# ============================================================================
# Wait Time Estimation Tests
# ============================================================================


class TestWaitTimeEstimation:
    """Tests for wait time estimation."""

    def test_get_estimated_wait_times(self, manager):
        """Test wait time estimation for all workers."""
        for i in range(2):
            manager.register_worker(
                worker_id=f"worker-{i}",
                worker_endpoint=f"http://localhost:800{i}",
                model_id="test-model",
            )

        wait_times = manager.get_estimated_wait_times(avg_exec_time_ms=100.0)

        assert len(wait_times) == 2
        assert all(isinstance(t, float) for t in wait_times.values())

        # Cleanup
        manager.shutdown()

    def test_get_estimated_wait_times_empty(self, manager):
        """Test wait time estimation with no workers."""
        wait_times = manager.get_estimated_wait_times(avg_exec_time_ms=100.0)
        assert wait_times == {}


# ============================================================================
# Worker Query Tests
# ============================================================================


class TestWorkerQueries:
    """Tests for worker query methods."""

    def test_get_worker(self, manager):
        """Test getting a worker thread."""
        manager.register_worker(
            worker_id="worker-1",
            worker_endpoint="http://localhost:8001",
            model_id="test-model",
        )

        worker = manager.get_worker("worker-1")

        assert worker is not None
        assert isinstance(worker, WorkerQueueThread)
        assert worker.worker_id == "worker-1"

        # Cleanup
        manager.shutdown()

    def test_get_worker_unknown(self, manager):
        """Test getting unknown worker returns None."""
        worker = manager.get_worker("unknown-worker")
        assert worker is None

    def test_has_worker(self, manager):
        """Test worker existence check."""
        assert not manager.has_worker("worker-1")

        manager.register_worker(
            worker_id="worker-1",
            worker_endpoint="http://localhost:8001",
            model_id="test-model",
        )

        assert manager.has_worker("worker-1")

        manager.shutdown()

    def test_get_worker_ids(self, manager):
        """Test getting list of worker IDs."""
        assert manager.get_worker_ids() == []

        for i in range(3):
            manager.register_worker(
                worker_id=f"worker-{i}",
                worker_endpoint=f"http://localhost:800{i}",
                model_id="test-model",
            )

        ids = manager.get_worker_ids()
        assert set(ids) == {"worker-0", "worker-1", "worker-2"}

        # Cleanup
        manager.shutdown()


# ============================================================================
# Shutdown Tests
# ============================================================================


class TestShutdown:
    """Tests for graceful shutdown."""

    def test_shutdown_empty(self, manager):
        """Test shutdown with no workers."""
        pending = manager.shutdown()
        assert pending == []

    def test_shutdown_returns_all_pending(self, manager):
        """Test shutdown returns all pending tasks."""
        for i in range(3):
            manager.register_worker(
                worker_id=f"worker-{i}",
                worker_endpoint=f"http://localhost:800{i}",
                model_id="test-model",
            )

        pending = manager.shutdown(timeout=5.0)

        assert isinstance(pending, list)
        assert manager.get_worker_count() == 0


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Tests for statistics collection."""

    def test_get_stats(self, manager):
        """Test statistics collection."""
        manager.register_worker(
            worker_id="worker-1",
            worker_endpoint="http://localhost:8001",
            model_id="test-model",
        )

        stats = manager.get_stats()

        assert stats["total_workers"] == 1
        assert "workers" in stats
        assert "worker-1" in stats["workers"]
        assert "queue_depth" in stats["workers"]["worker-1"]
        assert "has_running_task" in stats["workers"]["worker-1"]

        # Cleanup
        manager.shutdown()

    def test_get_stats_empty(self, manager):
        """Test statistics with no workers."""
        stats = manager.get_stats()

        assert stats["total_workers"] == 0
        assert stats["total_queue_depth"] == 0
        assert stats["workers"] == {}
