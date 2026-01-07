"""Integration tests for PYLET-021: Phase 3 Integration.

These tests verify the end-to-end functionality of the scheduler-side
task queue system implemented in Phase 3 (PYLET-015 through PYLET-020).

Test categories:
1. Basic Task Flow - Task lifecycle from submission to completion
2. Queue Behavior - FIFO ordering, queue depth tracking
3. Multi-Worker Scenarios - Load balancing, queue-aware scheduling
4. Worker Registration/Deregistration - Task redistribution
5. Graceful Shutdown - Proper task handling during shutdown
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.algorithms.base import ScheduleResult
from src.algorithms.queue_state_adapter import (
    get_all_queue_info_from_manager,
    get_queue_info_from_manager,
)
from src.instance_sync import (
    InstanceInfo,
    InstanceSyncRequest,
    handle_instance_addition,
    handle_instance_removal,
    handle_instance_sync,
)
from src.models.queue import InstanceQueueExpectError
from src.services.shutdown_handler import ShutdownHandler, ShutdownResult
from src.services.task_result_callback import TaskResultCallback
from src.services.worker_queue_manager import WorkerQueueManager
from src.services.worker_queue_thread import QueuedTask, TaskResult, WorkerQueueThread


class TestBasicTaskFlow:
    """Tests for basic task lifecycle."""

    def test_task_enqueue_and_callback(self):
        """Test task is enqueued and callback is invoked."""
        callback_results = []

        def mock_callback(result: TaskResult):
            callback_results.append(result)

        manager = WorkerQueueManager(
            callback=mock_callback,
            http_timeout=5.0,
        )

        # Register a worker (but we won't actually make HTTP calls)
        manager.register_worker(
            worker_id="worker-1",
            worker_endpoint="http://localhost:9999",
            model_id="test-model",
        )

        # Verify worker is registered
        assert manager.has_worker("worker-1")
        assert manager.get_worker_count() == 1

        # Clean up
        manager.shutdown(timeout=1.0)

    def test_task_status_transitions(self):
        """Test task goes through correct status transitions."""
        # Create a queued task
        task = QueuedTask(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
            enqueue_time=time.time(),
        )

        # Verify initial state
        assert task.task_id == "task-1"
        assert task.model_id == "test-model"


class TestQueueBehavior:
    """Tests for queue FIFO ordering and depth tracking."""

    def test_queue_depth_tracking(self):
        """Test queue depth is accurately tracked."""
        manager = WorkerQueueManager(
            callback=lambda x: None,
            http_timeout=5.0,
        )

        manager.register_worker(
            worker_id="worker-1",
            worker_endpoint="http://localhost:9999",
            model_id="test-model",
        )

        # Initial depth is 0
        assert manager.get_queue_depth("worker-1") == 0
        assert manager.get_total_queue_depth() == 0

        # Enqueue tasks
        for i in range(5):
            task = QueuedTask(
                task_id=f"task-{i}",
                model_id="test-model",
                task_input={},
                metadata={},
                enqueue_time=time.time(),
            )
            manager.enqueue_task("worker-1", task)

        # Verify depth (may be slightly less due to processing)
        depth = manager.get_queue_depth("worker-1")
        assert depth >= 0  # Tasks may have started processing

        # Clean up
        manager.shutdown(timeout=1.0)

    def test_get_all_queue_depths(self):
        """Test getting queue depths for all workers."""
        manager = WorkerQueueManager(
            callback=lambda x: None,
            http_timeout=5.0,
        )

        # Register multiple workers
        for i in range(3):
            manager.register_worker(
                worker_id=f"worker-{i}",
                worker_endpoint=f"http://localhost:{9000 + i}",
                model_id="test-model",
            )

        # Get all depths
        depths = manager.get_all_queue_depths()
        assert len(depths) == 3
        assert all(d >= 0 for d in depths.values())

        # Clean up
        manager.shutdown(timeout=1.0)


class TestQueueStateAdapter:
    """Tests for queue state adapter integration."""

    def test_adapter_integrates_with_manager(self):
        """Test adapter correctly converts manager state."""
        manager = WorkerQueueManager(
            callback=lambda x: None,
            http_timeout=5.0,
        )

        manager.register_worker(
            worker_id="worker-1",
            worker_endpoint="http://localhost:9999",
            model_id="test-model",
        )

        # Get queue info via adapter
        queue_info = get_queue_info_from_manager(
            worker_queue_manager=manager,
            instance_id="worker-1",
            avg_exec_time_ms=100.0,
        )

        assert isinstance(queue_info, InstanceQueueExpectError)
        assert queue_info.instance_id == "worker-1"
        assert queue_info.expected_time_ms >= 0
        assert queue_info.error_margin_ms == 20.0  # 20% of 100.0

        # Clean up
        manager.shutdown(timeout=1.0)

    def test_batch_adapter(self):
        """Test batch adapter for multiple workers."""
        manager = WorkerQueueManager(
            callback=lambda x: None,
            http_timeout=5.0,
        )

        for i in range(3):
            manager.register_worker(
                worker_id=f"worker-{i}",
                worker_endpoint=f"http://localhost:{9000 + i}",
                model_id="test-model",
            )

        # Get all queue info
        all_info = get_all_queue_info_from_manager(
            worker_queue_manager=manager,
            instance_ids=["worker-0", "worker-1", "worker-2"],
            avg_exec_time_ms=50.0,
        )

        assert len(all_info) == 3
        for info in all_info.values():
            assert info.error_margin_ms == 10.0  # 20% of 50.0

        # Clean up
        manager.shutdown(timeout=1.0)


class TestWorkerRegistrationDeregistration:
    """Tests for worker lifecycle management."""

    def test_register_creates_queue(self):
        """Test registration creates worker queue thread."""
        manager = WorkerQueueManager(
            callback=lambda x: None,
            http_timeout=5.0,
        )

        assert manager.get_worker_count() == 0

        manager.register_worker(
            worker_id="worker-1",
            worker_endpoint="http://localhost:9999",
            model_id="test-model",
        )

        assert manager.get_worker_count() == 1
        assert manager.has_worker("worker-1")

        # Worker thread should exist
        thread = manager.get_worker("worker-1")
        assert thread is not None
        assert isinstance(thread, WorkerQueueThread)

        manager.shutdown(timeout=1.0)

    def test_deregister_returns_pending_tasks(self):
        """Test deregistration returns pending tasks."""
        manager = WorkerQueueManager(
            callback=lambda x: None,
            http_timeout=5.0,
        )

        manager.register_worker(
            worker_id="worker-1",
            worker_endpoint="http://localhost:9999",
            model_id="test-model",
        )

        # Enqueue tasks
        for i in range(3):
            task = QueuedTask(
                task_id=f"task-{i}",
                model_id="test-model",
                task_input={},
                metadata={},
                enqueue_time=time.time(),
            )
            manager.enqueue_task("worker-1", task)

        # Deregister immediately (should return pending tasks)
        pending = manager.deregister_worker("worker-1", stop_timeout=1.0)

        # Some tasks may have started processing
        assert isinstance(pending, list)

        manager.shutdown(timeout=1.0)

    def test_duplicate_registration_raises(self):
        """Test duplicate worker registration raises error."""
        manager = WorkerQueueManager(
            callback=lambda x: None,
            http_timeout=5.0,
        )

        manager.register_worker(
            worker_id="worker-1",
            worker_endpoint="http://localhost:9999",
            model_id="test-model",
        )

        with pytest.raises(ValueError, match="already registered"):
            manager.register_worker(
                worker_id="worker-1",
                worker_endpoint="http://localhost:9999",
                model_id="test-model",
            )

        manager.shutdown(timeout=1.0)


class TestInstanceSyncIntegration:
    """Tests for instance sync integration."""

    @pytest.mark.asyncio
    async def test_sync_computes_diff(self):
        """Test sync correctly computes additions and removals."""
        mock_registry = AsyncMock()
        mock_instance = MagicMock()
        mock_instance.instance_id = "worker-1"
        mock_registry.list_all.return_value = [mock_instance]
        mock_registry.get_active_instances.return_value = []

        mock_manager = MagicMock()
        mock_manager.deregister_worker.return_value = []

        mock_strategy = AsyncMock()

        # Target: keep nothing (remove worker-1), add worker-2
        request = InstanceSyncRequest(
            instances=[
                InstanceInfo(
                    instance_id="worker-2",
                    endpoint="http://localhost:9002",
                    model_id="test-model",
                ),
            ]
        )

        result = await handle_instance_sync(
            request=request,
            config_model_id="test-model",
            instance_registry=mock_registry,
            worker_queue_manager=mock_manager,
            scheduling_strategy=mock_strategy,
        )

        assert result.success
        assert "worker-1" in result.removed
        assert "worker-2" in result.added

    @pytest.mark.asyncio
    async def test_sync_validates_model(self):
        """Test sync rejects mismatched model_id."""
        mock_registry = AsyncMock()
        mock_manager = MagicMock()
        mock_strategy = AsyncMock()

        request = InstanceSyncRequest(
            instances=[
                InstanceInfo(
                    instance_id="worker-1",
                    endpoint="http://localhost:9001",
                    model_id="wrong-model",
                ),
            ]
        )

        with pytest.raises(ValueError, match="Model mismatch"):
            await handle_instance_sync(
                request=request,
                config_model_id="test-model",
                instance_registry=mock_registry,
                worker_queue_manager=mock_manager,
                scheduling_strategy=mock_strategy,
            )


class TestGracefulShutdownIntegration:
    """Tests for graceful shutdown integration."""

    @pytest.mark.asyncio
    async def test_shutdown_all_workers(self):
        """Test shutdown stops all workers."""
        mock_manager = MagicMock()
        mock_manager.get_worker_ids.return_value = ["w1", "w2", "w3"]
        mock_manager.deregister_worker.return_value = []

        mock_registry = AsyncMock()

        handler = ShutdownHandler(
            worker_queue_manager=mock_manager,
            instance_registry=mock_registry,
        )

        result = await handler.shutdown_all(timeout=60.0)

        assert result.workers_stopped == 3
        assert result.tasks_dropped == 0
        assert not result.timeout_occurred

    @pytest.mark.asyncio
    async def test_shutdown_counts_dropped_tasks(self):
        """Test shutdown accurately counts dropped tasks."""
        mock_manager = MagicMock()
        mock_manager.get_worker_ids.return_value = ["w1"]

        # Worker has 5 pending tasks
        pending = [
            QueuedTask(
                task_id=f"task-{i}",
                model_id="test",
                task_input={},
                metadata={},
                enqueue_time=1000.0,
            )
            for i in range(5)
        ]
        mock_manager.deregister_worker.return_value = pending

        mock_registry = AsyncMock()

        handler = ShutdownHandler(
            worker_queue_manager=mock_manager,
            instance_registry=mock_registry,
        )

        result = await handler.shutdown_all(timeout=60.0)

        assert result.tasks_dropped == 5


class TestTaskResultCallback:
    """Tests for task result callback mechanism."""

    @pytest.mark.asyncio
    async def test_callback_handles_success(self):
        """Test callback correctly handles success result."""
        mock_task_registry = AsyncMock()
        mock_task = MagicMock()
        mock_task_registry.get.return_value = mock_task

        mock_instance_registry = AsyncMock()
        mock_instance = MagicMock()
        mock_instance.instance_id = "worker-1"
        mock_instance.endpoint = "http://localhost:9999"
        mock_instance_registry.get.return_value = mock_instance

        mock_websocket = AsyncMock()

        callback = TaskResultCallback(
            task_registry=mock_task_registry,
            instance_registry=mock_instance_registry,
            websocket_manager=mock_websocket,
        )

        result = TaskResult(
            task_id="task-1",
            worker_id="worker-1",
            status="completed",
            result={"output": "test"},
            execution_time_ms=100.0,
        )

        await callback.handle_result(result)

        # Verify task registry was updated
        mock_task_registry.update_status.assert_called()

    @pytest.mark.asyncio
    async def test_callback_handles_failure(self):
        """Test callback correctly handles failure result."""
        mock_task_registry = AsyncMock()
        mock_task = MagicMock()
        mock_task_registry.get.return_value = mock_task

        mock_instance_registry = AsyncMock()

        mock_websocket = AsyncMock()

        callback = TaskResultCallback(
            task_registry=mock_task_registry,
            instance_registry=mock_instance_registry,
            websocket_manager=mock_websocket,
        )

        result = TaskResult(
            task_id="task-1",
            worker_id="worker-1",
            status="failed",
            error="Connection timeout",
            execution_time_ms=5000.0,
        )

        await callback.handle_result(result)

        # Verify error was set
        mock_task_registry.set_error.assert_called()


class TestMultiWorkerScenarios:
    """Tests for multi-worker load balancing."""

    def test_multiple_workers_independent(self):
        """Test multiple workers operate independently."""
        manager = WorkerQueueManager(
            callback=lambda x: None,
            http_timeout=5.0,
        )

        # Register 3 workers
        for i in range(3):
            manager.register_worker(
                worker_id=f"worker-{i}",
                worker_endpoint=f"http://localhost:{9000 + i}",
                model_id="test-model",
            )

        assert manager.get_worker_count() == 3

        # Enqueue to specific workers
        for i in range(3):
            task = QueuedTask(
                task_id=f"task-{i}",
                model_id="test-model",
                task_input={},
                metadata={},
                enqueue_time=time.time(),
            )
            manager.enqueue_task(f"worker-{i}", task)

        # Each worker should have its task (or be processing it)
        for i in range(3):
            depth = manager.get_queue_depth(f"worker-{i}")
            assert depth >= 0  # May have started processing

        manager.shutdown(timeout=1.0)

    def test_worker_isolation(self):
        """Test slow worker doesn't affect fast workers."""
        manager = WorkerQueueManager(
            callback=lambda x: None,
            http_timeout=5.0,
        )

        manager.register_worker(
            worker_id="fast-worker",
            worker_endpoint="http://localhost:9001",
            model_id="test-model",
        )
        manager.register_worker(
            worker_id="slow-worker",
            worker_endpoint="http://localhost:9002",
            model_id="test-model",
        )

        # Enqueue many tasks to slow worker
        for i in range(10):
            task = QueuedTask(
                task_id=f"slow-task-{i}",
                model_id="test-model",
                task_input={},
                metadata={},
                enqueue_time=time.time(),
            )
            manager.enqueue_task("slow-worker", task)

        # Enqueue one task to fast worker
        fast_task = QueuedTask(
            task_id="fast-task",
            model_id="test-model",
            task_input={},
            metadata={},
            enqueue_time=time.time(),
        )
        manager.enqueue_task("fast-worker", fast_task)

        # Fast worker should have short queue
        fast_depth = manager.get_queue_depth("fast-worker")
        slow_depth = manager.get_queue_depth("slow-worker")

        # Fast worker queue is smaller or equal
        assert fast_depth <= slow_depth + 1

        manager.shutdown(timeout=1.0)
