"""Unit tests for TaskResultCallback.

TDD tests for PYLET-016: Library-Based Callback Mechanism.
These tests define the expected behavior before implementation.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swarmpilot.scheduler.models import TaskStatus
from swarmpilot.scheduler.services.task_result_callback import (
    TaskResultCallback,
)
from swarmpilot.scheduler.services.worker_queue_thread import TaskResult

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_task_registry():
    """Create a mock task registry."""
    registry = MagicMock()
    registry.get = AsyncMock()
    registry.update_status = AsyncMock()
    registry.set_result = AsyncMock()
    registry.set_error = AsyncMock()
    return registry


@pytest.fixture
def mock_instance_registry():
    """Create a mock instance registry."""
    registry = MagicMock()
    registry.get = AsyncMock()
    registry.increment_completed = AsyncMock()
    registry.increment_failed = AsyncMock()
    return registry


@pytest.fixture
def mock_websocket_manager():
    """Create a mock websocket manager."""
    manager = MagicMock()
    manager.broadcast_task_result = AsyncMock()
    return manager


@pytest.fixture
def mock_throughput_tracker():
    """Create a mock throughput tracker."""
    tracker = MagicMock()
    tracker.record_execution_time = AsyncMock()
    return tracker


@pytest.fixture
def callback_handler(
    mock_task_registry,
    mock_instance_registry,
    mock_websocket_manager,
    mock_throughput_tracker,
):
    """Create a TaskResultCallback for testing."""
    return TaskResultCallback(
        task_registry=mock_task_registry,
        instance_registry=mock_instance_registry,
        websocket_manager=mock_websocket_manager,
        throughput_tracker=mock_throughput_tracker,
    )


@pytest.fixture
def mock_training_client():
    """Create a mock training client."""
    client = MagicMock()
    client.add_sample = MagicMock()
    client.flush_if_ready = AsyncMock(return_value=False)
    return client


@pytest.fixture
def sample_task_record():
    """Create a sample task record."""
    task = MagicMock()
    task.task_id = "task-1"
    task.status = TaskStatus.RUNNING
    task.model_id = "test-model"
    task.metadata = {"batch_size": 32}
    task.set_execution_time = MagicMock()
    task.get_timestamps = MagicMock(
        return_value={"submitted_at": "2024-01-01T00:00:00Z"}
    )
    return task


@pytest.fixture
def sample_instance():
    """Create a sample instance."""
    instance = MagicMock()
    instance.instance_id = "worker-1"
    instance.endpoint = "http://localhost:8001"
    instance.platform_info = {"gpu": "A100", "driver": "535"}
    return instance


@pytest.fixture
def success_result():
    """Create a successful TaskResult."""
    return TaskResult(
        task_id="task-1",
        worker_id="worker-1",
        status="completed",
        result={"output": "test result"},
        execution_time_ms=150.0,
    )


@pytest.fixture
def failure_result():
    """Create a failed TaskResult."""
    return TaskResult(
        task_id="task-1",
        worker_id="worker-1",
        status="failed",
        error="Connection timeout",
        execution_time_ms=30000.0,
    )


# ============================================================================
# Basic Initialization Tests
# ============================================================================


class TestTaskResultCallbackInit:
    """Tests for TaskResultCallback initialization."""

    def test_initialization(
        self,
        mock_task_registry,
        mock_instance_registry,
        mock_websocket_manager,
    ):
        """Test basic initialization."""
        callback = TaskResultCallback(
            task_registry=mock_task_registry,
            instance_registry=mock_instance_registry,
            websocket_manager=mock_websocket_manager,
        )

        assert callback.task_registry is mock_task_registry
        assert callback.instance_registry is mock_instance_registry
        assert callback.websocket_manager is mock_websocket_manager
        assert callback.throughput_tracker is None

    def test_initialization_with_throughput_tracker(
        self,
        mock_task_registry,
        mock_instance_registry,
        mock_websocket_manager,
        mock_throughput_tracker,
    ):
        """Test initialization with optional throughput tracker."""
        callback = TaskResultCallback(
            task_registry=mock_task_registry,
            instance_registry=mock_instance_registry,
            websocket_manager=mock_websocket_manager,
            throughput_tracker=mock_throughput_tracker,
        )

        assert callback.throughput_tracker is mock_throughput_tracker


# ============================================================================
# Success Handling Tests
# ============================================================================


class TestSuccessHandling:
    """Tests for successful task completion handling."""

    @pytest.mark.asyncio
    async def test_updates_task_status_on_success(
        self,
        callback_handler,
        mock_task_registry,
        sample_task_record,
        sample_instance,
        success_result,
    ):
        """Test task registry status updated to COMPLETED on success."""
        mock_task_registry.get.return_value = sample_task_record
        callback_handler.instance_registry.get.return_value = sample_instance

        await callback_handler.handle_result(success_result)

        mock_task_registry.update_status.assert_called_once_with(
            "task-1", TaskStatus.COMPLETED
        )

    @pytest.mark.asyncio
    async def test_sets_result_on_success(
        self,
        callback_handler,
        mock_task_registry,
        sample_task_record,
        sample_instance,
        success_result,
    ):
        """Test task result is stored in registry."""
        mock_task_registry.get.return_value = sample_task_record
        callback_handler.instance_registry.get.return_value = sample_instance

        await callback_handler.handle_result(success_result)

        mock_task_registry.set_result.assert_called_once_with(
            "task-1", {"output": "test result"}
        )

    @pytest.mark.asyncio
    async def test_updates_instance_stats_on_success(
        self,
        callback_handler,
        mock_task_registry,
        mock_instance_registry,
        sample_task_record,
        sample_instance,
        success_result,
    ):
        """Test instance completed counter incremented."""
        mock_task_registry.get.return_value = sample_task_record
        mock_instance_registry.get.return_value = sample_instance

        await callback_handler.handle_result(success_result)

        mock_instance_registry.increment_completed.assert_called_once_with(
            "worker-1"
        )

    @pytest.mark.asyncio
    async def test_records_throughput_on_success(
        self,
        callback_handler,
        mock_task_registry,
        mock_throughput_tracker,
        sample_task_record,
        sample_instance,
        success_result,
    ):
        """Test throughput tracker receives execution data."""
        mock_task_registry.get.return_value = sample_task_record
        callback_handler.instance_registry.get.return_value = sample_instance

        await callback_handler.handle_result(success_result)

        mock_throughput_tracker.record_execution_time.assert_called_once_with(
            instance_endpoint="http://localhost:8001",
            execution_time_ms=150.0,
        )


# ============================================================================
# Failure Handling Tests
# ============================================================================


class TestFailureHandling:
    """Tests for failed task handling."""

    @pytest.mark.asyncio
    async def test_updates_task_status_on_failure(
        self,
        callback_handler,
        mock_task_registry,
        sample_task_record,
        sample_instance,
        failure_result,
    ):
        """Test task registry status updated to FAILED on failure."""
        mock_task_registry.get.return_value = sample_task_record
        callback_handler.instance_registry.get.return_value = sample_instance

        await callback_handler.handle_result(failure_result)

        mock_task_registry.update_status.assert_called_once_with(
            "task-1", TaskStatus.FAILED
        )

    @pytest.mark.asyncio
    async def test_sets_error_on_failure(
        self,
        callback_handler,
        mock_task_registry,
        sample_task_record,
        sample_instance,
        failure_result,
    ):
        """Test error message is stored in registry."""
        mock_task_registry.get.return_value = sample_task_record
        callback_handler.instance_registry.get.return_value = sample_instance

        await callback_handler.handle_result(failure_result)

        mock_task_registry.set_error.assert_called_once_with(
            "task-1", "Connection timeout"
        )

    @pytest.mark.asyncio
    async def test_updates_instance_stats_on_failure(
        self,
        callback_handler,
        mock_task_registry,
        mock_instance_registry,
        sample_task_record,
        sample_instance,
        failure_result,
    ):
        """Test instance failed counter incremented."""
        mock_task_registry.get.return_value = sample_task_record
        mock_instance_registry.get.return_value = sample_instance

        await callback_handler.handle_result(failure_result)

        mock_instance_registry.increment_failed.assert_called_once_with(
            "worker-1"
        )


# ============================================================================
# WebSocket Notification Tests
# ============================================================================


class TestWebSocketNotification:
    """Tests for WebSocket broadcast functionality."""

    @pytest.mark.asyncio
    async def test_broadcasts_success_to_websocket(
        self,
        callback_handler,
        mock_task_registry,
        mock_websocket_manager,
        sample_task_record,
        sample_instance,
        success_result,
    ):
        """Test WebSocket notification sent on success."""
        mock_task_registry.get.return_value = sample_task_record
        callback_handler.instance_registry.get.return_value = sample_instance

        await callback_handler.handle_result(success_result)

        mock_websocket_manager.broadcast_task_result.assert_called_once()
        call_kwargs = mock_websocket_manager.broadcast_task_result.call_args[1]
        assert call_kwargs["task_id"] == "task-1"
        assert call_kwargs["result"] == {"output": "test result"}

    @pytest.mark.asyncio
    async def test_broadcasts_failure_to_websocket(
        self,
        callback_handler,
        mock_task_registry,
        mock_websocket_manager,
        sample_task_record,
        sample_instance,
        failure_result,
    ):
        """Test WebSocket notification sent on failure."""
        mock_task_registry.get.return_value = sample_task_record
        callback_handler.instance_registry.get.return_value = sample_instance

        await callback_handler.handle_result(failure_result)

        mock_websocket_manager.broadcast_task_result.assert_called_once()
        call_kwargs = mock_websocket_manager.broadcast_task_result.call_args[1]
        assert call_kwargs["task_id"] == "task-1"
        assert call_kwargs["error"] == "Connection timeout"


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread-safe callback creation."""

    def test_create_thread_callback(self, callback_handler):
        """Test thread-safe callback creation."""
        loop = asyncio.new_event_loop()
        try:
            thread_callback = callback_handler.create_thread_callback(loop)
            assert callable(thread_callback)
            assert callback_handler._loop is loop
        finally:
            loop.close()

    @pytest.mark.asyncio
    async def test_thread_callback_schedules_on_loop(self, callback_handler):
        """Test callback from thread schedules on event loop."""
        # Use the running event loop
        loop = asyncio.get_running_loop()
        thread_callback = callback_handler.create_thread_callback(loop)

        # Mock handle_result to track if it's called
        with patch.object(
            callback_handler, "handle_result", new_callable=AsyncMock
        ) as mock_handle:
            result = TaskResult(
                task_id="task-1",
                worker_id="worker-1",
                status="completed",
                result={"output": "test"},
                execution_time_ms=100.0,
            )

            # Simulate call from another thread
            thread_callback(result)

            # Give the event loop a chance to process
            await asyncio.sleep(0.1)

            mock_handle.assert_called_once_with(result)


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in callback."""

    @pytest.mark.asyncio
    async def test_handles_missing_task_gracefully(
        self,
        callback_handler,
        mock_task_registry,
        success_result,
    ):
        """Test graceful handling of unknown task_id."""
        mock_task_registry.get.return_value = None

        # Should not raise
        await callback_handler.handle_result(success_result)

        # Should not try to update status for missing task
        mock_task_registry.update_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_missing_instance_gracefully(
        self,
        callback_handler,
        mock_task_registry,
        mock_instance_registry,
        sample_task_record,
        success_result,
    ):
        """Test graceful handling when instance not found."""
        mock_task_registry.get.return_value = sample_task_record
        mock_instance_registry.get.return_value = None

        # Should not raise
        await callback_handler.handle_result(success_result)

        # Should still update task status
        mock_task_registry.update_status.assert_called_once()
        # But should not try to update instance stats
        mock_instance_registry.increment_completed.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_registry_error_gracefully(
        self,
        callback_handler,
        mock_task_registry,
        success_result,
    ):
        """Test graceful handling of registry errors."""
        mock_task_registry.get.side_effect = Exception("Database error")

        # Should not raise - errors are logged
        await callback_handler.handle_result(success_result)


# ============================================================================
# No Throughput Tracker Tests
# ============================================================================


class TestWithoutThroughputTracker:
    """Tests when throughput tracker is not configured."""

    @pytest.mark.asyncio
    async def test_works_without_throughput_tracker(
        self,
        mock_task_registry,
        mock_instance_registry,
        mock_websocket_manager,
        sample_task_record,
        sample_instance,
        success_result,
    ):
        """Test callback works when throughput tracker is None."""
        callback = TaskResultCallback(
            task_registry=mock_task_registry,
            instance_registry=mock_instance_registry,
            websocket_manager=mock_websocket_manager,
            throughput_tracker=None,
        )

        mock_task_registry.get.return_value = sample_task_record
        mock_instance_registry.get.return_value = sample_instance

        # Should not raise
        await callback.handle_result(success_result)

        # Other operations should still work
        mock_task_registry.update_status.assert_called_once()


# ============================================================================
# Future Pool Tests
# ============================================================================


class TestFuturePool:
    """Tests for per-task Future pool in TaskResultCallback."""

    @pytest.mark.asyncio
    async def test_register_future_creates_future(self, callback_handler):
        """Test register_future() creates and returns an asyncio.Future."""
        future = callback_handler.register_future("task-42")

        assert isinstance(future, asyncio.Future)
        assert not future.done()

    @pytest.mark.asyncio
    async def test_register_future_duplicate_raises(self, callback_handler):
        """Test register_future() raises if task_id already has a Future."""
        callback_handler.register_future("task-42")

        with pytest.raises(ValueError, match="already registered"):
            callback_handler.register_future("task-42")

    @pytest.mark.asyncio
    async def test_handle_result_resolves_future(
        self,
        callback_handler,
        mock_task_registry,
        sample_task_record,
        sample_instance,
    ):
        """Test handle_result() resolves the registered Future with TaskResult."""
        mock_task_registry.get.return_value = sample_task_record
        callback_handler.instance_registry.get.return_value = sample_instance

        future = callback_handler.register_future("task-1")

        result = TaskResult(
            task_id="task-1",
            worker_id="worker-1",
            status="completed",
            result={"choices": [{"text": "hello"}]},
            execution_time_ms=100.0,
            http_status_code=200,
            response_headers={"content-type": "application/json"},
        )

        await callback_handler.handle_result(result)

        assert future.done()
        resolved = future.result()
        assert resolved.task_id == "task-1"
        assert resolved.http_status_code == 200
        assert resolved.response_headers == {"content-type": "application/json"}

    @pytest.mark.asyncio
    async def test_handle_result_resolves_future_on_failure(
        self,
        callback_handler,
        mock_task_registry,
        sample_task_record,
        sample_instance,
    ):
        """Test handle_result() resolves the Future even on failure."""
        mock_task_registry.get.return_value = sample_task_record
        callback_handler.instance_registry.get.return_value = sample_instance

        future = callback_handler.register_future("task-1")

        result = TaskResult(
            task_id="task-1",
            worker_id="worker-1",
            status="failed",
            error="timeout",
            execution_time_ms=5000.0,
            http_status_code=504,
        )

        await callback_handler.handle_result(result)

        assert future.done()
        resolved = future.result()
        assert resolved.status == "failed"
        assert resolved.http_status_code == 504

    @pytest.mark.asyncio
    async def test_handle_result_without_future_still_works(
        self,
        callback_handler,
        mock_task_registry,
        sample_task_record,
        sample_instance,
        success_result,
    ):
        """Test handle_result() still works when no Future is registered."""
        mock_task_registry.get.return_value = sample_task_record
        callback_handler.instance_registry.get.return_value = sample_instance

        # No future registered - should not raise
        await callback_handler.handle_result(success_result)

        mock_task_registry.update_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_future_removes_unresolved(self, callback_handler):
        """Test cleanup_future() removes and cancels an unresolved Future."""
        future = callback_handler.register_future("task-42")

        callback_handler.cleanup_future("task-42")

        assert future.cancelled()
        # Should not be in the pool anymore
        assert not callback_handler.has_future("task-42")

    @pytest.mark.asyncio
    async def test_cleanup_future_noop_for_missing(self, callback_handler):
        """Test cleanup_future() is a no-op for non-existent task_id."""
        # Should not raise
        callback_handler.cleanup_future("nonexistent")

    @pytest.mark.asyncio
    async def test_has_future(self, callback_handler):
        """Test has_future() returns correct state."""
        assert not callback_handler.has_future("task-42")
        callback_handler.register_future("task-42")
        assert callback_handler.has_future("task-42")

    @pytest.mark.asyncio
    async def test_future_resolved_and_removed_after_handle(
        self,
        callback_handler,
        mock_task_registry,
        sample_task_record,
        sample_instance,
    ):
        """Test Future is removed from pool after handle_result resolves it."""
        mock_task_registry.get.return_value = sample_task_record
        callback_handler.instance_registry.get.return_value = sample_instance

        callback_handler.register_future("task-1")

        result = TaskResult(
            task_id="task-1",
            worker_id="worker-1",
            status="completed",
            result={"output": "done"},
            execution_time_ms=50.0,
        )

        await callback_handler.handle_result(result)

        # Future should be removed from pool after resolution
        assert not callback_handler.has_future("task-1")


# ============================================================================
# Training Integration Tests (Issue 3)
# ============================================================================


class TestTrainingIntegration:
    """Tests for training client integration in task completion."""

    @pytest.fixture
    def callback_with_training(
        self,
        mock_task_registry,
        mock_instance_registry,
        mock_websocket_manager,
        mock_throughput_tracker,
        mock_training_client,
    ):
        """Create a TaskResultCallback with training client."""
        return TaskResultCallback(
            task_registry=mock_task_registry,
            instance_registry=mock_instance_registry,
            websocket_manager=mock_websocket_manager,
            throughput_tracker=mock_throughput_tracker,
            training_client=mock_training_client,
        )

    @pytest.mark.asyncio
    async def test_adds_training_sample_on_success(
        self,
        callback_with_training,
        mock_task_registry,
        mock_training_client,
        sample_task_record,
        sample_instance,
        success_result,
    ):
        """Test training sample added on successful task completion."""
        mock_task_registry.get.return_value = sample_task_record
        callback_with_training.instance_registry.get.return_value = (
            sample_instance
        )

        await callback_with_training.handle_result(success_result)

        mock_training_client.add_sample.assert_called_once_with(
            model_id="test-model",
            platform_info={"gpu": "A100", "driver": "535"},
            features={"batch_size": 32},
            actual_runtime_ms=150.0,
        )

    @pytest.mark.asyncio
    async def test_flush_if_ready_called_on_success(
        self,
        callback_with_training,
        mock_task_registry,
        mock_training_client,
        sample_task_record,
        sample_instance,
        success_result,
    ):
        """Test flush_if_ready called after adding sample."""
        mock_task_registry.get.return_value = sample_task_record
        callback_with_training.instance_registry.get.return_value = (
            sample_instance
        )

        await callback_with_training.handle_result(success_result)

        mock_training_client.flush_if_ready.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_training_sample_on_failure(
        self,
        callback_with_training,
        mock_task_registry,
        mock_training_client,
        sample_task_record,
        sample_instance,
        failure_result,
    ):
        """Test no training sample added on failure."""
        mock_task_registry.get.return_value = sample_task_record
        callback_with_training.instance_registry.get.return_value = (
            sample_instance
        )

        await callback_with_training.handle_result(failure_result)

        mock_training_client.add_sample.assert_not_called()

    @pytest.mark.asyncio
    async def test_training_error_does_not_break_callback(
        self,
        callback_with_training,
        mock_task_registry,
        mock_training_client,
        sample_task_record,
        sample_instance,
        success_result,
    ):
        """Test training errors don't break task completion."""
        mock_task_registry.get.return_value = sample_task_record
        callback_with_training.instance_registry.get.return_value = (
            sample_instance
        )
        mock_training_client.add_sample.side_effect = RuntimeError(
            "Training error"
        )

        # Should not raise
        await callback_with_training.handle_result(success_result)

        # Task status should still be updated
        mock_task_registry.update_status.assert_called_once_with(
            "task-1", TaskStatus.COMPLETED
        )

    @pytest.mark.asyncio
    async def test_none_training_client_is_safe(
        self,
        mock_task_registry,
        mock_instance_registry,
        mock_websocket_manager,
        sample_task_record,
        sample_instance,
        success_result,
    ):
        """Test callback works when training_client is None."""
        callback = TaskResultCallback(
            task_registry=mock_task_registry,
            instance_registry=mock_instance_registry,
            websocket_manager=mock_websocket_manager,
            training_client=None,
        )
        mock_task_registry.get.return_value = sample_task_record
        mock_instance_registry.get.return_value = sample_instance

        # Should not raise
        await callback.handle_result(success_result)
        mock_task_registry.update_status.assert_called_once()


# ============================================================================
# Task Started Callback Tests (Issue 2)
# ============================================================================


class TestTaskStartedCallback:
    """Tests for _handle_task_started and create_thread_start_callback."""

    @pytest.mark.asyncio
    async def test_handle_task_started_sets_running(
        self,
        callback_handler,
        mock_task_registry,
    ):
        """Test _handle_task_started sets RUNNING status."""
        await callback_handler._handle_task_started("task-1")

        mock_task_registry.update_status.assert_called_once_with(
            "task-1", TaskStatus.RUNNING
        )

    @pytest.mark.asyncio
    async def test_handle_task_started_missing_task(
        self,
        callback_handler,
        mock_task_registry,
    ):
        """Test _handle_task_started handles missing task gracefully."""
        mock_task_registry.update_status.side_effect = KeyError("not found")

        # Should not raise
        await callback_handler._handle_task_started("nonexistent")

    def test_create_thread_start_callback(self, callback_handler):
        """Test thread-safe start callback creation."""
        loop = asyncio.new_event_loop()
        try:
            start_callback = callback_handler.create_thread_start_callback(loop)
            assert callable(start_callback)
        finally:
            loop.close()

    @pytest.mark.asyncio
    async def test_thread_start_callback_schedules_on_loop(
        self, callback_handler
    ):
        """Test start callback from thread schedules on event loop."""
        loop = asyncio.get_running_loop()
        start_callback = callback_handler.create_thread_start_callback(loop)

        with patch.object(
            callback_handler,
            "_handle_task_started",
            new_callable=AsyncMock,
        ) as mock_started:
            start_callback("task-42")
            await asyncio.sleep(0.1)
            mock_started.assert_called_once_with("task-42")
