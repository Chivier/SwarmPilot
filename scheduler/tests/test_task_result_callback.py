"""Unit tests for TaskResultCallback.

TDD tests for PYLET-016: Library-Based Callback Mechanism.
These tests define the expected behavior before implementation.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.model import TaskStatus
from src.services.task_result_callback import TaskResultCallback
from src.services.worker_queue_thread import TaskResult


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
def sample_task_record():
    """Create a sample task record."""
    task = MagicMock()
    task.task_id = "task-1"
    task.status = TaskStatus.RUNNING
    task.set_execution_time = MagicMock()
    task.get_timestamps = MagicMock(return_value={"submitted_at": "2024-01-01T00:00:00Z"})
    return task


@pytest.fixture
def sample_instance():
    """Create a sample instance."""
    instance = MagicMock()
    instance.instance_id = "worker-1"
    instance.endpoint = "http://localhost:8001"
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
