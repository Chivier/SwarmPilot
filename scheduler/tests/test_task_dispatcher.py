"""
Unit tests for TaskDispatcher.

Tests task dispatching, error handling, and WebSocket notifications.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from src.task_dispatcher import TaskDispatcher
from src.model import TaskStatus


# ============================================================================
# Initialization Tests
# ============================================================================

class TestTaskDispatcherInit:
    """Tests for TaskDispatcher initialization."""

    async def test_initialization(
        self,
        task_registry,
        instance_registry,
        websocket_manager
    ):
        """Test TaskDispatcher initialization."""
        dispatcher = TaskDispatcher(
            task_registry=task_registry,
            instance_registry=instance_registry,
            websocket_manager=websocket_manager,
            timeout=30.0
        )

        assert dispatcher.task_registry == task_registry
        assert dispatcher.instance_registry == instance_registry
        assert dispatcher.websocket_manager == websocket_manager
        assert dispatcher.timeout == 30.0

    async def test_default_timeout(
        self,
        task_registry,
        instance_registry,
        websocket_manager
    ):
        """Test default timeout value."""
        dispatcher = TaskDispatcher(
            task_registry=task_registry,
            instance_registry=instance_registry,
            websocket_manager=websocket_manager
        )

        assert dispatcher.timeout == 60.0


# ============================================================================
# Successful Dispatch Tests
# ============================================================================

class TestSuccessfulDispatch:
    """Tests for successful task dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_successful_task(
        self,
        task_dispatcher,
        instance_registry,
        task_registry,
        sample_instance,
        websocket_manager
    ):
        """Test dispatching a task successfully."""
        # Setup
        await instance_registry.register(sample_instance)
        task = await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        await instance_registry.increment_pending(sample_instance.instance_id)

        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True, "message": "Task accepted"}
        mock_response.raise_for_status = MagicMock()
        task_dispatcher._http_client.post = AsyncMock(return_value=mock_response)

        # Execute
        await task_dispatcher.dispatch_task("task-1")

        # Verify task status - should be RUNNING (not COMPLETED - that happens via callback)
        task = await task_registry.get("task-1")
        assert task.status == TaskStatus.RUNNING

        # Verify pending was decremented
        stats = await instance_registry.get_stats(sample_instance.instance_id)
        assert stats.pending_tasks == 0

    @pytest.mark.asyncio
    async def test_dispatch_updates_status_to_running(
        self,
        task_dispatcher,
        instance_registry,
        task_registry,
        sample_instance
    ):
        """Test that task status is updated to RUNNING before execution."""
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        await instance_registry.increment_pending(sample_instance.instance_id)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"result": "ok"}
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await task_dispatcher.dispatch_task("task-1")

        task = await task_registry.get("task-1")
        assert task.started_at is not None


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling during dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_nonexistent_task(self, task_dispatcher):
        """Test dispatching a task that doesn't exist."""
        # Should not raise error
        await task_dispatcher.dispatch_task("nonexistent")

    @pytest.mark.asyncio
    async def test_dispatch_instance_not_found(
        self,
        task_dispatcher,
        task_registry,
        websocket_manager
    ):
        """Test dispatching when assigned instance doesn't exist."""
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="nonexistent-instance"
        )

        await task_dispatcher.dispatch_task("task-1")

        task = await task_registry.get("task-1")
        assert task.status == TaskStatus.FAILED
        assert "not found" in task.error

    @pytest.mark.asyncio
    async def test_dispatch_timeout(
        self,
        task_dispatcher,
        instance_registry,
        task_registry,
        sample_instance
    ):
        """Test handling task execution timeout."""
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        await instance_registry.increment_pending(sample_instance.instance_id)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await task_dispatcher.dispatch_task("task-1")

        task = await task_registry.get("task-1")
        assert task.status == TaskStatus.FAILED
        # The dispatcher catches all exceptions and returns a generic error
        assert "Task dispatch failed" in task.error or "Timeout" in task.error

        stats = await instance_registry.get_stats(sample_instance.instance_id)
        assert stats.pending_tasks == 0
        assert stats.failed_tasks == 1

    @pytest.mark.asyncio
    async def test_dispatch_http_error(
        self,
        task_dispatcher,
        instance_registry,
        task_registry,
        sample_instance
    ):
        """Test handling HTTP errors from instance."""
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        await instance_registry.increment_pending(sample_instance.instance_id)

        # Mock HTTP client to raise HTTPStatusError
        task_dispatcher._http_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "500 Server Error",
                request=MagicMock(),
                response=MagicMock()
            )
        )

        await task_dispatcher.dispatch_task("task-1")

        task = await task_registry.get("task-1")
        assert task.status == TaskStatus.FAILED
        assert "500 Server Error" in task.error

        stats = await instance_registry.get_stats(sample_instance.instance_id)
        assert stats.failed_tasks == 1

    @pytest.mark.asyncio
    async def test_dispatch_unexpected_error(
        self,
        task_dispatcher,
        instance_registry,
        task_registry,
        sample_instance
    ):
        """Test handling unexpected errors."""
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        await instance_registry.increment_pending(sample_instance.instance_id)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=Exception("Unexpected error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await task_dispatcher.dispatch_task("task-1")

        task = await task_registry.get("task-1")
        assert task.status == TaskStatus.FAILED
        # The dispatcher catches all exceptions and returns a generic error
        assert "Task dispatch failed" in task.error or "Unexpected error" in task.error

        stats = await instance_registry.get_stats(sample_instance.instance_id)
        assert stats.failed_tasks == 1


# ============================================================================
# Stats Update Tests
# ============================================================================

class TestStatsUpdates:
    """Tests for instance statistics updates."""

    @pytest.mark.asyncio
    async def test_decrement_pending_on_dispatch(
        self,
        task_dispatcher,
        instance_registry,
        task_registry,
        sample_instance
    ):
        """Test that pending count is decremented when task starts."""
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        await instance_registry.increment_pending(sample_instance.instance_id)
        await instance_registry.increment_pending(sample_instance.instance_id)

        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status = MagicMock()
        task_dispatcher._http_client.post = AsyncMock(return_value=mock_response)

        await task_dispatcher.dispatch_task("task-1")

        stats = await instance_registry.get_stats(sample_instance.instance_id)
        # Started with 2, decremented to 1
        assert stats.pending_tasks == 1

    @pytest.mark.asyncio
    async def test_increment_completed_on_success(
        self,
        task_dispatcher,
        instance_registry,
        task_registry,
        sample_instance
    ):
        """Test that task is dispatched successfully (completion happens via callback)."""
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        await instance_registry.increment_pending(sample_instance.instance_id)

        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status = MagicMock()
        task_dispatcher._http_client.post = AsyncMock(return_value=mock_response)

        await task_dispatcher.dispatch_task("task-1")

        # Task should be RUNNING (completion happens via callback, not during dispatch)
        task = await task_registry.get("task-1")
        assert task.status == TaskStatus.RUNNING

        # Pending should be decremented
        stats = await instance_registry.get_stats(sample_instance.instance_id)
        assert stats.pending_tasks == 0
        assert stats.completed_tasks == 0  # Completion happens via callback

    @pytest.mark.asyncio
    async def test_increment_failed_on_error(
        self,
        task_dispatcher,
        instance_registry,
        task_registry,
        sample_instance
    ):
        """Test that failed count is incremented on error."""
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=Exception("Error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await task_dispatcher.dispatch_task("task-1")

        stats = await instance_registry.get_stats(sample_instance.instance_id)
        assert stats.failed_tasks == 1


# ============================================================================
# WebSocket Notification Tests
# ============================================================================

class TestWebSocketNotification:
    """Tests for WebSocket notifications."""

    @pytest.mark.asyncio
    async def test_notify_on_completion(
        self,
        task_dispatcher,
        instance_registry,
        task_registry,
        sample_instance,
        websocket_manager
    ):
        """Test WebSocket notification on task completion."""
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )

        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()
        await websocket_manager.subscribe(mock_websocket, ["task-1"])

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"result": "ok"}
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await task_dispatcher.dispatch_task("task-1")

        # Verify notification was sent
        mock_websocket.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_notify_on_failure(
        self,
        task_dispatcher,
        instance_registry,
        task_registry,
        sample_instance,
        websocket_manager
    ):
        """Test WebSocket notification on task failure."""
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )

        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()
        await websocket_manager.subscribe(mock_websocket, ["task-1"])

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=Exception("Error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await task_dispatcher.dispatch_task("task-1")

        # Verify notification was sent
        mock_websocket.send_json.assert_called_once()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["status"] == "failed"

    @pytest.mark.asyncio
    async def test_notification_with_nonexistent_task(self, task_dispatcher):
        """Test notification when task doesn't exist."""
        # Should not raise error
        await task_dispatcher._notify_task_completion("nonexistent")

    @pytest.mark.asyncio
    async def test_notification_only_for_terminal_states(
        self,
        task_dispatcher,
        task_registry
    ):
        """Test that notification only happens for terminal states."""
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="inst-1"
        )

        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()
        await task_dispatcher.websocket_manager.subscribe(mock_websocket, ["task-1"])

        # Task is PENDING - should not notify
        await task_dispatcher._notify_task_completion("task-1")
        mock_websocket.send_json.assert_not_called()

        # Update to RUNNING - should not notify
        await task_registry.update_status("task-1", TaskStatus.RUNNING)
        await task_dispatcher._notify_task_completion("task-1")
        mock_websocket.send_json.assert_not_called()

        # Update to COMPLETED - should notify
        await task_registry.update_status("task-1", TaskStatus.COMPLETED)
        await task_dispatcher._notify_task_completion("task-1")
        mock_websocket.send_json.assert_called_once()


# ============================================================================
# Handle Task Result Tests
# ============================================================================

class TestHandleTaskResult:
    """Tests for handle_task_result callback handler."""

    @pytest.mark.asyncio
    async def test_handle_completed_task(
        self,
        task_dispatcher,
        sample_instance
    ):
        """Test handling a completed task result."""
        # Setup
        await task_dispatcher.instance_registry.register(sample_instance)
        task = await task_dispatcher.task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={"prompt": "test"},
            metadata={"input_size": 1000},
            assigned_instance=sample_instance.instance_id
        )
        task.predicted_time_ms = 100.0

        # Handle completion
        await task_dispatcher.handle_task_result(
            task_id="task-1",
            status="completed",
            result={"output": "result"},
            execution_time_ms=120.0
        )

        # Verify task was updated
        updated_task = await task_dispatcher.task_registry.get("task-1")
        assert updated_task.status == TaskStatus.COMPLETED
        assert updated_task.result == {"output": "result"}
        assert updated_task.execution_time_ms == 120.0

        # Verify instance stats were updated
        stats = await task_dispatcher.instance_registry.get_stats(sample_instance.instance_id)
        assert stats.completed_tasks == 1

    @pytest.mark.asyncio
    async def test_handle_completed_with_training_data(
        self,
        task_dispatcher,
        sample_instance
    ):
        """Test that training data is collected on completion."""
        # Setup with training client
        mock_training_client = MagicMock()
        mock_training_client.add_sample = MagicMock()
        mock_training_client.flush_if_ready = AsyncMock()
        task_dispatcher.training_client = mock_training_client

        await task_dispatcher.instance_registry.register(sample_instance)
        task = await task_dispatcher.task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={"input_size": 1000},
            assigned_instance=sample_instance.instance_id
        )

        # Handle completion
        await task_dispatcher.handle_task_result(
            task_id="task-1",
            status="completed",
            execution_time_ms=150.0
        )

        # Verify training data was collected
        mock_training_client.add_sample.assert_called_once_with(
            model_id="model-1",
            platform_info=sample_instance.platform_info,
            features={"input_size": 1000},
            actual_runtime_ms=150.0
        )
        mock_training_client.flush_if_ready.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_completed_without_training(
        self,
        task_dispatcher,
        sample_instance
    ):
        """Test completion without training client (no errors)."""
        task_dispatcher.training_client = None

        await task_dispatcher.instance_registry.register(sample_instance)
        await task_dispatcher.task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )

        # Should not raise error
        await task_dispatcher.handle_task_result(
            task_id="task-1",
            status="completed",
            execution_time_ms=150.0
        )

    @pytest.mark.asyncio
    async def test_handle_failed_task(
        self,
        task_dispatcher,
        sample_instance
    ):
        """Test handling a failed task result."""
        await task_dispatcher.instance_registry.register(sample_instance)
        await task_dispatcher.task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )

        # Handle failure
        await task_dispatcher.handle_task_result(
            task_id="task-1",
            status="failed",
            error="Task execution failed"
        )

        # Verify task was marked as failed
        task = await task_dispatcher.task_registry.get("task-1")
        assert task.status == TaskStatus.FAILED
        assert task.error == "Task execution failed"

        # Verify instance failed count increased
        stats = await task_dispatcher.instance_registry.get_stats(sample_instance.instance_id)
        assert stats.failed_tasks == 1

    @pytest.mark.asyncio
    async def test_handle_nonexistent_task(
        self,
        task_dispatcher,
        task_registry
    ):
        """Test handling result for nonexistent task."""
        # Should not raise error
        await task_dispatcher.handle_task_result(
            task_id="nonexistent",
            status="completed"
        )

    @pytest.mark.asyncio
    async def test_handle_triggers_queue_update(
        self,
        task_dispatcher,
        sample_instance
    ):
        """Test that queue update is triggered on completion."""
        from src.model import InstanceQueueExpectError

        await task_dispatcher.instance_registry.register(sample_instance)
        await task_dispatcher.instance_registry.update_queue_info(
            sample_instance.instance_id,
            InstanceQueueExpectError(
                instance_id=sample_instance.instance_id,
                expected_time_ms=200.0,
                error_margin_ms=50.0
            )
        )

        task = await task_dispatcher.task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        task.predicted_time_ms = 100.0

        # Handle completion
        await task_dispatcher.handle_task_result(
            task_id="task-1",
            status="completed",
            execution_time_ms=120.0
        )

        # Verify queue was updated
        queue = await task_dispatcher.instance_registry.get_queue_info(sample_instance.instance_id)
        # Expected: 200 - 120 = 80 (new formula: old - actual)
        assert queue.expected_time_ms == 80.0


# ============================================================================
# Queue Update on Completion Tests
# ============================================================================

class TestQueueUpdateOnCompletion:
    """Tests for _update_queue_on_completion method."""

    @pytest.mark.asyncio
    async def test_update_expect_error_queue(
        self,
        task_dispatcher,
        sample_instance
    ):
        """Test updating ExpectError queue on completion."""
        from src.model import InstanceQueueExpectError

        # Use task_dispatcher's instance_registry to ensure consistency
        await task_dispatcher.instance_registry.register(sample_instance)
        await task_dispatcher.instance_registry.update_queue_info(
            sample_instance.instance_id,
            InstanceQueueExpectError(
                instance_id=sample_instance.instance_id,
                expected_time_ms=300.0,
                error_margin_ms=50.0
            )
        )

        # Update: predicted=150ms, actual=180ms
        await task_dispatcher._update_queue_on_completion(
            instance_id=sample_instance.instance_id,
            predicted_time_ms=150.0,
            actual_time_ms=180.0,
            predicted_error_margin_ms=30.0
        )

        # Verify: 300 - 180 = 120 (new formula: old - actual)
        queue = await task_dispatcher.instance_registry.get_queue_info(sample_instance.instance_id)
        assert queue.expected_time_ms == 120.0
        assert queue.error_margin_ms == 50.0  # Error unchanged

    @pytest.mark.asyncio
    async def test_update_expect_error_queue_ensures_non_negative(
        self,
        task_dispatcher,
        sample_instance
    ):
        """Test that queue update ensures non-negative values."""
        from src.model import InstanceQueueExpectError

        await task_dispatcher.instance_registry.register(sample_instance)
        await task_dispatcher.instance_registry.update_queue_info(
            sample_instance.instance_id,
            InstanceQueueExpectError(
                instance_id=sample_instance.instance_id,
                expected_time_ms=50.0,
                error_margin_ms=10.0
            )
        )

        # actual_time > expected would result in negative
        await task_dispatcher._update_queue_on_completion(
            instance_id=sample_instance.instance_id,
            predicted_time_ms=100.0,
            actual_time_ms=60.0
        )

        # Verify: 50 - 60 = -10, clamped to 0
        queue = await task_dispatcher.instance_registry.get_queue_info(sample_instance.instance_id)
        assert queue.expected_time_ms == 0.0

    @pytest.mark.asyncio
    async def test_update_probabilistic_queue_with_quantiles(
        self,
        task_dispatcher,
        sample_instance
    ):
        """Test updating Probabilistic queue with Monte Carlo."""
        from src.model import InstanceQueueProbabilistic

        await task_dispatcher.instance_registry.register(sample_instance)
        await task_dispatcher.instance_registry.update_queue_info(
            sample_instance.instance_id,
            InstanceQueueProbabilistic(
                instance_id=sample_instance.instance_id,
                quantiles=[0.5, 0.9, 0.95],
                values=[100.0, 200.0, 250.0]
            )
        )

        # Update with quantiles (triggers Monte Carlo)
        await task_dispatcher._update_queue_on_completion(
            instance_id=sample_instance.instance_id,
            predicted_time_ms=80.0,
            actual_time_ms=100.0,
            predicted_quantiles={0.5: 70.0, 0.9: 90.0, 0.95: 100.0}
        )

        # Verify queue was updated
        queue = await task_dispatcher.instance_registry.get_queue_info(sample_instance.instance_id)
        assert isinstance(queue, InstanceQueueProbabilistic)
        assert len(queue.values) == 3
        # Values should be updated (difficult to test exact values due to Monte Carlo)
        assert all(v >= 0 for v in queue.values)

    @pytest.mark.asyncio
    async def test_update_probabilistic_queue_fallback(
        self,
        task_dispatcher,
        sample_instance
    ):
        """Test Probabilistic queue update without quantiles (fallback)."""
        from src.model import InstanceQueueProbabilistic

        await task_dispatcher.instance_registry.register(sample_instance)
        await task_dispatcher.instance_registry.update_queue_info(
            sample_instance.instance_id,
            InstanceQueueProbabilistic(
                instance_id=sample_instance.instance_id,
                quantiles=[0.5, 0.9, 0.95],
                values=[150.0, 250.0, 300.0]
            )
        )

        # Update without quantiles (triggers fallback)
        await task_dispatcher._update_queue_on_completion(
            instance_id=sample_instance.instance_id,
            predicted_time_ms=100.0,
            actual_time_ms=120.0
        )

        # Verify simple subtraction/addition: values - 100 + 120 = values + 20
        queue = await task_dispatcher.instance_registry.get_queue_info(sample_instance.instance_id)
        assert queue.values[0] == 170.0  # 150 - 100 + 120
        assert queue.values[1] == 270.0  # 250 - 100 + 120
        assert queue.values[2] == 320.0  # 300 - 100 + 120

    @pytest.mark.asyncio
    async def test_update_missing_queue_info(
        self,
        task_dispatcher,
        instance_registry,
        sample_instance
    ):
        """Test queue update when no queue info exists."""
        await instance_registry.register(sample_instance)
        # No queue info set

        # Should not raise error
        await task_dispatcher._update_queue_on_completion(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            actual_time_ms=120.0
        )

    @pytest.mark.asyncio
    async def test_update_without_predictions(
        self,
        task_dispatcher,
        sample_instance
    ):
        """Test that update handles missing prediction data gracefully."""
        from src.model import InstanceQueueExpectError

        await task_dispatcher.instance_registry.register(sample_instance)
        await task_dispatcher.instance_registry.update_queue_info(
            sample_instance.instance_id,
            InstanceQueueExpectError(
                instance_id=sample_instance.instance_id,
                expected_time_ms=200.0,
                error_margin_ms=50.0
            )
        )

        # Update with minimal data
        await task_dispatcher._update_queue_on_completion(
            instance_id=sample_instance.instance_id,
            predicted_time_ms=100.0,
            actual_time_ms=110.0
        )

        # Should still update: 200 - 110 = 90 (new formula: old - actual)
        queue = await task_dispatcher.instance_registry.get_queue_info(sample_instance.instance_id)
        assert queue.expected_time_ms == 90.0


# ============================================================================
# Async Dispatch Tests
# ============================================================================

class TestAsyncDispatch:
    """Tests for async dispatch functionality."""

    @pytest.mark.asyncio
    async def test_dispatch_task_async_creates_task(
        self,
        task_dispatcher,
        task_registry,
        instance_registry,
        sample_instance
    ):
        """Test that dispatch_task_async creates an async task."""
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="inst-1"
        )

        # Mock dispatch_task to track if it was called
        with patch.object(task_dispatcher, 'dispatch_task', new=AsyncMock()) as mock_dispatch:
            # Call async dispatch
            task_dispatcher.dispatch_task_async("task-1")

            # Wait a bit for the task to be created
            import asyncio
            await asyncio.sleep(0.1)

            # Verify dispatch_task was called
            mock_dispatch.assert_called_once_with("task-1")


# ============================================================================
# Close Tests
# ============================================================================

class TestClose:
    """Tests for resource cleanup."""

    @pytest.mark.asyncio
    async def test_close_http_client(
        self,
        task_dispatcher
    ):
        """Test that close properly closes the HTTP client."""
        # Mock the aclose method
        task_dispatcher._http_client.aclose = AsyncMock()

        await task_dispatcher.close()

        task_dispatcher._http_client.aclose.assert_called_once()


# ============================================================================
# WebSocket Error Paths Tests
# ============================================================================

class TestWebSocketErrorPaths:
    """Tests for error paths in dispatch_task."""

    @pytest.mark.asyncio
    async def test_dispatch_timeout_error(
        self,
        task_dispatcher,
        instance_registry,
        task_registry,
        sample_instance,
        websocket_manager
    ):
        """Test dispatch timeout error handling (lines 133-138)."""
        # Setup
        await instance_registry.register(sample_instance)
        task = await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        await instance_registry.increment_pending(sample_instance.instance_id)

        # Mock HTTP client to timeout
        task_dispatcher._http_client.post = AsyncMock(
            side_effect=httpx.TimeoutException("Timeout")
        )

        # Execute
        await task_dispatcher.dispatch_task("task-1")

        # Verify task was marked as failed
        task = await task_registry.get("task-1")
        assert task.status == TaskStatus.FAILED
        assert "timed out" in task.error.lower()

        # Verify instance stats updated
        stats = await instance_registry.get_stats(sample_instance.instance_id)
        assert stats.failed_tasks == 1

    @pytest.mark.asyncio
    async def test_dispatch_http_error(
        self,
        task_dispatcher,
        instance_registry,
        task_registry,
        sample_instance,
        websocket_manager
    ):
        """Test dispatch HTTP error handling (lines 147-152)."""
        # Setup
        await instance_registry.register(sample_instance)
        task = await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        await instance_registry.increment_pending(sample_instance.instance_id)

        # Mock HTTP client to raise HTTP error
        task_dispatcher._http_client.post = AsyncMock(
            side_effect=httpx.HTTPError("HTTP error")
        )

        # Execute
        await task_dispatcher.dispatch_task("task-1")

        # Verify task was marked as failed
        task = await task_registry.get("task-1")
        assert task.status == TaskStatus.FAILED
        assert "dispatch failed" in task.error.lower()

        # Verify instance stats updated
        stats = await instance_registry.get_stats(sample_instance.instance_id)
        assert stats.failed_tasks == 1


# ============================================================================
# Transient Error Retry Tests
# ============================================================================

class TestTransientErrorRetry:
    """Tests for transient connection error retry logic."""

    def test_is_transient_connection_error_read_error(
        self,
        task_dispatcher
    ):
        """Test that ReadError is identified as transient."""
        # Create a mock ReadError (httpx internal error)
        class MockReadError(httpx.HTTPError):
            pass
        MockReadError.__name__ = "ReadError"

        error = MockReadError("Connection reset")
        assert task_dispatcher._is_transient_connection_error(error) is True

    def test_is_transient_connection_error_connect_error(
        self,
        task_dispatcher
    ):
        """Test that ConnectError is identified as transient."""
        class MockConnectError(httpx.HTTPError):
            pass
        MockConnectError.__name__ = "ConnectError"

        error = MockConnectError("Connection refused")
        assert task_dispatcher._is_transient_connection_error(error) is True

    def test_is_transient_connection_error_remote_protocol_error(
        self,
        task_dispatcher
    ):
        """Test that RemoteProtocolError is identified as transient."""
        class MockRemoteProtocolError(httpx.HTTPError):
            pass
        MockRemoteProtocolError.__name__ = "RemoteProtocolError"

        error = MockRemoteProtocolError("Protocol error")
        assert task_dispatcher._is_transient_connection_error(error) is True

    def test_is_transient_connection_error_http_status_error(
        self,
        task_dispatcher
    ):
        """Test that HTTPStatusError is NOT identified as transient."""
        error = httpx.HTTPStatusError(
            "500 Server Error",
            request=MagicMock(),
            response=MagicMock()
        )
        assert task_dispatcher._is_transient_connection_error(error) is False

    def test_is_transient_connection_error_timeout(
        self,
        task_dispatcher
    ):
        """Test that TimeoutException is NOT identified as transient."""
        error = httpx.TimeoutException("Timeout")
        assert task_dispatcher._is_transient_connection_error(error) is False

    @pytest.mark.asyncio
    async def test_retry_on_read_error_success(
        self,
        task_dispatcher,
        instance_registry,
        task_registry,
        sample_instance
    ):
        """Test that ReadError triggers retry and succeeds on second attempt."""
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        await instance_registry.increment_pending(sample_instance.instance_id)

        # Create a mock ReadError
        class MockReadError(httpx.HTTPError):
            pass
        MockReadError.__name__ = "ReadError"

        # Create successful response mock
        success_response = MagicMock()
        success_response.raise_for_status = MagicMock()
        success_response.json = MagicMock(return_value={"success": True})

        # First call raises ReadError, second succeeds
        call_count = 0
        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise MockReadError("Connection reset by peer")
            return success_response

        task_dispatcher._http_client.post = mock_post

        # Execute
        await task_dispatcher.dispatch_task("task-1")

        # Verify retry happened (2 calls total)
        assert call_count == 2

        # Task should be successful (RUNNING status after dispatch)
        task = await task_registry.get("task-1")
        assert task.status == TaskStatus.RUNNING

    @pytest.mark.asyncio
    async def test_retry_exhausted_on_read_error(
        self,
        task_dispatcher,
        instance_registry,
        task_registry,
        sample_instance
    ):
        """Test that task fails after all retries are exhausted."""
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        await instance_registry.increment_pending(sample_instance.instance_id)

        # Create a mock ReadError
        class MockReadError(httpx.HTTPError):
            pass
        MockReadError.__name__ = "ReadError"

        # All calls fail with ReadError
        call_count = 0
        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise MockReadError("Connection reset by peer")

        task_dispatcher._http_client.post = mock_post

        # Execute
        await task_dispatcher.dispatch_task("task-1")

        # Verify all retries were attempted (default is 3)
        assert call_count == task_dispatcher.dispatch_retries

        # Task should be failed
        task = await task_registry.get("task-1")
        assert task.status == TaskStatus.FAILED
        assert "dispatch failed" in task.error.lower()

    @pytest.mark.asyncio
    async def test_no_retry_on_http_status_error(
        self,
        task_dispatcher,
        instance_registry,
        task_registry,
        sample_instance
    ):
        """Test that HTTPStatusError (non-transient) does not trigger retry."""
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        await instance_registry.increment_pending(sample_instance.instance_id)

        # All calls fail with HTTPStatusError (non-transient)
        call_count = 0
        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx.HTTPStatusError(
                "500 Server Error",
                request=MagicMock(),
                response=MagicMock()
            )

        task_dispatcher._http_client.post = mock_post

        # Execute
        await task_dispatcher.dispatch_task("task-1")

        # Verify NO retry happened (only 1 call)
        assert call_count == 1

        # Task should be failed
        task = await task_registry.get("task-1")
        assert task.status == TaskStatus.FAILED

    def test_initialization_with_retry_params(
        self,
        task_registry,
        instance_registry,
        websocket_manager
    ):
        """Test TaskDispatcher initialization with custom retry parameters."""
        dispatcher = TaskDispatcher(
            task_registry=task_registry,
            instance_registry=instance_registry,
            websocket_manager=websocket_manager,
            dispatch_retries=5,
            dispatch_retry_delay=0.2
        )

        assert dispatcher.dispatch_retries == 5
        assert dispatcher.dispatch_retry_delay == 0.2

    def test_default_retry_params(
        self,
        task_registry,
        instance_registry,
        websocket_manager
    ):
        """Test TaskDispatcher uses default retry parameters."""
        dispatcher = TaskDispatcher(
            task_registry=task_registry,
            instance_registry=instance_registry,
            websocket_manager=websocket_manager
        )

        assert dispatcher.dispatch_retries == 3
        assert dispatcher.dispatch_retry_delay == 0.1
