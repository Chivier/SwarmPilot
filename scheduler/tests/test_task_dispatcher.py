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

    def test_initialization(
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

    def test_default_timeout(
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
        instance_registry.register(sample_instance)
        task = task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        instance_registry.increment_pending(sample_instance.instance_id)

        # Mock HTTP response
        mock_result = {"output": "test output", "tokens": 100}
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_result
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            # Execute
            await task_dispatcher.dispatch_task("task-1")

        # Verify task status
        task = task_registry.get("task-1")
        assert task.status == TaskStatus.COMPLETED
        assert task.result == mock_result

        # Verify stats
        stats = instance_registry.get_stats(sample_instance.instance_id)
        assert stats.pending_tasks == 0
        assert stats.completed_tasks == 1
        assert stats.failed_tasks == 0

    @pytest.mark.asyncio
    async def test_dispatch_updates_status_to_running(
        self,
        task_dispatcher,
        instance_registry,
        task_registry,
        sample_instance
    ):
        """Test that task status is updated to RUNNING before execution."""
        instance_registry.register(sample_instance)
        task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        instance_registry.increment_pending(sample_instance.instance_id)

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

        task = task_registry.get("task-1")
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
        task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="nonexistent-instance"
        )

        await task_dispatcher.dispatch_task("task-1")

        task = task_registry.get("task-1")
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
        instance_registry.register(sample_instance)
        task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        instance_registry.increment_pending(sample_instance.instance_id)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await task_dispatcher.dispatch_task("task-1")

        task = task_registry.get("task-1")
        assert task.status == TaskStatus.FAILED
        assert "timed out" in task.error

        stats = instance_registry.get_stats(sample_instance.instance_id)
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
        instance_registry.register(sample_instance)
        task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        instance_registry.increment_pending(sample_instance.instance_id)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "500 Server Error",
                    request=MagicMock(),
                    response=MagicMock()
                )
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await task_dispatcher.dispatch_task("task-1")

        task = task_registry.get("task-1")
        assert task.status == TaskStatus.FAILED
        assert "request failed" in task.error

        stats = instance_registry.get_stats(sample_instance.instance_id)
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
        instance_registry.register(sample_instance)
        task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        instance_registry.increment_pending(sample_instance.instance_id)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=Exception("Unexpected error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await task_dispatcher.dispatch_task("task-1")

        task = task_registry.get("task-1")
        assert task.status == TaskStatus.FAILED
        assert "Unexpected error" in task.error

        stats = instance_registry.get_stats(sample_instance.instance_id)
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
        instance_registry.register(sample_instance)
        task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        instance_registry.increment_pending(sample_instance.instance_id)
        instance_registry.increment_pending(sample_instance.instance_id)

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

        stats = instance_registry.get_stats(sample_instance.instance_id)
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
        """Test that completed count is incremented on success."""
        instance_registry.register(sample_instance)
        task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )

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

        stats = instance_registry.get_stats(sample_instance.instance_id)
        assert stats.completed_tasks == 1

    @pytest.mark.asyncio
    async def test_increment_failed_on_error(
        self,
        task_dispatcher,
        instance_registry,
        task_registry,
        sample_instance
    ):
        """Test that failed count is incremented on error."""
        instance_registry.register(sample_instance)
        task_registry.create_task(
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

        stats = instance_registry.get_stats(sample_instance.instance_id)
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
        instance_registry.register(sample_instance)
        task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )

        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()
        websocket_manager.subscribe(mock_websocket, ["task-1"])

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
        instance_registry.register(sample_instance)
        task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )

        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()
        websocket_manager.subscribe(mock_websocket, ["task-1"])

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
        task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="inst-1"
        )

        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()
        task_dispatcher.websocket_manager.subscribe(mock_websocket, ["task-1"])

        # Task is PENDING - should not notify
        await task_dispatcher._notify_task_completion("task-1")
        mock_websocket.send_json.assert_not_called()

        # Update to RUNNING - should not notify
        task_registry.update_status("task-1", TaskStatus.RUNNING)
        await task_dispatcher._notify_task_completion("task-1")
        mock_websocket.send_json.assert_not_called()

        # Update to COMPLETED - should notify
        task_registry.update_status("task-1", TaskStatus.COMPLETED)
        await task_dispatcher._notify_task_completion("task-1")
        mock_websocket.send_json.assert_called_once()
