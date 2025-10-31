"""
Unit tests for ConnectionManager (WebSocket management).

Tests connection management, subscriptions, and broadcasting.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.websocket_manager import ConnectionManager
from src.model import TaskStatus, TaskTimestamps


# ============================================================================
# Connection Management Tests
# ============================================================================

class TestConnectionManagement:
    """Tests for connection registration and cleanup."""

    def test_connect_websocket(self, websocket_manager, mock_websocket):
        """Test registering a WebSocket connection."""
        websocket_manager.connect(mock_websocket)

        subscribed = websocket_manager.get_subscribed_tasks(mock_websocket)
        assert subscribed == []

    def test_disconnect_websocket(self, websocket_manager, mock_websocket):
        """Test disconnecting a WebSocket."""
        websocket_manager.connect(mock_websocket)
        websocket_manager.subscribe(mock_websocket, ["task-1", "task-2"])

        websocket_manager.disconnect(mock_websocket)

        # Verify connection is gone
        subscribed = websocket_manager.get_subscribed_tasks(mock_websocket)
        assert subscribed == []

        # Verify no subscribers for tasks
        assert websocket_manager.get_subscribers("task-1") == []
        assert websocket_manager.get_subscribers("task-2") == []

    def test_disconnect_cleans_up_subscriptions(self, websocket_manager, mock_websocket):
        """Test that disconnect removes all subscriptions."""
        websocket_manager.connect(mock_websocket)
        websocket_manager.subscribe(mock_websocket, ["task-1", "task-2", "task-3"])

        websocket_manager.disconnect(mock_websocket)

        # All subscriptions should be cleaned up
        assert websocket_manager.get_subscribers("task-1") == []
        assert websocket_manager.get_subscribers("task-2") == []
        assert websocket_manager.get_subscribers("task-3") == []

    def test_disconnect_nonexistent_connection(self, websocket_manager, mock_websocket):
        """Test disconnecting a connection that doesn't exist."""
        # Should not raise error
        websocket_manager.disconnect(mock_websocket)

    def test_multiple_connections(self, websocket_manager):
        """Test managing multiple WebSocket connections."""
        ws1 = MagicMock()
        ws2 = MagicMock()
        ws3 = MagicMock()

        websocket_manager.connect(ws1)
        websocket_manager.connect(ws2)
        websocket_manager.connect(ws3)

        websocket_manager.subscribe(ws1, ["task-1"])
        websocket_manager.subscribe(ws2, ["task-1"])
        websocket_manager.subscribe(ws3, ["task-2"])

        # Verify subscriptions
        assert len(websocket_manager.get_subscribers("task-1")) == 2
        assert len(websocket_manager.get_subscribers("task-2")) == 1


# ============================================================================
# Subscription Management Tests
# ============================================================================

class TestSubscriptionManagement:
    """Tests for subscribing and unsubscribing."""

    def test_subscribe_to_tasks(self, websocket_manager, mock_websocket):
        """Test subscribing to task updates."""
        websocket_manager.subscribe(mock_websocket, ["task-1", "task-2"])

        subscribed = websocket_manager.get_subscribed_tasks(mock_websocket)
        assert set(subscribed) == {"task-1", "task-2"}

        assert mock_websocket in websocket_manager.get_subscribers("task-1")
        assert mock_websocket in websocket_manager.get_subscribers("task-2")

    def test_subscribe_auto_registers_connection(self, websocket_manager, mock_websocket):
        """Test that subscribe automatically registers connection if needed."""
        # Don't call connect() first
        websocket_manager.subscribe(mock_websocket, ["task-1"])

        subscribed = websocket_manager.get_subscribed_tasks(mock_websocket)
        assert "task-1" in subscribed

    def test_subscribe_multiple_times(self, websocket_manager, mock_websocket):
        """Test subscribing to additional tasks."""
        websocket_manager.subscribe(mock_websocket, ["task-1"])
        websocket_manager.subscribe(mock_websocket, ["task-2", "task-3"])

        subscribed = websocket_manager.get_subscribed_tasks(mock_websocket)
        assert set(subscribed) == {"task-1", "task-2", "task-3"}

    def test_subscribe_to_same_task_twice(self, websocket_manager, mock_websocket):
        """Test subscribing to the same task multiple times."""
        websocket_manager.subscribe(mock_websocket, ["task-1"])
        websocket_manager.subscribe(mock_websocket, ["task-1"])

        # Should only be subscribed once
        subscribers = websocket_manager.get_subscribers("task-1")
        assert subscribers.count(mock_websocket) == 1

    def test_unsubscribe_from_tasks(self, websocket_manager, mock_websocket):
        """Test unsubscribing from task updates."""
        websocket_manager.subscribe(mock_websocket, ["task-1", "task-2", "task-3"])
        websocket_manager.unsubscribe(mock_websocket, ["task-1", "task-3"])

        subscribed = websocket_manager.get_subscribed_tasks(mock_websocket)
        assert subscribed == ["task-2"]

        assert mock_websocket not in websocket_manager.get_subscribers("task-1")
        assert mock_websocket in websocket_manager.get_subscribers("task-2")
        assert mock_websocket not in websocket_manager.get_subscribers("task-3")

    def test_unsubscribe_from_non_subscribed_task(self, websocket_manager, mock_websocket):
        """Test unsubscribing from a task not subscribed to."""
        websocket_manager.subscribe(mock_websocket, ["task-1"])

        # Should not raise error
        websocket_manager.unsubscribe(mock_websocket, ["task-2"])

        subscribed = websocket_manager.get_subscribed_tasks(mock_websocket)
        assert subscribed == ["task-1"]

    def test_unsubscribe_all(self, websocket_manager, mock_websocket):
        """Test unsubscribing from all tasks."""
        websocket_manager.subscribe(mock_websocket, ["task-1", "task-2"])
        websocket_manager.unsubscribe(mock_websocket, ["task-1", "task-2"])

        subscribed = websocket_manager.get_subscribed_tasks(mock_websocket)
        assert subscribed == []

    def test_subscribe_empty_list(self, websocket_manager, mock_websocket):
        """Test subscribing to empty task list."""
        websocket_manager.subscribe(mock_websocket, [])

        subscribed = websocket_manager.get_subscribed_tasks(mock_websocket)
        assert subscribed == []

    def test_multiple_subscribers_per_task(self, websocket_manager):
        """Test multiple WebSockets subscribing to same task."""
        ws1 = MagicMock()
        ws2 = MagicMock()
        ws3 = MagicMock()

        websocket_manager.subscribe(ws1, ["task-1"])
        websocket_manager.subscribe(ws2, ["task-1"])
        websocket_manager.subscribe(ws3, ["task-1"])

        subscribers = websocket_manager.get_subscribers("task-1")
        assert len(subscribers) == 3
        assert ws1 in subscribers
        assert ws2 in subscribers
        assert ws3 in subscribers

    def test_get_subscribed_tasks_nonexistent_connection(self, websocket_manager, mock_websocket):
        """Test getting subscribed tasks for connection that doesn't exist."""
        subscribed = websocket_manager.get_subscribed_tasks(mock_websocket)
        assert subscribed == []

    def test_get_subscribers_nonexistent_task(self, websocket_manager):
        """Test getting subscribers for task with no subscribers."""
        subscribers = websocket_manager.get_subscribers("nonexistent-task")
        assert subscribers == []


# ============================================================================
# Broadcasting Tests
# ============================================================================

class TestBroadcasting:
    """Tests for broadcasting task results."""

    @pytest.mark.asyncio
    async def test_broadcast_to_single_subscriber(self, websocket_manager, mock_websocket):
        """Test broadcasting to a single subscriber."""
        websocket_manager.subscribe(mock_websocket, ["task-1"])

        timestamps = TaskTimestamps(
            submitted_at="2024-01-01T00:00:00Z",
            started_at="2024-01-01T00:00:01Z",
            completed_at="2024-01-01T00:00:02Z"
        )

        await websocket_manager.broadcast_task_result(
            task_id="task-1",
            status=TaskStatus.COMPLETED,
            result={"output": "test result"},
            timestamps=timestamps,
            execution_time_ms=1000
        )

        # Verify message was sent
        mock_websocket.send_json.assert_called_once()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["task_id"] == "task-1"
        assert call_args["status"] == "completed"
        assert call_args["result"] == {"output": "test result"}

    @pytest.mark.asyncio
    async def test_broadcast_to_multiple_subscribers(self, websocket_manager):
        """Test broadcasting to multiple subscribers."""
        ws1 = MagicMock()
        ws1.send_json = AsyncMock()
        ws2 = MagicMock()
        ws2.send_json = AsyncMock()
        ws3 = MagicMock()
        ws3.send_json = AsyncMock()

        websocket_manager.subscribe(ws1, ["task-1"])
        websocket_manager.subscribe(ws2, ["task-1"])
        websocket_manager.subscribe(ws3, ["task-1"])

        timestamps = TaskTimestamps(submitted_at="2024-01-01T00:00:00Z")

        await websocket_manager.broadcast_task_result(
            task_id="task-1",
            status=TaskStatus.COMPLETED,
            timestamps=timestamps
        )

        # All should receive the message
        ws1.send_json.assert_called_once()
        ws2.send_json.assert_called_once()
        ws3.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_with_no_subscribers(self, websocket_manager):
        """Test broadcasting when no subscribers exist."""
        timestamps = TaskTimestamps(submitted_at="2024-01-01T00:00:00Z")

        # Should not raise error
        await websocket_manager.broadcast_task_result(
            task_id="task-1",
            status=TaskStatus.COMPLETED,
            timestamps=timestamps
        )

    @pytest.mark.asyncio
    async def test_broadcast_failed_task(self, websocket_manager, mock_websocket):
        """Test broadcasting failed task result."""
        websocket_manager.subscribe(mock_websocket, ["task-1"])

        timestamps = TaskTimestamps(
            submitted_at="2024-01-01T00:00:00Z",
            started_at="2024-01-01T00:00:01Z",
            completed_at="2024-01-01T00:00:02Z"
        )

        await websocket_manager.broadcast_task_result(
            task_id="task-1",
            status=TaskStatus.FAILED,
            error="Execution failed",
            timestamps=timestamps
        )

        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["status"] == "failed"
        assert call_args["error"] == "Execution failed"
        assert call_args["result"] is None

    @pytest.mark.asyncio
    async def test_broadcast_removes_subscriptions(self, websocket_manager, mock_websocket):
        """Test that broadcasting auto-unsubscribes all connections."""
        websocket_manager.subscribe(mock_websocket, ["task-1"])

        timestamps = TaskTimestamps(submitted_at="2024-01-01T00:00:00Z")

        await websocket_manager.broadcast_task_result(
            task_id="task-1",
            status=TaskStatus.COMPLETED,
            timestamps=timestamps
        )

        # After broadcast, subscriptions should be removed
        subscribers = websocket_manager.get_subscribers("task-1")
        assert subscribers == []

        subscribed = websocket_manager.get_subscribed_tasks(mock_websocket)
        assert "task-1" not in subscribed

    @pytest.mark.asyncio
    async def test_broadcast_handles_send_failure(self, websocket_manager):
        """Test that broadcast handles send failures gracefully."""
        ws1 = MagicMock()
        ws1.send_json = AsyncMock(side_effect=Exception("Send failed"))
        ws2 = MagicMock()
        ws2.send_json = AsyncMock()

        websocket_manager.subscribe(ws1, ["task-1"])
        websocket_manager.subscribe(ws2, ["task-1"])

        timestamps = TaskTimestamps(submitted_at="2024-01-01T00:00:00Z")

        # Should not raise error
        await websocket_manager.broadcast_task_result(
            task_id="task-1",
            status=TaskStatus.COMPLETED,
            timestamps=timestamps
        )

        # ws2 should still receive the message
        ws2.send_json.assert_called_once()

        # ws1 should be disconnected
        subscribed = websocket_manager.get_subscribed_tasks(ws1)
        assert subscribed == []

    @pytest.mark.asyncio
    async def test_broadcast_only_to_task_subscribers(self, websocket_manager):
        """Test that broadcast only sends to subscribers of specific task."""
        ws1 = MagicMock()
        ws1.send_json = AsyncMock()
        ws2 = MagicMock()
        ws2.send_json = AsyncMock()

        websocket_manager.subscribe(ws1, ["task-1"])
        websocket_manager.subscribe(ws2, ["task-2"])

        timestamps = TaskTimestamps(submitted_at="2024-01-01T00:00:00Z")

        await websocket_manager.broadcast_task_result(
            task_id="task-1",
            status=TaskStatus.COMPLETED,
            timestamps=timestamps
        )

        # Only ws1 should receive
        ws1.send_json.assert_called_once()
        ws2.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_with_all_fields(self, websocket_manager, mock_websocket):
        """Test broadcasting with all optional fields populated."""
        websocket_manager.subscribe(mock_websocket, ["task-1"])

        timestamps = TaskTimestamps(
            submitted_at="2024-01-01T00:00:00Z",
            started_at="2024-01-01T00:00:01Z",
            completed_at="2024-01-01T00:00:02Z"
        )

        await websocket_manager.broadcast_task_result(
            task_id="task-1",
            status=TaskStatus.COMPLETED,
            result={"output": "data", "tokens": 100},
            error=None,
            timestamps=timestamps,
            execution_time_ms=1500
        )

        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["task_id"] == "task-1"
        assert call_args["status"] == "completed"
        assert call_args["result"] == {"output": "data", "tokens": 100}
        assert call_args["execution_time_ms"] == 1500
        assert "timestamps" in call_args
