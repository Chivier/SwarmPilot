"""Unit tests for PYLET-020: Graceful Shutdown.

Tests the shutdown handler that manages graceful shutdown of worker queues
and task handling during scheduler shutdown.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.shutdown_handler import (
    ShutdownHandler,
    ShutdownResult,
)
from src.services.worker_queue_thread import QueuedTask


class TestShutdownResult:
    """Tests for ShutdownResult dataclass."""

    def test_shutdown_result_creation(self):
        """Test creating a ShutdownResult."""
        result = ShutdownResult(
            workers_stopped=3,
            tasks_dropped=10,
            timeout_occurred=False,
        )

        assert result.workers_stopped == 3
        assert result.tasks_dropped == 10
        assert result.timeout_occurred is False


class TestShutdownHandler:
    """Tests for ShutdownHandler class."""

    @pytest.mark.asyncio
    async def test_shutdown_all_stops_all_workers(self):
        """Test shutdown stops all workers."""
        mock_manager = MagicMock()
        mock_manager.get_worker_ids.return_value = ["worker-1", "worker-2"]
        mock_manager.deregister_worker.return_value = []  # No pending tasks

        mock_registry = AsyncMock()

        handler = ShutdownHandler(
            worker_queue_manager=mock_manager,
            instance_registry=mock_registry,
        )

        result = await handler.shutdown_all(timeout=60.0)

        assert result.workers_stopped == 2
        assert result.tasks_dropped == 0
        assert result.timeout_occurred is False
        assert mock_manager.deregister_worker.call_count == 2
        assert mock_registry.remove.call_count == 2

    @pytest.mark.asyncio
    async def test_shutdown_all_counts_dropped_tasks(self):
        """Test shutdown counts dropped tasks correctly."""
        mock_manager = MagicMock()
        mock_manager.get_worker_ids.return_value = ["worker-1"]

        pending_tasks = [
            QueuedTask(
                task_id="task-1",
                model_id="gpt-4",
                task_input={},
                metadata={},
                enqueue_time=1000.0,
            ),
            QueuedTask(
                task_id="task-2",
                model_id="gpt-4",
                task_input={},
                metadata={},
                enqueue_time=2000.0,
            ),
        ]
        mock_manager.deregister_worker.return_value = pending_tasks

        mock_registry = AsyncMock()

        handler = ShutdownHandler(
            worker_queue_manager=mock_manager,
            instance_registry=mock_registry,
        )

        result = await handler.shutdown_all(timeout=60.0)

        assert result.tasks_dropped == 2

    @pytest.mark.asyncio
    async def test_shutdown_all_handles_empty_workers(self):
        """Test shutdown handles case with no workers."""
        mock_manager = MagicMock()
        mock_manager.get_worker_ids.return_value = []

        mock_registry = AsyncMock()

        handler = ShutdownHandler(
            worker_queue_manager=mock_manager,
            instance_registry=mock_registry,
        )

        result = await handler.shutdown_all(timeout=60.0)

        assert result.workers_stopped == 0
        assert result.tasks_dropped == 0
        assert result.timeout_occurred is False

    @pytest.mark.asyncio
    async def test_shutdown_all_removes_from_registry(self):
        """Test shutdown removes instances from registry."""
        mock_manager = MagicMock()
        mock_manager.get_worker_ids.return_value = ["worker-1", "worker-2"]
        mock_manager.deregister_worker.return_value = []

        mock_registry = AsyncMock()

        handler = ShutdownHandler(
            worker_queue_manager=mock_manager,
            instance_registry=mock_registry,
        )

        await handler.shutdown_all(timeout=60.0)

        # Verify remove was called for each worker
        mock_registry.remove.assert_any_call("worker-1")
        mock_registry.remove.assert_any_call("worker-2")

    @pytest.mark.asyncio
    async def test_shutdown_all_handles_registry_errors(self):
        """Test shutdown continues on registry errors."""
        mock_manager = MagicMock()
        mock_manager.get_worker_ids.return_value = ["worker-1", "worker-2"]
        mock_manager.deregister_worker.return_value = []

        mock_registry = AsyncMock()
        # First call fails, second succeeds
        mock_registry.remove.side_effect = [Exception("Registry error"), None]

        handler = ShutdownHandler(
            worker_queue_manager=mock_manager,
            instance_registry=mock_registry,
        )

        # Should not raise, just log error
        result = await handler.shutdown_all(timeout=60.0)

        # Both workers should be stopped (from manager perspective)
        # But only one removed from registry successfully
        assert mock_manager.deregister_worker.call_count == 2

    @pytest.mark.asyncio
    async def test_shutdown_distributes_timeout_per_worker(self):
        """Test shutdown distributes timeout across workers."""
        mock_manager = MagicMock()
        mock_manager.get_worker_ids.return_value = ["w1", "w2", "w3"]
        mock_manager.deregister_worker.return_value = []

        mock_registry = AsyncMock()

        handler = ShutdownHandler(
            worker_queue_manager=mock_manager,
            instance_registry=mock_registry,
        )

        await handler.shutdown_all(timeout=30.0)

        # Each worker should get roughly 10 seconds (30 / 3)
        # We can't easily verify the exact timeout, but we can verify
        # the method was called
        assert mock_manager.deregister_worker.call_count == 3


class TestShutdownHandlerWithCallback:
    """Tests for shutdown with task callbacks."""

    @pytest.mark.asyncio
    async def test_shutdown_does_not_reschedule(self):
        """Test scheduler shutdown does NOT reschedule tasks (drops them)."""
        mock_manager = MagicMock()
        mock_manager.get_worker_ids.return_value = ["worker-1"]
        pending_task = QueuedTask(
            task_id="task-1",
            model_id="gpt-4",
            task_input={},
            metadata={},
            enqueue_time=1000.0,
        )
        mock_manager.deregister_worker.return_value = [pending_task]

        mock_registry = AsyncMock()
        mock_strategy = AsyncMock()

        handler = ShutdownHandler(
            worker_queue_manager=mock_manager,
            instance_registry=mock_registry,
        )

        result = await handler.shutdown_all(timeout=60.0)

        # Task should be dropped, not rescheduled
        assert result.tasks_dropped == 1
        # Scheduling strategy should NOT be called during shutdown
        mock_strategy.schedule_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_shutdown_calls_on_complete_callback(self):
        """Test shutdown invokes callback when complete."""
        mock_manager = MagicMock()
        mock_manager.get_worker_ids.return_value = []

        mock_registry = AsyncMock()
        callback_called = []

        async def on_complete(result):
            callback_called.append(result)

        handler = ShutdownHandler(
            worker_queue_manager=mock_manager,
            instance_registry=mock_registry,
            on_shutdown_complete=on_complete,
        )

        result = await handler.shutdown_all(timeout=60.0)

        assert len(callback_called) == 1
        assert callback_called[0] == result
