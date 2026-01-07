"""Unit tests for PYLET-019: Instance Sync API.

Tests the declarative instance sync endpoint where the Planner submits
the complete target instance list and the scheduler computes the diff.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.instance_sync import (
    InstanceInfo,
    InstanceSyncRequest,
    handle_instance_sync,
    handle_instance_addition,
    handle_instance_removal,
    reschedule_task,
)
from src.services.worker_queue_thread import QueuedTask


class TestInstanceInfo:
    """Tests for InstanceInfo dataclass."""

    def test_instance_info_creation(self):
        """Test creating an InstanceInfo."""
        info = InstanceInfo(
            instance_id="worker-1",
            endpoint="http://localhost:8001",
            model_id="gpt-4",
        )

        assert info.instance_id == "worker-1"
        assert info.endpoint == "http://localhost:8001"
        assert info.model_id == "gpt-4"


class TestInstanceSyncRequest:
    """Tests for InstanceSyncRequest dataclass."""

    def test_sync_request_creation(self):
        """Test creating an InstanceSyncRequest."""
        instances = [
            InstanceInfo(
                instance_id="worker-1",
                endpoint="http://localhost:8001",
                model_id="gpt-4",
            ),
            InstanceInfo(
                instance_id="worker-2",
                endpoint="http://localhost:8002",
                model_id="gpt-4",
            ),
        ]
        request = InstanceSyncRequest(instances=instances)

        assert len(request.instances) == 2


class TestHandleInstanceAddition:
    """Tests for handle_instance_addition function."""

    @pytest.mark.asyncio
    async def test_registers_in_instance_registry(self):
        """Test that instance is registered in instance registry."""
        mock_registry = AsyncMock()
        mock_manager = MagicMock()

        info = InstanceInfo(
            instance_id="worker-1",
            endpoint="http://localhost:8001",
            model_id="gpt-4",
        )

        await handle_instance_addition(
            instance_info=info,
            instance_registry=mock_registry,
            worker_queue_manager=mock_manager,
        )

        mock_registry.register.assert_called_once()

    @pytest.mark.asyncio
    async def test_creates_worker_queue_thread(self):
        """Test that worker queue thread is created."""
        mock_registry = AsyncMock()
        mock_manager = MagicMock()

        info = InstanceInfo(
            instance_id="worker-1",
            endpoint="http://localhost:8001",
            model_id="gpt-4",
        )

        await handle_instance_addition(
            instance_info=info,
            instance_registry=mock_registry,
            worker_queue_manager=mock_manager,
        )

        mock_manager.register_worker.assert_called_once_with(
            worker_id="worker-1",
            worker_endpoint="http://localhost:8001",
            model_id="gpt-4",
        )


class TestHandleInstanceRemoval:
    """Tests for handle_instance_removal function."""

    @pytest.mark.asyncio
    async def test_stops_worker_queue_thread(self):
        """Test that worker queue thread is stopped."""
        mock_registry = AsyncMock()
        mock_manager = MagicMock()
        mock_manager.deregister_worker.return_value = []  # No pending tasks

        result = await handle_instance_removal(
            instance_id="worker-1",
            instance_registry=mock_registry,
            worker_queue_manager=mock_manager,
            scheduling_strategy=MagicMock(),
        )

        mock_manager.deregister_worker.assert_called_once()
        assert result["rescheduled"] == 0

    @pytest.mark.asyncio
    async def test_returns_pending_tasks_for_rescheduling(self):
        """Test that pending tasks are returned for rescheduling."""
        mock_registry = AsyncMock()
        mock_registry.get_active_instances.return_value = []  # No other workers
        mock_manager = MagicMock()

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

        result = await handle_instance_removal(
            instance_id="worker-1",
            instance_registry=mock_registry,
            worker_queue_manager=mock_manager,
            scheduling_strategy=MagicMock(),
        )

        # No other workers, so tasks can't be rescheduled
        assert result["pending_tasks"] == 2
        assert result["rescheduled"] == 0

    @pytest.mark.asyncio
    async def test_removes_from_instance_registry(self):
        """Test that instance is removed from registry."""
        mock_registry = AsyncMock()
        mock_manager = MagicMock()
        mock_manager.deregister_worker.return_value = []

        await handle_instance_removal(
            instance_id="worker-1",
            instance_registry=mock_registry,
            worker_queue_manager=mock_manager,
            scheduling_strategy=MagicMock(),
        )

        mock_registry.remove.assert_called_once_with("worker-1")


class TestRescheduleTask:
    """Tests for reschedule_task function."""

    @pytest.mark.asyncio
    async def test_uses_scheduling_strategy(self):
        """Test that rescheduling uses the scheduling strategy."""
        mock_registry = AsyncMock()
        mock_instance = MagicMock()
        mock_instance.instance_id = "worker-2"
        mock_registry.get_active_instances.return_value = [mock_instance]

        mock_manager = MagicMock()
        mock_strategy = AsyncMock()
        mock_strategy.schedule_task.return_value = MagicMock(
            selected_instance_id="worker-2"
        )

        task = QueuedTask(
            task_id="task-1",
            model_id="gpt-4",
            task_input={},
            metadata={},
            enqueue_time=1000.0,
        )

        success = await reschedule_task(
            task=task,
            exclude_instance="worker-1",
            instance_registry=mock_registry,
            worker_queue_manager=mock_manager,
            scheduling_strategy=mock_strategy,
        )

        assert success is True
        mock_strategy.schedule_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_preserves_enqueue_time(self):
        """Test that rescheduled tasks keep original enqueue_time."""
        mock_registry = AsyncMock()
        mock_instance = MagicMock()
        mock_instance.instance_id = "worker-2"
        mock_registry.get_active_instances.return_value = [mock_instance]

        mock_manager = MagicMock()
        mock_strategy = AsyncMock()
        mock_strategy.schedule_task.return_value = MagicMock(
            selected_instance_id="worker-2"
        )

        original_time = 1000.0
        task = QueuedTask(
            task_id="task-1",
            model_id="gpt-4",
            task_input={},
            metadata={},
            enqueue_time=original_time,
        )

        await reschedule_task(
            task=task,
            exclude_instance="worker-1",
            instance_registry=mock_registry,
            worker_queue_manager=mock_manager,
            scheduling_strategy=mock_strategy,
        )

        # Verify the enqueue call uses the original task (with original time)
        mock_manager.enqueue_task.assert_called_once()
        enqueued_task = mock_manager.enqueue_task.call_args[0][1]
        assert enqueued_task.enqueue_time == original_time

    @pytest.mark.asyncio
    async def test_returns_false_when_no_workers(self):
        """Test that rescheduling returns False when no workers available."""
        mock_registry = AsyncMock()
        mock_registry.get_active_instances.return_value = []  # No workers

        mock_manager = MagicMock()
        mock_strategy = AsyncMock()

        task = QueuedTask(
            task_id="task-1",
            model_id="gpt-4",
            task_input={},
            metadata={},
            enqueue_time=1000.0,
        )

        success = await reschedule_task(
            task=task,
            exclude_instance="worker-1",
            instance_registry=mock_registry,
            worker_queue_manager=mock_manager,
            scheduling_strategy=mock_strategy,
        )

        assert success is False
        mock_strategy.schedule_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_excludes_removed_instance(self):
        """Test that the removed instance is excluded from scheduling."""
        mock_registry = AsyncMock()
        mock_instance_1 = MagicMock()
        mock_instance_1.instance_id = "worker-1"  # The removed one
        mock_instance_2 = MagicMock()
        mock_instance_2.instance_id = "worker-2"
        mock_registry.get_active_instances.return_value = [
            mock_instance_1,
            mock_instance_2,
        ]

        mock_manager = MagicMock()
        mock_strategy = AsyncMock()
        mock_strategy.schedule_task.return_value = MagicMock(
            selected_instance_id="worker-2"
        )

        task = QueuedTask(
            task_id="task-1",
            model_id="gpt-4",
            task_input={},
            metadata={},
            enqueue_time=1000.0,
        )

        await reschedule_task(
            task=task,
            exclude_instance="worker-1",
            instance_registry=mock_registry,
            worker_queue_manager=mock_manager,
            scheduling_strategy=mock_strategy,
        )

        # Check that schedule_task was called with filtered instances
        call_args = mock_strategy.schedule_task.call_args
        available = call_args.kwargs.get(
            "available_instances", call_args[1].get("available_instances")
        )
        assert len(available) == 1
        assert available[0].instance_id == "worker-2"


class TestHandleInstanceSync:
    """Tests for handle_instance_sync function."""

    @pytest.mark.asyncio
    async def test_computes_diff_correctly(self):
        """Test that sync computes the correct diff."""
        mock_registry = AsyncMock()
        mock_instance_1 = MagicMock()
        mock_instance_1.instance_id = "worker-1"
        mock_instance_2 = MagicMock()
        mock_instance_2.instance_id = "worker-2"
        mock_registry.list_all.return_value = [mock_instance_1, mock_instance_2]

        mock_manager = MagicMock()
        mock_manager.deregister_worker.return_value = []

        mock_strategy = AsyncMock()

        # Target list: keep worker-1, remove worker-2, add worker-3
        request = InstanceSyncRequest(
            instances=[
                InstanceInfo(
                    instance_id="worker-1",
                    endpoint="http://localhost:8001",
                    model_id="gpt-4",
                ),
                InstanceInfo(
                    instance_id="worker-3",
                    endpoint="http://localhost:8003",
                    model_id="gpt-4",
                ),
            ]
        )

        result = await handle_instance_sync(
            request=request,
            config_model_id="gpt-4",
            instance_registry=mock_registry,
            worker_queue_manager=mock_manager,
            scheduling_strategy=mock_strategy,
        )

        assert result.success is True
        assert "worker-3" in result.added
        assert "worker-2" in result.removed
        assert "worker-1" not in result.added
        assert "worker-1" not in result.removed

    @pytest.mark.asyncio
    async def test_validates_model_id(self):
        """Test that sync validates model_id matches config."""
        mock_registry = AsyncMock()
        mock_registry.list_all.return_value = []

        mock_manager = MagicMock()
        mock_strategy = AsyncMock()

        request = InstanceSyncRequest(
            instances=[
                InstanceInfo(
                    instance_id="worker-1",
                    endpoint="http://localhost:8001",
                    model_id="wrong-model",  # Wrong model
                ),
            ]
        )

        with pytest.raises(ValueError, match="Model mismatch"):
            await handle_instance_sync(
                request=request,
                config_model_id="gpt-4",
                instance_registry=mock_registry,
                worker_queue_manager=mock_manager,
                scheduling_strategy=mock_strategy,
            )

    @pytest.mark.asyncio
    async def test_removes_before_adds(self):
        """Test that removals happen before additions for clean state."""
        mock_registry = AsyncMock()
        mock_instance_1 = MagicMock()
        mock_instance_1.instance_id = "worker-1"
        mock_registry.list_all.return_value = [mock_instance_1]
        mock_registry.get_active_instances.return_value = []  # No reschedule targets

        mock_manager = MagicMock()
        mock_manager.deregister_worker.return_value = []

        mock_strategy = AsyncMock()

        call_order = []

        async def track_register(*args, **kwargs):
            call_order.append("register")

        def track_deregister(*args, **kwargs):
            call_order.append("deregister")
            return []

        mock_registry.register = track_register
        mock_manager.deregister_worker = track_deregister

        request = InstanceSyncRequest(
            instances=[
                InstanceInfo(
                    instance_id="worker-2",
                    endpoint="http://localhost:8002",
                    model_id="gpt-4",
                ),
            ]
        )

        await handle_instance_sync(
            request=request,
            config_model_id="gpt-4",
            instance_registry=mock_registry,
            worker_queue_manager=mock_manager,
            scheduling_strategy=mock_strategy,
        )

        # Deregister should come before register
        assert call_order == ["deregister", "register"]

    @pytest.mark.asyncio
    async def test_reschedules_tasks_from_removed_instances(self):
        """Test that tasks are rescheduled when instance is removed."""
        mock_registry = AsyncMock()
        mock_instance_1 = MagicMock()
        mock_instance_1.instance_id = "worker-1"
        mock_instance_2 = MagicMock()
        mock_instance_2.instance_id = "worker-2"
        mock_registry.list_all.return_value = [mock_instance_1, mock_instance_2]

        # worker-2 will be the reschedule target
        mock_registry.get_active_instances.return_value = [mock_instance_2]

        mock_manager = MagicMock()
        # worker-1 has pending tasks
        pending_task = QueuedTask(
            task_id="task-1",
            model_id="gpt-4",
            task_input={},
            metadata={},
            enqueue_time=1000.0,
        )
        mock_manager.deregister_worker.side_effect = lambda *args, **kwargs: (
            [pending_task] if args[0] == "worker-1" else []
        )

        mock_strategy = AsyncMock()
        mock_strategy.schedule_task.return_value = MagicMock(
            selected_instance_id="worker-2"
        )

        # Remove worker-1, keep worker-2
        request = InstanceSyncRequest(
            instances=[
                InstanceInfo(
                    instance_id="worker-2",
                    endpoint="http://localhost:8002",
                    model_id="gpt-4",
                ),
            ]
        )

        result = await handle_instance_sync(
            request=request,
            config_model_id="gpt-4",
            instance_registry=mock_registry,
            worker_queue_manager=mock_manager,
            scheduling_strategy=mock_strategy,
        )

        assert result.success is True
        assert result.rescheduled == 1
        mock_manager.enqueue_task.assert_called_once()
