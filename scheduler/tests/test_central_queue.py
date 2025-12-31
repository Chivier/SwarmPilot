"""Unit tests for CentralTaskQueue.

Tests the central task queue for task dispatch management.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.central_queue import CentralTaskQueue, QueuedTask
from src.instance_registry import InstanceRegistry
from src.task_registry import TaskRegistry


class TestQueuedTask:
    """Tests for QueuedTask data class."""

    def test_queued_task_creation(self):
        """Test basic QueuedTask creation."""
        task = QueuedTask(
            task_id="task-1",
            model_id="model-a",
            task_input={"data": "test"},
            metadata={"key": "value"},
            enqueue_time=1000.0,
            generation=0,
        )

        assert task.task_id == "task-1"
        assert task.model_id == "model-a"
        assert task.task_input == {"data": "test"}
        assert task.metadata == {"key": "value"}
        assert task.enqueue_time == 1000.0
        assert task.generation == 0

    def test_queued_task_default_generation(self):
        """Test QueuedTask with default generation."""
        task = QueuedTask(
            task_id="task-1",
            model_id="model-a",
            task_input={},
            metadata={},
            enqueue_time=1000.0,
        )
        assert task.generation == 0


class TestCentralQueueBasic:
    """Basic tests for CentralTaskQueue."""

    @pytest.fixture
    def task_registry(self):
        """Create a mock task registry."""
        return MagicMock(spec=TaskRegistry)

    @pytest.fixture
    def instance_registry(self):
        """Create a mock instance registry."""
        return MagicMock(spec=InstanceRegistry)

    @pytest.fixture
    def central_queue(self, task_registry, instance_registry):
        """Create a CentralTaskQueue instance."""
        return CentralTaskQueue(
            task_registry=task_registry,
            instance_registry=instance_registry,
            max_concurrent_dispatch=10,
        )

    def test_initialization(self, task_registry, instance_registry):
        """Test queue initialization."""
        queue = CentralTaskQueue(
            task_registry=task_registry,
            instance_registry=instance_registry,
            max_concurrent_dispatch=20,
        )

        assert queue._max_concurrent_dispatch == 20
        assert len(queue._queue) == 0
        assert queue._shutdown is False
        assert queue._generation == 0

    def test_set_scheduling_strategy(self, central_queue):
        """Test setting scheduling strategy."""
        strategy = MagicMock()
        central_queue.set_scheduling_strategy(strategy)
        assert central_queue._scheduling_strategy is strategy

    def test_set_task_dispatcher(self, central_queue):
        """Test setting task dispatcher."""
        dispatcher = MagicMock()
        central_queue.set_task_dispatcher(dispatcher)
        assert central_queue._task_dispatcher is dispatcher


class TestCentralQueueEnqueue:
    """Tests for enqueue functionality."""

    @pytest.fixture
    def task_registry(self):
        return MagicMock(spec=TaskRegistry)

    @pytest.fixture
    def instance_registry(self):
        return MagicMock(spec=InstanceRegistry)

    @pytest.fixture
    def central_queue(self, task_registry, instance_registry):
        return CentralTaskQueue(
            task_registry=task_registry,
            instance_registry=instance_registry,
        )

    @pytest.mark.asyncio
    async def test_enqueue_single_task(self, central_queue):
        """Test enqueueing a single task."""
        position = await central_queue.enqueue(
            task_id="task-1",
            model_id="model-a",
            task_input={"data": "test"},
            metadata={"key": "value"},
        )

        assert position == 1
        assert await central_queue.get_queue_size() == 1

    @pytest.mark.asyncio
    async def test_enqueue_multiple_tasks(self, central_queue):
        """Test enqueueing multiple tasks returns incrementing positions."""
        pos1 = await central_queue.enqueue(
            task_id="task-1",
            model_id="model-a",
            task_input={},
            metadata={},
        )
        pos2 = await central_queue.enqueue(
            task_id="task-2",
            model_id="model-a",
            task_input={},
            metadata={},
        )
        pos3 = await central_queue.enqueue(
            task_id="task-3",
            model_id="model-b",
            task_input={},
            metadata={},
        )

        assert pos1 == 1
        assert pos2 == 2
        assert pos3 == 3
        assert await central_queue.get_queue_size() == 3

    @pytest.mark.asyncio
    async def test_enqueue_with_custom_time(self, central_queue):
        """Test enqueueing with custom enqueue time."""
        await central_queue.enqueue(
            task_id="task-1",
            model_id="model-a",
            task_input={},
            metadata={},
            enqueue_time=12345.0,
        )

        assert central_queue._queue[0].enqueue_time == 12345.0

    @pytest.mark.asyncio
    async def test_enqueue_sets_dispatch_event(self, central_queue):
        """Test that enqueueing signals the dispatch event."""
        central_queue._dispatch_event.clear()

        await central_queue.enqueue(
            task_id="task-1",
            model_id="model-a",
            task_input={},
            metadata={},
        )

        assert central_queue._dispatch_event.is_set()

    @pytest.mark.asyncio
    async def test_enqueue_uses_current_generation(self, central_queue):
        """Test that enqueued tasks get the current generation."""
        central_queue._generation = 5

        await central_queue.enqueue(
            task_id="task-1",
            model_id="model-a",
            task_input={},
            metadata={},
        )

        assert central_queue._queue[0].generation == 5


class TestCentralQueueClear:
    """Tests for queue clear functionality."""

    @pytest.fixture
    def task_registry(self):
        return MagicMock(spec=TaskRegistry)

    @pytest.fixture
    def instance_registry(self):
        return MagicMock(spec=InstanceRegistry)

    @pytest.fixture
    def central_queue(self, task_registry, instance_registry):
        return CentralTaskQueue(
            task_registry=task_registry,
            instance_registry=instance_registry,
        )

    @pytest.mark.asyncio
    async def test_clear_empty_queue(self, central_queue):
        """Test clearing an empty queue."""
        count = await central_queue.clear()

        assert count == 0
        assert await central_queue.get_queue_size() == 0

    @pytest.mark.asyncio
    async def test_clear_with_tasks(self, central_queue):
        """Test clearing queue with tasks."""
        await central_queue.enqueue("task-1", "model-a", {}, {})
        await central_queue.enqueue("task-2", "model-a", {}, {})
        await central_queue.enqueue("task-3", "model-b", {}, {})

        count = await central_queue.clear()

        assert count == 3
        assert await central_queue.get_queue_size() == 0

    @pytest.mark.asyncio
    async def test_clear_increments_generation(self, central_queue):
        """Test that clear increments the generation counter."""
        initial_gen = central_queue._generation

        await central_queue.clear()

        assert central_queue._generation == initial_gen + 1

    @pytest.mark.asyncio
    async def test_multiple_clears_increment_generation(self, central_queue):
        """Test multiple clears increment generation each time."""
        await central_queue.clear()
        await central_queue.clear()
        await central_queue.clear()

        assert central_queue._generation == 3


class TestCentralQueueInfo:
    """Tests for queue info functionality."""

    @pytest.fixture
    def task_registry(self):
        return MagicMock(spec=TaskRegistry)

    @pytest.fixture
    def instance_registry(self):
        return MagicMock(spec=InstanceRegistry)

    @pytest.fixture
    def central_queue(self, task_registry, instance_registry):
        return CentralTaskQueue(
            task_registry=task_registry,
            instance_registry=instance_registry,
        )

    @pytest.mark.asyncio
    async def test_get_queue_info_empty(self, central_queue):
        """Test queue info for empty queue."""
        info = await central_queue.get_queue_info()

        assert info["total_size"] == 0
        assert info["by_model"] == {}

    @pytest.mark.asyncio
    async def test_get_queue_info_with_tasks(self, central_queue):
        """Test queue info with tasks grouped by model."""
        await central_queue.enqueue("task-1", "model-a", {}, {})
        await central_queue.enqueue("task-2", "model-a", {}, {})
        await central_queue.enqueue("task-3", "model-b", {}, {})
        await central_queue.enqueue("task-4", "model-c", {}, {})
        await central_queue.enqueue("task-5", "model-a", {}, {})

        info = await central_queue.get_queue_info()

        assert info["total_size"] == 5
        assert info["by_model"]["model-a"] == 3
        assert info["by_model"]["model-b"] == 1
        assert info["by_model"]["model-c"] == 1


class TestCentralQueueNotify:
    """Tests for notify_capacity_available."""

    @pytest.fixture
    def central_queue(self):
        return CentralTaskQueue(
            task_registry=MagicMock(spec=TaskRegistry),
            instance_registry=MagicMock(spec=InstanceRegistry),
        )

    @pytest.mark.asyncio
    async def test_notify_capacity_available_sets_event(self, central_queue):
        """Test notify_capacity_available sets the dispatch event."""
        central_queue._dispatch_event.clear()

        await central_queue.notify_capacity_available()

        assert central_queue._dispatch_event.is_set()


class TestCentralQueueStart:
    """Tests for start functionality."""

    @pytest.fixture
    def central_queue(self):
        return CentralTaskQueue(
            task_registry=MagicMock(spec=TaskRegistry),
            instance_registry=MagicMock(spec=InstanceRegistry),
        )

    @pytest.mark.asyncio
    async def test_start_creates_dispatcher_task(self, central_queue):
        """Test start creates dispatcher task."""
        await central_queue.start()

        assert central_queue._dispatcher_task is not None
        assert not central_queue._shutdown

        # Cleanup
        await central_queue.shutdown()

    @pytest.mark.asyncio
    async def test_start_already_running_warns(self, central_queue):
        """Test start when dispatcher already running logs warning."""
        await central_queue.start()

        # Second start should warn
        with patch("src.services.central_queue.logger") as mock_logger:
            await central_queue.start()
            mock_logger.warning.assert_called_with("Dispatcher already running")

        # Cleanup
        await central_queue.shutdown()


class TestCentralQueueShutdown:
    """Tests for shutdown functionality."""

    @pytest.fixture
    def central_queue(self):
        return CentralTaskQueue(
            task_registry=MagicMock(spec=TaskRegistry),
            instance_registry=MagicMock(spec=InstanceRegistry),
        )

    @pytest.mark.asyncio
    async def test_shutdown_graceful(self, central_queue):
        """Test graceful shutdown."""
        await central_queue.start()
        await central_queue.shutdown()

        assert central_queue._shutdown is True
        assert central_queue._dispatcher_task is None

    @pytest.mark.asyncio
    async def test_shutdown_timeout_cancels_task(self, central_queue):
        """Test shutdown timeout cancels the task."""

        # Create a dispatcher task that hangs
        async def hanging_dispatcher():
            try:
                await asyncio.sleep(100)  # Long sleep
            except asyncio.CancelledError:
                raise

        central_queue._dispatcher_task = asyncio.create_task(
            hanging_dispatcher()
        )
        central_queue._shutdown = False

        # Patch wait_for to timeout immediately
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            await central_queue.shutdown()

        assert central_queue._dispatcher_task is None

    @pytest.mark.asyncio
    async def test_shutdown_without_start(self, central_queue):
        """Test shutdown when dispatcher was never started."""
        await central_queue.shutdown()

        assert central_queue._shutdown is True
        assert central_queue._dispatcher_task is None


class TestCentralQueueDispatchLoop:
    """Tests for the dispatch loop."""

    @pytest.fixture
    def task_registry(self):
        return MagicMock(spec=TaskRegistry)

    @pytest.fixture
    def instance_registry(self):
        registry = MagicMock(spec=InstanceRegistry)
        registry.has_active_instance = AsyncMock(return_value=False)
        return registry

    @pytest.fixture
    def central_queue(self, task_registry, instance_registry):
        return CentralTaskQueue(
            task_registry=task_registry,
            instance_registry=instance_registry,
        )

    @pytest.mark.asyncio
    async def test_dispatch_loop_exits_on_shutdown(self, central_queue):
        """Test dispatch loop exits when shutdown is set."""
        await central_queue.start()

        # Give it time to start
        await asyncio.sleep(0.1)

        await central_queue.shutdown()

        # Verify it stopped
        assert central_queue._dispatcher_task is None

    @pytest.mark.asyncio
    async def test_dispatch_loop_continues_on_error(self, central_queue):
        """Test dispatch loop continues after error."""
        central_queue.set_scheduling_strategy(MagicMock())
        central_queue.set_task_dispatcher(MagicMock())

        # Make _try_dispatch_tasks raise an exception
        with patch.object(
            central_queue,
            "_try_dispatch_tasks",
            side_effect=Exception("Test error"),
        ):
            await central_queue.start()

            # Trigger dispatch
            central_queue._dispatch_event.set()
            await asyncio.sleep(0.2)  # Let it process

            # Should still be running
            assert central_queue._dispatcher_task is not None

            await central_queue.shutdown()


class TestCentralQueueTryDispatch:
    """Tests for _try_dispatch_tasks."""

    @pytest.fixture
    def task_registry(self):
        registry = MagicMock(spec=TaskRegistry)
        registry.get = AsyncMock(return_value=None)
        return registry

    @pytest.fixture
    def instance_registry(self):
        registry = MagicMock(spec=InstanceRegistry)
        registry.has_active_instance = AsyncMock(return_value=True)
        registry.get_active_instances = AsyncMock(return_value=[])
        return registry

    @pytest.fixture
    def central_queue(self, task_registry, instance_registry):
        return CentralTaskQueue(
            task_registry=task_registry,
            instance_registry=instance_registry,
        )

    @pytest.mark.asyncio
    async def test_try_dispatch_without_strategy_warns(self, central_queue):
        """Test _try_dispatch_tasks warns when strategy not set."""
        with patch("src.services.central_queue.logger") as mock_logger:
            await central_queue._try_dispatch_tasks()
            mock_logger.warning.assert_called_with(
                "Scheduling strategy or task dispatcher not set"
            )

    @pytest.mark.asyncio
    async def test_try_dispatch_without_dispatcher_warns(self, central_queue):
        """Test _try_dispatch_tasks warns when dispatcher not set."""
        central_queue.set_scheduling_strategy(MagicMock())

        with patch("src.services.central_queue.logger") as mock_logger:
            await central_queue._try_dispatch_tasks()
            mock_logger.warning.assert_called_with(
                "Scheduling strategy or task dispatcher not set"
            )

    @pytest.mark.asyncio
    async def test_try_dispatch_skips_task_without_instance(
        self, central_queue, instance_registry
    ):
        """Test tasks without available instances are skipped."""
        instance_registry.has_active_instance = AsyncMock(return_value=False)

        central_queue.set_scheduling_strategy(MagicMock())
        central_queue.set_task_dispatcher(MagicMock())

        await central_queue.enqueue("task-1", "model-a", {}, {})

        await central_queue._try_dispatch_tasks()

        # Task should still be in queue (skipped)
        assert await central_queue.get_queue_size() == 1

    @pytest.mark.asyncio
    async def test_try_dispatch_discards_stale_tasks(
        self, central_queue, instance_registry
    ):
        """Test stale tasks from previous generations are discarded."""
        instance_registry.has_active_instance = AsyncMock(return_value=False)

        central_queue.set_scheduling_strategy(MagicMock())
        central_queue.set_task_dispatcher(MagicMock())

        # Enqueue task with generation 0
        await central_queue.enqueue("task-1", "model-a", {}, {})

        # Pop the task manually
        async with central_queue._queue_lock:
            task = central_queue._queue.popleft()

        # Clear queue (increments generation)
        await central_queue.clear()

        # Simulate _try_dispatch_tasks trying to re-add the stale task
        # The task has generation 0, but current generation is 1
        async with central_queue._queue_lock:
            current_generation = central_queue._generation
            if task.generation == current_generation:
                central_queue._queue.appendleft(task)

        # Task should NOT have been re-added (generation mismatch)
        assert await central_queue.get_queue_size() == 0


class TestCentralQueueDispatchSingleTask:
    """Tests for _dispatch_single_task."""

    @pytest.fixture
    def task_registry(self):
        registry = MagicMock(spec=TaskRegistry)
        task_record = MagicMock()
        registry.get = AsyncMock(return_value=task_record)
        return registry

    @pytest.fixture
    def instance_registry(self):
        registry = MagicMock(spec=InstanceRegistry)
        registry.get_active_instances = AsyncMock(return_value=[])
        registry.increment_pending = AsyncMock()
        return registry

    @pytest.fixture
    def mock_instance(self):
        """Create a mock instance."""
        instance = MagicMock()
        instance.instance_id = "inst-1"
        instance.endpoint = "http://localhost:8000"
        return instance

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock scheduling strategy."""
        strategy = MagicMock()
        result = MagicMock()
        result.selected_instance_id = "inst-1"
        result.selected_prediction = None
        strategy.schedule_task = AsyncMock(return_value=result)
        return strategy

    @pytest.fixture
    def mock_dispatcher(self):
        """Create a mock task dispatcher."""
        dispatcher = MagicMock()
        dispatcher.dispatch_task_async = MagicMock()
        return dispatcher

    @pytest.fixture
    def central_queue(self, task_registry, instance_registry):
        return CentralTaskQueue(
            task_registry=task_registry,
            instance_registry=instance_registry,
        )

    @pytest.mark.asyncio
    async def test_dispatch_single_task_success(
        self,
        central_queue,
        instance_registry,
        mock_instance,
        mock_strategy,
        mock_dispatcher,
    ):
        """Test successful single task dispatch."""
        instance_registry.get_active_instances = AsyncMock(
            return_value=[mock_instance]
        )
        central_queue.set_scheduling_strategy(mock_strategy)
        central_queue.set_task_dispatcher(mock_dispatcher)

        task = QueuedTask(
            task_id="task-1",
            model_id="model-a",
            task_input={},
            metadata={},
            enqueue_time=1000.0,
            generation=0,
        )

        success = await central_queue._dispatch_single_task(task)

        assert success is True
        mock_strategy.schedule_task.assert_called_once()
        mock_dispatcher.dispatch_task_async.assert_called_once_with("task-1")
        instance_registry.increment_pending.assert_called_once_with("inst-1")

    @pytest.mark.asyncio
    async def test_dispatch_single_task_no_instances(
        self, central_queue, instance_registry, mock_strategy, mock_dispatcher
    ):
        """Test dispatch fails when no instances available."""
        instance_registry.get_active_instances = AsyncMock(return_value=[])
        central_queue.set_scheduling_strategy(mock_strategy)
        central_queue.set_task_dispatcher(mock_dispatcher)

        task = QueuedTask(
            task_id="task-1",
            model_id="model-a",
            task_input={},
            metadata={},
            enqueue_time=1000.0,
        )

        success = await central_queue._dispatch_single_task(task)

        assert success is False
        mock_strategy.schedule_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_single_task_stale_generation(
        self,
        central_queue,
        instance_registry,
        mock_instance,
        mock_strategy,
        mock_dispatcher,
    ):
        """Test dispatch returns True for stale tasks to not re-queue."""
        instance_registry.get_active_instances = AsyncMock(
            return_value=[mock_instance]
        )
        central_queue.set_scheduling_strategy(mock_strategy)
        central_queue.set_task_dispatcher(mock_dispatcher)

        # Task has generation 0, but queue is at generation 5
        central_queue._generation = 5

        task = QueuedTask(
            task_id="task-1",
            model_id="model-a",
            task_input={},
            metadata={},
            enqueue_time=1000.0,
            generation=0,  # Stale generation
        )

        success = await central_queue._dispatch_single_task(task)

        # Returns True so the task is not re-queued
        assert success is True
        mock_strategy.schedule_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_single_task_scheduling_error(
        self, central_queue, instance_registry, mock_instance, mock_dispatcher
    ):
        """Test dispatch handles scheduling errors gracefully."""
        instance_registry.get_active_instances = AsyncMock(
            return_value=[mock_instance]
        )

        strategy = MagicMock()
        strategy.schedule_task = AsyncMock(
            side_effect=Exception("Scheduling failed")
        )

        central_queue.set_scheduling_strategy(strategy)
        central_queue.set_task_dispatcher(mock_dispatcher)

        task = QueuedTask(
            task_id="task-1",
            model_id="model-a",
            task_input={},
            metadata={},
            enqueue_time=1000.0,
        )

        success = await central_queue._dispatch_single_task(task)

        assert success is False

    @pytest.mark.asyncio
    async def test_dispatch_single_task_no_instance_selected(
        self, central_queue, instance_registry, mock_instance, mock_dispatcher
    ):
        """Test dispatch fails when strategy returns no instance."""
        instance_registry.get_active_instances = AsyncMock(
            return_value=[mock_instance]
        )

        strategy = MagicMock()
        result = MagicMock()
        result.selected_instance_id = None
        strategy.schedule_task = AsyncMock(return_value=result)

        central_queue.set_scheduling_strategy(strategy)
        central_queue.set_task_dispatcher(mock_dispatcher)

        task = QueuedTask(
            task_id="task-1",
            model_id="model-a",
            task_input={},
            metadata={},
            enqueue_time=1000.0,
        )

        success = await central_queue._dispatch_single_task(task)

        assert success is False

    @pytest.mark.asyncio
    async def test_dispatch_single_task_instance_not_found(
        self, central_queue, instance_registry, mock_instance, mock_dispatcher
    ):
        """Test dispatch fails when selected instance not in list."""
        instance_registry.get_active_instances = AsyncMock(
            return_value=[mock_instance]
        )

        strategy = MagicMock()
        result = MagicMock()
        result.selected_instance_id = (
            "different-inst"  # Not in available instances
        )
        strategy.schedule_task = AsyncMock(return_value=result)

        central_queue.set_scheduling_strategy(strategy)
        central_queue.set_task_dispatcher(mock_dispatcher)

        task = QueuedTask(
            task_id="task-1",
            model_id="model-a",
            task_input={},
            metadata={},
            enqueue_time=1000.0,
        )

        success = await central_queue._dispatch_single_task(task)

        assert success is False

    @pytest.mark.asyncio
    async def test_dispatch_single_task_with_prediction(
        self,
        central_queue,
        task_registry,
        instance_registry,
        mock_instance,
        mock_dispatcher,
    ):
        """Test dispatch updates task with prediction info."""
        instance_registry.get_active_instances = AsyncMock(
            return_value=[mock_instance]
        )

        prediction = MagicMock()
        prediction.predicted_time_ms = 1500.0
        prediction.error_margin_ms = 200.0
        prediction.quantiles = {0.5: 1400.0, 0.9: 1800.0}

        strategy = MagicMock()
        result = MagicMock()
        result.selected_instance_id = "inst-1"
        result.selected_prediction = prediction
        strategy.schedule_task = AsyncMock(return_value=result)

        central_queue.set_scheduling_strategy(strategy)
        central_queue.set_task_dispatcher(mock_dispatcher)

        task = QueuedTask(
            task_id="task-1",
            model_id="model-a",
            task_input={},
            metadata={},
            enqueue_time=1000.0,
        )

        success = await central_queue._dispatch_single_task(task)

        assert success is True

        # Verify task record was updated with prediction
        task_record = await task_registry.get("task-1")
        assert task_record.predicted_time_ms == 1500.0
        assert task_record.predicted_error_margin_ms == 200.0
        assert task_record.predicted_quantiles == {0.5: 1400.0, 0.9: 1800.0}

    @pytest.mark.asyncio
    async def test_dispatch_single_task_exception_handling(
        self,
        central_queue,
        instance_registry,
        mock_instance,
        mock_strategy,
        mock_dispatcher,
    ):
        """Test dispatch handles unexpected exceptions."""
        instance_registry.get_active_instances = AsyncMock(
            side_effect=Exception("Unexpected error")
        )
        central_queue.set_scheduling_strategy(mock_strategy)
        central_queue.set_task_dispatcher(mock_dispatcher)

        task = QueuedTask(
            task_id="task-1",
            model_id="model-a",
            task_input={},
            metadata={},
            enqueue_time=1000.0,
        )

        success = await central_queue._dispatch_single_task(task)

        assert success is False


class TestCentralQueueIntegration:
    """Integration tests for the full dispatch flow."""

    @pytest.fixture
    def task_registry(self):
        registry = MagicMock(spec=TaskRegistry)
        task_record = MagicMock()
        registry.get = AsyncMock(return_value=task_record)
        return registry

    @pytest.fixture
    def instance_registry(self):
        registry = MagicMock(spec=InstanceRegistry)
        registry.increment_pending = AsyncMock()
        return registry

    @pytest.fixture
    def mock_instance(self):
        instance = MagicMock()
        instance.instance_id = "inst-1"
        instance.endpoint = "http://localhost:8000"
        return instance

    @pytest.mark.asyncio
    async def test_full_dispatch_flow(
        self, task_registry, instance_registry, mock_instance
    ):
        """Test full flow from enqueue to dispatch."""
        instance_registry.has_active_instance = AsyncMock(return_value=True)
        instance_registry.get_active_instances = AsyncMock(
            return_value=[mock_instance]
        )

        queue = CentralTaskQueue(
            task_registry=task_registry,
            instance_registry=instance_registry,
        )

        # Set up strategy and dispatcher
        strategy = MagicMock()
        result = MagicMock()
        result.selected_instance_id = "inst-1"
        result.selected_prediction = None
        strategy.schedule_task = AsyncMock(return_value=result)

        dispatcher = MagicMock()
        dispatcher.dispatch_task_async = MagicMock()

        queue.set_scheduling_strategy(strategy)
        queue.set_task_dispatcher(dispatcher)

        # Enqueue a task
        await queue.enqueue(
            "task-1", "model-a", {"data": "test"}, {"key": "value"}
        )

        # Manually trigger dispatch
        await queue._try_dispatch_tasks()

        # Verify task was dispatched
        dispatcher.dispatch_task_async.assert_called_once_with("task-1")
        assert await queue.get_queue_size() == 0

    @pytest.mark.asyncio
    async def test_fifo_order_maintained(
        self, task_registry, instance_registry, mock_instance
    ):
        """Test FIFO order is maintained during dispatch."""
        instance_registry.has_active_instance = AsyncMock(return_value=True)
        instance_registry.get_active_instances = AsyncMock(
            return_value=[mock_instance]
        )

        queue = CentralTaskQueue(
            task_registry=task_registry,
            instance_registry=instance_registry,
        )

        strategy = MagicMock()
        result = MagicMock()
        result.selected_instance_id = "inst-1"
        result.selected_prediction = None
        strategy.schedule_task = AsyncMock(return_value=result)

        dispatched_tasks = []
        dispatcher = MagicMock()
        dispatcher.dispatch_task_async = MagicMock(
            side_effect=lambda task_id: dispatched_tasks.append(task_id)
        )

        queue.set_scheduling_strategy(strategy)
        queue.set_task_dispatcher(dispatcher)

        # Enqueue tasks in order
        await queue.enqueue("task-1", "model-a", {}, {})
        await queue.enqueue("task-2", "model-a", {}, {})
        await queue.enqueue("task-3", "model-a", {}, {})

        # Dispatch all
        await queue._try_dispatch_tasks()

        # Verify FIFO order
        assert dispatched_tasks == ["task-1", "task-2", "task-3"]

    @pytest.mark.asyncio
    async def test_mixed_model_dispatch(self, task_registry, instance_registry):
        """Test dispatch with mixed models - some have instances, some don't."""
        mock_inst_a = MagicMock()
        mock_inst_a.instance_id = "inst-a"

        async def has_active_instance(model_id):
            return model_id == "model-a"

        async def get_active_instances(model_id):
            if model_id == "model-a":
                return [mock_inst_a]
            return []

        instance_registry.has_active_instance = AsyncMock(
            side_effect=has_active_instance
        )
        instance_registry.get_active_instances = AsyncMock(
            side_effect=get_active_instances
        )

        queue = CentralTaskQueue(
            task_registry=task_registry,
            instance_registry=instance_registry,
        )

        strategy = MagicMock()
        result = MagicMock()
        result.selected_instance_id = "inst-a"
        result.selected_prediction = None
        strategy.schedule_task = AsyncMock(return_value=result)

        dispatcher = MagicMock()
        dispatcher.dispatch_task_async = MagicMock()

        queue.set_scheduling_strategy(strategy)
        queue.set_task_dispatcher(dispatcher)

        # Enqueue tasks for different models
        await queue.enqueue("task-a1", "model-a", {}, {})
        await queue.enqueue("task-b1", "model-b", {}, {})  # No instance
        await queue.enqueue("task-a2", "model-a", {}, {})

        await queue._try_dispatch_tasks()

        # Only model-a tasks should be dispatched
        assert dispatcher.dispatch_task_async.call_count == 2

        # model-b task should remain in queue
        assert await queue.get_queue_size() == 1
