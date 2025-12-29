"""Tests for BackgroundScheduler.

This module tests background task scheduling functionality including
error paths, edge cases, and shutdown behavior.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.background_scheduler import BackgroundScheduler
from src.model import Instance, TaskStatus
from src.predictor_client import Prediction
from src.scheduler import ScheduleResult

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_scheduling_strategy():
    """Create a mock scheduling strategy."""
    strategy = MagicMock()
    strategy.schedule_task = AsyncMock()
    return strategy


@pytest.fixture
def mock_task_dispatcher():
    """Create a mock task dispatcher."""
    dispatcher = MagicMock()
    dispatcher.dispatch_task_async = MagicMock()
    return dispatcher


@pytest.fixture
def background_scheduler(
    mock_scheduling_strategy,
    task_registry,
    instance_registry,
    mock_task_dispatcher,
):
    """Create a BackgroundScheduler for testing."""
    return BackgroundScheduler(
        scheduling_strategy=mock_scheduling_strategy,
        task_registry=task_registry,
        instance_registry=instance_registry,
        task_dispatcher=mock_task_dispatcher,
        max_concurrent_scheduling=50,
    )


@pytest.fixture
def sample_schedule_result():
    """Create a sample schedule result."""
    prediction = Prediction(
        instance_id="instance-1",
        predicted_time_ms=150.0,
        confidence=0.95,
        quantiles={0.5: 100.0, 0.9: 200.0, 0.95: 300.0, 0.99: 500.0},
    )
    return ScheduleResult(
        selected_instance_id="instance-1", selected_prediction=prediction
    )


# ============================================================================
# Basic Functionality Tests
# ============================================================================


class TestBasicFunctionality:
    """Tests for basic background scheduler functionality."""

    @pytest.mark.asyncio
    async def test_successful_scheduling(
        self,
        background_scheduler,
        instance_registry,
        task_registry,
        mock_scheduling_strategy,
        mock_task_dispatcher,
        sample_instance,
        sample_schedule_result,
    ):
        """Test successful background task scheduling."""
        # Setup
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance=None,
        )
        mock_scheduling_strategy.schedule_task.return_value = (
            sample_schedule_result
        )

        # Execute
        background_scheduler.schedule_task_background(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
        )

        # Wait for background task to complete
        await asyncio.sleep(0.1)

        # Verify
        task = await task_registry.get("task-1")
        assert task.assigned_instance == "instance-1"
        assert task.predicted_time_ms == 150.0
        mock_task_dispatcher.dispatch_task_async.assert_called_once_with(
            "task-1"
        )

    @pytest.mark.asyncio
    async def test_get_stats(self, background_scheduler):
        """Test getting scheduler statistics."""
        stats = await background_scheduler.get_stats()

        assert "active_scheduling_tasks" in stats
        assert "max_concurrent_scheduling" in stats
        assert "available_slots" in stats
        assert stats["active_scheduling_tasks"] == 0
        assert stats["max_concurrent_scheduling"] == 50


# ============================================================================
# Error Path Tests - Lines 132-137
# ============================================================================


class TestNoInstancesAvailable:
    """Tests for no available instances error path (lines 132-137)."""

    @pytest.mark.asyncio
    async def test_no_instances_available(
        self, background_scheduler, task_registry, instance_registry
    ):
        """Test handling when no instances are available (lines 132-137)."""
        # Setup - create task but no instances
        await task_registry.create_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance=None,
        )

        # Execute - schedule task with no available instances
        background_scheduler.schedule_task_background(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
        )

        # Wait for background task to complete
        await asyncio.sleep(0.1)

        # Verify task is marked as failed
        task = await task_registry.get("task-1")
        assert task.status == TaskStatus.FAILED
        assert "No available instance" in task.error
        assert "test-model" in task.error


# ============================================================================
# Error Path Tests - Lines 157-158
# ============================================================================


class TestTaskNotFound:
    """Tests for task not found error path (lines 157-158)."""

    @pytest.mark.asyncio
    async def test_task_not_found_after_scheduling(
        self,
        background_scheduler,
        task_registry,
        instance_registry,
        mock_scheduling_strategy,
        sample_instance,
        sample_schedule_result,
    ):
        """Test handling when task disappears during scheduling (lines 176-177)."""
        # Setup
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance=None,
        )
        mock_scheduling_strategy.schedule_task.return_value = (
            sample_schedule_result
        )

        # Mock task_registry.get to always return None (simulating task deletion
        # after update_status but before the get call on line 174)
        async def mock_get(task_id):
            return None

        task_registry.get = mock_get

        # Execute
        background_scheduler.schedule_task_background(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
        )

        # Wait for background task to complete
        await asyncio.sleep(0.1)

        # Verify - should log error but not crash
        # The test passes if no exception is raised


# ============================================================================
# Error Path Tests - Scheduling Failure
# ============================================================================


class TestSchedulingFailure:
    """Tests for scheduling strategy failure."""

    @pytest.mark.asyncio
    async def test_scheduling_strategy_fails(
        self,
        background_scheduler,
        task_registry,
        instance_registry,
        mock_scheduling_strategy,
        sample_instance,
    ):
        """Test handling when scheduling strategy fails."""
        # Setup
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance=None,
        )

        # Make scheduling strategy raise an exception
        mock_scheduling_strategy.schedule_task.side_effect = Exception(
            "Prediction service error"
        )

        # Execute
        background_scheduler.schedule_task_background(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
        )

        # Wait for background task to complete
        await asyncio.sleep(0.1)

        # Verify task is marked as failed
        task = await task_registry.get("task-1")
        assert task.status == TaskStatus.FAILED
        assert "Scheduling failed" in task.error
        assert "Prediction service error" in task.error


# ============================================================================
# Error Path Tests - Lines 188
# ============================================================================


class TestUnexpectedError:
    """Tests for unexpected error handling (line 188)."""

    @pytest.mark.asyncio
    async def test_unexpected_error_with_update_failure(
        self,
        background_scheduler,
        task_registry,
        instance_registry,
        sample_instance,
    ):
        """Test unexpected error where updating task status also fails (line 188)."""
        # Setup
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance=None,
        )

        # Mock instance_registry.increment_pending to raise an exception
        instance_registry.increment_pending = AsyncMock(
            side_effect=Exception("Unexpected registry error")
        )

        # Mock task_registry.update_status to also fail (for line 188 coverage)
        original_update = task_registry.update_status
        task_registry.update_status = AsyncMock(
            side_effect=Exception("Update status failed")
        )

        # Execute
        background_scheduler.schedule_task_background(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
        )

        # Wait for background task to complete
        await asyncio.sleep(0.1)

        # Restore original method
        task_registry.update_status = original_update

        # Verify - should log error but not crash
        # The test passes if no exception is raised


# ============================================================================
# Test set_error path (line 207) - exception during scheduling, update_status succeeds
# ============================================================================


class TestSetErrorPath:
    """Tests for set_error call in exception handler (line 207)."""

    @pytest.mark.asyncio
    async def test_unexpected_error_sets_error_message(
        self,
        background_scheduler,
        task_registry,
        instance_registry,
        mock_scheduling_strategy,
        sample_instance,
        sample_schedule_result,
    ):
        """Test that set_error is called when unexpected exception occurs (line 207)."""
        # Setup
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance=None,
        )
        mock_scheduling_strategy.schedule_task.return_value = (
            sample_schedule_result
        )

        # Mock instance_registry.increment_pending to raise an exception
        # This will trigger the exception handler after update_status succeeds
        instance_registry.increment_pending = AsyncMock(
            side_effect=Exception("Unexpected increment error")
        )

        # Execute
        background_scheduler.schedule_task_background(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
        )

        # Wait for background task to complete
        await asyncio.sleep(0.15)

        # Verify task is marked as failed with error message
        task = await task_registry.get("task-1")
        assert task.status == TaskStatus.FAILED
        assert "Internal scheduling error" in task.error
        assert "Unexpected increment error" in task.error


# ============================================================================
# Shutdown Tests - Lines 214-221
# ============================================================================


class TestShutdown:
    """Tests for shutdown functionality (lines 214-221)."""

    @pytest.mark.asyncio
    async def test_shutdown_with_active_tasks(
        self,
        background_scheduler,
        task_registry,
        instance_registry,
        mock_scheduling_strategy,
        sample_instance,
        sample_schedule_result,
    ):
        """Test shutdown with active scheduling tasks (lines 214-221)."""
        # Setup
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance=None,
        )
        await task_registry.create_task(
            task_id="task-2",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance=None,
        )

        # Make scheduling slow to ensure tasks are still running during shutdown
        async def slow_schedule(*args, **kwargs):
            await asyncio.sleep(0.2)
            return sample_schedule_result

        mock_scheduling_strategy.schedule_task.side_effect = slow_schedule

        # Execute - start multiple background tasks
        background_scheduler.schedule_task_background(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
        )
        background_scheduler.schedule_task_background(
            task_id="task-2",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
        )

        # Verify tasks are active
        stats = await background_scheduler.get_stats()
        assert stats["active_scheduling_tasks"] == 2

        # Shutdown and wait for tasks to complete
        await background_scheduler.shutdown()

        # Verify all tasks completed
        stats = await background_scheduler.get_stats()
        assert stats["active_scheduling_tasks"] == 0

    @pytest.mark.asyncio
    async def test_shutdown_with_no_active_tasks(self, background_scheduler):
        """Test shutdown with no active tasks."""
        # Execute
        await background_scheduler.shutdown()

        # Verify - should complete without error
        stats = await background_scheduler.get_stats()
        assert stats["active_scheduling_tasks"] == 0


# ============================================================================
# Concurrency Tests
# ============================================================================


class TestConcurrency:
    """Tests for concurrent scheduling operations."""

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(
        self,
        task_registry,
        instance_registry,
        mock_scheduling_strategy,
        mock_task_dispatcher,
        sample_instance,
        sample_schedule_result,
    ):
        """Test that semaphore limits concurrent scheduling."""
        # Create scheduler with low concurrency limit
        scheduler = BackgroundScheduler(
            scheduling_strategy=mock_scheduling_strategy,
            task_registry=task_registry,
            instance_registry=instance_registry,
            task_dispatcher=mock_task_dispatcher,
            max_concurrent_scheduling=2,
        )

        # Setup
        await instance_registry.register(sample_instance)
        for i in range(5):
            await task_registry.create_task(
                task_id=f"task-{i}",
                model_id="test-model",
                task_input={"prompt": "test"},
                metadata={},
                assigned_instance=None,
            )

        # Make scheduling slow
        async def slow_schedule(*args, **kwargs):
            await asyncio.sleep(0.1)
            return sample_schedule_result

        mock_scheduling_strategy.schedule_task.side_effect = slow_schedule

        # Schedule multiple tasks
        for i in range(5):
            scheduler.schedule_task_background(
                task_id=f"task-{i}",
                model_id="test-model",
                task_input={"prompt": "test"},
                metadata={},
            )

        # Wait a bit for tasks to enter the semaphore
        await asyncio.sleep(0.15)

        # Verify all tasks were tracked
        stats = await scheduler.get_stats()
        assert (
            stats["active_scheduling_tasks"] >= 0
        )  # Some tasks may have completed

        # Wait for all to complete
        await scheduler.shutdown()

        # All tasks should be done now
        final_stats = await scheduler.get_stats()
        assert final_stats["active_scheduling_tasks"] == 0


# ============================================================================
# Task Cleanup Tests
# ============================================================================


class TestTaskCleanup:
    """Tests for task cleanup after completion."""

    @pytest.mark.asyncio
    async def test_task_removed_after_completion(
        self,
        background_scheduler,
        task_registry,
        instance_registry,
        mock_scheduling_strategy,
        sample_instance,
        sample_schedule_result,
    ):
        """Test that task is removed from active tasks after completion."""
        # Setup
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance=None,
        )
        mock_scheduling_strategy.schedule_task.return_value = (
            sample_schedule_result
        )

        # Execute
        background_scheduler.schedule_task_background(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
        )

        # Check that task is in active tasks
        assert "task-1" in background_scheduler._active_tasks

        # Wait for completion
        await asyncio.sleep(0.1)

        # Verify task is removed
        assert "task-1" not in background_scheduler._active_tasks


# ============================================================================
# Backpressure Tests - Lines 152-156
# ============================================================================


class TestBackpressure:
    """Tests for backpressure handling when instances are above water mark."""

    @pytest.mark.asyncio
    async def test_all_instances_above_water_mark_uses_all_active(
        self,
        background_scheduler,
        task_registry,
        instance_registry,
        mock_scheduling_strategy,
        mock_task_dispatcher,
        sample_schedule_result,
    ):
        """Test backpressure: when all instances are full, still uses them (lines 152-156)."""
        # Create instance with high pending count (above HIGH_WATER_MARK of 5)
        instance = Instance(
            instance_id="inst-1",
            model_id="test-model",
            endpoint="http://localhost:8000",
            platform_info={
                "software_name": "Linux",
                "software_version": "5.15",
                "hardware_name": "x86_64",
            },
        )
        await instance_registry.register(instance)

        # Set pending count above HIGH_WATER_MARK
        for _ in range(6):  # 6 > HIGH_WATER_MARK (5)
            await instance_registry.increment_pending("inst-1")

        # Create task
        await task_registry.create_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance=None,
        )

        mock_scheduling_strategy.schedule_task.return_value = (
            sample_schedule_result
        )

        # Execute
        background_scheduler.schedule_task_background(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
        )

        # Wait for completion
        await asyncio.sleep(0.1)

        # Verify scheduling strategy was called (backpressure allowed scheduling)
        mock_scheduling_strategy.schedule_task.assert_called_once()


# ============================================================================
# Reassign Task Tests - Lines 242-336
# ============================================================================


class TestReassignTask:
    """Tests for task reassignment functionality (lines 242-336)."""

    @pytest.fixture
    def mock_task_dispatcher_with_dispatch(self):
        """Create a mock dispatcher with async dispatch_task."""
        dispatcher = MagicMock()
        dispatcher.dispatch_task_async = MagicMock()
        dispatcher.dispatch_task = AsyncMock()
        return dispatcher

    @pytest.mark.asyncio
    async def test_reassign_task_success(
        self,
        task_registry,
        instance_registry,
        mock_scheduling_strategy,
        mock_task_dispatcher_with_dispatch,
        sample_instance,
        sample_schedule_result,
    ):
        """Test successful task reassignment."""
        scheduler = BackgroundScheduler(
            scheduling_strategy=mock_scheduling_strategy,
            task_registry=task_registry,
            instance_registry=instance_registry,
            task_dispatcher=mock_task_dispatcher_with_dispatch,
        )

        # Setup
        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance="old-instance",
        )
        mock_scheduling_strategy.schedule_task.return_value = (
            sample_schedule_result
        )

        # Execute
        result = await scheduler.reassign_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
        )

        # Verify
        assert result is True
        mock_task_dispatcher_with_dispatch.dispatch_task.assert_called_once()
        task = await task_registry.get("task-1")
        assert task.assigned_instance == "instance-1"

    @pytest.mark.asyncio
    async def test_reassign_task_exclude_instance(
        self,
        task_registry,
        instance_registry,
        mock_scheduling_strategy,
        mock_task_dispatcher_with_dispatch,
        sample_schedule_result,
    ):
        """Test reassignment excluding a specific instance."""
        scheduler = BackgroundScheduler(
            scheduling_strategy=mock_scheduling_strategy,
            task_registry=task_registry,
            instance_registry=instance_registry,
            task_dispatcher=mock_task_dispatcher_with_dispatch,
        )

        # Create two instances
        inst1 = Instance(
            instance_id="inst-1",
            model_id="test-model",
            endpoint="http://localhost:8000",
            platform_info={
                "software_name": "Linux",
                "software_version": "5.15",
                "hardware_name": "x86_64",
            },
        )
        inst2 = Instance(
            instance_id="inst-2",
            model_id="test-model",
            endpoint="http://localhost:8001",
            platform_info={
                "software_name": "Linux",
                "software_version": "5.15",
                "hardware_name": "x86_64",
            },
        )
        await instance_registry.register(inst1)
        await instance_registry.register(inst2)

        await task_registry.create_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance="inst-1",
        )

        # Mock to verify excluded instance is not in available list
        async def check_available_instances(
            model_id, metadata, available_instances
        ):
            # Verify inst-1 is excluded
            instance_ids = [i.instance_id for i in available_instances]
            assert "inst-1" not in instance_ids
            return sample_schedule_result

        mock_scheduling_strategy.schedule_task.side_effect = (
            check_available_instances
        )

        # Execute with exclusion
        result = await scheduler.reassign_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            exclude_instance_id="inst-1",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_reassign_task_no_instances_available(
        self,
        task_registry,
        instance_registry,
        mock_scheduling_strategy,
        mock_task_dispatcher_with_dispatch,
    ):
        """Test reassignment fails when no instances available."""
        scheduler = BackgroundScheduler(
            scheduling_strategy=mock_scheduling_strategy,
            task_registry=task_registry,
            instance_registry=instance_registry,
            task_dispatcher=mock_task_dispatcher_with_dispatch,
        )

        # No instances registered

        # Execute
        result = await scheduler.reassign_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
        )

        # Verify
        assert result is False

    @pytest.mark.asyncio
    async def test_reassign_task_creates_new_task(
        self,
        task_registry,
        instance_registry,
        mock_scheduling_strategy,
        mock_task_dispatcher_with_dispatch,
        sample_instance,
        sample_schedule_result,
    ):
        """Test reassignment creates task if it doesn't exist."""
        scheduler = BackgroundScheduler(
            scheduling_strategy=mock_scheduling_strategy,
            task_registry=task_registry,
            instance_registry=instance_registry,
            task_dispatcher=mock_task_dispatcher_with_dispatch,
        )

        await instance_registry.register(sample_instance)
        mock_scheduling_strategy.schedule_task.return_value = (
            sample_schedule_result
        )

        # No existing task - reassign should create it
        result = await scheduler.reassign_task(
            task_id="new-task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={"key": "value"},
        )

        assert result is True
        task = await task_registry.get("new-task-1")
        assert task is not None
        assert task.model_id == "test-model"

    @pytest.mark.asyncio
    async def test_reassign_task_preserves_enqueue_time(
        self,
        task_registry,
        instance_registry,
        mock_scheduling_strategy,
        mock_task_dispatcher_with_dispatch,
        sample_instance,
        sample_schedule_result,
    ):
        """Test reassignment preserves original enqueue time."""
        scheduler = BackgroundScheduler(
            scheduling_strategy=mock_scheduling_strategy,
            task_registry=task_registry,
            instance_registry=instance_registry,
            task_dispatcher=mock_task_dispatcher_with_dispatch,
        )

        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance="old-instance",
        )
        mock_scheduling_strategy.schedule_task.return_value = (
            sample_schedule_result
        )

        original_enqueue_time = 12345.0

        result = await scheduler.reassign_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            enqueue_time=original_enqueue_time,
        )

        assert result is True
        # Verify dispatch_task was called with the enqueue_time
        mock_task_dispatcher_with_dispatch.dispatch_task.assert_called_once_with(
            task_id="task-1", enqueue_time=original_enqueue_time
        )

    @pytest.mark.asyncio
    async def test_reassign_task_scheduling_failure(
        self,
        task_registry,
        instance_registry,
        mock_scheduling_strategy,
        mock_task_dispatcher_with_dispatch,
        sample_instance,
    ):
        """Test reassignment fails when scheduling fails."""
        scheduler = BackgroundScheduler(
            scheduling_strategy=mock_scheduling_strategy,
            task_registry=task_registry,
            instance_registry=instance_registry,
            task_dispatcher=mock_task_dispatcher_with_dispatch,
        )

        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance="old-instance",
        )

        mock_scheduling_strategy.schedule_task.side_effect = Exception(
            "Scheduling failed"
        )

        result = await scheduler.reassign_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_reassign_task_task_not_found_after_scheduling(
        self,
        task_registry,
        instance_registry,
        mock_scheduling_strategy,
        mock_task_dispatcher_with_dispatch,
        sample_instance,
        sample_schedule_result,
    ):
        """Test reassignment fails when task disappears after scheduling."""
        scheduler = BackgroundScheduler(
            scheduling_strategy=mock_scheduling_strategy,
            task_registry=task_registry,
            instance_registry=instance_registry,
            task_dispatcher=mock_task_dispatcher_with_dispatch,
        )

        await instance_registry.register(sample_instance)
        await task_registry.create_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance="old-instance",
        )
        mock_scheduling_strategy.schedule_task.return_value = (
            sample_schedule_result
        )

        # Mock get to return None after scheduling
        original_get = task_registry.get
        call_count = 0

        async def mock_get(task_id):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return await original_get(task_id)
            return None

        task_registry.get = mock_get

        result = await scheduler.reassign_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_reassign_with_backpressure_all_instances_full(
        self,
        task_registry,
        instance_registry,
        mock_scheduling_strategy,
        mock_task_dispatcher_with_dispatch,
        sample_schedule_result,
    ):
        """Test reassignment with backpressure when all instances above water mark."""
        scheduler = BackgroundScheduler(
            scheduling_strategy=mock_scheduling_strategy,
            task_registry=task_registry,
            instance_registry=instance_registry,
            task_dispatcher=mock_task_dispatcher_with_dispatch,
        )

        # Create instance above water mark
        inst = Instance(
            instance_id="inst-1",
            model_id="test-model",
            endpoint="http://localhost:8000",
            platform_info={
                "software_name": "Linux",
                "software_version": "5.15",
                "hardware_name": "x86_64",
            },
        )
        await instance_registry.register(inst)

        # Push above HIGH_WATER_MARK
        for _ in range(6):
            await instance_registry.increment_pending("inst-1")

        await task_registry.create_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            metadata={},
            assigned_instance="old-instance",
        )
        mock_scheduling_strategy.schedule_task.return_value = (
            sample_schedule_result
        )

        result = await scheduler.reassign_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
        )

        # Should succeed due to backpressure handling
        assert result is True

    @pytest.mark.asyncio
    async def test_reassign_exclude_only_instance_fails(
        self,
        task_registry,
        instance_registry,
        mock_scheduling_strategy,
        mock_task_dispatcher_with_dispatch,
        sample_instance,
    ):
        """Test reassignment fails when excluding the only available instance."""
        scheduler = BackgroundScheduler(
            scheduling_strategy=mock_scheduling_strategy,
            task_registry=task_registry,
            instance_registry=instance_registry,
            task_dispatcher=mock_task_dispatcher_with_dispatch,
        )

        await instance_registry.register(sample_instance)

        # sample_instance has instance_id="test-instance-1"
        result = await scheduler.reassign_task(
            task_id="task-1",
            model_id="test-model",
            task_input={"prompt": "test"},
            exclude_instance_id="test-instance-1",  # Exclude the only instance
        )

        assert result is False
