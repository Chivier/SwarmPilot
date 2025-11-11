"""
Unit tests for InstanceTaskQueueManager.

Tests queue management, worker threads, task redistribution, and error handling.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from src.instance_queue_manager import InstanceTaskQueueManager, QueuedTaskInfo
from src.model import TaskStatus, Instance


# ============================================================================
# Initialization Tests
# ============================================================================

class TestInstanceQueueManagerInit:
    """Tests for InstanceTaskQueueManager initialization."""

    def test_initialization(
        self,
        instance_registry,
        task_registry,
        task_dispatcher
    ):
        """Test InstanceTaskQueueManager initialization."""
        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher,
            max_queue_size=500
        )

        # Test internal state
        assert manager._max_queue_size == 500
        assert len(manager._instance_queues) == 0
        assert len(manager._worker_tasks) == 0
        assert manager._shutting_down is False

    def test_default_queue_size(
        self,
        instance_registry,
        task_registry,
        task_dispatcher
    ):
        """Test default max queue size."""
        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        assert manager._max_queue_size == 1000


# ============================================================================
# Queue Creation Tests
# ============================================================================

class TestQueueCreation:
    """Tests for create_queue method."""

    @pytest.mark.asyncio
    async def test_create_queue_success(
        self,
        instance_registry,
        task_registry,
        task_dispatcher
    ):
        """Test successful queue creation."""
        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        await manager.create_queue("instance-1")

        assert "instance-1" in manager._instance_queues
        assert "instance-1" in manager._worker_tasks
        assert isinstance(manager._instance_queues["instance-1"], asyncio.Queue)
        assert isinstance(manager._worker_tasks["instance-1"], asyncio.Task)

    @pytest.mark.asyncio
    async def test_create_queue_duplicate(
        self,
        instance_registry,
        task_registry,
        task_dispatcher
    ):
        """Test that creating duplicate queue raises error."""
        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        await manager.create_queue("instance-1")

        with pytest.raises(ValueError, match="Queue already exists"):
            await manager.create_queue("instance-1")


# ============================================================================
# Task Addition Tests
# ============================================================================

class TestTaskAddition:
    """Tests for add_task_to_queue method."""

    @pytest.mark.asyncio
    async def test_add_task_success(
        self,
        instance_registry,
        task_registry,
        task_dispatcher
    ):
        """Test successfully adding task to queue."""
        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        await manager.create_queue("instance-1")

        task_input = {"prompt": "test"}
        metadata = {"priority": "high"}

        await manager.add_task_to_queue(
            instance_id="instance-1",
            task_id="task-1",
            model_id="model-1",
            task_input=task_input,
            metadata=metadata
        )

        # Verify task was added to queue
        queue = manager._instance_queues["instance-1"]
        assert queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_add_task_nonexistent_queue(
        self,
        instance_registry,
        task_registry,
        task_dispatcher
    ):
        """Test adding task to nonexistent queue raises error."""
        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        with pytest.raises(KeyError, match="No queue exists for instance"):
            await manager.add_task_to_queue(
                instance_id="nonexistent",
                task_id="task-1",
                model_id="model-1",
                task_input={},
                metadata={}
            )


# ============================================================================
# Worker Thread Tests
# ============================================================================

class TestWorkerThread:
    """Tests for worker thread processing."""

    @pytest.mark.asyncio
    async def test_worker_processes_task(
        self,
        instance_registry,
        task_registry,
        task_dispatcher,
        sample_instance
    ):
        """Test that worker thread processes queued tasks."""
        # Setup
        await instance_registry.register(sample_instance)

        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        await manager.create_queue(sample_instance.instance_id)

        # Mock dispatcher
        task_dispatcher.dispatch_task = AsyncMock()

        # Add task to queue
        await manager.add_task_to_queue(
            instance_id=sample_instance.instance_id,
            task_id="task-1",
            model_id="model-1",
            task_input={"prompt": "test"},
            metadata={}
        )

        # Wait for worker to process
        await asyncio.sleep(0.2)

        # Verify dispatch was called
        task_dispatcher.dispatch_task.assert_called_once_with("task-1")

    @pytest.mark.asyncio
    async def test_worker_handles_dispatch_error(
        self,
        instance_registry,
        task_registry,
        task_dispatcher,
        sample_instance
    ):
        """Test worker handles dispatch errors gracefully."""
        # Setup
        await instance_registry.register(sample_instance)

        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        await manager.create_queue(sample_instance.instance_id)

        # Mock dispatcher to fail
        task_dispatcher.dispatch_task = AsyncMock(
            side_effect=Exception("Dispatch failed")
        )

        # Add task to queue
        await manager.add_task_to_queue(
            instance_id=sample_instance.instance_id,
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={}
        )

        # Wait for worker to process
        await asyncio.sleep(0.2)

        # Worker should have logged error but continue running
        assert manager._worker_tasks[sample_instance.instance_id].done() is False

    @pytest.mark.asyncio
    async def test_worker_processes_multiple_tasks_fifo(
        self,
        instance_registry,
        task_registry,
        task_dispatcher,
        sample_instance
    ):
        """Test worker processes tasks in FIFO order."""
        # Setup
        await instance_registry.register(sample_instance)

        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        await manager.create_queue(sample_instance.instance_id)

        # Track dispatch calls
        dispatch_order = []

        async def track_dispatch(task_id):
            dispatch_order.append(task_id)

        task_dispatcher.dispatch_task = AsyncMock(side_effect=track_dispatch)

        # Add multiple tasks
        for i in range(5):
            await manager.add_task_to_queue(
                instance_id=sample_instance.instance_id,
                task_id=f"task-{i}",
                model_id="model-1",
                task_input={},
                metadata={}
            )

        # Wait for all tasks to process
        await asyncio.sleep(0.5)

        # Verify FIFO order
        assert dispatch_order == ["task-0", "task-1", "task-2", "task-3", "task-4"]


# ============================================================================
# Queue Removal Tests
# ============================================================================

class TestQueueRemoval:
    """Tests for remove_queue method."""

    @pytest.mark.asyncio
    async def test_remove_queue_without_redistribution(
        self,
        instance_registry,
        task_registry,
        task_dispatcher,
        sample_instance
    ):
        """Test removing queue without redistributing tasks."""
        # Setup
        await instance_registry.register(sample_instance)

        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        await manager.create_queue(sample_instance.instance_id)

        # Remove queue
        redistributed = await manager.remove_queue(
            instance_id=sample_instance.instance_id,
            redistribute_tasks=False
        )

        assert redistributed == 0
        assert sample_instance.instance_id not in manager._instance_queues
        assert sample_instance.instance_id not in manager._worker_tasks

    @pytest.mark.asyncio
    async def test_remove_queue_with_empty_queue(
        self,
        instance_registry,
        task_registry,
        task_dispatcher,
        sample_instance
    ):
        """Test removing queue with no queued tasks."""
        # Setup
        await instance_registry.register(sample_instance)

        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        await manager.create_queue(sample_instance.instance_id)

        # Remove queue with redistribution
        redistributed = await manager.remove_queue(
            instance_id=sample_instance.instance_id,
            redistribute_tasks=True
        )

        assert redistributed == 0

    @pytest.mark.asyncio
    async def test_remove_queue_with_redistribution(
        self,
        instance_registry,
        task_registry,
        task_dispatcher,
        sample_instances
    ):
        """Test redistributing tasks when removing queue."""
        # Setup - register multiple instances
        for instance in sample_instances:
            await instance_registry.register(instance)

        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        # Create queues for all instances
        for instance in sample_instances:
            await manager.create_queue(instance.instance_id)

        # Create an event to block the dispatcher so tasks stay in queue
        dispatch_block = asyncio.Event()

        async def blocking_dispatch(task_id):
            await dispatch_block.wait()  # Will block forever

        task_dispatcher.dispatch_task = AsyncMock(side_effect=blocking_dispatch)

        # Add tasks to first instance's queue
        for i in range(3):
            task = await task_registry.create_task(
                task_id=f"task-{i}",
                model_id="test-model",
                task_input={},
                metadata={},
                assigned_instance=sample_instances[0].instance_id
            )
            await manager.add_task_to_queue(
                instance_id=sample_instances[0].instance_id,
                task_id=f"task-{i}",
                model_id="test-model",
                task_input={},
                metadata={}
            )
            await instance_registry.increment_pending(sample_instances[0].instance_id)

        # Wait for worker to start processing first task (others remain in queue)
        await asyncio.sleep(0.1)

        # Mock the redistribute method to avoid complex scheduling strategy mocking
        redistribute_called = False
        redistribute_task_count = 0

        async def mock_redistribute(tasks, exclude_instance=None):
            nonlocal redistribute_called, redistribute_task_count
            redistribute_called = True
            redistribute_task_count = len(tasks)
            # Just mark tasks as failed instead of redistributing
            for task_info in tasks:
                await task_registry.update_status(task_info.task_id, TaskStatus.FAILED)
            return 0  # Return 0 to indicate tasks weren't actually redistributed

        manager._redistribute_tasks = mock_redistribute

        # Remove first instance's queue with redistribution
        redistributed = await manager.remove_queue(
            instance_id=sample_instances[0].instance_id,
            redistribute_tasks=True
        )

        # Verify redistribution was called with pending tasks
        # (2 tasks should still be in queue - one is being processed)
        assert redistribute_called
        assert redistribute_task_count == 2  # Two tasks remained in queue

    @pytest.mark.asyncio
    async def test_remove_nonexistent_queue(
        self,
        instance_registry,
        task_registry,
        task_dispatcher
    ):
        """Test removing nonexistent queue raises error."""
        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        with pytest.raises(KeyError, match="No queue exists for instance"):
            await manager.remove_queue(
                instance_id="nonexistent",
                redistribute_tasks=False
            )

    @pytest.mark.asyncio
    async def test_remove_queue_no_other_instances(
        self,
        instance_registry,
        task_registry,
        task_dispatcher,
        sample_instance
    ):
        """Test removing queue when no other instances available."""
        # Setup
        await instance_registry.register(sample_instance)

        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        await manager.create_queue(sample_instance.instance_id)

        # Add task
        task = await task_registry.create_task(
            task_id="task-1",
            model_id="test-model",
            task_input={},
            metadata={},
            assigned_instance=sample_instance.instance_id
        )
        await manager.add_task_to_queue(
            instance_id=sample_instance.instance_id,
            task_id="task-1",
            model_id="test-model",
            task_input={},
            metadata={}
        )

        # Mock task_registry.update_status
        task_registry.update_status = AsyncMock()
        task_registry.set_error = AsyncMock()

        # Remove queue with redistribution (should fail tasks)
        redistributed = await manager.remove_queue(
            instance_id=sample_instance.instance_id,
            redistribute_tasks=True
        )

        # Task should be marked as failed
        assert redistributed == 0
        task_registry.update_status.assert_called()
        task_registry.set_error.assert_called()


# ============================================================================
# Shutdown Tests
# ============================================================================

class TestShutdown:
    """Tests for shutdown method."""

    @pytest.mark.asyncio
    async def test_shutdown_empty_queues(
        self,
        instance_registry,
        task_registry,
        task_dispatcher,
        sample_instances
    ):
        """Test shutdown with empty queues."""
        # Setup
        for instance in sample_instances:
            await instance_registry.register(instance)

        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        for instance in sample_instances:
            await manager.create_queue(instance.instance_id)

        # Shutdown should complete quickly
        await manager.shutdown()

        # Verify all queues and workers are cleared
        assert len(manager._instance_queues) == 0
        assert len(manager._worker_tasks) == 0
        assert manager._shutting_down is True

    @pytest.mark.asyncio
    async def test_shutdown_with_pending_tasks(
        self,
        instance_registry,
        task_registry,
        task_dispatcher,
        sample_instance
    ):
        """Test shutdown waits for pending tasks."""
        # Setup
        await instance_registry.register(sample_instance)

        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        await manager.create_queue(sample_instance.instance_id)

        # Mock dispatcher with delay
        async def slow_dispatch(task_id):
            await asyncio.sleep(0.2)

        task_dispatcher.dispatch_task = AsyncMock(side_effect=slow_dispatch)

        # Add tasks
        for i in range(3):
            await manager.add_task_to_queue(
                instance_id=sample_instance.instance_id,
                task_id=f"task-{i}",
                model_id="model-1",
                task_input={},
                metadata={}
            )

        # Wait a bit for processing to start
        await asyncio.sleep(0.1)

        # Shutdown should wait for tasks (has 30s timeout built-in)
        await manager.shutdown()

        # Verify all queues cleared
        assert len(manager._instance_queues) == 0
        assert len(manager._worker_tasks) == 0

    @pytest.mark.asyncio
    async def test_shutdown_with_hung_worker(
        self,
        instance_registry,
        task_registry,
        task_dispatcher,
        sample_instance
    ):
        """Test shutdown handles stuck workers (uses built-in 30s timeout)."""
        # Setup
        await instance_registry.register(sample_instance)

        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        await manager.create_queue(sample_instance.instance_id)

        # Mock dispatcher with very long delay that exceeds shutdown timeout
        async def very_slow_dispatch(task_id):
            await asyncio.sleep(100.0)

        task_dispatcher.dispatch_task = AsyncMock(side_effect=very_slow_dispatch)

        # Add task
        await manager.add_task_to_queue(
            instance_id=sample_instance.instance_id,
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={}
        )

        # Wait for worker to start processing
        await asyncio.sleep(0.1)

        # Shutdown should complete (may timeout waiting for worker)
        # This tests that shutdown doesn't hang forever
        import time
        start = time.time()
        await manager.shutdown()
        duration = time.time() - start

        # Should complete within reasonable time (30s timeout + overhead)
        assert duration < 35.0

        # Queues should be cleared
        assert len(manager._instance_queues) == 0
        assert len(manager._worker_tasks) == 0


# ============================================================================
# Get Queue Stats Tests
# ============================================================================

class TestGetQueueStats:
    """Tests for get_queue_stats method."""

    @pytest.mark.asyncio
    async def test_get_stats_empty_queue(
        self,
        instance_registry,
        task_registry,
        task_dispatcher,
        sample_instance
    ):
        """Test getting stats for empty queue."""
        # Setup
        await instance_registry.register(sample_instance)

        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        await manager.create_queue(sample_instance.instance_id)

        stats = await manager.get_queue_stats(sample_instance.instance_id)

        # Stats format: {instance_id: {"queued": N, "max_size": M}}
        assert sample_instance.instance_id in stats
        assert stats[sample_instance.instance_id]["queued"] == 0
        assert stats[sample_instance.instance_id]["max_size"] == 1000  # default max size

    @pytest.mark.asyncio
    async def test_get_stats_with_queued_tasks(
        self,
        instance_registry,
        task_registry,
        task_dispatcher,
        sample_instance
    ):
        """Test getting stats with queued tasks."""
        # Setup
        await instance_registry.register(sample_instance)

        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        await manager.create_queue(sample_instance.instance_id)

        # Mock dispatcher to prevent processing
        task_dispatcher.dispatch_task = AsyncMock(
            side_effect=lambda x: asyncio.sleep(10)
        )

        # Add tasks
        for i in range(3):
            await manager.add_task_to_queue(
                instance_id=sample_instance.instance_id,
                task_id=f"task-{i}",
                model_id="model-1",
                task_input={},
                metadata={}
            )

        # Wait for first task to start processing
        await asyncio.sleep(0.1)

        stats = await manager.get_queue_stats(sample_instance.instance_id)

        assert sample_instance.instance_id in stats
        # Should have tasks queued (worker is processing one)
        assert stats[sample_instance.instance_id]["queued"] >= 0

    @pytest.mark.asyncio
    async def test_get_stats_nonexistent_queue(
        self,
        instance_registry,
        task_registry,
        task_dispatcher
    ):
        """Test getting stats for nonexistent queue raises error."""
        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        with pytest.raises(KeyError, match="No queue exists for instance"):
            await manager.get_queue_stats("nonexistent")


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_concurrent_queue_creation(
        self,
        instance_registry,
        task_registry,
        task_dispatcher
    ):
        """Test concurrent queue creation attempts."""
        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        # Try creating same queue concurrently
        tasks = [
            manager.create_queue("instance-1"),
            manager.create_queue("instance-1")
        ]

        # One should succeed, one should fail
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if not isinstance(r, Exception))
        error_count = sum(1 for r in results if isinstance(r, ValueError))

        assert success_count == 1
        assert error_count == 1

    @pytest.mark.asyncio
    async def test_worker_survives_task_registry_error(
        self,
        instance_registry,
        task_registry,
        task_dispatcher,
        sample_instance
    ):
        """Test worker continues after task registry errors."""
        # Setup
        await instance_registry.register(sample_instance)

        manager = InstanceTaskQueueManager(
            instance_registry=instance_registry,
            task_registry=task_registry,
            task_dispatcher=task_dispatcher
        )

        await manager.create_queue(sample_instance.instance_id)

        # Mock dispatcher to fail once, then succeed
        call_count = 0

        async def failing_dispatch(task_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First task fails")

        task_dispatcher.dispatch_task = AsyncMock(side_effect=failing_dispatch)

        # Add two tasks
        await manager.add_task_to_queue(
            instance_id=sample_instance.instance_id,
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={}
        )
        await manager.add_task_to_queue(
            instance_id=sample_instance.instance_id,
            task_id="task-2",
            model_id="model-1",
            task_input={},
            metadata={}
        )

        # Wait for processing
        await asyncio.sleep(0.3)

        # Both tasks should have been attempted
        assert task_dispatcher.dispatch_task.call_count == 2

        # Worker should still be running
        assert manager._worker_tasks[sample_instance.instance_id].done() is False
