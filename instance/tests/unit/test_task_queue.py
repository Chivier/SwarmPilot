"""
Unit tests for src/task_queue.py
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.task_queue import TaskQueue, get_task_queue
from src.models import Task, TaskStatus


@pytest.mark.unit
@pytest.mark.asyncio
class TestTaskQueue:
    """Test suite for TaskQueue class"""

    async def test_submit_task(self, sample_task):
        """Test successful task submission"""
        queue = TaskQueue()

        position = await queue.submit_task(sample_task)

        # Check task was added
        assert sample_task.task_id in queue.tasks
        assert sample_task.task_id in queue.queue
        assert position == 1  # First task, position 1

    async def test_submit_duplicate_task(self, sample_task):
        """Test that submitting duplicate task_id raises ValueError"""
        queue = TaskQueue()

        # Submit first time
        await queue.submit_task(sample_task)

        # Submit again with same task_id
        duplicate_task = Task(
            task_id=sample_task.task_id,
            model_id="different-model",
            task_input={"different": "input"}
        )

        with pytest.raises(ValueError) as exc_info:
            await queue.submit_task(duplicate_task)

        assert "already exists" in str(exc_info.value)

    async def test_submit_task_queue_position(self):
        """Test that queue positions are correctly assigned"""
        queue = TaskQueue()

        task1 = Task(task_id="task-1", model_id="model", task_input={})
        task2 = Task(task_id="task-2", model_id="model", task_input={})
        task3 = Task(task_id="task-3", model_id="model", task_input={})

        pos1 = await queue.submit_task(task1)
        pos2 = await queue.submit_task(task2)
        pos3 = await queue.submit_task(task3)

        assert pos1 == 1
        assert pos2 == 2
        assert pos3 == 3

    async def test_submit_task_starts_processing(self, sample_task):
        """Test that submitting a task starts background processing"""
        queue = TaskQueue()

        assert queue.is_processing is False
        assert queue._processing_task is None

        # Mock _process_queue to avoid actual processing
        with patch.object(queue, '_process_queue', new=AsyncMock()):
            await queue.submit_task(sample_task)

            # Give asyncio time to create the task
            await asyncio.sleep(0.01)

            # Processing task should be created
            assert queue._processing_task is not None

    async def test_get_task_exists(self, sample_task):
        """Test retrieving an existing task"""
        queue = TaskQueue()
        await queue.submit_task(sample_task)

        retrieved_task = await queue.get_task(sample_task.task_id)

        assert retrieved_task is not None
        assert retrieved_task.task_id == sample_task.task_id
        assert retrieved_task is sample_task  # Same object

    async def test_get_task_not_exists(self):
        """Test retrieving a non-existent task returns None"""
        queue = TaskQueue()

        task = await queue.get_task("non-existent-task")

        assert task is None

    async def test_list_tasks_no_filter(self):
        """Test listing all tasks without filter"""
        queue = TaskQueue()

        task1 = Task(task_id="task-1", model_id="model", task_input={})
        task2 = Task(task_id="task-2", model_id="model", task_input={})

        await queue.submit_task(task1)
        await queue.submit_task(task2)

        tasks = await queue.list_tasks()

        assert len(tasks) == 2
        assert task1 in tasks
        assert task2 in tasks

    async def test_list_tasks_with_status_filter(self):
        """Test listing tasks with status filter"""
        queue = TaskQueue()

        task1 = Task(task_id="task-1", model_id="model", task_input={})
        task2 = Task(task_id="task-2", model_id="model", task_input={})
        task3 = Task(task_id="task-3", model_id="model", task_input={})

        await queue.submit_task(task1)
        await queue.submit_task(task2)
        await queue.submit_task(task3)

        # Mark some tasks with different statuses
        task1.mark_started()
        task2.mark_completed({"result": "success"})

        # Filter by QUEUED
        queued_tasks = await queue.list_tasks(status_filter=TaskStatus.QUEUED)
        assert len(queued_tasks) == 1
        assert task3 in queued_tasks

        # Filter by RUNNING
        running_tasks = await queue.list_tasks(status_filter=TaskStatus.RUNNING)
        assert len(running_tasks) == 1
        assert task1 in running_tasks

        # Filter by COMPLETED
        completed_tasks = await queue.list_tasks(status_filter=TaskStatus.COMPLETED)
        assert len(completed_tasks) == 1
        assert task2 in completed_tasks

    async def test_list_tasks_with_limit(self):
        """Test listing tasks with limit"""
        queue = TaskQueue()

        for i in range(5):
            task = Task(task_id=f"task-{i}", model_id="model", task_input={})
            await queue.submit_task(task)

        # Get only 3 tasks
        tasks = await queue.list_tasks(limit=3)

        assert len(tasks) == 3

    async def test_list_tasks_sorting(self):
        """Test that tasks are sorted by submission time (most recent first)"""
        queue = TaskQueue()

        task1 = Task(task_id="task-1", model_id="model", task_input={})
        task2 = Task(task_id="task-2", model_id="model", task_input={})
        task3 = Task(task_id="task-3", model_id="model", task_input={})

        await queue.submit_task(task1)
        await asyncio.sleep(0.01)  # Ensure different timestamps
        await queue.submit_task(task2)
        await asyncio.sleep(0.01)
        await queue.submit_task(task3)

        tasks = await queue.list_tasks()

        # Most recent first
        assert tasks[0].task_id == "task-3"
        assert tasks[1].task_id == "task-2"
        assert tasks[2].task_id == "task-1"

    async def test_delete_task_queued(self):
        """Test deleting a queued task"""
        queue = TaskQueue()

        task = Task(task_id="task-1", model_id="model", task_input={})
        await queue.submit_task(task)

        # Delete the task
        result = await queue.delete_task(task.task_id)

        assert result is True
        assert task.task_id not in queue.tasks
        assert task.task_id not in queue.queue

    async def test_delete_task_completed(self):
        """Test deleting a completed task"""
        queue = TaskQueue()

        task = Task(task_id="task-1", model_id="model", task_input={})
        await queue.submit_task(task)
        task.mark_completed({"result": "success"})

        # Delete the task
        result = await queue.delete_task(task.task_id)

        assert result is True
        assert task.task_id not in queue.tasks

    async def test_delete_task_failed(self):
        """Test deleting a failed task"""
        queue = TaskQueue()

        task = Task(task_id="task-1", model_id="model", task_input={})
        await queue.submit_task(task)
        task.mark_failed("error message")

        # Delete the task
        result = await queue.delete_task(task.task_id)

        assert result is True
        assert task.task_id not in queue.tasks

    async def test_delete_task_running(self):
        """Test that deleting a running task raises ValueError"""
        queue = TaskQueue()

        task = Task(task_id="task-1", model_id="model", task_input={})
        await queue.submit_task(task)
        task.mark_started()  # Mark as running

        with pytest.raises(ValueError) as exc_info:
            await queue.delete_task(task.task_id)

        assert "Cannot delete a running task" in str(exc_info.value)

    async def test_delete_task_not_found(self):
        """Test deleting a non-existent task returns False"""
        queue = TaskQueue()

        result = await queue.delete_task("non-existent-task")

        assert result is False

    async def test_get_queue_stats(self):
        """Test getting queue statistics"""
        queue = TaskQueue()

        # Create tasks with different statuses
        task1 = Task(task_id="task-1", model_id="model", task_input={})
        task2 = Task(task_id="task-2", model_id="model", task_input={})
        task3 = Task(task_id="task-3", model_id="model", task_input={})
        task4 = Task(task_id="task-4", model_id="model", task_input={})

        await queue.submit_task(task1)
        await queue.submit_task(task2)
        await queue.submit_task(task3)
        await queue.submit_task(task4)

        # Set different statuses
        task1.status = TaskStatus.QUEUED
        task2.mark_started()
        task3.mark_completed({"result": "success"})
        task4.mark_failed("error")

        stats = await queue.get_queue_stats()

        assert stats["total"] == 4
        assert stats["queued"] == 1
        assert stats["running"] == 1
        assert stats["completed"] == 1
        assert stats["failed"] == 1

    async def test_execute_task_success(self, sample_task):
        """Test successful task execution"""
        queue = TaskQueue()

        # Mock docker manager
        mock_docker_manager = AsyncMock()
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = Mock(model_id=sample_task.model_id)
        mock_docker_manager.invoke_inference.return_value = {"output": "result"}

        with patch("src.task_queue.get_docker_manager", return_value=mock_docker_manager):
            await queue._execute_task(sample_task)

        # Check task was marked as completed
        assert sample_task.status == TaskStatus.COMPLETED
        assert sample_task.result == {"output": "result"}
        assert sample_task.error is None

    async def test_execute_task_no_model_running(self, sample_task):
        """Test task execution fails when no model is running"""
        queue = TaskQueue()

        # Mock docker manager - no model running
        mock_docker_manager = AsyncMock()
        mock_docker_manager.is_model_running.return_value = False

        with patch("src.task_queue.get_docker_manager", return_value=mock_docker_manager):
            await queue._execute_task(sample_task)

        # Check task was marked as failed
        assert sample_task.status == TaskStatus.FAILED
        assert "No model is currently running" in sample_task.error

    async def test_execute_task_model_mismatch(self, sample_task):
        """Test task execution fails when model doesn't match"""
        queue = TaskQueue()

        # Mock docker manager - different model running
        mock_docker_manager = AsyncMock()
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = Mock(model_id="different-model")

        with patch("src.task_queue.get_docker_manager", return_value=mock_docker_manager):
            await queue._execute_task(sample_task)

        # Check task was marked as failed
        assert sample_task.status == TaskStatus.FAILED
        assert "Model mismatch" in sample_task.error

    async def test_execute_task_inference_error(self, sample_task):
        """Test task execution handles inference errors"""
        queue = TaskQueue()

        # Mock docker manager - inference fails
        mock_docker_manager = AsyncMock()
        mock_docker_manager.is_model_running.return_value = True
        mock_docker_manager.get_current_model.return_value = Mock(model_id=sample_task.model_id)
        mock_docker_manager.invoke_inference.side_effect = Exception("Inference failed")

        with patch("src.task_queue.get_docker_manager", return_value=mock_docker_manager):
            await queue._execute_task(sample_task)

        # Check task was marked as failed
        assert sample_task.status == TaskStatus.FAILED
        assert "Inference failed" in sample_task.error

    async def test_process_queue_fifo_order(self):
        """Test that tasks are processed in FIFO order"""
        queue = TaskQueue()
        processed_order = []

        # Mock _execute_task to track order
        async def mock_execute(task):
            processed_order.append(task.task_id)
            task.mark_completed({"output": "done"})

        queue._execute_task = mock_execute

        # Submit multiple tasks
        task1 = Task(task_id="task-1", model_id="model", task_input={})
        task2 = Task(task_id="task-2", model_id="model", task_input={})
        task3 = Task(task_id="task-3", model_id="model", task_input={})

        await queue.submit_task(task1)
        await queue.submit_task(task2)
        await queue.submit_task(task3)

        # Wait for processing to complete
        await asyncio.sleep(0.1)

        # Verify FIFO order
        assert processed_order == ["task-1", "task-2", "task-3"]

    async def test_stop_processing(self):
        """Test graceful shutdown of queue processing"""
        queue = TaskQueue()

        # Start processing with a long-running task
        task = Task(task_id="task-1", model_id="model", task_input={})

        async def slow_execute(task):
            await asyncio.sleep(10)  # Simulate long task
            task.mark_completed({"output": "done"})

        queue._execute_task = slow_execute
        await queue.submit_task(task)

        # Give it time to start
        await asyncio.sleep(0.01)

        # Stop processing
        await queue.stop_processing()

        # Processing task should be cancelled
        assert queue._processing_task.done()


@pytest.mark.unit
class TestGetTaskQueue:
    """Test suite for get_task_queue function"""

    async def test_get_task_queue_singleton(self):
        """Test that get_task_queue returns a singleton instance"""
        # Reset the global queue
        import src.task_queue
        src.task_queue._task_queue = None

        # First call creates the instance
        queue1 = get_task_queue()
        assert queue1 is not None

        # Second call returns the same instance
        queue2 = get_task_queue()
        assert queue2 is queue1

    async def test_get_task_queue_creates_instance(self):
        """Test that get_task_queue creates queue on first call"""
        # Reset the global queue
        import src.task_queue
        src.task_queue._task_queue = None

        queue = get_task_queue()

        assert queue is not None
        assert isinstance(queue, TaskQueue)

    async def test_get_task_queue_preserves_state(self):
        """Test that get_task_queue preserves queue state across calls"""
        # Reset the global queue
        import src.task_queue
        src.task_queue._task_queue = None

        # Get queue and add a task
        queue1 = get_task_queue()
        task = Task(task_id="task-1", model_id="model", task_input={})
        await queue1.submit_task(task)

        # Get queue again
        queue2 = get_task_queue()

        # Should have the task we added
        assert "task-1" in queue2.tasks
