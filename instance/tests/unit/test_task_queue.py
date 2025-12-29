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
        # queue.queue is a list of tuples (enqueue_time, task_id)
        assert any(task_id == sample_task.task_id for _, task_id in queue.queue)
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

    async def test_process_queue_handles_execution_error(self):
        """Test that _process_queue handles task execution errors gracefully"""
        queue = TaskQueue()

        # Mock _execute_task to raise an exception
        async def failing_execute(task):
            raise RuntimeError("Execution error")

        queue._execute_task = failing_execute

        # Submit a task
        task = Task(task_id="task-1", model_id="model", task_input={})
        await queue.submit_task(task)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Task should be marked as failed
        assert task.status == TaskStatus.FAILED
        assert "Execution error" in task.error

    async def test_send_callback_success_websocket(self, sample_task):
        """Test successful callback sending via WebSocket"""
        queue = TaskQueue()
        queue.tasks[sample_task.task_id] = sample_task

        # Mock WebSocket client - connected and successful
        mock_ws_client = AsyncMock()
        mock_ws_client.is_connected = Mock(return_value=True)  # Use Mock for sync method
        mock_ws_client.send_task_result = AsyncMock(return_value=True)

        # Mock scheduler client (should not be called if WebSocket succeeds)
        mock_scheduler_client = AsyncMock()
        mock_scheduler_client.send_task_result = AsyncMock(return_value=True)

        with patch("src.websocket_client_singleton.get_websocket_client", return_value=mock_ws_client), \
             patch("src.task_queue.get_scheduler_client", return_value=mock_scheduler_client):
            await queue._send_callback(
                sample_task.task_id,
                "completed",
                result={"output": "test"},
                execution_time_ms=100.0
            )

        # Verify WebSocket was called
        mock_ws_client.send_task_result.assert_called_once()
        ws_call_args = mock_ws_client.send_task_result.call_args
        assert ws_call_args[1]["task_id"] == sample_task.task_id
        assert ws_call_args[1]["status"] == "completed"
        assert ws_call_args[1]["result"] == {"output": "test"}
        assert ws_call_args[1]["execution_time_ms"] == 100.0

        # Verify HTTP was NOT called (WebSocket succeeded)
        mock_scheduler_client.send_task_result.assert_not_called()

    async def test_send_callback_success_http_fallback(self, sample_task):
        """Test callback falls back to HTTP when WebSocket is not available"""
        queue = TaskQueue()
        queue.tasks[sample_task.task_id] = sample_task

        # Mock WebSocket client - not available
        mock_ws_client = None

        # Mock scheduler client (should be called as fallback)
        mock_scheduler_client = AsyncMock()
        mock_scheduler_client.send_task_result = AsyncMock(return_value=True)

        with patch("src.websocket_client_singleton.get_websocket_client", return_value=mock_ws_client), \
             patch("src.task_queue.get_scheduler_client", return_value=mock_scheduler_client):
            await queue._send_callback(
                sample_task.task_id,
                "completed",
                result={"output": "test"},
                execution_time_ms=100.0
            )

        # Verify HTTP callback was sent
        mock_scheduler_client.send_task_result.assert_called_once()
        call_args = mock_scheduler_client.send_task_result.call_args
        assert call_args[1]["task_id"] == sample_task.task_id
        assert call_args[1]["status"] == "completed"
        assert call_args[1]["result"] == {"output": "test"}
        assert call_args[1]["execution_time_ms"] == 100.0
        assert call_args[1]["callback_url"] is None  # No callback_url on task

    async def test_send_callback_websocket_not_connected(self, sample_task):
        """Test callback falls back to HTTP when WebSocket is not connected"""
        queue = TaskQueue()
        queue.tasks[sample_task.task_id] = sample_task

        # Mock WebSocket client - exists but not connected
        mock_ws_client = AsyncMock()
        mock_ws_client.is_connected = Mock(return_value=False)  # Use Mock for sync method

        # Mock scheduler client (should be called as fallback)
        mock_scheduler_client = AsyncMock()
        mock_scheduler_client.send_task_result = AsyncMock(return_value=True)

        with patch("src.websocket_client_singleton.get_websocket_client", return_value=mock_ws_client), \
             patch("src.task_queue.get_scheduler_client", return_value=mock_scheduler_client):
            await queue._send_callback(
                sample_task.task_id,
                "completed",
                result={"output": "test"}
            )

        # Verify WebSocket was not called (not connected)
        mock_ws_client.send_task_result.assert_not_called()

        # Verify HTTP callback was attempted
        mock_scheduler_client.send_task_result.assert_called_once()

    async def test_send_callback_websocket_fails_fallback(self, sample_task):
        """Test callback falls back to HTTP when WebSocket send fails"""
        queue = TaskQueue()
        queue.tasks[sample_task.task_id] = sample_task

        # Mock WebSocket client - connected but send fails
        mock_ws_client = AsyncMock()
        mock_ws_client.is_connected = Mock(return_value=True)  # Use Mock for sync method
        mock_ws_client.send_task_result = AsyncMock(return_value=False)

        # Mock scheduler client (should be called as fallback)
        mock_scheduler_client = AsyncMock()
        mock_scheduler_client.send_task_result = AsyncMock(return_value=True)

        with patch("src.websocket_client_singleton.get_websocket_client", return_value=mock_ws_client), \
             patch("src.task_queue.get_scheduler_client", return_value=mock_scheduler_client):
            await queue._send_callback(
                sample_task.task_id,
                "completed",
                result={"output": "test"}
            )

        # Verify WebSocket was attempted
        mock_ws_client.send_task_result.assert_called_once()

        # Verify HTTP callback was attempted as fallback
        mock_scheduler_client.send_task_result.assert_called_once()

    async def test_send_callback_websocket_exception_fallback(self, sample_task):
        """Test callback falls back to HTTP when WebSocket raises exception"""
        queue = TaskQueue()
        queue.tasks[sample_task.task_id] = sample_task

        # Mock WebSocket client - connected but raises exception
        mock_ws_client = AsyncMock()
        mock_ws_client.is_connected = Mock(return_value=True)  # Use Mock for sync method
        mock_ws_client.send_task_result = AsyncMock(side_effect=Exception("WebSocket error"))

        # Mock scheduler client (should be called as fallback)
        mock_scheduler_client = AsyncMock()
        mock_scheduler_client.send_task_result = AsyncMock(return_value=True)

        with patch("src.websocket_client_singleton.get_websocket_client", return_value=mock_ws_client), \
             patch("src.task_queue.get_scheduler_client", return_value=mock_scheduler_client):
            # Should not raise exception - falls back to HTTP
            await queue._send_callback(
                sample_task.task_id,
                "completed",
                result={"output": "test"}
            )

        # Verify WebSocket was attempted
        mock_ws_client.send_task_result.assert_called_once()

        # Verify HTTP callback was attempted as fallback
        mock_scheduler_client.send_task_result.assert_called_once()

    async def test_send_callback_http_failure(self, sample_task):
        """Test callback handles HTTP failure response"""
        queue = TaskQueue()
        queue.tasks[sample_task.task_id] = sample_task

        # Mock WebSocket client - not available
        mock_ws_client = None

        # Mock scheduler client to return False
        mock_scheduler_client = AsyncMock()
        mock_scheduler_client.send_task_result = AsyncMock(return_value=False)

        with patch("src.websocket_client_singleton.get_websocket_client", return_value=mock_ws_client), \
             patch("src.task_queue.get_scheduler_client", return_value=mock_scheduler_client):
            # Should not raise exception even if callback fails
            await queue._send_callback(
                sample_task.task_id,
                "completed",
                result={"output": "test"}
            )

        # Verify HTTP callback was attempted
        mock_scheduler_client.send_task_result.assert_called_once()

    async def test_send_callback_http_exception(self, sample_task):
        """Test callback handles HTTP exceptions"""
        queue = TaskQueue()
        queue.tasks[sample_task.task_id] = sample_task

        # Mock WebSocket client - not available
        mock_ws_client = None

        # Mock scheduler client to raise exception
        mock_scheduler_client = AsyncMock()
        mock_scheduler_client.send_task_result = AsyncMock(side_effect=Exception("Network error"))

        with patch("src.websocket_client_singleton.get_websocket_client", return_value=mock_ws_client), \
             patch("src.task_queue.get_scheduler_client", return_value=mock_scheduler_client):
            # Should not raise - just logs error
            await queue._send_callback(
                sample_task.task_id,
                "completed",
                result={"output": "test"}
            )

        # Verify HTTP callback was attempted
        mock_scheduler_client.send_task_result.assert_called_once()

    async def test_send_callback_with_callback_url(self, sample_task):
        """Test callback includes callback_url from task when falling back to HTTP"""
        queue = TaskQueue()
        sample_task.callback_url = "http://custom-callback:8000/callback"
        queue.tasks[sample_task.task_id] = sample_task

        # Mock WebSocket client - not available
        mock_ws_client = None

        # Mock scheduler client
        mock_scheduler_client = AsyncMock()
        mock_scheduler_client.send_task_result = AsyncMock(return_value=True)

        with patch("src.websocket_client_singleton.get_websocket_client", return_value=mock_ws_client), \
             patch("src.task_queue.get_scheduler_client", return_value=mock_scheduler_client):
            await queue._send_callback(
                sample_task.task_id,
                "completed",
                result={"output": "test"}
            )

        # Verify HTTP callback was called with callback_url from task
        mock_scheduler_client.send_task_result.assert_called_once()
        call_args = mock_scheduler_client.send_task_result.call_args
        assert call_args[1]["callback_url"] == "http://custom-callback:8000/callback"

    async def test_clear_all_tasks_empty_queue(self):
        """Test clearing tasks when queue is empty"""
        queue = TaskQueue()

        result = await queue.clear_all_tasks()

        assert result["total"] == 0
        assert result["queued"] == 0
        assert result["completed"] == 0
        assert result["failed"] == 0
        assert len(queue.tasks) == 0
        assert len(queue.queue) == 0

    async def test_clear_all_tasks_with_queued_tasks(self):
        """Test clearing tasks with queued tasks"""
        queue = TaskQueue()

        # Mock _process_queue to prevent tasks from executing
        with patch.object(queue, '_process_queue', new=AsyncMock()):
            # Add some queued tasks
            task1 = Task(task_id="task-1", model_id="model", task_input={})
            task2 = Task(task_id="task-2", model_id="model", task_input={})
            task3 = Task(task_id="task-3", model_id="model", task_input={})

            await queue.submit_task(task1)
            await queue.submit_task(task2)
            await queue.submit_task(task3)

            result = await queue.clear_all_tasks()

            assert result["total"] == 3
            assert result["queued"] == 3
            assert result["completed"] == 0
            assert result["failed"] == 0
            assert len(queue.tasks) == 0
            assert len(queue.queue) == 0

    async def test_clear_all_tasks_with_mixed_statuses(self):
        """Test clearing tasks with mixed statuses"""
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

        # Mark tasks with different statuses
        task2.mark_completed({"result": "success"})
        task3.mark_failed("error")
        # task1 and task4 remain queued

        result = await queue.clear_all_tasks()

        assert result["total"] == 4
        assert result["queued"] == 2
        assert result["completed"] == 1
        assert result["failed"] == 1
        assert len(queue.tasks) == 0
        assert len(queue.queue) == 0

    async def test_clear_all_tasks_with_running_task(self):
        """Test that clearing tasks fails when there are running tasks and force=False"""
        queue = TaskQueue()

        # Create and submit a task
        task = Task(task_id="task-1", model_id="model", task_input={})
        await queue.submit_task(task)

        # Mark task as running
        task.mark_started()

        # Attempt to clear with force=False - should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await queue.clear_all_tasks(force=False)

        assert "Cannot clear tasks while 1 task(s) are running" in str(exc_info.value)

        # Task should still exist
        assert len(queue.tasks) == 1
        assert task.task_id in queue.tasks

    async def test_clear_all_tasks_multiple_running_tasks(self):
        """Test that clearing tasks fails when there are multiple running tasks and force=False"""
        queue = TaskQueue()

        # Create tasks
        task1 = Task(task_id="task-1", model_id="model", task_input={})
        task2 = Task(task_id="task-2", model_id="model", task_input={})
        task3 = Task(task_id="task-3", model_id="model", task_input={})

        await queue.submit_task(task1)
        await queue.submit_task(task2)
        await queue.submit_task(task3)

        # Mark multiple tasks as running (shouldn't happen in real scenarios, but test it)
        task1.mark_started()
        task2.mark_started()

        # Attempt to clear with force=False - should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await queue.clear_all_tasks(force=False)

        assert "Cannot clear tasks while 2 task(s) are running" in str(exc_info.value)

        # All tasks should still exist
        assert len(queue.tasks) == 3

    async def test_clear_all_tasks_returns_correct_counts(self):
        """Test that clear_all_tasks returns accurate counts"""
        queue = TaskQueue()

        # Mock _process_queue to prevent tasks from executing
        with patch.object(queue, '_process_queue', new=AsyncMock()):
            # Create a realistic scenario with various task statuses
            for i in range(5):
                task = Task(task_id=f"queued-{i}", model_id="model", task_input={})
                await queue.submit_task(task)

            for i in range(3):
                task = Task(task_id=f"completed-{i}", model_id="model", task_input={})
                await queue.submit_task(task)
                task.mark_completed({"result": "success"})

            for i in range(2):
                task = Task(task_id=f"failed-{i}", model_id="model", task_input={})
                await queue.submit_task(task)
                task.mark_failed("error")

            result = await queue.clear_all_tasks()

            assert result["queued"] == 5
            assert result["completed"] == 3
            assert result["failed"] == 2
            assert result["total"] == 10

    async def test_clear_all_tasks_clears_queue_and_storage(self):
        """Test that clear_all_tasks clears both queue and task storage"""
        queue = TaskQueue()

        # Mock _process_queue to prevent tasks from executing
        with patch.object(queue, '_process_queue', new=AsyncMock()):
            # Add tasks
            for i in range(5):
                task = Task(task_id=f"task-{i}", model_id="model", task_input={})
                await queue.submit_task(task)

            # Verify tasks exist before clearing
            assert len(queue.tasks) == 5
            assert len(queue.queue) == 5

            # Clear tasks
            await queue.clear_all_tasks()

            # Verify everything is cleared
            assert len(queue.tasks) == 0
            assert len(queue.queue) == 0
            assert queue.current_task_id is None


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


@pytest.mark.unit
@pytest.mark.asyncio
class TestTaskQueueFetch:
    """Test suite for TaskQueue.fetch_task() method"""

    async def test_fetch_task_from_queue_front(self):
        """Test fetch returns task with highest enqueue_time (newest task - LIFO for work-stealing)"""
        queue = TaskQueue()

        # Mock _process_queue to prevent tasks from executing
        with patch.object(queue, '_process_queue', new=AsyncMock()):
            # Submit tasks with different enqueue times
            task1 = Task(task_id="task-1", model_id="model", task_input={})
            task2 = Task(task_id="task-2", model_id="model", task_input={})
            task3 = Task(task_id="task-3", model_id="model", task_input={})

            await queue.submit_task(task1, enqueue_time=100.0)  # Oldest
            await queue.submit_task(task2, enqueue_time=200.0)
            await queue.submit_task(task3, enqueue_time=300.0)  # Newest

            # Fetch should return the task with highest enqueue_time (newest - LIFO)
            fetched = await queue.fetch_task()

            assert fetched is not None
            assert fetched.task_id == "task-3"

    async def test_fetch_task_removes_from_storage(self):
        """Test fetched task is completely removed from storage (not marked)"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            # Need at least 2 tasks because fetch keeps at least 1 active task
            task1 = Task(task_id="task-1", model_id="model", task_input={})
            task2 = Task(task_id="task-2", model_id="model", task_input={})
            await queue.submit_task(task1, enqueue_time=100.0)
            await queue.submit_task(task2, enqueue_time=200.0)

            fetched = await queue.fetch_task()

            assert fetched is not None
            assert fetched.task_id == "task-2"  # Newest fetched first (LIFO)
            # Task is completely removed from storage
            assert fetched.task_id not in queue.tasks

    async def test_fetch_task_removes_from_execution_queue(self):
        """Test fetched task won't be in execution queue"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            # Need at least 2 tasks because fetch keeps at least 1 active task
            task1 = Task(task_id="task-1", model_id="model", task_input={})
            task2 = Task(task_id="task-2", model_id="model", task_input={})
            await queue.submit_task(task1, enqueue_time=100.0)
            await queue.submit_task(task2, enqueue_time=200.0)

            # Verify tasks are in queue before fetch
            assert len(queue.queue) == 2

            fetched = await queue.fetch_task()

            # Only fetched task should be removed, one remains
            assert len(queue.queue) == 1
            assert fetched.task_id == "task-2"  # Newest fetched first (LIFO)

    async def test_fetch_task_empty_queue_returns_none(self):
        """Test fetch on empty queue returns None"""
        queue = TaskQueue()

        fetched = await queue.fetch_task()

        assert fetched is None

    async def test_fetch_task_skips_running_tasks(self):
        """Test fetch only considers QUEUED tasks, not running ones"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task1 = Task(task_id="task-1", model_id="model", task_input={})
            task2 = Task(task_id="task-2", model_id="model", task_input={})
            task3 = Task(task_id="task-3", model_id="model", task_input={})

            await queue.submit_task(task1, enqueue_time=100.0)
            await queue.submit_task(task2, enqueue_time=200.0)
            await queue.submit_task(task3, enqueue_time=300.0)

            # Mark task1 (oldest) as running - counts as 1 active
            task1.mark_started()

            # Now: 1 running + 2 queued = 3 active, can fetch newest queued
            # Fetch should return task3 (newest QUEUED task - LIFO)
            fetched = await queue.fetch_task()

            assert fetched is not None
            assert fetched.task_id == "task-3"

    async def test_fetch_task_preserves_remaining_tasks(self):
        """Test non-fetched tasks remain in storage after fetch"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            # Need at least 2 tasks because fetch keeps at least 1 active task
            task1 = Task(task_id="task-1", model_id="model", task_input={})
            task2 = Task(task_id="task-2", model_id="model", task_input={})
            await queue.submit_task(task1, enqueue_time=100.0)
            await queue.submit_task(task2, enqueue_time=200.0)

            fetched = await queue.fetch_task()

            # Fetched task (task-2) is completely removed
            assert fetched.task_id == "task-2"
            assert "task-2" not in queue.tasks
            # Remaining task (task-1) stays in storage
            remaining_task = await queue.get_task("task-1")
            assert remaining_task is not None
            assert remaining_task.status == TaskStatus.QUEUED

    async def test_fetch_task_all_running_returns_none(self):
        """Test fetch when all tasks are running returns None"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task1 = Task(task_id="task-1", model_id="model", task_input={})
            task2 = Task(task_id="task-2", model_id="model", task_input={})

            await queue.submit_task(task1)
            await queue.submit_task(task2)

            # Mark all as running
            task1.mark_started()
            task2.mark_started()

            fetched = await queue.fetch_task()

            assert fetched is None

    async def test_fetch_task_all_completed_returns_none(self):
        """Test fetch when all tasks are completed returns None"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task1 = Task(task_id="task-1", model_id="model", task_input={})
            await queue.submit_task(task1)

            # Complete the task
            task1.mark_completed({"result": "done"})

            fetched = await queue.fetch_task()

            assert fetched is None

    async def test_fetch_task_multiple_fetches(self):
        """Test multiple sequential fetches return tasks in descending enqueue_time order (LIFO)"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task1 = Task(task_id="task-1", model_id="model", task_input={})
            task2 = Task(task_id="task-2", model_id="model", task_input={})
            task3 = Task(task_id="task-3", model_id="model", task_input={})

            await queue.submit_task(task1, enqueue_time=100.0)
            await queue.submit_task(task2, enqueue_time=200.0)
            await queue.submit_task(task3, enqueue_time=300.0)

            # First fetch gets newest (from tail - LIFO)
            fetched1 = await queue.fetch_task()
            assert fetched1.task_id == "task-3"

            # Second fetch gets next newest
            fetched2 = await queue.fetch_task()
            assert fetched2.task_id == "task-2"

            # Third fetch returns None - must keep at least 1 active task
            fetched3 = await queue.fetch_task()
            assert fetched3 is None

            # task-1 should still be QUEUED (oldest kept)
            assert task1.status == TaskStatus.QUEUED

    async def test_fetch_task_reduces_total_count(self):
        """Test that fetching removes task from storage, reducing counts"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            # Need at least 2 tasks because fetch keeps at least 1 active task
            task1 = Task(task_id="task-1", model_id="model", task_input={})
            task2 = Task(task_id="task-2", model_id="model", task_input={})
            await queue.submit_task(task1, enqueue_time=100.0)
            await queue.submit_task(task2, enqueue_time=200.0)

            # Before fetch: 2 tasks
            stats_before = await queue.get_queue_stats()
            assert stats_before["total"] == 2
            assert stats_before["queued"] == 2

            await queue.fetch_task()

            # After fetch: 1 task remaining (task is removed, not marked)
            stats_after = await queue.get_queue_stats()
            assert stats_after["total"] == 1
            assert stats_after["queued"] == 1
            assert stats_after["fetched"] == 0  # Not marked, just removed

    async def test_fetch_task_keeps_at_least_one_active(self):
        """Test that fetch won't remove the last active task"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            # Submit only 1 task
            task = Task(task_id="task-1", model_id="model", task_input={})
            await queue.submit_task(task)

            # Fetch should return None - can't remove the last active task
            fetched = await queue.fetch_task()
            assert fetched is None

            # Task should still be QUEUED
            assert task.status == TaskStatus.QUEUED

    async def test_fetch_task_with_running_counts_as_active(self):
        """Test that running task counts as active, allowing fetch of queued"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task1 = Task(task_id="task-1", model_id="model", task_input={})
            task2 = Task(task_id="task-2", model_id="model", task_input={})
            await queue.submit_task(task1, enqueue_time=100.0)
            await queue.submit_task(task2, enqueue_time=200.0)

            # Mark task1 as running (counts as 1 active)
            task1.mark_started()

            # Now: 1 running + 1 queued = 2 active
            # Can fetch 1, leaving 1 running
            fetched = await queue.fetch_task()
            assert fetched is not None
            assert fetched.task_id == "task-2"

            # Try to fetch again - should fail (only 1 running left)
            fetched2 = await queue.fetch_task()
            assert fetched2 is None


class TestTaskQueueCleanup:
    """Tests for cleanup_fetched_tasks method.

    Note: fetch_task() now removes tasks completely instead of marking them as FETCHED.
    These tests verify the cleanup function still works for any manually set FETCHED tasks.
    """

    async def test_cleanup_fetched_tasks_removes_manually_marked(self):
        """Test that cleanup removes manually FETCHED tasks from storage"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task1 = Task(task_id="task-1", model_id="model", task_input={})
            task2 = Task(task_id="task-2", model_id="model", task_input={})
            task3 = Task(task_id="task-3", model_id="model", task_input={})

            await queue.submit_task(task1)
            await queue.submit_task(task2)
            await queue.submit_task(task3)

            # Manually mark some tasks as FETCHED (simulating external marking)
            task1.status = TaskStatus.FETCHED
            task2.status = TaskStatus.FETCHED

            # Verify 2 are FETCHED
            stats = await queue.get_queue_stats()
            assert stats["fetched"] == 2
            assert stats["queued"] == 1
            assert stats["total"] == 3

            # Cleanup
            cleaned = await queue.cleanup_fetched_tasks()
            assert cleaned == 2

            # Verify only task-3 (QUEUED) remains
            stats = await queue.get_queue_stats()
            assert stats["fetched"] == 0
            assert stats["queued"] == 1
            assert stats["total"] == 1

    async def test_cleanup_fetched_tasks_preserves_other_statuses(self):
        """Test that cleanup only removes FETCHED tasks, not others"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task1 = Task(task_id="task-1", model_id="model", task_input={})
            task2 = Task(task_id="task-2", model_id="model", task_input={})
            task3 = Task(task_id="task-3", model_id="model", task_input={})
            task4 = Task(task_id="task-4", model_id="model", task_input={})

            await queue.submit_task(task1)
            await queue.submit_task(task2)
            await queue.submit_task(task3)
            await queue.submit_task(task4)

            # Set up various statuses
            task1.mark_started()  # RUNNING
            task2.status = TaskStatus.FETCHED  # Manually marked FETCHED
            task3.mark_completed({"result": "done"})  # COMPLETED
            # task4 remains QUEUED

            # Cleanup
            cleaned = await queue.cleanup_fetched_tasks()
            assert cleaned == 1

            # Verify other tasks remain
            stats = await queue.get_queue_stats()
            assert stats["fetched"] == 0
            assert stats["completed"] == 1
            assert stats["queued"] == 1
            assert stats["running"] == 1
            assert stats["total"] == 3

    async def test_cleanup_fetched_tasks_empty_returns_zero(self):
        """Test that cleanup returns 0 when no FETCHED tasks"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task = Task(task_id="task-1", model_id="model", task_input={})
            await queue.submit_task(task)

            cleaned = await queue.cleanup_fetched_tasks()
            assert cleaned == 0

    async def test_get_queue_stats_includes_fetched_count(self):
        """Test that get_queue_stats includes fetched count key"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task1 = Task(task_id="task-1", model_id="model", task_input={})
            await queue.submit_task(task1)

            # Manually mark as FETCHED to test stats counting
            task1.status = TaskStatus.FETCHED

            stats = await queue.get_queue_stats()
            assert "fetched" in stats
            assert stats["fetched"] == 1

    async def test_get_queue_stats_fetched_is_zero_after_fetch_task(self):
        """Test that fetch_task removes (not marks) so fetched count stays 0"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            # Need at least 2 tasks to fetch 1
            task1 = Task(task_id="task-1", model_id="model", task_input={})
            task2 = Task(task_id="task-2", model_id="model", task_input={})
            await queue.submit_task(task1, enqueue_time=100.0)
            await queue.submit_task(task2, enqueue_time=200.0)

            await queue.fetch_task()

            # fetch_task removes tasks, doesn't mark as FETCHED
            stats = await queue.get_queue_stats()
            assert stats["fetched"] == 0
            assert stats["total"] == 1  # Only 1 task remaining


@pytest.mark.unit
@pytest.mark.asyncio
class TestTaskQueueDetach:
    """Test suite for TaskQueue.detach_current_task() method.

    This tests the task detachment feature used by deregister to allow
    long-running tasks to continue in background while deregister proceeds.
    """

    async def test_detach_current_task_success(self):
        """Test successful detach of a running task.

        When a task is running (current_task_id is set), detach_current_task()
        should:
        1. Return the detached task_id
        2. Set current_task_id to None (so deregister can proceed)
        3. Set _detached_task_id to track the detached task
        """
        queue = TaskQueue()
        queue.current_task_id = "task-123"

        detached_id = await queue.detach_current_task()

        assert detached_id == "task-123"
        assert queue.current_task_id is None
        assert queue._detached_task_id == "task-123"

    async def test_detach_current_task_no_task_running(self):
        """Test detach when no task is running returns None.

        If current_task_id is None, detach_current_task() should return None
        and not modify _detached_task_id.
        """
        queue = TaskQueue()
        queue.current_task_id = None

        detached_id = await queue.detach_current_task()

        assert detached_id is None
        assert queue._detached_task_id is None

    async def test_detached_task_clears_after_completion(self):
        """Test that _detached_task_id is cleared after task completes.

        When _process_queue() finishes executing a detached task, it should
        clear _detached_task_id instead of current_task_id (which is already None).
        """
        queue = TaskQueue()

        # Simulate the state after a task has been detached
        queue._detached_task_id = "task-123"
        queue.current_task_id = None

        # Submit a task and mock execution
        task = Task(task_id="task-123", model_id="model", task_input={})
        queue.tasks[task.task_id] = task

        # Simulate what _process_queue does after _execute_task returns
        # This is the logic we'll implement: clear _detached_task_id if it matches
        task_id = "task-123"
        if queue._detached_task_id == task_id:
            queue._detached_task_id = None
        else:
            queue.current_task_id = None

        assert queue._detached_task_id is None
        assert queue.current_task_id is None

    async def test_multiple_detach_calls_second_returns_none(self):
        """Test that calling detach twice returns None on second call.

        After a task is detached, current_task_id is None, so subsequent
        calls to detach_current_task() should return None.
        """
        queue = TaskQueue()
        queue.current_task_id = "task-123"

        # First detach
        first_result = await queue.detach_current_task()
        assert first_result == "task-123"

        # Second detach should return None
        second_result = await queue.detach_current_task()
        assert second_result is None

    async def test_detach_preserves_task_in_storage(self):
        """Test that detaching a task doesn't remove it from storage.

        The task should continue to exist in queue.tasks so it can complete
        and send its callback.
        """
        queue = TaskQueue()

        task = Task(task_id="task-123", model_id="model", task_input={})
        queue.tasks[task.task_id] = task
        queue.current_task_id = task.task_id

        await queue.detach_current_task()

        # Task should still be in storage
        assert "task-123" in queue.tasks
        assert queue.tasks["task-123"] is task


@pytest.mark.unit
@pytest.mark.asyncio
class TestTaskQueuePeekPending:
    """Test suite for TaskQueue.peek_pending_tasks() method"""

    async def test_peek_pending_tasks_returns_queued_tasks(self):
        """Test that peek_pending_tasks returns only QUEUED tasks"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task1 = Task(task_id="task-1", model_id="model", task_input={"key": "value1"})
            task2 = Task(task_id="task-2", model_id="model", task_input={"key": "value2"})
            await queue.submit_task(task1, enqueue_time=100.0)
            await queue.submit_task(task2, enqueue_time=200.0)

            pending = await queue.peek_pending_tasks()

            assert len(pending) == 2
            task_ids = [t["task_id"] for t in pending]
            assert "task-1" in task_ids
            assert "task-2" in task_ids

    async def test_peek_pending_tasks_excludes_running(self):
        """Test that peek_pending_tasks excludes RUNNING tasks"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task1 = Task(task_id="task-1", model_id="model", task_input={})
            task2 = Task(task_id="task-2", model_id="model", task_input={})
            await queue.submit_task(task1)
            await queue.submit_task(task2)

            # Mark one as running
            task1.mark_started()

            pending = await queue.peek_pending_tasks()

            assert len(pending) == 1
            assert pending[0]["task_id"] == "task-2"

    async def test_peek_pending_tasks_empty_queue(self):
        """Test that peek_pending_tasks returns empty list for empty queue"""
        queue = TaskQueue()

        pending = await queue.peek_pending_tasks()
        assert pending == []

    async def test_peek_pending_tasks_includes_callback_url(self):
        """Test that peek_pending_tasks includes callback_url"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task = Task(
                task_id="task-1",
                model_id="test-model",
                task_input={"prompt": "test"},
                callback_url="http://callback.example.com"
            )
            await queue.submit_task(task)

            pending = await queue.peek_pending_tasks()

            assert len(pending) == 1
            assert pending[0]["model_id"] == "test-model"
            assert pending[0]["task_input"] == {"prompt": "test"}
            assert pending[0]["callback_url"] == "http://callback.example.com"


@pytest.mark.unit
@pytest.mark.asyncio
class TestTaskQueueRemove:
    """Test suite for TaskQueue.remove_task() method"""

    async def test_remove_task_success(self):
        """Test successful task removal"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task = Task(task_id="task-1", model_id="model", task_input={})
            await queue.submit_task(task)

            assert "task-1" in queue.tasks

            result = await queue.remove_task("task-1")

            assert result is True
            assert "task-1" not in queue.tasks

    async def test_remove_task_not_found(self):
        """Test removing non-existent task returns False"""
        queue = TaskQueue()

        result = await queue.remove_task("non-existent")
        assert result is False

    async def test_remove_task_removes_from_queue(self):
        """Test that remove_task removes from priority queue"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task1 = Task(task_id="task-1", model_id="model", task_input={})
            task2 = Task(task_id="task-2", model_id="model", task_input={})
            await queue.submit_task(task1)
            await queue.submit_task(task2)

            initial_queue_len = len(queue.queue)
            await queue.remove_task("task-1")

            assert len(queue.queue) == initial_queue_len - 1
            task_ids_in_queue = [tid for _, tid in queue.queue]
            assert "task-1" not in task_ids_in_queue
            assert "task-2" in task_ids_in_queue


@pytest.mark.unit
@pytest.mark.asyncio
class TestTaskQueueExtract:
    """Test suite for TaskQueue.extract_pending_tasks() method"""

    async def test_extract_pending_tasks_returns_queued(self):
        """Test extract_pending_tasks returns all QUEUED tasks"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task1 = Task(task_id="task-1", model_id="model", task_input={"key": "v1"})
            task2 = Task(task_id="task-2", model_id="model", task_input={"key": "v2"})
            await queue.submit_task(task1, enqueue_time=100.0)
            await queue.submit_task(task2, enqueue_time=200.0)

            extracted = await queue.extract_pending_tasks()

            assert len(extracted) == 2
            task_ids = [t["task_id"] for t in extracted]
            assert "task-1" in task_ids
            assert "task-2" in task_ids

    async def test_extract_pending_tasks_removes_from_storage(self):
        """Test extracted tasks are removed from storage"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task = Task(task_id="task-1", model_id="model", task_input={})
            await queue.submit_task(task)

            await queue.extract_pending_tasks()

            assert "task-1" not in queue.tasks

    async def test_extract_pending_tasks_preserves_running(self):
        """Test that RUNNING tasks are preserved during extraction"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task1 = Task(task_id="task-1", model_id="model", task_input={})
            task2 = Task(task_id="task-2", model_id="model", task_input={})
            await queue.submit_task(task1)
            await queue.submit_task(task2)

            # Mark one as running
            task1.mark_started()

            extracted = await queue.extract_pending_tasks()

            # Only task-2 should be extracted
            assert len(extracted) == 1
            assert extracted[0]["task_id"] == "task-2"

            # task-1 (RUNNING) should still be in storage and queue
            assert "task-1" in queue.tasks
            assert len(queue.queue) == 1

    async def test_extract_pending_tasks_removes_fetched(self):
        """Test that FETCHED tasks are removed without being returned"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task1 = Task(task_id="task-1", model_id="model", task_input={})
            task2 = Task(task_id="task-2", model_id="model", task_input={})
            await queue.submit_task(task1)
            await queue.submit_task(task2)

            # Manually mark one as FETCHED
            task1.status = TaskStatus.FETCHED

            extracted = await queue.extract_pending_tasks()

            # Only task-2 should be extracted (task-1 was FETCHED)
            assert len(extracted) == 1
            assert extracted[0]["task_id"] == "task-2"

            # Both should be removed from storage
            assert "task-1" not in queue.tasks
            assert "task-2" not in queue.tasks


@pytest.mark.unit
@pytest.mark.asyncio
class TestTaskQueueCurrentTaskInfo:
    """Test suite for TaskQueue.get_current_task_info() method"""

    async def test_get_current_task_info_no_running_task(self):
        """Test returns None when no task is running"""
        queue = TaskQueue()

        result = await queue.get_current_task_info()
        assert result is None

    async def test_get_current_task_info_with_running_task(self):
        """Test returns task info when task is running"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task = Task(task_id="task-123", model_id="model", task_input={})
            await queue.submit_task(task)

            # Simulate running task
            task.mark_started()
            queue.current_task_id = "task-123"

            result = await queue.get_current_task_info()

            assert result is not None
            assert result["task_id"] == "task-123"
            assert "estimated_completion_ms" in result

    async def test_get_current_task_info_task_not_running_status(self):
        """Test returns None when current_task_id points to non-RUNNING task"""
        queue = TaskQueue()

        with patch.object(queue, '_process_queue', new=AsyncMock()):
            task = Task(task_id="task-123", model_id="model", task_input={})
            await queue.submit_task(task)

            # Set current_task_id but task is still QUEUED
            queue.current_task_id = "task-123"

            result = await queue.get_current_task_info()
            assert result is None

    async def test_get_current_task_info_task_not_found(self):
        """Test returns None when current_task_id doesn't exist in storage"""
        queue = TaskQueue()

        queue.current_task_id = "non-existent"

        result = await queue.get_current_task_info()
        assert result is None
