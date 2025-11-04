"""
Unit tests for TaskRegistry and TaskRecord.

Tests task lifecycle management, status transitions, filtering, and thread safety.
"""

import pytest
import threading
from datetime import datetime, timedelta
from time import sleep

from src.task_registry import TaskRegistry, TaskRecord
from src.model import TaskStatus, TaskTimestamps


# ============================================================================
# TaskRecord Tests
# ============================================================================

class TestTaskRecord:
    """Tests for TaskRecord class."""

    async def test_initialization(self):
        """Test TaskRecord initialization."""
        task = TaskRecord(
            task_id="task-1",
            model_id="model-1",
            task_input={"prompt": "test"},
            metadata={"key": "value"},
            assigned_instance="inst-1"
        )

        assert task.task_id == "task-1"
        assert task.model_id == "model-1"
        assert task.task_input == {"prompt": "test"}
        assert task.metadata == {"key": "value"}
        assert task.assigned_instance == "inst-1"
        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.error is None
        assert task.submitted_at is not None
        assert task.started_at is None
        assert task.completed_at is None

    async def test_execution_time_not_started(self):
        """Test execution time when task hasn't started."""
        task = TaskRecord(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="inst-1"
        )
        assert task.execution_time_ms is None

    async def test_execution_time_calculation(self):
        """Test execution time calculation."""
        task = TaskRecord(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="inst-1"
        )

        # Manually set timestamps
        task.started_at = "2024-01-01T12:00:00.000Z"
        task.completed_at = "2024-01-01T12:00:01.500Z"

        execution_time = task.execution_time_ms
        assert execution_time == 1500

    async def test_execution_time_with_invalid_timestamps(self):
        """Test execution time with malformed timestamps."""
        task = TaskRecord(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="inst-1"
        )

        task.started_at = "invalid"
        task.completed_at = "also-invalid"

        assert task.execution_time_ms is None

    async def test_get_timestamps(self):
        """Test getting timestamps as TaskTimestamps model."""
        task = TaskRecord(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="inst-1"
        )

        task.started_at = "2024-01-01T12:00:00Z"
        task.completed_at = "2024-01-01T12:00:01Z"

        timestamps = task.get_timestamps()
        assert isinstance(timestamps, TaskTimestamps)
        assert timestamps.submitted_at is not None
        assert timestamps.started_at == "2024-01-01T12:00:00Z"
        assert timestamps.completed_at == "2024-01-01T12:00:01Z"


# ============================================================================
# Task Creation and Retrieval Tests
# ============================================================================

class TestTaskCreation:
    """Tests for task creation and retrieval."""

    async def test_create_task(self, task_registry):
        """Test creating a new task."""
        task = await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={"prompt": "test"},
            metadata={"key": "value"},
            assigned_instance="inst-1"
        )

        assert task is not None
        assert task.task_id == "task-1"
        assert task.status == TaskStatus.PENDING

    async def test_create_duplicate_task(self, task_registry):
        """Test that creating duplicate task raises ValueError."""
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="inst-1"
        )

        with pytest.raises(ValueError, match="already exists"):
            await task_registry.create_task(
                task_id="task-1",
                model_id="model-1",
                task_input={},
                metadata={},
                assigned_instance="inst-1"
            )

    async def test_get_existing_task(self, task_registry):
        """Test getting an existing task."""
        created = await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={"test": "data"},
            metadata={},
            assigned_instance="inst-1"
        )

        retrieved = await task_registry.get("task-1")
        assert retrieved is not None
        assert retrieved.task_id == created.task_id
        assert retrieved.task_input == {"test": "data"}

    async def test_get_nonexistent_task(self, task_registry):
        """Test getting a non-existent task returns None."""
        result = await task_registry.get("nonexistent-task")
        assert result is None


# ============================================================================
# Status Update Tests
# ============================================================================

class TestStatusUpdates:
    """Tests for task status updates."""

    async def test_update_status_to_running(self, task_registry):
        """Test updating status to RUNNING sets started_at."""
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="inst-1"
        )

        await task_registry.update_status("task-1", TaskStatus.RUNNING)

        task = await task_registry.get("task-1")
        assert task.status == TaskStatus.RUNNING
        assert task.started_at is not None
        assert task.completed_at is None

    async def test_update_status_to_completed(self, task_registry):
        """Test updating status to COMPLETED sets completed_at."""
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="inst-1"
        )

        await task_registry.update_status("task-1", TaskStatus.RUNNING)
        await task_registry.update_status("task-1", TaskStatus.COMPLETED)

        task = await task_registry.get("task-1")
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None

    async def test_update_status_to_failed(self, task_registry):
        """Test updating status to FAILED sets completed_at."""
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="inst-1"
        )

        await task_registry.update_status("task-1", TaskStatus.RUNNING)
        await task_registry.update_status("task-1", TaskStatus.FAILED)

        task = await task_registry.get("task-1")
        assert task.status == TaskStatus.FAILED
        assert task.completed_at is not None

    async def test_update_status_nonexistent_task(self, task_registry):
        """Test updating status of non-existent task raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            await task_registry.update_status("nonexistent", TaskStatus.RUNNING)

    async def test_status_transitions(self, task_registry):
        """Test full status transition lifecycle."""
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="inst-1"
        )

        # PENDING -> RUNNING
        await task_registry.update_status("task-1", TaskStatus.RUNNING)
        task = await task_registry.get("task-1")
        assert task.status == TaskStatus.RUNNING
        assert task.started_at is not None

        # RUNNING -> COMPLETED
        await task_registry.update_status("task-1", TaskStatus.COMPLETED)
        task = await task_registry.get("task-1")
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None

    async def test_started_at_only_set_once(self, task_registry):
        """Test that started_at is only set once."""
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="inst-1"
        )

        await task_registry.update_status("task-1", TaskStatus.RUNNING)
        task = await task_registry.get("task-1")
        first_started_at = task.started_at

        sleep(0.01)
        await task_registry.update_status("task-1", TaskStatus.RUNNING)
        task = await task_registry.get("task-1")
        assert task.started_at == first_started_at

    async def test_completed_at_only_set_once(self, task_registry):
        """Test that completed_at is only set once."""
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="inst-1"
        )

        await task_registry.update_status("task-1", TaskStatus.RUNNING)
        await task_registry.update_status("task-1", TaskStatus.COMPLETED)
        task = await task_registry.get("task-1")
        first_completed_at = task.completed_at

        sleep(0.01)
        await task_registry.update_status("task-1", TaskStatus.COMPLETED)
        task = await task_registry.get("task-1")
        assert task.completed_at == first_completed_at


# ============================================================================
# Result and Error Tests
# ============================================================================

class TestResultAndError:
    """Tests for setting task results and errors."""

    async def test_set_result(self, task_registry):
        """Test setting task result."""
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="inst-1"
        )

        result = {"output": "test output", "tokens": 100}
        await task_registry.set_result("task-1", result)

        task = await task_registry.get("task-1")
        assert task.result == result

    async def test_set_result_nonexistent_task(self, task_registry):
        """Test setting result for non-existent task raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            await task_registry.set_result("nonexistent", {"output": "test"})

    async def test_set_error(self, task_registry):
        """Test setting task error."""
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={},
            assigned_instance="inst-1"
        )

        error_msg = "Task execution failed"
        await task_registry.set_error("task-1", error_msg)

        task = await task_registry.get("task-1")
        assert task.error == error_msg

    async def test_set_error_nonexistent_task(self, task_registry):
        """Test setting error for non-existent task raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            await task_registry.set_error("nonexistent", "error message")


# ============================================================================
# List and Filter Tests
# ============================================================================

class TestListAndFilter:
    """Tests for listing and filtering tasks."""

    async def test_list_all_empty(self, task_registry):
        """Test listing tasks from empty registry."""
        tasks, total = await task_registry.list_all()
        assert tasks == []
        assert total == 0

    async def test_list_all_with_tasks(self, task_registry):
        """Test listing all tasks."""
        for i in range(5):
            await task_registry.create_task(
                task_id=f"task-{i}",
                model_id="model-1",
                task_input={},
                metadata={},
                assigned_instance="inst-1"
            )

        tasks, total = await task_registry.list_all()
        assert len(tasks) == 5
        assert total == 5

    async def test_filter_by_status(self, task_registry):
        """Test filtering tasks by status."""
        # Create tasks with different statuses
        for i in range(3):
            await task_registry.create_task(
                task_id=f"pending-{i}",
                model_id="model-1",
                task_input={},
                metadata={},
                assigned_instance="inst-1"
            )

        for i in range(2):
            task_id = f"running-{i}"
            await task_registry.create_task(
                task_id=task_id,
                model_id="model-1",
                task_input={},
                metadata={},
                assigned_instance="inst-1"
            )
            await task_registry.update_status(task_id, TaskStatus.RUNNING)

        # Filter by PENDING
        tasks, total = await task_registry.list_all(status=TaskStatus.PENDING)
        assert len(tasks) == 3
        assert total == 3
        assert all(t.status == TaskStatus.PENDING for t in tasks)

        # Filter by RUNNING
        tasks, total = await task_registry.list_all(status=TaskStatus.RUNNING)
        assert len(tasks) == 2
        assert total == 2
        assert all(t.status == TaskStatus.RUNNING for t in tasks)

    async def test_filter_by_model_id(self, task_registry):
        """Test filtering tasks by model_id."""
        for i in range(3):
            await task_registry.create_task(
                task_id=f"task-a-{i}",
                model_id="model-a",
                task_input={},
                metadata={},
                assigned_instance="inst-1"
            )

        for i in range(2):
            await task_registry.create_task(
                task_id=f"task-b-{i}",
                model_id="model-b",
                task_input={},
                metadata={},
                assigned_instance="inst-1"
            )

        tasks, total = await task_registry.list_all(model_id="model-a")
        assert len(tasks) == 3
        assert total == 3
        assert all(t.model_id == "model-a" for t in tasks)

    async def test_filter_by_instance_id(self, task_registry):
        """Test filtering tasks by instance_id."""
        for i in range(2):
            await task_registry.create_task(
                task_id=f"task-inst1-{i}",
                model_id="model-1",
                task_input={},
                metadata={},
                assigned_instance="inst-1"
            )

        for i in range(3):
            await task_registry.create_task(
                task_id=f"task-inst2-{i}",
                model_id="model-1",
                task_input={},
                metadata={},
                assigned_instance="inst-2"
            )

        tasks, total = await task_registry.list_all(instance_id="inst-2")
        assert len(tasks) == 3
        assert total == 3
        assert all(t.assigned_instance == "inst-2" for t in tasks)

    async def test_combined_filters(self, task_registry):
        """Test combining multiple filters."""
        # Create diverse tasks
        await task_registry.create_task(
            task_id="task-1",
            model_id="model-a",
            task_input={},
            metadata={},
            assigned_instance="inst-1"
        )
        await task_registry.update_status("task-1", TaskStatus.RUNNING)

        await task_registry.create_task(
            task_id="task-2",
            model_id="model-a",
            task_input={},
            metadata={},
            assigned_instance="inst-2"
        )
        await task_registry.update_status("task-2", TaskStatus.RUNNING)

        await task_registry.create_task(
            task_id="task-3",
            model_id="model-a",
            task_input={},
            metadata={},
            assigned_instance="inst-1"
        )
        # Keep PENDING

        # Filter by model_id=model-a, status=RUNNING, instance_id=inst-1
        tasks, total = await task_registry.list_all(
            status=TaskStatus.RUNNING,
            model_id="model-a",
            instance_id="inst-1"
        )
        assert len(tasks) == 1
        assert total == 1
        assert tasks[0].task_id == "task-1"

    async def test_pagination(self, task_registry):
        """Test pagination with limit and offset."""
        # Create 10 tasks
        for i in range(10):
            await task_registry.create_task(
                task_id=f"task-{i}",
                model_id="model-1",
                task_input={},
                metadata={},
                assigned_instance="inst-1"
            )
            sleep(0.001)  # Ensure different timestamps

        # Get first page
        tasks, total = await task_registry.list_all(limit=3, offset=0)
        assert len(tasks) == 3
        assert total == 10

        # Get second page
        tasks, total = await task_registry.list_all(limit=3, offset=3)
        assert len(tasks) == 3
        assert total == 10

        # Get last page
        tasks, total = await task_registry.list_all(limit=3, offset=9)
        assert len(tasks) == 1
        assert total == 10

    async def test_sorting_newest_first(self, task_registry):
        """Test that tasks are sorted newest first."""
        task_ids = []
        for i in range(5):
            task_id = f"task-{i}"
            await task_registry.create_task(
                task_id=task_id,
                model_id="model-1",
                task_input={},
                metadata={},
                assigned_instance="inst-1"
            )
            task_ids.append(task_id)
            sleep(0.001)

        tasks, _ = await task_registry.list_all()

        # Should be in reverse order (newest first)
        assert tasks[0].task_id == task_ids[-1]
        assert tasks[-1].task_id == task_ids[0]


# ============================================================================
# Count Operations Tests
# ============================================================================

class TestCountOperations:
    """Tests for task counting operations."""

    async def test_get_total_count_empty(self, task_registry):
        """Test total count on empty registry."""
        assert await task_registry.get_total_count() == 0

    async def test_get_total_count(self, task_registry):
        """Test total count with tasks."""
        for i in range(7):
            await task_registry.create_task(
                task_id=f"task-{i}",
                model_id="model-1",
                task_input={},
                metadata={},
                assigned_instance="inst-1"
            )

        assert await task_registry.get_total_count() == 7

    async def test_get_count_by_status(self, task_registry):
        """Test counting tasks by status."""
        # Create tasks with different statuses
        for i in range(5):
            task_id = f"task-{i}"
            await task_registry.create_task(
                task_id=task_id,
                model_id="model-1",
                task_input={},
                metadata={},
                assigned_instance="inst-1"
            )

            if i < 2:
                await task_registry.update_status(task_id, TaskStatus.RUNNING)
            elif i < 4:
                await task_registry.update_status(task_id, TaskStatus.RUNNING)
                await task_registry.update_status(task_id, TaskStatus.COMPLETED)
            # else: keep PENDING

        assert await task_registry.get_count_by_status(TaskStatus.PENDING) == 1
        assert await task_registry.get_count_by_status(TaskStatus.RUNNING) == 2
        assert await task_registry.get_count_by_status(TaskStatus.COMPLETED) == 2
        assert await task_registry.get_count_by_status(TaskStatus.FAILED) == 0

    async def test_get_count_by_status_empty(self, task_registry):
        """Test counting by status when no tasks exist."""
        assert await task_registry.get_count_by_status(TaskStatus.PENDING) == 0
        assert await task_registry.get_count_by_status(TaskStatus.COMPLETED) == 0


# ============================================================================
# Thread Safety Tests
# ============================================================================

@pytest.mark.skip(reason="ThreadSafety tests need async rewrite")
class TestThreadSafety:
    """Tests for thread-safe operations - DISABLED pending async rewrite."""
    pass
