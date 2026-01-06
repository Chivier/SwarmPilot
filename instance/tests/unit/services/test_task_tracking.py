"""Unit tests for TaskTrackingService.

Tests follow TDD principle - written before implementation.
"""

import pytest


@pytest.fixture
def task_service():
    """Create a TaskTrackingService instance."""
    from src.services.task_tracking import TaskTrackingService

    return TaskTrackingService()


class TestTaskTrackingServiceInit:
    """Tests for TaskTrackingService initialization."""

    def test_init_creates_service(self):
        """Test that service can be instantiated."""
        from src.services.task_tracking import TaskTrackingService

        service = TaskTrackingService()
        assert service is not None


class TestCreateTask:
    """Tests for create_task method."""

    @pytest.mark.asyncio
    async def test_create_task_returns_task_id(self, task_service):
        """Test that create_task returns a task_id."""
        task_id = await task_service.create_task(
            model_id="model_abc123",
            model_type="llm",
            operation="pull",
        )

        assert task_id.startswith("task_")

    @pytest.mark.asyncio
    async def test_create_task_unique_ids(self, task_service):
        """Test that each task gets unique ID."""
        task_id1 = await task_service.create_task(
            model_id="model_abc123",
            model_type="llm",
            operation="pull",
        )
        task_id2 = await task_service.create_task(
            model_id="model_def456",
            model_type="llm",
            operation="pull",
        )

        assert task_id1 != task_id2

    @pytest.mark.asyncio
    async def test_create_task_initial_status(self, task_service):
        """Test that new tasks have 'pending' status."""
        task_id = await task_service.create_task(
            model_id="model_abc123",
            model_type="llm",
            operation="pull",
        )

        task = await task_service.get_task(task_id)
        assert task is not None
        assert task.status == "pending"


class TestGetTask:
    """Tests for get_task method."""

    @pytest.mark.asyncio
    async def test_get_task_returns_task_info(self, task_service):
        """Test that get_task returns TaskProgressResponse."""
        from src.api.schemas import TaskProgressResponse

        task_id = await task_service.create_task(
            model_id="model_abc123",
            model_type="llm",
            operation="pull",
        )

        task = await task_service.get_task(task_id)
        assert task is not None
        assert isinstance(task, TaskProgressResponse)
        assert task.task_id == task_id
        assert task.model_id == "model_abc123"

    @pytest.mark.asyncio
    async def test_get_task_not_found(self, task_service):
        """Test that get_task returns None for non-existent task."""
        task = await task_service.get_task("task_nonexistent")
        assert task is None

    @pytest.mark.asyncio
    async def test_get_task_includes_progress(self, task_service):
        """Test that task includes progress information."""
        task_id = await task_service.create_task(
            model_id="model_abc123",
            model_type="llm",
            operation="pull",
        )

        task = await task_service.get_task(task_id)
        assert task.progress_percent == 0
        assert task.bytes_completed == 0
        assert task.bytes_total == 0
        assert task.current_step == "Initializing"


class TestUpdateTaskStatus:
    """Tests for update_task_status method."""

    @pytest.mark.asyncio
    async def test_update_status_changes_status(self, task_service):
        """Test that update_task_status changes the status."""
        task_id = await task_service.create_task(
            model_id="model_abc123",
            model_type="llm",
            operation="pull",
        )

        await task_service.update_task_status(task_id, "pulling")

        task = await task_service.get_task(task_id)
        assert task.status == "pulling"

    @pytest.mark.asyncio
    async def test_update_status_not_found(self, task_service):
        """Test update_task_status raises for non-existent task."""
        from src.services.task_tracking import TaskNotFoundError

        with pytest.raises(TaskNotFoundError):
            await task_service.update_task_status("task_nonexistent", "pulling")


class TestUpdateTaskProgress:
    """Tests for update_task_progress method."""

    @pytest.mark.asyncio
    async def test_update_progress_changes_values(self, task_service):
        """Test that update_task_progress updates progress."""
        task_id = await task_service.create_task(
            model_id="model_abc123",
            model_type="llm",
            operation="pull",
        )

        await task_service.update_task_progress(
            task_id,
            progress_percent=50,
            bytes_completed=500,
            bytes_total=1000,
            current_step="Downloading model files",
        )

        task = await task_service.get_task(task_id)
        assert task.progress_percent == 50
        assert task.bytes_completed == 500
        assert task.bytes_total == 1000
        assert task.current_step == "Downloading model files"

    @pytest.mark.asyncio
    async def test_update_progress_not_found(self, task_service):
        """Test update_task_progress raises for non-existent task."""
        from src.services.task_tracking import TaskNotFoundError

        with pytest.raises(TaskNotFoundError):
            await task_service.update_task_progress(
                "task_nonexistent",
                progress_percent=50,
                bytes_completed=500,
                bytes_total=1000,
                current_step="Test",
            )


class TestSetTaskError:
    """Tests for set_task_error method."""

    @pytest.mark.asyncio
    async def test_set_error_updates_task(self, task_service):
        """Test that set_task_error updates status and error message."""
        task_id = await task_service.create_task(
            model_id="model_abc123",
            model_type="llm",
            operation="pull",
        )

        await task_service.set_task_error(task_id, "Download failed: network error")

        task = await task_service.get_task(task_id)
        assert task.status == "error"
        assert task.error == "Download failed: network error"

    @pytest.mark.asyncio
    async def test_set_error_not_found(self, task_service):
        """Test set_task_error raises for non-existent task."""
        from src.services.task_tracking import TaskNotFoundError

        with pytest.raises(TaskNotFoundError):
            await task_service.set_task_error("task_nonexistent", "Error")


class TestCompleteTask:
    """Tests for complete_task method."""

    @pytest.mark.asyncio
    async def test_complete_task_sets_ready(self, task_service):
        """Test that complete_task sets status to ready."""
        task_id = await task_service.create_task(
            model_id="model_abc123",
            model_type="llm",
            operation="pull",
        )

        await task_service.complete_task(task_id)

        task = await task_service.get_task(task_id)
        assert task.status == "ready"
        assert task.progress_percent == 100

    @pytest.mark.asyncio
    async def test_complete_task_not_found(self, task_service):
        """Test complete_task raises for non-existent task."""
        from src.services.task_tracking import TaskNotFoundError

        with pytest.raises(TaskNotFoundError):
            await task_service.complete_task("task_nonexistent")


class TestListTasks:
    """Tests for list_tasks method."""

    @pytest.mark.asyncio
    async def test_list_tasks_empty(self, task_service):
        """Test list_tasks returns empty list when no tasks."""
        tasks = await task_service.list_tasks()
        assert tasks == []

    @pytest.mark.asyncio
    async def test_list_tasks_returns_all(self, task_service):
        """Test list_tasks returns all tasks."""
        await task_service.create_task(
            model_id="model_abc", model_type="llm", operation="pull"
        )
        await task_service.create_task(
            model_id="model_def", model_type="llm", operation="upload"
        )

        tasks = await task_service.list_tasks()
        assert len(tasks) == 2

    @pytest.mark.asyncio
    async def test_list_tasks_filter_by_model_id(self, task_service):
        """Test filtering tasks by model_id."""
        await task_service.create_task(
            model_id="model_abc", model_type="llm", operation="pull"
        )
        await task_service.create_task(
            model_id="model_def", model_type="llm", operation="pull"
        )

        tasks = await task_service.list_tasks(model_id="model_abc")
        assert len(tasks) == 1
        assert tasks[0].model_id == "model_abc"

    @pytest.mark.asyncio
    async def test_list_tasks_filter_by_status(self, task_service):
        """Test filtering tasks by status."""
        task_id = await task_service.create_task(
            model_id="model_abc", model_type="llm", operation="pull"
        )
        await task_service.update_task_status(task_id, "pulling")

        await task_service.create_task(
            model_id="model_def", model_type="llm", operation="pull"
        )

        tasks = await task_service.list_tasks(status="pulling")
        assert len(tasks) == 1
        assert tasks[0].status == "pulling"


class TestDeleteTask:
    """Tests for delete_task method."""

    @pytest.mark.asyncio
    async def test_delete_task_removes_task(self, task_service):
        """Test that delete_task removes the task."""
        task_id = await task_service.create_task(
            model_id="model_abc123",
            model_type="llm",
            operation="pull",
        )

        await task_service.delete_task(task_id)

        task = await task_service.get_task(task_id)
        assert task is None

    @pytest.mark.asyncio
    async def test_delete_task_not_found(self, task_service):
        """Test delete_task raises for non-existent task."""
        from src.services.task_tracking import TaskNotFoundError

        with pytest.raises(TaskNotFoundError):
            await task_service.delete_task("task_nonexistent")
