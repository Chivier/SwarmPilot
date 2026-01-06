"""Task tracking service for managing background tasks.

Handles task creation, progress tracking, and status updates for
long-running operations like model pulls and uploads.
"""

import uuid
from dataclasses import dataclass, field

from src.api.schemas import TaskProgressResponse


class TaskNotFoundError(Exception):
    """Raised when a requested task does not exist."""

    def __init__(self, task_id: str):
        """Initialize with task ID.

        Args:
            task_id: The ID of the task that was not found.
        """
        self.task_id = task_id
        super().__init__(f"Task not found: {task_id}")


@dataclass
class TaskState:
    """Internal state for a task.

    Attributes:
        task_id: Unique task identifier.
        model_id: Associated model identifier.
        model_type: Type of the model.
        operation: Type of operation (pull, upload).
        status: Current task status.
        progress_percent: Progress percentage (0-100).
        bytes_completed: Bytes processed so far.
        bytes_total: Total bytes to process.
        current_step: Human-readable current step description.
        error: Error message if task failed.
    """

    task_id: str
    model_id: str
    model_type: str
    operation: str
    status: str = "pending"
    progress_percent: int = 0
    bytes_completed: int = 0
    bytes_total: int = 0
    current_step: str = "Initializing"
    error: str | None = None


class TaskTrackingService:
    """Service for tracking background tasks.

    Provides in-memory task state management for operations like
    model pulling and uploading. Tasks can be queried for progress
    and updated by background workers.

    Attributes:
        _tasks: Dictionary mapping task IDs to task state.
    """

    def __init__(self):
        """Initialize the task tracking service."""
        self._tasks: dict[str, TaskState] = {}

    def _generate_task_id(self) -> str:
        """Generate a unique task ID with prefix.

        Returns:
            A unique task ID in the format 'task_<uuid>'.
        """
        return f"task_{uuid.uuid4().hex[:12]}"

    async def create_task(
        self,
        model_id: str,
        model_type: str,
        operation: str,
    ) -> str:
        """Create a new task.

        Args:
            model_id: Associated model identifier.
            model_type: Type of the model.
            operation: Type of operation (pull, upload).

        Returns:
            The generated task_id.
        """
        task_id = self._generate_task_id()
        self._tasks[task_id] = TaskState(
            task_id=task_id,
            model_id=model_id,
            model_type=model_type,
            operation=operation,
        )
        return task_id

    async def get_task(self, task_id: str) -> TaskProgressResponse | None:
        """Retrieve a task by ID.

        Args:
            task_id: The task identifier.

        Returns:
            TaskProgressResponse or None if not found.
        """
        state = self._tasks.get(task_id)
        if state is None:
            return None

        return TaskProgressResponse(
            task_id=state.task_id,
            model_id=state.model_id,
            type=state.model_type,
            operation=state.operation,
            status=state.status,
            progress_percent=state.progress_percent,
            current_step=state.current_step,
            bytes_completed=state.bytes_completed,
            bytes_total=state.bytes_total,
            error=state.error,
        )

    async def update_task_status(self, task_id: str, status: str) -> None:
        """Update a task's status.

        Args:
            task_id: The task identifier.
            status: New status value.

        Raises:
            TaskNotFoundError: If task doesn't exist.
        """
        if task_id not in self._tasks:
            raise TaskNotFoundError(task_id)

        self._tasks[task_id].status = status

    async def update_task_progress(
        self,
        task_id: str,
        progress_percent: int,
        bytes_completed: int,
        bytes_total: int,
        current_step: str,
    ) -> None:
        """Update a task's progress.

        Args:
            task_id: The task identifier.
            progress_percent: Progress percentage (0-100).
            bytes_completed: Bytes processed so far.
            bytes_total: Total bytes to process.
            current_step: Human-readable current step.

        Raises:
            TaskNotFoundError: If task doesn't exist.
        """
        if task_id not in self._tasks:
            raise TaskNotFoundError(task_id)

        state = self._tasks[task_id]
        state.progress_percent = progress_percent
        state.bytes_completed = bytes_completed
        state.bytes_total = bytes_total
        state.current_step = current_step

    async def set_task_error(self, task_id: str, error: str) -> None:
        """Set a task as failed with error message.

        Args:
            task_id: The task identifier.
            error: Error message describing the failure.

        Raises:
            TaskNotFoundError: If task doesn't exist.
        """
        if task_id not in self._tasks:
            raise TaskNotFoundError(task_id)

        state = self._tasks[task_id]
        state.status = "error"
        state.error = error

    async def complete_task(self, task_id: str) -> None:
        """Mark a task as completed.

        Args:
            task_id: The task identifier.

        Raises:
            TaskNotFoundError: If task doesn't exist.
        """
        if task_id not in self._tasks:
            raise TaskNotFoundError(task_id)

        state = self._tasks[task_id]
        state.status = "ready"
        state.progress_percent = 100

    async def list_tasks(
        self,
        model_id: str | None = None,
        status: str | None = None,
    ) -> list[TaskProgressResponse]:
        """List tasks with optional filtering.

        Args:
            model_id: Filter by model ID.
            status: Filter by status.

        Returns:
            List of TaskProgressResponse objects.
        """
        tasks = []

        for state in self._tasks.values():
            # Apply filters
            if model_id and state.model_id != model_id:
                continue
            if status and state.status != status:
                continue

            tasks.append(
                TaskProgressResponse(
                    task_id=state.task_id,
                    model_id=state.model_id,
                    type=state.model_type,
                    operation=state.operation,
                    status=state.status,
                    progress_percent=state.progress_percent,
                    current_step=state.current_step,
                    bytes_completed=state.bytes_completed,
                    bytes_total=state.bytes_total,
                    error=state.error,
                )
            )

        return tasks

    async def delete_task(self, task_id: str) -> None:
        """Delete a task.

        Args:
            task_id: The task identifier.

        Raises:
            TaskNotFoundError: If task doesn't exist.
        """
        if task_id not in self._tasks:
            raise TaskNotFoundError(task_id)

        del self._tasks[task_id]
