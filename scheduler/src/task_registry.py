"""
Task registry for managing task lifecycle and state.

This module provides thread-safe storage and management of tasks
throughout their execution lifecycle.
"""

from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime

from .model import TaskStatus, TaskTimestamps


class TaskRecord:
    """Internal representation of a task with full state."""

    def __init__(
        self,
        task_id: str,
        model_id: str,
        task_input: Dict[str, Any],
        metadata: Dict[str, Any],
        assigned_instance: str,
        predicted_time_ms: Optional[float] = None,
        predicted_error_margin_ms: Optional[float] = None,
        predicted_quantiles: Optional[Dict[float, float]] = None,
    ):
        self.task_id = task_id
        self.model_id = model_id
        self.task_input = task_input
        self.metadata = metadata
        self.assigned_instance = assigned_instance
        self.status = TaskStatus.PENDING
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None

        # Prediction information (for queue updates on task completion)
        self.predicted_time_ms = predicted_time_ms
        self.predicted_error_margin_ms = predicted_error_margin_ms
        self.predicted_quantiles = predicted_quantiles

        # Timestamps
        self.submitted_at = datetime.now().isoformat() + "Z"
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None

        # Actual execution time reported by instance (more accurate than timestamp calculation)
        self._actual_execution_time_ms: Optional[float] = None

    @property
    def execution_time_ms(self) -> Optional[float]:
        """
        Get execution time in milliseconds.

        Returns the actual execution time reported by the instance if available,
        otherwise calculates from timestamps.
        """
        # Prefer actual execution time reported by instance
        if self._actual_execution_time_ms is not None:
            return self._actual_execution_time_ms

        # Fallback to timestamp-based calculation
        if self.started_at and self.completed_at:
            try:
                start = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
                end = datetime.fromisoformat(self.completed_at.replace("Z", "+00:00"))
                return (end - start).total_seconds() * 1000
            except Exception:
                return None
        return None

    def set_execution_time(self, execution_time_ms: float) -> None:
        """
        Set the actual execution time reported by the instance.

        Args:
            execution_time_ms: Actual execution time in milliseconds
        """
        self._actual_execution_time_ms = execution_time_ms

    def get_timestamps(self) -> TaskTimestamps:
        """Get task timestamps as model."""
        return TaskTimestamps(
            submitted_at=self.submitted_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
        )


class TaskRegistry:
    """Thread-safe registry for managing tasks."""

    def __init__(self):
        self._tasks: Dict[str, TaskRecord] = {}
        self._lock = asyncio.Lock()

    async def create_task(
        self,
        task_id: str,
        model_id: str,
        task_input: Dict[str, Any],
        metadata: Dict[str, Any],
        assigned_instance: str,
        predicted_time_ms: Optional[float] = None,
        predicted_error_margin_ms: Optional[float] = None,
        predicted_quantiles: Optional[Dict[float, float]] = None,
    ) -> TaskRecord:
        """
        Create and register a new task.

        Args:
            task_id: Unique task identifier
            model_id: Model/tool to use
            task_input: Input data for the task
            metadata: Metadata for prediction
            assigned_instance: Instance assigned to execute this task
            predicted_time_ms: Predicted execution time in milliseconds
            predicted_error_margin_ms: Predicted error margin for expect_error strategy
            predicted_quantiles: Predicted quantiles for probabilistic strategy

        Returns:
            Created task record

        Raises:
            ValueError: If task with this ID already exists
        """
        async with self._lock:
            if task_id in self._tasks:
                raise ValueError(f"Task {task_id} already exists")

            task = TaskRecord(
                task_id=task_id,
                model_id=model_id,
                task_input=task_input,
                metadata=metadata,
                assigned_instance=assigned_instance,
                predicted_time_ms=predicted_time_ms,
                predicted_error_margin_ms=predicted_error_margin_ms,
                predicted_quantiles=predicted_quantiles,
            )
            self._tasks[task_id] = task
            return task

    async def get(self, task_id: str) -> Optional[TaskRecord]:
        """
        Get a task by ID.

        Args:
            task_id: ID of task to retrieve

        Returns:
            TaskRecord if found, None otherwise
        """
        async with self._lock:
            return self._tasks.get(task_id)

    async def list_all(
        self,
        status: Optional[TaskStatus] = None,
        model_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[List[TaskRecord], int]:
        """
        List tasks with optional filtering and pagination.

        Args:
            status: Optional status filter
            model_id: Optional model ID filter
            instance_id: Optional instance ID filter
            limit: Maximum number of tasks to return
            offset: Pagination offset

        Returns:
            Tuple of (filtered tasks, total count)
        """
        async with self._lock:
            tasks = list(self._tasks.values())

            # Apply filters
            if status:
                tasks = [t for t in tasks if t.status == status]
            if model_id:
                tasks = [t for t in tasks if t.model_id == model_id]
            if instance_id:
                tasks = [t for t in tasks if t.assigned_instance == instance_id]

            total = len(tasks)

            # Sort by submission time (newest first)
            tasks.sort(key=lambda t: t.submitted_at, reverse=True)

            # Apply pagination
            tasks = tasks[offset : offset + limit]

            return tasks, total

    async def update_status(self, task_id: str, status: TaskStatus) -> None:
        """
        Update task status.

        Args:
            task_id: ID of task to update
            status: New status

        Raises:
            KeyError: If task not found
        """
        async with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"Task {task_id} not found")

            task = self._tasks[task_id]
            task.status = status

            # Update timestamps
            if status == TaskStatus.RUNNING and task.started_at is None:
                task.started_at = datetime.now().isoformat() + "Z"
            elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                if task.completed_at is None:
                    task.completed_at = datetime.now().isoformat() + "Z"

    async def set_result(self, task_id: str, result: Dict[str, Any]) -> None:
        """
        Set task result.

        Args:
            task_id: ID of task
            result: Task result data

        Raises:
            KeyError: If task not found
        """
        async with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"Task {task_id} not found")

            self._tasks[task_id].result = result

    async def set_error(self, task_id: str, error: str) -> None:
        """
        Set task error.

        Args:
            task_id: ID of task
            error: Error message

        Raises:
            KeyError: If task not found
        """
        async with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"Task {task_id} not found")

            self._tasks[task_id].error = error

    async def update_assigned_instance(self, task_id: str, new_instance_id: str) -> None:
        """
        Update the assigned instance for a task.

        This is used during work stealing when a task is moved from one
        instance to another.

        Args:
            task_id: ID of task to update
            new_instance_id: ID of the new assigned instance

        Raises:
            KeyError: If task not found
        """
        async with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"Task {task_id} not found")

            self._tasks[task_id].assigned_instance = new_instance_id

    async def update_metadata(self, task_id: str, metadata: Dict[str, Any]) -> TaskRecord:
        """
        Update metadata for a task (replaces existing metadata).

        Args:
            task_id: ID of task to update
            metadata: New metadata dictionary (replaces existing)

        Returns:
            Updated TaskRecord

        Raises:
            KeyError: If task not found
        """
        async with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"Task {task_id} not found")

            task = self._tasks[task_id]
            task.metadata = metadata
            return task

    async def update_prediction(
        self,
        task_id: str,
        predicted_time_ms: Optional[float],
        predicted_error_margin_ms: Optional[float] = None,
        predicted_quantiles: Optional[Dict[float, float]] = None,
    ) -> None:
        """
        Update prediction fields for a task.

        Args:
            task_id: ID of task to update
            predicted_time_ms: New predicted execution time in milliseconds
            predicted_error_margin_ms: New error margin (for expect_error strategy)
            predicted_quantiles: New quantile predictions (for probabilistic strategy)

        Raises:
            KeyError: If task not found
        """
        async with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"Task {task_id} not found")

            task = self._tasks[task_id]
            task.predicted_time_ms = predicted_time_ms
            task.predicted_error_margin_ms = predicted_error_margin_ms
            task.predicted_quantiles = predicted_quantiles

    async def get_count_by_status(self, status: TaskStatus) -> int:
        """
        Get count of tasks with specific status.

        Args:
            status: Status to count

        Returns:
            Count of tasks with this status
        """
        async with self._lock:
            return sum(1 for t in self._tasks.values() if t.status == status)

    async def get_total_count(self) -> int:
        """Get total number of tasks."""
        async with self._lock:
            return len(self._tasks)

    async def clear_all(self) -> int:
        """
        Clear all tasks from the registry.

        Returns:
            Count of tasks that were cleared
        """
        async with self._lock:
            count = len(self._tasks)
            self._tasks.clear()
            return count

    async def reset_for_resubmit(self, task_id: str) -> TaskRecord:
        """
        Reset a task for resubmission by clearing result/error and timestamps.

        This method is used during instance migration to prepare a task for
        rescheduling to a new instance.

        Args:
            task_id: ID of task to reset

        Returns:
            The reset TaskRecord

        Raises:
            KeyError: If task not found
        """
        async with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"Task {task_id} not found")

            task = self._tasks[task_id]

            # Clear result and error
            task.result = None
            task.error = None

            # Reset timestamps (keep submitted_at)
            task.started_at = None
            task.completed_at = None

            # Clear actual execution time
            task._actual_execution_time_ms = None

            # Clear assigned instance (will be reassigned)
            task.assigned_instance = ""

            # Reset status to PENDING
            task.status = TaskStatus.PENDING

            return task
