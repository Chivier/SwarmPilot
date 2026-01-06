"""Tasks API routes.

Provides endpoints for task progress tracking.
"""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import TaskProgressResponse
from src.services.task_tracking import TaskTrackingService

router = APIRouter(tags=["tasks"])

# Task tracking service instance (set during app startup)
_task_tracking_service: TaskTrackingService | None = None


def set_task_tracking_service(service: TaskTrackingService) -> None:
    """Set the task tracking service instance.

    Args:
        service: TaskTrackingService instance to use.
    """
    global _task_tracking_service
    _task_tracking_service = service


def get_task_tracking_service() -> TaskTrackingService:
    """Get the task tracking service instance.

    Returns:
        The TaskTrackingService instance.

    Raises:
        RuntimeError: If task tracking service not initialized.
    """
    if _task_tracking_service is None:
        raise RuntimeError("Task tracking service not initialized")
    return _task_tracking_service


class TaskListResponse:
    """Response for task listing."""

    def __init__(self, tasks: list[TaskProgressResponse]):
        """Initialize with tasks list."""
        self.tasks = tasks


@router.get("", response_model=dict)
async def list_tasks(
    status: Annotated[str | None, Query()] = None,
    model_id: Annotated[str | None, Query()] = None,
) -> dict:
    """List tasks with optional filtering.

    Args:
        status: Filter by task status.
        model_id: Filter by model ID.

    Returns:
        Task list response with tasks array.
    """
    service = get_task_tracking_service()
    tasks = await service.list_tasks(model_id=model_id, status=status)

    return {"tasks": [task.model_dump() for task in tasks]}


@router.get("/{task_id}", response_model=TaskProgressResponse)
async def get_task(task_id: str) -> TaskProgressResponse:
    """Get task progress information.

    Args:
        task_id: The task identifier.

    Returns:
        Task progress response.

    Raises:
        HTTPException: 404 if task not found.
    """
    service = get_task_tracking_service()

    task = await service.get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "task_not_found",
                "message": f"Task not found: {task_id}",
            },
        )

    return task
