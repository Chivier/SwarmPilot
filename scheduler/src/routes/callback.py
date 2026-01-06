"""Callback endpoints for task result notifications.

Provides endpoints for instances to report task completion status.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from ..model import TaskResultCallbackRequest, TaskResultCallbackResponse
from .deps import get_background_scheduler, get_task_dispatcher, get_task_registry

router = APIRouter(prefix="/callback", tags=["callback"])


@router.post("/task_result", response_model=TaskResultCallbackResponse)
async def callback_task_result(request: TaskResultCallbackRequest):
    """Callback endpoint for instances to report task completion.

    This endpoint is called by instances when a task completes or fails.
    The instance sends the task result, and the scheduler updates its state
    and notifies WebSocket subscribers.

    Args:
        request: Task result callback data

    Returns:
        TaskResultCallbackResponse with acknowledgment

    Raises:
        HTTPException 404: If task not found
        HTTPException 400: If task status is invalid
    """
    task_registry = get_task_registry()
    task_dispatcher = get_task_dispatcher()
    background_scheduler = get_background_scheduler()

    # Validate task exists
    task = await task_registry.get(request.task_id)

    # TODO: Restore it after experiment
    if (
        background_scheduler.scheduling_strategy.__class__.__name__
        == "PowerOfTwoStrategy"
    ):
        request.execution_time_ms = 1.0

    if not task:
        error_msg = "Task not found"
        logger.error(f"[callback_task_result] {error_msg} | task_id={request.task_id}")
        raise HTTPException(
            status_code=404,
            detail={"success": False, "error": error_msg},
        )

    # Validate status
    if request.status not in ("completed", "failed"):
        error_msg = (
            f"Invalid status: {request.status}. Must be 'completed' or 'failed'"
        )
        logger.error(
            f"[callback_task_result] {error_msg} | task_id={request.task_id} | status={request.status}"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": error_msg,
            },
        )

    # Handle task result via task dispatcher
    await task_dispatcher.handle_task_result(
        task_id=request.task_id,
        status=request.status,
        result=request.result,
        error=request.error,
        execution_time_ms=request.execution_time_ms,
    )

    return TaskResultCallbackResponse(
        success=True,
        message="Task result received successfully",
    )
