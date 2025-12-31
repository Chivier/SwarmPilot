"""Scheduler-compatible dummy endpoints.

These endpoints provide compatibility with the Scheduler interface,
allowing instances registered to Planner to complete their deregister
flow without errors.
"""

from fastapi import APIRouter
from loguru import logger

from ..models import (
    InstanceDrainRequest,
    InstanceDrainResponse,
    InstanceDrainStatusResponse,
    InstanceRemoveRequest,
    InstanceRemoveResponse,
    InstanceStatus,
    TaskResubmitRequest,
    TaskResubmitResponse,
)

router = APIRouter(tags=["scheduler_compat"])


@router.post("/instance/drain", response_model=InstanceDrainResponse)
async def dummy_drain_instance(request: InstanceDrainRequest):
    """Dummy drain endpoint for instances registered to Planner.

    Always returns success since Planner doesn't manage task queues.
    This allows instances registered to Planner to complete their
    deregister flow without errors.

    Args:
        request: Instance drain request with instance_id

    Returns:
        InstanceDrainResponse with success status
    """
    logger.info(f"[Dummy] Drain requested for instance: {request.instance_id}")
    return InstanceDrainResponse(
        success=True,
        message=f"Instance {request.instance_id} drain acknowledged (Planner dummy)",
        instance_id=request.instance_id,
        status=InstanceStatus.DRAINING,
        pending_tasks=0,
        running_tasks=0,
        estimated_completion_time_ms=0.0,
    )


@router.get("/instance/drain/status", response_model=InstanceDrainStatusResponse)
async def dummy_drain_status(instance_id: str):
    """Dummy drain status endpoint for instances registered to Planner.

    Always returns can_remove=True since Planner doesn't manage task queues.

    Args:
        instance_id: Instance ID to check drain status

    Returns:
        InstanceDrainStatusResponse with can_remove=True
    """
    logger.info(f"[Dummy] Drain status check for instance: {instance_id}")
    return InstanceDrainStatusResponse(
        success=True,
        instance_id=instance_id,
        status=InstanceStatus.REMOVING,
        pending_tasks=0,
        running_tasks=0,
        can_remove=True,
        drain_initiated_at=None,
    )


@router.post("/instance/remove", response_model=InstanceRemoveResponse)
async def dummy_remove_instance(request: InstanceRemoveRequest):
    """Dummy remove endpoint for instances registered to Planner.

    Always returns success. This allows instances registered to Planner
    to complete their deregister flow without errors.

    Args:
        request: Instance remove request with instance_id

    Returns:
        InstanceRemoveResponse with success status
    """
    logger.info(f"[Dummy] Remove requested for instance: {request.instance_id}")
    return InstanceRemoveResponse(
        success=True,
        message=f"Instance {request.instance_id} removed (Planner dummy)",
        instance_id=request.instance_id,
    )


@router.post("/task/resubmit", response_model=TaskResubmitResponse)
async def dummy_resubmit_task(request: TaskResubmitRequest):
    """Dummy task resubmit endpoint for instances registered to Planner.

    Always returns success. Planner doesn't manage task queues,
    so resubmission is a no-op.

    Args:
        request: Task resubmit request with task_id and original_instance_id

    Returns:
        TaskResubmitResponse with success status
    """
    logger.info(
        f"[Dummy] Task resubmit requested: task={request.task_id}, "
        f"original_instance={request.original_instance_id}"
    )
    return TaskResubmitResponse(
        success=True,
        message=f"Task {request.task_id} resubmit acknowledged (Planner dummy)",
    )
