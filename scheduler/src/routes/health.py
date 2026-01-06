"""Health check endpoint.

Provides a health check endpoint for monitoring the scheduler service status.
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException
from loguru import logger

from ..model import HealthResponse, HealthStats, TaskStatus
from .deps import get_instance_registry, get_task_registry

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify scheduler status.

    Returns:
        HealthResponse with status and statistics

    Returns 503 if service is unhealthy
    """
    instance_registry = get_instance_registry()
    task_registry = get_task_registry()

    try:
        # Collect statistics
        stats = HealthStats(
            total_instances=await instance_registry.get_total_count(),
            active_instances=await instance_registry.get_active_count(),
            total_tasks=await task_registry.get_total_count(),
            pending_tasks=await task_registry.get_count_by_status(
                TaskStatus.PENDING
            ),
            running_tasks=await task_registry.get_count_by_status(
                TaskStatus.RUNNING
            ),
            completed_tasks=await task_registry.get_count_by_status(
                TaskStatus.COMPLETED
            ),
            failed_tasks=await task_registry.get_count_by_status(TaskStatus.FAILED),
        )

        return HealthResponse(
            success=True,
            status="healthy",
            timestamp=datetime.now().isoformat() + "Z",
            version="1.0.0",
            stats=stats,
        )

    except Exception as e:
        # Log the health check failure
        error_msg = str(e)
        logger.error(
            f"[health_check] Health check failed | error={error_msg}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "status": "unhealthy",
                "error": error_msg,
                "timestamp": datetime.now().isoformat() + "Z",
            },
        ) from e
