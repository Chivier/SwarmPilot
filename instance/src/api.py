"""
Instance Service API

This module implements the FastAPI application for the Instance Service.
It provides endpoints for model management, task management, and instance monitoring.
"""

import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Response, status
from pydantic import BaseModel, Field
from loguru import logger

from .config import config
from .docker_manager import get_docker_manager
from .model_registry import get_registry
from .models import InstanceStatus, Task, TaskStatus
from .task_queue import get_task_queue
from .scheduler_client import get_scheduler_client
from . import logger as _  # Import logger module to initialize logging

# =============================================================================
# Pydantic Models for Request/Response Schemas
# =============================================================================


# Model Management Schemas
class ModelStartRequest(BaseModel):
    """Request schema for starting a model"""
    model_id: str = Field(..., description="Unique identifier of the model/tool to start")
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Model-specific initialization parameters"
    )
    scheduler_url: Optional[str] = Field(
        None,
        description="Scheduler URL to register with (updates current scheduler if provided)"
    )


class ModelStartResponse(BaseModel):
    """Response schema for model start endpoint"""
    success: bool
    message: str
    model_id: str
    status: str


class ModelStopResponse(BaseModel):
    """Response schema for model stop endpoint"""
    success: bool
    message: str
    model_id: str


# Task Management Schemas
class TaskSubmitRequest(BaseModel):
    """Request schema for submitting a task"""
    task_id: str = Field(..., description="Unique identifier for this task")
    model_id: str = Field(..., description="Model/tool ID to use for this task")
    task_input: Dict[str, Any] = Field(..., description="Model-specific input data")


class TaskSubmitResponse(BaseModel):
    """Response schema for task submission"""
    success: bool
    message: str
    task_id: str
    status: str
    position: int = Field(..., description="Position in the queue")


class TaskDetail(BaseModel):
    """Schema for detailed task information"""
    task_id: str
    model_id: str
    status: str = Field(..., description="Status: queued, running, completed, failed")
    task_input: Optional[Dict[str, Any]] = None
    submitted_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TaskListResponse(BaseModel):
    """Response schema for task list endpoint"""
    success: bool
    total: int
    tasks: List[TaskDetail]


class TaskGetResponse(BaseModel):
    """Response schema for getting a specific task"""
    success: bool
    task: TaskDetail


class TaskDeleteResponse(BaseModel):
    """Response schema for task deletion"""
    success: bool
    message: str
    task_id: str


# Management Schemas
class ModelInfo(BaseModel):
    """Current model information"""
    model_id: str
    started_at: str
    parameters: Dict[str, Any]


class TaskQueueStats(BaseModel):
    """Task queue statistics"""
    total: int
    queued: int
    running: int
    completed: int
    failed: int


class InstanceInfo(BaseModel):
    """Instance information"""
    instance_id: str
    status: str = Field(..., description="Status: idle, running, busy, error")
    current_model: Optional[ModelInfo] = None
    task_queue: TaskQueueStats
    uptime: int = Field(..., description="Uptime in seconds")
    version: str


class InfoResponse(BaseModel):
    """Response schema for info endpoint"""
    success: bool
    instance: InstanceInfo


class HealthResponse(BaseModel):
    """Response schema for health check endpoint"""
    status: str
    timestamp: str
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response format"""
    success: bool = False
    error: str


# =============================================================================
# FastAPI Application
# =============================================================================

# Track startup time for uptime calculation
_startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan: startup and shutdown events."""
    # Startup
    global _startup_time
    _startup_time = time.time()

    logger.info("Instance Service starting...")
    logger.info(f"Instance ID: {config.instance_id}")
    logger.info(f"Instance Port: {config.instance_port}")
    logger.info(f"Model Port: {config.model_port}")
    logger.info(f"Registry Path: {config.registry_path}")

    # Load model registry
    try:
        registry = get_registry()
        logger.info(f"Loaded {len(registry.models)} models from registry")
    except Exception as e:
        logger.warning(f"Failed to load model registry: {e}")

    yield

    # Shutdown
    logger.info("Instance Service shutting down...")

    # Deregister from scheduler if enabled
    scheduler_client = get_scheduler_client()
    if scheduler_client.is_enabled:
        try:
            await scheduler_client.deregister_instance()
        except Exception as e:
            logger.warning(f"Failed to deregister from scheduler: {str(e)}")

    # Stop task processing
    task_queue = get_task_queue()
    await task_queue.stop_processing()

    # Stop running model
    docker_manager = get_docker_manager()
    if await docker_manager.is_model_running():
        await docker_manager.stop_model()

    # Close HTTP client
    await docker_manager.close()

    logger.info("Instance Service stopped")


app = FastAPI(
    title="Instance Service API",
    description="API for managing model instances and task queues in SwarmPilot",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Model Management Endpoints
# =============================================================================

@app.post(
    "/model/start",
    response_model=ModelStartResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def start_model(request: ModelStartRequest):
    """
    Start a model/tool on this instance.

    Each instance can only serve one model at a time. If a model is already
    running, it must be stopped before starting a new one.

    If scheduler_url is provided, the instance will update its scheduler configuration
    and register with the new scheduler instead of the current one.
    """
    docker_manager = get_docker_manager()
    registry = get_registry()

    # Check if another model is already running
    if await docker_manager.is_model_running():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Another model is already running. Stop the current model first."
        )

    # Validate model exists in registry
    if not registry.model_exists(request.model_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model not found in registry: {request.model_id}"
        )

    # Start the model
    try:
        model_info = await docker_manager.start_model(
            request.model_id,
            request.parameters
        )

        # Update scheduler URL if provided and register with scheduler
        scheduler_client = get_scheduler_client()

        # If new scheduler_url is provided, update the scheduler configuration
        if request.scheduler_url:
            logger.info(f"Updating scheduler URL to: {request.scheduler_url}")
            scheduler_client.scheduler_url = request.scheduler_url
            scheduler_client._registered = False  # Reset registration status

        # Register with scheduler if enabled
        if scheduler_client.is_enabled:
            try:
                await scheduler_client.register_instance(model_id=request.model_id)
                logger.info(f"Successfully registered with scheduler: {scheduler_client.scheduler_url}")
            except Exception as e:
                # Log error but don't fail model start
                logger.warning(f"Failed to register with scheduler: {str(e)}")

        return ModelStartResponse(
            success=True,
            message="Model started successfully",
            model_id=model_info.model_id,
            status="running"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start model: {str(e)}"
        )


@app.get(
    "/model/stop",
    response_model=ModelStopResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse},
    },
)
async def stop_model():
    """
    Stop the currently running model on this instance.

    This will gracefully shutdown the model container and free up resources.
    """
    docker_manager = get_docker_manager()

    # Check if a model is running
    if not await docker_manager.is_model_running():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model is currently running"
        )

    # Stop the model
    try:
        model_id = await docker_manager.stop_model()

        # Deregister from scheduler if enabled
        scheduler_client = get_scheduler_client()
        if scheduler_client.is_enabled:
            try:
                await scheduler_client.deregister_instance()
            except Exception as e:
                # Log error but don't fail model stop
                logger.warning(f"Failed to deregister from scheduler: {str(e)}")

        return ModelStopResponse(
            success=True,
            message="Model stopped successfully",
            model_id=model_id or "unknown"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop model: {str(e)}"
        )


# =============================================================================
# Task Management Endpoints
# =============================================================================

@app.post(
    "/task/submit",
    response_model=TaskSubmitResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def submit_task(request: TaskSubmitRequest):
    """
    Submit a new task to the task queue.

    Tasks are processed sequentially in FIFO order. The task will be queued
    and executed when all previous tasks are completed.
    """
    task_queue = get_task_queue()
    docker_manager = get_docker_manager()

    # Check if model is running
    if not await docker_manager.is_model_running():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="No model is currently running"
        )

    # Validate model_id matches currently running model
    current_model = await docker_manager.get_current_model()
    if current_model and current_model.model_id != request.model_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Model ID does not match the currently running model. "
                   f"Expected: {current_model.model_id}, Got: {request.model_id}"
        )

    # Create task
    task = Task(
        task_id=request.task_id,
        model_id=request.model_id,
        task_input=request.task_input
    )

    # Submit to queue
    try:
        position = await task_queue.submit_task(task)

        return TaskSubmitResponse(
            success=True,
            message="Task submitted successfully",
            task_id=task.task_id,
            status=task.status.value,
            position=position
        )
    except ValueError as e:
        logger.error(f"Task submission error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.get(
    "/task/list",
    response_model=TaskListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_tasks(
    status_filter: Optional[str] = Query(
        None,
        alias="status",
        description="Filter by status: queued, running, completed, failed"
    ),
    limit: int = Query(
        100,
        description="Maximum number of tasks to return",
        ge=1,
        le=1000
    )
):
    """
    List all tasks and their current status in the queue.

    Results can be filtered by status and limited in count.
    """
    task_queue = get_task_queue()

    # Parse status filter
    status_enum = None
    if status_filter:
        try:
            status_enum = TaskStatus(status_filter)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}. "
                       f"Valid values: queued, running, completed, failed"
            )

    # Get tasks
    tasks = await task_queue.list_tasks(status_filter=status_enum, limit=limit)

    # Convert to response format
    task_details = [
        TaskDetail(
            task_id=task.task_id,
            model_id=task.model_id,
            status=task.status.value,
            task_input=task.task_input,
            submitted_at=task.submitted_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            result=task.result,
            error=task.error
        )
        for task in tasks
    ]

    return TaskListResponse(
        success=True,
        total=len(task_details),
        tasks=task_details
    )


@app.get(
    "/task/{task_id}",
    response_model=TaskGetResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse},
    },
)
async def get_task(task_id: str):
    """
    Get detailed information about a specific task.

    Returns the complete task information including input, output, timestamps,
    and status.
    """
    task_queue = get_task_queue()

    # Get task
    task = await task_queue.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    # Convert to response format
    task_detail = TaskDetail(
        task_id=task.task_id,
        model_id=task.model_id,
        status=task.status.value,
        task_input=task.task_input,
        submitted_at=task.submitted_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        result=task.result,
        error=task.error
    )

    return TaskGetResponse(
        success=True,
        task=task_detail
    )


@app.delete(
    "/task/{task_id}",
    response_model=TaskDeleteResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def delete_task(task_id: str):
    """
    Cancel a queued task or remove a completed/failed task from the list.

    Running tasks cannot be cancelled. Only queued tasks can be cancelled,
    and completed/failed tasks can be removed from history.
    """
    task_queue = get_task_queue()

    # Delete task
    try:
        deleted = await task_queue.delete_task(task_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )

        return TaskDeleteResponse(
            success=True,
            message="Task cancelled/removed successfully",
            task_id=task_id
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# =============================================================================
# Management Endpoints
# =============================================================================

@app.get(
    "/info",
    response_model=InfoResponse,
    status_code=status.HTTP_200_OK,
)
async def get_info():
    """
    Get current status and information about this instance.

    Returns comprehensive information including current model, task queue
    statistics, and instance metadata.
    """
    docker_manager = get_docker_manager()
    task_queue = get_task_queue()

    # Get current model info
    current_model = await docker_manager.get_current_model()
    current_model_info = None
    if current_model:
        current_model_info = ModelInfo(
            model_id=current_model.model_id,
            started_at=current_model.started_at,
            parameters=current_model.parameters
        )

    # Get task queue stats
    stats = await task_queue.get_queue_stats()
    task_queue_stats = TaskQueueStats(**stats)

    # Determine instance status
    if current_model is None:
        instance_status = InstanceStatus.IDLE
    elif stats["running"] > 0:
        instance_status = InstanceStatus.BUSY
    else:
        instance_status = InstanceStatus.RUNNING

    # Calculate uptime
    uptime = int(time.time() - _startup_time)

    # Build response
    instance_info = InstanceInfo(
        instance_id=config.instance_id,
        status=instance_status.value,
        current_model=current_model_info,
        task_queue=task_queue_stats,
        uptime=uptime,
        version="1.0.0"
    )

    return InfoResponse(
        success=True,
        instance=instance_info
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    responses={
        503: {"model": HealthResponse},
    },
)
async def health_check(response: Response):
    """
    Health check endpoint for monitoring and load balancing.

    Returns 200 if the instance is healthy and ready to accept requests.
    Returns 503 if the instance is unhealthy or not ready.
    """
    docker_manager = get_docker_manager()

    # Check if model is running
    if await docker_manager.is_model_running():
        # Check model health
        if not await docker_manager.check_model_health():
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return HealthResponse(
                status="unhealthy",
                error="Model container is not responding",
                timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z")
            )

    # Instance is healthy
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # Logger is already initialized by the import at the top
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.instance_port,
        log_config=None  # Disable uvicorn's default logging config, let loguru handle it
    )
