"""
Instance Service API

This module implements the FastAPI application for the Instance Service.
It provides endpoints for model management, task management, and instance monitoring.
"""

import asyncio
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Response, status
from pydantic import BaseModel, Field
from loguru import logger


def log_error_with_traceback(
    error: Exception,
    context: str,
    client_message: str,
) -> None:
    """
    Log error with detailed information, client message, and traceback.

    Args:
        error: The exception that occurred
        context: Context description of where the error occurred
        client_message: The message that will be returned to the client
    """
    tb_str = traceback.format_exc()
    logger.error(
        f"[{context}] Error occurred:\n"
        f"  Internal error: {type(error).__name__}: {error}\n"
        f"  Client message: {client_message}\n"
        f"  Traceback:\n{tb_str}"
    )

from .config import config
from .manager_factory import get_docker_manager
from .model_registry import get_registry
from .models import InstanceStatus, Task, TaskStatus, RestartOperation, RestartStatus, DeregisterOperation, DeregisterStatus
from .task_queue import get_task_queue
from .scheduler_client import get_scheduler_client, _get_gpu0_name
from .websocket_client import WebSocketClient
from .websocket_client_singleton import get_websocket_client, set_websocket_client
from . import logger as _  # Import logger module to initialize logging

# =============================================================================
# Pydantic Models for Request/Response Schemas
# =============================================================================


# Model Management Schemas
class StandbyConfig(BaseModel):
    """Configuration for standby (hot-standby) behavior"""
    port_offset: Optional[int] = Field(
        default=None,
        description="Port offset for standby process (default: from env INSTANCE_STANDBY_PORT_OFFSET or 1000)"
    )
    max_retries: Optional[int] = Field(
        default=None,
        description="Maximum retries for standby startup (default: from env INSTANCE_HOT_STANDBY_MAX_RETRIES or 3)"
    )
    initial_delay: Optional[float] = Field(
        default=None,
        description="Initial delay before retry in seconds (default: from env INSTANCE_HOT_STANDBY_INITIAL_DELAY or 5.0)"
    )
    max_delay: Optional[float] = Field(
        default=None,
        description="Maximum delay between retries in seconds (default: from env INSTANCE_HOT_STANDBY_MAX_DELAY or 30.0)"
    )
    backoff_multiplier: Optional[float] = Field(
        default=None,
        description="Backoff multiplier for retry delays (default: from env INSTANCE_HOT_STANDBY_BACKOFF_MULTIPLIER or 2.0)"
    )
    restart_delay: Optional[int] = Field(
        default=None,
        description="Delay before restarting standby after hot-switch in seconds (default: from env INSTANCE_STANDBY_RESTART_DELAY or 30)"
    )
    health_check_timeout: Optional[int] = Field(
        default=None,
        description="Health check timeout for standby process in seconds (default: from env INSTANCE_BACKUP_HEALTH_TIMEOUT or 600)"
    )
    traditional_restart_delay: Optional[int] = Field(
        default=None,
        description="Delay for traditional restart when standby disabled in seconds (default: from env INSTANCE_TRADITIONAL_RESTART_DELAY or 30)"
    )


class ModelStartRequest(BaseModel):
    """Request schema for starting a model"""
    model_id: str = Field(..., description="Unique identifier of the model/tool to start")
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Model-specific initialization parameters"
    )
    scheduler_url: str = Field(..., description="Scheduler URL to register with (must not be empty)")
    standby: Optional[bool] = Field(
        default=None,
        description="Enable/disable standby mode. If None, uses env INSTANCE_STANDBY_ENABLED (default: true)"
    )
    standby_config: Optional[StandbyConfig] = Field(
        default=None,
        description="Standby configuration overrides. Values override environment variable defaults."
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


class ModelRestartRequest(BaseModel):
    """Request schema for restarting a model"""
    model_id: str = Field(..., description="Unique identifier of the new model/tool to start")
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Model-specific initialization parameters"
    )
    scheduler_url: str = Field(..., description="New scheduler URL to register with (must not be empty)")


class ModelRestartResponse(BaseModel):
    """Response schema for model restart endpoint"""
    success: bool
    message: str
    operation_id: str = Field(..., description="Unique ID to track this restart operation")
    status: str = Field(..., description="Initial status (should be 'pending')")


class RestartStatusResponse(BaseModel):
    """Response schema for restart status endpoint"""
    success: bool
    operation_id: str
    status: str = Field(..., description="Current restart status")
    old_model_id: Optional[str] = None
    new_model_id: str
    initiated_at: str
    completed_at: Optional[str] = None
    pending_tasks_at_start: int
    pending_tasks_completed: int
    redistributed_tasks_count: int = Field(
        default=0,
        description="Number of pending tasks redistributed to scheduler"
    )
    error: Optional[str] = None


# Task Management Schemas
class TaskSubmitRequest(BaseModel):
    """Request schema for submitting a task"""
    task_id: str = Field(..., description="Unique identifier for this task")
    model_id: str = Field(..., description="Model/tool ID to use for this task")
    task_input: Dict[str, Any] = Field(..., description="Model-specific input data")
    enqueue_time: Optional[float] = Field(None, description="Optional Unix timestamp for task priority ordering")
    callback_url: Optional[str] = Field(None, description="Optional callback URL for task result")


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


class TaskClearResponse(BaseModel):
    """Response schema for clearing all tasks"""
    success: bool
    message: str
    cleared_count: Dict[str, int] = Field(
        ...,
        description="Number of tasks cleared by status (queued, completed, failed, total)"
    )


class TaskFetchResponse(BaseModel):
    """Response schema for fetching the first queued task with full details"""
    exist: bool = Field(..., description="Whether a task was found and fetched")
    task_id: str = Field(..., description="ID of the fetched task, empty string if none")
    # Additional fields for task redistribution (optional for backward compatibility)
    model_id: Optional[str] = Field(None, description="Model ID of the fetched task")
    task_input: Optional[Dict[str, Any]] = Field(None, description="Task input data")
    enqueue_time: Optional[float] = Field(None, description="Original enqueue timestamp for priority ordering")
    submitted_at: Optional[str] = Field(None, description="Original submission timestamp (ISO 8601)")


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


class StandbyInfo(BaseModel):
    """Hot-standby port information"""
    enabled: bool = Field(..., description="Whether hot-standby is enabled")
    primary_port: Optional[int] = Field(None, description="Current primary (active) port")
    standby_port: Optional[int] = Field(None, description="Current standby port")
    standby_ready: bool = Field(False, description="Whether standby is healthy and ready for hot-switch")
    standby_state: Optional[str] = Field(None, description="Standby port state: uninitialized, starting, healthy, stopping, stopped, failed")


class InstanceInfo(BaseModel):
    """Instance information"""
    instance_id: str
    status: str = Field(..., description="Status: idle, running, busy, error")
    current_model: Optional[ModelInfo] = None
    task_queue: TaskQueueStats
    uptime: int = Field(..., description="Uptime in seconds")
    version: str
    hardware_name: Optional[str] = Field(None, description="Hardware name (e.g., GPU name)")
    software_name: Optional[str] = Field(None, description="Software platform name (e.g., sglang, vllm)")
    software_version: Optional[str] = Field(None, description="Software platform version")
    active_port: Optional[int] = Field(None, description="Currently active model port")
    standby: Optional[StandbyInfo] = Field(None, description="Hot-standby port information (SubprocessManager only)")


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


class SuccessResponse(BaseModel):
    """Standard success response format"""
    success: bool = True
    message: str

class ModelRegisterRequest(BaseModel):
    """Model register request format"""
    scheduler_url: str = Field(..., description="Scheduler URL to register with")


class ModelRegisterResponse(BaseModel):
    """Response schema for model register endpoint"""
    success: bool = True
    message: str = ""


class DeregisterStatusResponse(BaseModel):
    """Response schema for deregister status endpoint"""
    success: bool
    operation_id: str
    status: str = Field(..., description="Current deregister status")
    old_model_id: Optional[str] = None
    initiated_at: str
    completed_at: Optional[str] = None
    pending_tasks_at_start: int
    pending_tasks_completed: int
    redistributed_tasks_count: int = Field(
        default=0,
        description="Number of pending tasks redistributed to scheduler"
    )
    error: Optional[str] = None


class ModelDeregisterResponse(BaseModel):
    """Response schema for synchronous model deregister endpoint"""
    success: bool
    message: str
    model_id: Optional[str] = Field(None, description="The model ID that was deregistered")
    redistributed_tasks_count: int = Field(
        default=0,
        description="Number of pending tasks redistributed to scheduler"
    )

# =============================================================================
# FastAPI Application
# =============================================================================

# Track startup time for uptime calculation
_startup_time = time.time()

# Track restart operations
_restart_operations: Dict[str, RestartOperation] = {}
_deregister_operations: Dict[str, DeregisterOperation] = {}
_restart_operation_lock = asyncio.Lock()
_deregister_operation_lock = asyncio.Lock()


def construct_websocket_url(http_url: str) -> str:
    """
    Construct WebSocket URL from HTTP scheduler URL.

    WebSocket server runs on HTTP port + 1.
    Example: http://scheduler:8000 -> ws://scheduler:8001/instance/ws

    Args:
        http_url: HTTP URL of the scheduler (e.g., "http://scheduler:8000")

    Returns:
        WebSocket URL (e.g., "ws://scheduler:8001/instance/ws")
    """
    import re

    # Convert protocol
    ws_url = http_url.replace("http://", "ws://").replace("https://", "wss://")

    # Parse and increment port
    match = re.match(r"(wss?://[^:]+):(\d+)", ws_url)
    if match:
        protocol_host = match.group(1)
        http_port = int(match.group(2))
        ws_port = http_port + 1
        return f"{protocol_host}:{ws_port}/instance/ws"
    else:
        # No port specified, add default WebSocket port (8001)
        return f"{ws_url}:8001/instance/ws"


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


    logger.info("Using HTTP-only communication with Scheduler (WebSocket disabled)")

    yield

    # Shutdown
    logger.info("Instance Service shutting down...")

    # Deregister from scheduler if enabled (HTTP fallback)
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

    The scheduler_url must be provided and cannot be empty.
    """
    # Validate scheduler_url is not empty
    if not request.scheduler_url or request.scheduler_url.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="scheduler_url is required and cannot be empty"
        )

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

    # Build standby configuration dict from request (only include non-None values)
    standby_config_dict = None
    if request.standby_config is not None:
        standby_config_dict = {
            k: v for k, v in request.standby_config.model_dump().items()
            if v is not None
        }

    # Start the model
    try:
        model_info = await docker_manager.start_model(
            request.model_id,
            request.parameters,
            standby_enabled=request.standby,
            standby_config=standby_config_dict
        )

        software_name = request.parameters.get("software_name", None)
        software_version = request.parameters.get("software_version", None)
        hardware_name = request.parameters.get("hardware_name", None)
        
        if software_name:
            config.platform_software_name = software_name
        if software_version:
            config.platform_software_version = software_version
        if hardware_name:
            config.platform_hardware_name = hardware_name

        # Update scheduler URL and register with scheduler
        scheduler_client = get_scheduler_client()

        # Update the scheduler configuration
        logger.info(f"Updating scheduler URL to: {request.scheduler_url}")
        scheduler_client.scheduler_url = request.scheduler_url
        scheduler_client._registered = False  # Reset registration status

        # Register with scheduler if enabled
        if scheduler_client.is_enabled:
            try:
                await scheduler_client.register_instance(model_id=request.model_id)
                logger.info(f"Successfully registered with scheduler via HTTP: {scheduler_client.scheduler_url}")
            except Exception as e:
                # Log error but don't fail model start
                logger.warning(f"Failed to register with scheduler via HTTP: {str(e)}")

        return ModelStartResponse(
            success=True,
            message="Model started successfully",
            model_id=model_info.model_id,
            status="running"
        )
    except Exception as e:
        client_message = f"Failed to start model: {str(e)}"
        log_error_with_traceback(
            error=e,
            context=f"start_model(model_id={request.model_id})",
            client_message=client_message,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=client_message
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
        client_message = f"Failed to stop model: {str(e)}"
        log_error_with_traceback(
            error=e,
            context="stop_model",
            client_message=client_message,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=client_message
        )

@app.post(
    "/model/deregister",
    response_model=ModelDeregisterResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def deregister_model():
    """
    Deregister the current model from the scheduler synchronously.

    This is a **blocking** operation that:
    1. Drains from the current scheduler (stops accepting new tasks)
    2. Extracts pending queued tasks and redistributes them to scheduler
    3. Waits for all currently running tasks to complete
    4. Deregisters from the current scheduler

    The endpoint will block until all running tasks are completed and the
    deregister operation is finished. Use this when you need to ensure
    the instance is fully drained before proceeding.
    """

    docker_manager = get_docker_manager()

    # Check if a model is running
    if not await docker_manager.is_model_running():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model is currently running"
        )

    # Get current model info
    current_model = await docker_manager.get_current_model()
    model_id = current_model.model_id if current_model else None

    # Check if there's already a deregister operation in progress
    async with _deregister_operation_lock:
        for op in _deregister_operations.values():
            if op.status not in (DeregisterStatus.COMPLETED, DeregisterStatus.FAILED):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"A deregister operation is already in progress (operation_id: {op.operation_id})"
                )

        # Create deregister operation for tracking
        operation_id = str(uuid.uuid4())
        operation = DeregisterOperation(
            operation_id=operation_id,
            old_model_id=model_id,
        )
        operation.update_status(DeregisterStatus.DRAINING)
        _deregister_operations[operation_id] = operation

    logger.info(f"Starting synchronous deregister operation {operation_id} for model: {model_id}")

    try:
        scheduler_client = get_scheduler_client()
        task_queue = get_task_queue()
        redistributed_tasks_count = 0

        # Step 1: Drain from scheduler (if enabled)
        if scheduler_client.is_enabled:
            try:
                await scheduler_client.drain_instance()
                logger.info("Instance draining from scheduler")
            except Exception as e:
                logger.warning(f"Failed to drain from scheduler: {str(e)}")
        else:
            logger.info("Scheduler integration disabled, skipping drain step")

        # Step 2: Extract pending queued tasks and redistribute to scheduler
        operation.update_status(DeregisterStatus.EXTRACTING_TASKS)

        if scheduler_client.is_enabled:
            try:
                pending_tasks = await task_queue.extract_pending_tasks()
                logger.info(f"Extracted {len(pending_tasks)} pending tasks from queue")

                successful_redistributions = 0
                failed_redistributions = 0

                for task_data in pending_tasks:
                    try:
                        success = await scheduler_client.resubmit_task(
                            task_id=task_data["task_id"],
                            original_instance_id=config.instance_id,
                        )

                        if success:
                            successful_redistributions += 1
                            logger.debug(f"Successfully redistributed task {task_data['task_id']}")
                        else:
                            failed_redistributions += 1
                            logger.warning(f"Failed to redistribute task {task_data['task_id']}")

                    except Exception as e:
                        failed_redistributions += 1
                        logger.error(f"Error redistributing task {task_data['task_id']}: {str(e)}")

                redistributed_tasks_count = successful_redistributions
                operation.redistributed_tasks_count = successful_redistributions

                if failed_redistributions > 0:
                    logger.warning(
                        f"Task redistribution: {successful_redistributions} succeeded, "
                        f"{failed_redistributions} failed"
                    )
                else:
                    logger.info(f"Successfully redistributed all {successful_redistributions} tasks")

            except Exception as e:
                logger.error(f"Failed to extract and redistribute tasks: {str(e)}")
        else:
            logger.info("Scheduler integration disabled, skipping task extraction")

        # Step 3: Wait for currently running task to complete
        operation.update_status(DeregisterStatus.WAITING_RUNNING_TASK)
        stats = await task_queue.get_queue_stats()

        if stats["running"] > 0:
            logger.info(f"Waiting for {stats['running']} running task(s) to complete (no timeout)")

            while True:
                stats = await task_queue.get_queue_stats()
                running = stats["running"]

                if running == 0:
                    logger.info("All running tasks completed")
                    break

                await asyncio.sleep(1)
        else:
            logger.info("No running tasks to wait for")

        # Step 4: Deregister from scheduler
        operation.update_status(DeregisterStatus.DEREGISTERING)
        if scheduler_client.is_enabled:
            try:
                await scheduler_client.deregister_instance()
                logger.info("Successfully deregistered from scheduler")
            except Exception as e:
                logger.warning(f"Failed to deregister from scheduler: {str(e)}")

        # Mark operation as completed
        operation.update_status(DeregisterStatus.COMPLETED)
        logger.info(f"Synchronous deregister operation {operation_id} completed successfully")

        return ModelDeregisterResponse(
            success=True,
            message="Model deregistered successfully",
            model_id=model_id,
            redistributed_tasks_count=redistributed_tasks_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        tb_str = traceback.format_exc()
        logger.error(
            f"Deregister operation {operation_id} failed:\n"
            f"  Internal error: {type(e).__name__}: {error_msg}\n"
            f"  Traceback:\n{tb_str}"
        )
        async with _deregister_operation_lock:
            op = _deregister_operations.get(operation_id)
            if op:
                op.update_status(DeregisterStatus.FAILED, error=error_msg)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deregister operation failed: {error_msg}"
        )


@app.get(
    "/model/deregister/status",
    response_model=DeregisterStatusResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse},
    },
)
async def get_deregister_status(operation_id: str = Query(..., description="Deregister operation ID")):
    """
    Get the status of a model deregister operation.

    Returns the current state of the deregister operation including progress,
    status, and any errors that occurred.
    """
    async with _deregister_operation_lock:
        operation = _deregister_operations.get(operation_id)

    if not operation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deregister operation not found: {operation_id}"
        )

    return DeregisterStatusResponse(
        success=True,
        operation_id=operation.operation_id,
        status=operation.status.value,
        old_model_id=operation.old_model_id,
        initiated_at=operation.initiated_at,
        completed_at=operation.completed_at,
        pending_tasks_at_start=operation.pending_tasks_at_start,
        pending_tasks_completed=operation.pending_tasks_completed,
        redistributed_tasks_count=operation.redistributed_tasks_count,
        error=operation.error,
    )


@app.post(
    "/model/register",
    response_model=ModelRegisterResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse},
    },
)
async def register_model(request: ModelRegisterRequest):
    """
    Register current instance to a specific scheduler without stopping the model.

    This allows an instance to dynamically register to a new scheduler while
    keeping the current model running.

    Requirements:
    - A model must be currently running
    - The scheduler_url must be provided
    """
    docker_manager = get_docker_manager()
    scheduler_client = get_scheduler_client()

    # Check if a model is running
    if not await docker_manager.is_model_running():
        logger.error("No model is currently running")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model is currently running. Start a model first."
        )

    # Validate scheduler_url is not empty
    if not request.scheduler_url or request.scheduler_url.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="scheduler_url is required and cannot be empty"
        )

    # Get current model info
    current_model = await docker_manager.get_current_model()
    if not current_model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to get current model information"
        )

    # Update scheduler URL
    logger.info(f"Updating scheduler URL to: {request.scheduler_url}")
    scheduler_client.scheduler_url = request.scheduler_url
    scheduler_client._registered = False  # Reset registration status

    # Register with scheduler
    if scheduler_client.is_enabled:
        try:
            success = await scheduler_client.register_instance(model_id=current_model.model_id)
            if success:
                logger.info(f"Successfully registered with scheduler: {request.scheduler_url}")
                return ModelRegisterResponse(
                    success=True,
                    message=f"Successfully registered model '{current_model.model_id}' with scheduler at {request.scheduler_url}"
                )
            else:
                logger.warning(f"Failed to register with scheduler: {request.scheduler_url}")
                return ModelRegisterResponse(
                    success=False,
                    message=f"Failed to register with scheduler at {request.scheduler_url}"
                )
        except Exception as e:
            client_message = f"Failed to register with scheduler: {str(e)}"
            log_error_with_traceback(
                error=e,
                context=f"register_model(scheduler_url={request.scheduler_url})",
                client_message=client_message,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=client_message
            )

    return ModelRegisterResponse(
        success=False,
        message="Scheduler client is not enabled"
    )


@app.post(
    "/model/restart",
    response_model=ModelRestartResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
async def restart_model(request: ModelRestartRequest):
    """
    Restart the model with a new model and scheduler.

    This is a non-blocking operation that:
    1. Drains from the current scheduler (stops accepting new tasks)
    2. Waits for all pending tasks to complete
    3. Stops the current model
    4. Deregisters from the current scheduler
    5. Starts the new model
    6. Registers with the new scheduler

    The scheduler_url must be provided and cannot be empty.
    The operation runs in the background. Use GET /model/restart/status to monitor progress.
    """
    # Validate scheduler_url is not empty
    if not request.scheduler_url or request.scheduler_url.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="scheduler_url is required and cannot be empty"
        )

    docker_manager = get_docker_manager()
    registry = get_registry()

    # Check if a model is running
    if not await docker_manager.is_model_running():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model is currently running"
        )

    # Validate new model exists in registry
    if not registry.model_exists(request.model_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model not found in registry: {request.model_id}"
        )

    # Get current model info
    current_model = await docker_manager.get_current_model()

    # Check if there's already a restart operation in progress and create new operation atomically
    async with _restart_operation_lock:
        for op in _restart_operations.values():
            if op.status not in (RestartStatus.COMPLETED, RestartStatus.FAILED):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"A restart operation is already in progress (operation_id: {op.operation_id})"
                )

        # Create restart operation
        operation_id = str(uuid.uuid4())
        operation = RestartOperation(
            operation_id=operation_id,
            old_model_id=current_model.model_id if current_model else None,
            new_model_id=request.model_id,
            new_parameters=request.parameters or {},
            new_scheduler_url=request.scheduler_url,
        )

        # Mark operation as in progress immediately to prevent concurrent restarts
        operation.update_status(RestartStatus.DRAINING)

        # Store operation (in same lock block to ensure atomicity)
        _restart_operations[operation_id] = operation

    # Start background task
    asyncio.create_task(_perform_restart_operation(operation_id))

    logger.info(
        f"Restart operation {operation_id} initiated: "
        f"{operation.old_model_id} -> {operation.new_model_id}"
    )

    return ModelRestartResponse(
        success=True,
        message="Model restart operation initiated",
        operation_id=operation_id,
        status=operation.status.value,
    )


async def _perform_restart_operation(operation_id: str):
    """
    Background task to perform the model restart operation.

    This function executes the following steps:
    1. Drain from scheduler (if enabled)
    2. Extract pending queued tasks and redistribute to scheduler
    3. Wait for currently running task to complete
    4. Stop current model
    5. Deregister from current scheduler
    6. Start new model
    7. Register with new scheduler (if URL provided)

    Args:
        operation_id: Unique identifier for this restart operation
    """
    async with _restart_operation_lock:
        operation = _restart_operations.get(operation_id)
        if not operation:
            logger.error(f"Restart operation {operation_id} not found")
            return

    try:
        logger.info(f"Starting restart operation {operation_id}")
        docker_manager = get_docker_manager()
        scheduler_client = get_scheduler_client()
        task_queue = get_task_queue()

        # Step 1: Drain from scheduler (if enabled)
        operation.update_status(RestartStatus.DRAINING)
        if scheduler_client.is_enabled:
            try:
                await scheduler_client.drain_instance()
                logger.info(f"Instance draining from scheduler")
            except Exception as e:
                # Log warning but continue - instance might already be draining or not registered
                logger.warning(f"Failed to drain from scheduler: {str(e)}")
        else:
            logger.info("Scheduler integration disabled, skipping drain step")

        # Step 2: Extract pending queued tasks and redistribute to scheduler
        operation.update_status(RestartStatus.EXTRACTING_TASKS)

        if scheduler_client.is_enabled:
            try:
                # Extract all pending QUEUED tasks (preserves running task)
                pending_tasks = await task_queue.extract_pending_tasks()
                logger.info(f"Extracted {len(pending_tasks)} pending tasks from queue")

                # Redistribute tasks back to scheduler
                successful_redistributions = 0
                failed_redistributions = 0

                for task_data in pending_tasks:
                    try:
                        success = await scheduler_client.resubmit_task(
                            task_id=task_data["task_id"],
                            original_instance_id=config.instance_id,
                        )

                        if success:
                            successful_redistributions += 1
                            logger.debug(f"Successfully redistributed task {task_data['task_id']}")
                        else:
                            failed_redistributions += 1
                            logger.warning(f"Failed to redistribute task {task_data['task_id']}")

                    except Exception as e:
                        failed_redistributions += 1
                        logger.error(f"Error redistributing task {task_data['task_id']}: {str(e)}")

                operation.redistributed_tasks_count = successful_redistributions

                if failed_redistributions > 0:
                    logger.warning(
                        f"Task redistribution: {successful_redistributions} succeeded, "
                        f"{failed_redistributions} failed"
                    )
                else:
                    logger.info(f"Successfully redistributed all {successful_redistributions} tasks")

            except Exception as e:
                logger.error(f"Failed to extract and redistribute tasks: {str(e)}")
                # Continue with restart even if redistribution fails
        else:
            logger.info("Scheduler integration disabled, skipping task extraction")

        # Step 3: Wait for currently running task to complete
        operation.update_status(RestartStatus.WAITING_RUNNING_TASK)
        stats = await task_queue.get_queue_stats()

        # Only count running task (queued tasks have been extracted)
        if stats["running"] > 0:
            logger.info(f"Waiting for {stats['running']} running task to complete")

            # Poll task queue until running task completes
            max_wait_time = 300  # 5 minutes timeout
            start_wait_time = time.time()

            while True:
                stats = await task_queue.get_queue_stats()
                running = stats["running"]

                if running == 0:
                    logger.info("Running task completed")
                    break

                # Check timeout
                if time.time() - start_wait_time > max_wait_time:
                    raise TimeoutError(
                        f"Timeout waiting for running task to complete. "
                        f"Task still running after {max_wait_time}s"
                    )

                # Wait a bit before checking again
                await asyncio.sleep(1)
        else:
            logger.info("No running tasks to wait for")

        # Step 4: Stop current model
        operation.update_status(RestartStatus.STOPPING_MODEL)
        if await docker_manager.is_model_running():
            old_model_id = await docker_manager.stop_model()
            operation.old_model_id = old_model_id
            logger.info(f"Stopped old model: {old_model_id}")
        else:
            logger.warning("No model was running")

        # Step 5: Deregister from current scheduler
        operation.update_status(RestartStatus.DEREGISTERING)
        if scheduler_client.is_enabled:
            try:
                await scheduler_client.deregister_instance()
                logger.info("Deregistered from scheduler")
            except Exception as e:
                # Log warning but continue
                logger.warning(f"Failed to deregister: {str(e)}")

        # Step 6: Start new model
        operation.update_status(RestartStatus.STARTING_MODEL)
        registry = get_registry()

        # Validate model exists
        if not registry.model_exists(operation.new_model_id):
            raise ValueError(f"Model not found in registry: {operation.new_model_id}")

        model_info = await docker_manager.start_model(
            operation.new_model_id,
            operation.new_parameters
        )
        logger.info(f"Started new model: {operation.new_model_id}")

        # Step 7: Register with new scheduler
        operation.update_status(RestartStatus.REGISTERING)

        # Update scheduler URL (guaranteed to be non-empty due to validation)
        logger.info(f"Updating scheduler URL to: {operation.new_scheduler_url}")
        scheduler_client.scheduler_url = operation.new_scheduler_url
        scheduler_client._registered = False

        # Register with scheduler if enabled
        if scheduler_client.is_enabled:
            try:
                await scheduler_client.register_instance(model_id=operation.new_model_id)
                logger.info(f"Registered with scheduler: {scheduler_client.scheduler_url}")
            except Exception as e:
                # Log warning but mark operation as completed
                logger.warning(f"Failed to register with scheduler: {str(e)}")

        # Mark operation as completed
        operation.update_status(RestartStatus.COMPLETED)
        logger.info(f"Restart operation {operation_id} completed successfully")

    except Exception as e:
        error_msg = str(e)
        tb_str = traceback.format_exc()
        logger.error(
            f"Restart operation {operation_id} failed:\n"
            f"  Internal error: {type(e).__name__}: {error_msg}\n"
            f"  Traceback:\n{tb_str}"
        )
        async with _restart_operation_lock:
            operation = _restart_operations.get(operation_id)
            if operation:
                operation.update_status(RestartStatus.FAILED, error=error_msg)


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
        logger.warning("No model is currently running")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="No model is currently running"
        )

    # Validate model_id matches currently running model
    current_model = await docker_manager.get_current_model()
    if current_model and current_model.model_id != request.model_id:
        logger.warning(f"Model ID does not match the currently running model. "
                       f"Expected: {current_model.model_id}, Got: {request.model_id}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Model ID does not match the currently running model. "
                   f"Expected: {current_model.model_id}, Got: {request.model_id}"
        )

    # Create task
    task = Task(
        task_id=request.task_id,
        model_id=request.model_id,
        task_input=request.task_input,
        callback_url=request.callback_url
    )

    # Submit to queue with optional enqueue_time for priority ordering
    try:
        position = await task_queue.submit_task(task, enqueue_time=request.enqueue_time)

        return TaskSubmitResponse(
            success=True,
            message="Task submitted successfully",
            task_id=task.task_id,
            status=task.status.value,
            position=position
        )
    except ValueError as e:
        client_message = str(e)
        log_error_with_traceback(
            error=e,
            context=f"submit_task(task_id={request.task_id})",
            client_message=client_message,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=client_message
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
    "/task/fetch",
    response_model=TaskFetchResponse,
    status_code=status.HTTP_200_OK,
)
async def fetch_task():
    """
    Fetch the first queued task from the task queue.

    This endpoint allows clients to retrieve the oldest queued task
    (lowest enqueue_time) from the instance's task queue. The fetched
    task is marked as FETCHED and will not be executed by this instance.
    No callback will be sent for fetched tasks.

    Use this for work redistribution scenarios where another instance
    or external client needs to take over pending tasks.

    Returns:
        - exist: True if a task was fetched, False if no eligible task
        - task_id: The ID of the fetched task, or empty string if none
    """
    task_queue = get_task_queue()

    # Fetch the oldest queued task
    task = await task_queue.fetch_task()

    if task:
        return TaskFetchResponse(
            exist=True,
            task_id=task.task_id,
            model_id=task.model_id,
            task_input=task.task_input,
            enqueue_time=task.enqueue_time,
            submitted_at=task.submitted_at,
        )
    else:
        return TaskFetchResponse(
            exist=False,
            task_id=""
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
        client_message = str(e)
        log_error_with_traceback(
            error=e,
            context=f"delete_task(task_id={task_id})",
            client_message=client_message,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=client_message
        )


@app.post(
    "/task/clear",
    response_model=TaskClearResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse},
    },
)
async def clear_tasks():
    """
    Clear all tasks from the instance.

    This will:
    1. Restart the Docker container (if a model is running)
    2. Remove all tasks from the queue and task storage, regardless
       of their status (queued, completed, failed)

    Running tasks cannot be cleared - the endpoint will return an error
    if there are running tasks.

    This is useful for:
    - Resetting the model state by restarting the container
    - Cleaning up completed/failed task history
    - Resetting the instance state before starting new work
    - Removing queued tasks that are no longer needed

    Returns the count of tasks that were cleared by status.
    """
    task_queue = get_task_queue()
    docker_manager = get_docker_manager()

    try:
        # Step 1: Clear all tasks
        cleared_count = await task_queue.clear_all_tasks()
        
        # Step 2: Restart Docker container if a model is running
        if await docker_manager.is_model_running():
            logger.info("Restarting Docker container before clearing tasks")
            try:
                model_id = await docker_manager.restart_model()
                if model_id:
                    logger.info(f"Successfully restarted Docker container for model: {model_id}")
            except Exception as e:
                client_message = f"Failed to restart Docker container: {str(e)}"
                log_error_with_traceback(
                    error=e,
                    context="clear_tasks/restart_container",
                    client_message=client_message,
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=client_message
                )

        return TaskClearResponse(
            success=True,
            message=f"Successfully cleared {cleared_count['total']} task(s)",
            cleared_count=cleared_count
        )
    except RuntimeError as e:
        client_message = str(e)
        log_error_with_traceback(
            error=e,
            context="clear_tasks",
            client_message=client_message,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=client_message
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
        # Sanitize parameters to ensure JSON-serializable values only
        # Some parameters may contain non-serializable objects (locks, handles, etc.)
        import json
        try:
            # Deep copy and filter to only JSON-serializable values
            sanitized_params = json.loads(json.dumps(current_model.parameters, default=str))
        except (TypeError, ValueError):
            # If serialization fails, use empty dict as fallback
            sanitized_params = {}
            logger.warning("Failed to serialize model parameters, using empty dict")

        current_model_info = ModelInfo(
            model_id=current_model.model_id,
            started_at=current_model.started_at,
            parameters=sanitized_params
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
    
    # Get platform info (hardware and software)
    # Priority: config overrides > env vars > model params > auto-detection
    try:
        import platform as platform_module

        # Start with auto-detected values
        hardware_name = _get_gpu0_name()
        software_name = platform_module.system()
        software_version = platform_module.release()

        # Apply environment variable overrides (deprecated, kept for backward compatibility)
        import os
        env_software_name = os.getenv("INSTANCE_SOFTWARE_NAME")
        env_software_version = os.getenv("INSTANCE_SOFTWARE_VERSION")

        if env_software_name:
            software_name = env_software_name
        if env_software_version:
            software_version = env_software_version

        # If a model is running, try to infer software from model metadata
        # (e.g., if model uses sglang, vllm, etc.)
        if current_model and current_model.parameters:
            model_params = current_model.parameters
            if "software_name" in model_params:
                software_name = model_params["software_name"]
            if "software_version" in model_params:
                software_version = model_params["software_version"]

        # Finally, apply config overrides (highest priority)
        # These come from INSTANCE_PLATFORM_* env vars or CLI args
        if config.platform_software_name:
            software_name = config.platform_software_name
        if config.platform_software_version:
            software_version = config.platform_software_version
        if config.platform_hardware_name:
            hardware_name = config.platform_hardware_name

    except Exception as e:
        logger.warning(f"Failed to detect platform info: {e}")
        hardware_name = None
        software_name = None
        software_version = None

    # Get hot-standby information (SubprocessManager only)
    standby_info = None
    active_port = None

    # Check if using SubprocessManager with hot-standby support
    from .subprocess_manager import SubprocessManager
    if isinstance(docker_manager, SubprocessManager):
        active_port = docker_manager.active_port

        # Get dual-port state if available
        dual_state = docker_manager._dual_port_state
        if dual_state is not None:
            standby_info = StandbyInfo(
                enabled=True,
                primary_port=dual_state.primary.port,
                standby_port=dual_state.standby.port,
                standby_ready=docker_manager.is_standby_ready(),
                standby_state=dual_state.standby.state.value
            )
        else:
            standby_info = StandbyInfo(
                enabled=False,
                standby_ready=False
            )

    # Build response
    instance_info = InstanceInfo(
        instance_id=config.instance_id,
        status=instance_status.value,
        current_model=current_model_info,
        task_queue=task_queue_stats,
        uptime=uptime,
        version="1.0.0",
        hardware_name=hardware_name,
        software_name=software_name,
        software_version=software_version,
        active_port=active_port,
        standby=standby_info
    )

    return InfoResponse(
        success=True,
        instance=instance_info
    )


@app.get(
    "/model/restart/status",
    response_model=RestartStatusResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse},
    },
)
async def get_restart_status(operation_id: str = Query(..., description="Restart operation ID")):
    """
    Get the status of a model restart operation.

    Returns the current state of the restart operation including progress,
    status, and any errors that occurred.
    """
    async with _restart_operation_lock:
        operation = _restart_operations.get(operation_id)

    if not operation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Restart operation not found: {operation_id}"
        )

    return RestartStatusResponse(
        success=True,
        operation_id=operation.operation_id,
        status=operation.status.value,
        old_model_id=operation.old_model_id,
        new_model_id=operation.new_model_id,
        initiated_at=operation.initiated_at,
        completed_at=operation.completed_at,
        pending_tasks_at_start=operation.pending_tasks_at_start,
        pending_tasks_completed=operation.pending_tasks_completed,
        redistributed_tasks_count=operation.redistributed_tasks_count,
        error=operation.error,
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
