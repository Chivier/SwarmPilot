"""
FastAPI application for the scheduler service.

This module defines all API endpoints for instance management, task scheduling,
and WebSocket connections for real-time task result delivery.
"""

from typing import Optional
from datetime import datetime
import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query

from .model import (
    # Instance models
    InstanceRegisterRequest,
    InstanceRegisterResponse,
    InstanceRemoveRequest,
    InstanceRemoveResponse,
    InstanceListResponse,
    InstanceInfoResponse,
    Instance,
    InstanceStats,
    InstanceQueueBase,
    # Task models
    TaskSubmitRequest,
    TaskSubmitResponse,
    TaskListResponse,
    TaskDetailResponse,
    TaskStatus,
    TaskInfo,
    TaskSummary,
    TaskDetailInfo,
    TaskTimestamps,
    # Health models
    HealthResponse,
    HealthErrorResponse,
    HealthStats,
    # WebSocket models
    WSSubscribeMessage,
    WSUnsubscribeMessage,
    WSAckMessage,
    WSTaskResultMessage,
    WSErrorMessage,
    WSMessageType,
    # Common models
    ErrorResponse,
)

from .instance_registry import InstanceRegistry
from .task_registry import TaskRegistry
from .websocket_manager import ConnectionManager
from .predictor_client import PredictorClient
from .scheduler import get_strategy
from .task_dispatcher import TaskDispatcher


# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="Scheduler API",
    description="Task scheduling and instance management service",
    version="1.0.0",
)

# Global state - TODO: Consider dependency injection for better testability
instance_registry = InstanceRegistry()
task_registry = TaskRegistry()
websocket_manager = ConnectionManager()

# TODO: Get predictor URL from configuration
predictor_client = PredictorClient(predictor_url="http://predictor:8001")

# Initialize task dispatcher
task_dispatcher = TaskDispatcher(
    task_registry=task_registry,
    instance_registry=instance_registry,
    websocket_manager=websocket_manager,
)

# TODO: Get scheduling strategy from configuration
scheduling_strategy = get_strategy("probabilistic")


# ============================================================================
# Instance Management Endpoints
# ============================================================================

@app.post("/instance/register", response_model=InstanceRegisterResponse)
async def register_instance(request: InstanceRegisterRequest):
    """
    Register a new instance to the scheduler.

    Args:
        request: Instance registration details

    Returns:
        InstanceRegisterResponse with registration status and instance info

    Raises:
        HTTPException 400: If instance with this ID already exists
    """
    # Check if instance already exists
    if instance_registry.get(request.instance_id):
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": "Instance with this ID already exists"},
        )

    # TODO: Optional - validate that the endpoint is reachable
    # try:
    #     async with httpx.AsyncClient(timeout=5.0) as client:
    #         response = await client.get(f"{request.endpoint}/health")
    #         response.raise_for_status()
    # except Exception:
    #     raise HTTPException(
    #         status_code=400,
    #         detail={"success": False, "error": "Instance endpoint is not reachable"},
    #     )

    # Create Instance object
    instance = Instance(
        instance_id=request.instance_id,
        model_id=request.model_id,
        endpoint=request.endpoint,
    )

    # Register instance (this also initializes queue info and stats)
    try:
        instance_registry.register(instance)
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail={"success": False, "error": str(e)}
        )

    return InstanceRegisterResponse(
        success=True,
        message="Instance registered successfully",
        instance=instance,
    )


@app.post("/instance/remove", response_model=InstanceRemoveResponse)
async def remove_instance(request: InstanceRemoveRequest):
    """
    Remove an instance from the scheduler.

    Args:
        request: Instance removal request with instance_id

    Returns:
        InstanceRemoveResponse with removal status

    Raises:
        HTTPException 404: If instance not found
    """
    # Check if instance exists
    if not instance_registry.get(request.instance_id):
        raise HTTPException(
            status_code=404,
            detail={"success": False, "error": "Instance not found"},
        )

    # TODO: Optional - check if instance has pending tasks and warn or reject
    stats = instance_registry.get_stats(request.instance_id)
    if stats and stats.pending_tasks > 0:
        # For now, we allow removal even with pending tasks
        # In production, you might want to:
        # - Reject the removal
        # - Reassign pending tasks to other instances
        # - Wait for tasks to complete
        pass

    # Remove instance
    try:
        instance_registry.remove(request.instance_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={"success": False, "error": "Instance not found"},
        )

    return InstanceRemoveResponse(
        success=True,
        message="Instance removed successfully",
        instance_id=request.instance_id,
    )


@app.get("/instance/list", response_model=InstanceListResponse)
async def list_instances(model_id: Optional[str] = Query(None)):
    """
    List all registered instances with optional filtering.

    Args:
        model_id: Optional filter by model ID

    Returns:
        InstanceListResponse with list of instances
    """
    # Retrieve instances (with optional model_id filter)
    instances = instance_registry.list_all(model_id=model_id)

    return InstanceListResponse(
        success=True,
        count=len(instances),
        instances=instances,
    )


@app.get("/instance/info", response_model=InstanceInfoResponse)
async def get_instance_info(instance_id: str = Query(...)):
    """
    Get detailed information about a specific instance.

    Args:
        instance_id: ID of the instance to query

    Returns:
        InstanceInfoResponse with instance details, queue info, and stats

    Raises:
        HTTPException 404: If instance not found
    """
    # Get instance
    instance = instance_registry.get(instance_id)
    if not instance:
        raise HTTPException(
            status_code=404,
            detail={"success": False, "error": "Instance not found"},
        )

    # Get queue information
    queue_info = instance_registry.get_queue_info(instance_id)
    if not queue_info:
        # Shouldn't happen, but handle gracefully
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error": "Queue info not found"},
        )

    # Get statistics
    stats = instance_registry.get_stats(instance_id)
    if not stats:
        # Shouldn't happen, but handle gracefully
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error": "Statistics not found"},
        )

    return InstanceInfoResponse(
        success=True,
        instance=instance,
        queue_info=queue_info,
        stats=stats,
    )


# ============================================================================
# Task Management Endpoints
# ============================================================================

@app.post("/task/submit", response_model=TaskSubmitResponse)
async def submit_task(request: TaskSubmitRequest):
    """
    Submit a new task for execution.

    Args:
        request: Task submission details

    Returns:
        TaskSubmitResponse with task status and assigned instance

    Raises:
        HTTPException 400: If task with this ID already exists
        HTTPException 404: If no available instance for the model_id
    """
    # Validate that task doesn't already exist
    if task_registry.get(request.task_id):
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": "Task with this ID already exists"},
        )

    # Find available instances for the model
    available_instances = instance_registry.list_all(model_id=request.model_id)

    if not available_instances:
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": f"No available instance for model_id: {request.model_id}",
            },
        )

    # Get predictions from predictor
    instance_ids = [inst.instance_id for inst in available_instances]

    # TODO: Handle predictor errors gracefully
    # For now, if predictor fails, we could fall back to round-robin
    try:
        predictions = await predictor_client.predict(
            model_id=request.model_id,
            metadata=request.metadata,
            instance_ids=instance_ids,
        )
    except Exception as e:
        # Fallback: use first available instance
        # In production, you might want to use a different fallback strategy
        selected_instance_id = available_instances[0].instance_id
    else:
        # Apply scheduling strategy to select best instance
        selected_instance_id = scheduling_strategy.select_instance(predictions)

        if not selected_instance_id:
            # Shouldn't happen, but handle gracefully
            selected_instance_id = available_instances[0].instance_id

    # Create task record
    try:
        task_record = task_registry.create_task(
            task_id=request.task_id,
            model_id=request.model_id,
            task_input=request.task_input,
            metadata=request.metadata,
            assigned_instance=selected_instance_id,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail={"success": False, "error": str(e)}
        )

    # Update instance stats
    instance_registry.increment_pending(selected_instance_id)

    # TODO: Update instance queue information based on prediction
    # For probabilistic scheduling, you might want to update the quantiles

    # Dispatch task asynchronously
    task_dispatcher.dispatch_task_async(request.task_id)

    # Return task info
    task_info = TaskInfo(
        task_id=task_record.task_id,
        status=task_record.status,
        assigned_instance=task_record.assigned_instance,
        submitted_at=task_record.submitted_at,
    )

    return TaskSubmitResponse(
        success=True,
        message="Task submitted successfully",
        task=task_info,
    )


@app.get("/task/list", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[TaskStatus] = Query(None),
    model_id: Optional[str] = Query(None),
    instance_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    List tasks with optional filtering and pagination.

    Args:
        status: Optional filter by task status
        model_id: Optional filter by model ID
        instance_id: Optional filter by assigned instance
        limit: Maximum number of tasks to return (1-1000)
        offset: Pagination offset

    Returns:
        TaskListResponse with paginated task list
    """
    # Retrieve tasks with filters and pagination
    tasks, total = task_registry.list_all(
        status=status,
        model_id=model_id,
        instance_id=instance_id,
        limit=limit,
        offset=offset,
    )

    # Convert to TaskSummary objects
    task_summaries = [
        TaskSummary(
            task_id=task.task_id,
            model_id=task.model_id,
            status=task.status,
            assigned_instance=task.assigned_instance,
            submitted_at=task.submitted_at,
            completed_at=task.completed_at,
        )
        for task in tasks
    ]

    return TaskListResponse(
        success=True,
        count=len(task_summaries),
        total=total,
        offset=offset,
        limit=limit,
        tasks=task_summaries,
    )


@app.get("/task/info", response_model=TaskDetailResponse)
async def get_task_info(task_id: str = Query(...)):
    """
    Get detailed information about a specific task.

    Args:
        task_id: ID of the task to query

    Returns:
        TaskDetailResponse with complete task details

    Raises:
        HTTPException 404: If task not found
    """
    # Get task
    task = task_registry.get(task_id)
    if not task:
        raise HTTPException(
            status_code=404,
            detail={"success": False, "error": "Task not found"},
        )

    # Create detailed task info
    task_detail = TaskDetailInfo(
        task_id=task.task_id,
        model_id=task.model_id,
        status=task.status,
        assigned_instance=task.assigned_instance,
        task_input=task.task_input,
        metadata=task.metadata,
        result=task.result,
        error=task.error,
        timestamps=task.get_timestamps(),
        execution_time_ms=task.execution_time_ms,
    )

    return TaskDetailResponse(
        success=True,
        task=task_detail,
    )


@app.websocket("/task/get_result")
async def websocket_get_result(websocket: WebSocket):
    """
    WebSocket endpoint for real-time task result delivery.

    Clients can subscribe to multiple task IDs and receive results
    as soon as tasks complete or fail.

    Args:
        websocket: WebSocket connection
    """
    await websocket.accept()

    # Register connection
    websocket_manager.connect(websocket)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            # Parse message type
            message_type = data.get("type")

            if message_type == WSMessageType.SUBSCRIBE:
                # Parse task_ids
                task_ids = data.get("task_ids", [])

                if not isinstance(task_ids, list):
                    error_msg = WSErrorMessage(
                        error="task_ids must be a list of strings"
                    )
                    await websocket.send_json(error_msg.model_dump())
                    continue

                # Subscribe to tasks
                websocket_manager.subscribe(websocket, task_ids)

                # For each task_id, check if already completed and send result immediately
                for task_id in task_ids:
                    task = task_registry.get(task_id)
                    if task and task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                        # Send result immediately
                        result_msg = WSTaskResultMessage(
                            task_id=task.task_id,
                            status=task.status,
                            result=task.result,
                            error=task.error,
                            timestamps=task.get_timestamps(),
                            execution_time_ms=task.execution_time_ms,
                        )
                        await websocket.send_json(result_msg.model_dump())

                # Send acknowledgment
                subscribed = websocket_manager.get_subscribed_tasks(websocket)
                ack_msg = WSAckMessage(
                    message=f"Subscribed to {len(task_ids)} tasks",
                    subscribed_tasks=subscribed,
                )
                await websocket.send_json(ack_msg.model_dump())

            elif message_type == WSMessageType.UNSUBSCRIBE:
                # Parse task_ids
                task_ids = data.get("task_ids", [])

                if not isinstance(task_ids, list):
                    error_msg = WSErrorMessage(
                        error="task_ids must be a list of strings"
                    )
                    await websocket.send_json(error_msg.model_dump())
                    continue

                # Unsubscribe from tasks
                websocket_manager.unsubscribe(websocket, task_ids)

                # Send acknowledgment
                subscribed = websocket_manager.get_subscribed_tasks(websocket)
                ack_msg = WSAckMessage(
                    message=f"Unsubscribed from {len(task_ids)} tasks",
                    subscribed_tasks=subscribed,
                )
                await websocket.send_json(ack_msg.model_dump())

            else:
                # Unknown message type
                error_msg = WSErrorMessage(
                    error=f"Unknown message type: {message_type}"
                )
                await websocket.send_json(error_msg.model_dump())

    except WebSocketDisconnect:
        # Clean up subscriptions
        websocket_manager.disconnect(websocket)

    except Exception as e:
        # TODO: Add proper logging
        # Send error message to client
        try:
            error_msg = WSErrorMessage(error=f"Server error: {str(e)}")
            await websocket.send_json(error_msg.model_dump())
        except Exception:
            pass

        # Clean up
        websocket_manager.disconnect(websocket)


# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify scheduler status.

    Returns:
        HealthResponse with status and statistics

    Returns 503 if service is unhealthy
    """
    try:
        # Check if predictor service is reachable
        # TODO: Make this optional or configurable
        # predictor_healthy = await predictor_client.health_check()
        # if not predictor_healthy:
        #     raise Exception("Predictor service unavailable")

        # Collect statistics
        stats = HealthStats(
            total_instances=instance_registry.get_total_count(),
            active_instances=instance_registry.get_active_count(),
            total_tasks=task_registry.get_total_count(),
            pending_tasks=task_registry.get_count_by_status(TaskStatus.PENDING),
            running_tasks=task_registry.get_count_by_status(TaskStatus.RUNNING),
            completed_tasks=task_registry.get_count_by_status(TaskStatus.COMPLETED),
            failed_tasks=task_registry.get_count_by_status(TaskStatus.FAILED),
        )

        return HealthResponse(
            success=True,
            status="healthy",
            timestamp=datetime.now().isoformat() + "Z",
            version="1.0.0",
            stats=stats,
        )

    except Exception as e:
        # TODO: Add proper logging
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat() + "Z",
            },
        )


# ============================================================================
# Lifecycle Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    # TODO: Load persisted state if needed
    # TODO: Initialize background tasks
    # TODO: Connect to external services
    pass


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    # TODO: Persist state if needed
    # TODO: Close connections to external services
    # TODO: Cancel background tasks
    pass
