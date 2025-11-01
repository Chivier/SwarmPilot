"""
FastAPI application for the scheduler service.

This module defines all API endpoints for instance management, task scheduling,
and WebSocket connections for real-time task result delivery.
"""

from typing import Optional, Callable
from datetime import datetime
import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, Request
from pyinstrument import Profiler
from pyinstrument.renderers.html import HTMLRenderer
from pyinstrument.renderers.speedscope import SpeedscopeRenderer
from loguru import logger

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
    InstanceQueueProbabilistic,
    InstanceQueueExpectError,
    # Task models
    TaskSubmitRequest,
    TaskSubmitResponse,
    TaskListResponse,
    TaskDetailResponse,
    TaskClearResponse,
    TaskStatus,
    TaskInfo,
    TaskSummary,
    TaskDetailInfo,
    TaskTimestamps,
    # Strategy models
    StrategyType,
    StrategySetRequest,
    StrategySetResponse,
    StrategyGetResponse,
    StrategyInfo,
    # Callback models
    TaskResultCallbackRequest,
    TaskResultCallbackResponse,
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

# Import logger configuration to initialize loguru
from . import logger as logger_module  # noqa: F401


# ============================================================================
# Application Setup
# ============================================================================



app = FastAPI(
    title="Scheduler API",
    description="Task scheduling and instance management service",
    version="1.0.0",
)

logger.info("Initializing Scheduler API")

@app.middleware("http")
async def profile_request(request: Request, call_next: Callable):
    """Profile the current request

    Taken from https://pyinstrument.readthedocs.io/en/latest/guide.html#profile-a-web-request-in-fastapi
    with small improvements.

    """
    # we map a profile type to a file extension, as well as a pyinstrument profile renderer
    profile_type_to_ext = {"html": "html", "speedscope": "speedscope.json"}
    profile_type_to_renderer = {
        "html": HTMLRenderer,
        "speedscope": SpeedscopeRenderer,
    }

    # if the `profile=true` HTTP query argument is passed, we profile the request
    if request.query_params.get("profile", False):

        # The default profile format is speedscope
        profile_type = request.query_params.get("profile_format", "speedscope")

        # we profile the request along with all additional middlewares, by interrupting
        # the program every 1ms1 and records the entire stack at that point
        with Profiler(interval=0.001, async_mode="enabled") as profiler:
            response = await call_next(request)

        # we dump the profiling into a file
        extension = profile_type_to_ext[profile_type]
        renderer = profile_type_to_renderer[profile_type]()
        with open(f"profile.{extension}", "w") as out:
            out.write(profiler.output(renderer=renderer))
        profiler.print()
        return response

    # Proceed without profiling
    return await call_next(request)

# Initialize configuration
from .config import config
from .training_client import TrainingClient

# Determine queue info type based on scheduling strategy
queue_info_type = "expect_error" if config.scheduling.default_strategy == "min_time" else "probabilistic"

# Global state - TODO: Consider dependency injection for better testability
instance_registry = InstanceRegistry(queue_info_type=queue_info_type)
task_registry = TaskRegistry()
websocket_manager = ConnectionManager()

predictor_client = PredictorClient(
    predictor_url=config.predictor.url,
    timeout=config.predictor.timeout,
    max_retries=config.predictor.max_retries,
    retry_delay=config.predictor.retry_delay,
    cache_ttl=config.predictor.cache_ttl,
    enable_cache=config.predictor.enable_cache,
)

# Initialize training client
training_client = TrainingClient(
    predictor_url=config.predictor.url,
    batch_size=config.training.batch_size,
    min_samples=config.training.min_samples,
) if config.training.enable_auto_training else None

# Initialize task dispatcher
# Construct callback base URL from config
callback_base_url = f"http://{config.server.host}:{config.server.port}"
if config.server.host == "0.0.0.0":
    # For 0.0.0.0, use localhost for callbacks
    callback_base_url = f"http://localhost:{config.server.port}"

task_dispatcher = TaskDispatcher(
    task_registry=task_registry,
    instance_registry=instance_registry,
    websocket_manager=websocket_manager,
    training_client=training_client,
    callback_base_url=callback_base_url,
)

# Initialize scheduling strategy from configuration
# Note: strategy now receives predictor_client and instance_registry dependencies
scheduling_strategy = get_strategy(
    strategy_name=config.scheduling.default_strategy,
    predictor_client=predictor_client,
    instance_registry=instance_registry,
    target_quantile=config.scheduling.probabilistic_quantile,
)


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

    # Validate platform_info has required keys
    required_keys = {"software_name", "software_version", "hardware_name"}
    if not required_keys.issubset(request.platform_info.keys()):
        missing_keys = required_keys - request.platform_info.keys()
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": f"platform_info missing required keys: {missing_keys}",
            },
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
        platform_info=request.platform_info,
    )

    # Register instance (this also initializes queue info and stats)
    try:
        instance_registry.register(instance)
        logger.info(
            f"Registered instance {request.instance_id} for model {request.model_id} "
            f"on {request.platform_info['hardware_name']}"
        )
    except ValueError as e:
        logger.warning(f"Failed to register instance {request.instance_id}: {e}")
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
        logger.info(f"Removed instance {request.instance_id}")
    except KeyError:
        logger.warning(f"Attempted to remove non-existent instance {request.instance_id}")
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
        HTTPException 503: If predictor service errors
    """
    # 1. Validate that task doesn't already exist
    if task_registry.get(request.task_id):
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": "Task with this ID already exists"},
        )

    # 2. Find available instances for the model
    available_instances = instance_registry.list_all(model_id=request.model_id)

    if not available_instances:
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": f"No available instance for model_id: {request.model_id}",
            },
        )

    # 3. Schedule task using strategy (handles predictions, selection, queue updates)
    try:
        schedule_result = await scheduling_strategy.schedule_task(
            model_id=request.model_id,
            metadata=request.metadata,
            available_instances=available_instances,
        )
    except ValueError as e:
        # Model not found or invalid metadata
        error_msg = str(e)
        if "No trained model" in error_msg or "Model not found" in error_msg:
            raise HTTPException(
                status_code=404,
                detail={
                    "success": False,
                    "error": "No trained model available for this platform. "
                    "Please train the model first or use experiment mode.",
                },
            )
        else:
            raise HTTPException(
                status_code=400,
                detail={"success": False, "error": f"Invalid task metadata: {error_msg}"},
            )
    except (ConnectionError, TimeoutError) as e:
        # Predictor service errors
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": f"Predictor service unavailable: {str(e)}",
            },
        )

    # 4. Create task record with prediction information
    selected_pred = schedule_result.selected_prediction
    try:
        task_record = task_registry.create_task(
            task_id=request.task_id,
            model_id=request.model_id,
            task_input=request.task_input,
            metadata=request.metadata,
            assigned_instance=schedule_result.selected_instance_id,
            predicted_time_ms=selected_pred.predicted_time_ms if selected_pred else None,
            predicted_error_margin_ms=selected_pred.error_margin_ms if selected_pred else None,
            predicted_quantiles=selected_pred.quantiles if selected_pred else None,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": str(e)}
        )

    # 5. Update instance stats
    instance_registry.increment_pending(schedule_result.selected_instance_id)

    # 6. Dispatch task asynchronously
    task_dispatcher.dispatch_task_async(request.task_id)

    # 7. Return task info
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


@app.post("/task/clear", response_model=TaskClearResponse)
async def clear_tasks():
    """
    Clear all tasks from the scheduler.

    This endpoint removes all task records from the scheduler's registry.
    Use with caution as this operation cannot be undone.

    Returns:
        TaskClearResponse with count of cleared tasks
    """
    # Clear all tasks from registry
    cleared_count = task_registry.clear_all()
    logger.warning(f"Cleared {cleared_count} tasks from registry")

    return TaskClearResponse(
        success=True,
        message=f"Successfully cleared {cleared_count} task(s)",
        cleared_count=cleared_count,
    )


@app.post("/callback/task_result", response_model=TaskResultCallbackResponse)
async def callback_task_result(request: TaskResultCallbackRequest):
    """
    Callback endpoint for instances to report task completion.

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
    # Validate task exists
    task = task_registry.get(request.task_id)
    if not task:
        raise HTTPException(
            status_code=404,
            detail={"success": False, "error": "Task not found"},
        )

    # Validate status
    if request.status not in ("completed", "failed"):
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": f"Invalid status: {request.status}. Must be 'completed' or 'failed'",
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
        logger.debug(f"WebSocket client disconnected")

    except Exception as e:
        # Log the error
        logger.error(f"WebSocket error: {e}", exc_info=True)
        # Send error message to client
        try:
            error_msg = WSErrorMessage(error=f"Server error: {str(e)}")
            await websocket.send_json(error_msg.model_dump())
        except Exception:
            pass

        # Clean up
        websocket_manager.disconnect(websocket)


# ============================================================================
# Strategy Management Helpers
# ============================================================================


def reinitialize_instance_queues(strategy_name: str) -> int:
    """
    Reinitialize all instance queue info to match the new strategy type.

    Args:
        strategy_name: Name of the new scheduling strategy

    Returns:
        Number of instances whose queue info was reinitialized
    """
    # Determine the queue info type for the new strategy
    if strategy_name == "min_time":
        queue_info_type = "expect_error"
    elif strategy_name == "probabilistic":
        queue_info_type = "probabilistic"
    else:  # round_robin
        queue_info_type = "probabilistic"  # Default to probabilistic for round_robin

    # Get all registered instances
    all_instances = instance_registry.list_all()

    # Reinitialize queue info for each instance
    for instance in all_instances:
        if queue_info_type == "expect_error":
            # Initialize with InstanceQueueExpectError
            new_queue_info = InstanceQueueExpectError(
                instance_id=instance.instance_id,
                expected_time_ms=0.0,
                error_margin_ms=0.0,
            )
        else:  # probabilistic
            # Initialize with InstanceQueueProbabilistic
            new_queue_info = InstanceQueueProbabilistic(
                instance_id=instance.instance_id,
                quantiles=[0.5, 0.9, 0.95, 0.99],
                values=[0.0, 0.0, 0.0, 0.0],
            )

        # Update the queue info in the registry
        instance_registry.update_queue_info(instance.instance_id, new_queue_info)

    # Update the global queue_info_type
    instance_registry._queue_info_type = queue_info_type

    return len(all_instances)


def get_current_strategy_info() -> StrategyInfo:
    """
    Get information about the current scheduling strategy.

    Returns:
        StrategyInfo with strategy name and parameters
    """
    # Determine strategy name from the strategy instance
    strategy_class_name = scheduling_strategy.__class__.__name__

    if strategy_class_name == "MinimumExpectedTimeStrategy":
        strategy_name = "min_time"
        parameters = {}
    elif strategy_class_name == "ProbabilisticSchedulingStrategy":
        strategy_name = "probabilistic"
        # Extract target_quantile from strategy instance
        parameters = {"target_quantile": scheduling_strategy.target_quantile}
    elif strategy_class_name == "RoundRobinStrategy":
        strategy_name = "round_robin"
        parameters = {}
    else:
        strategy_name = "unknown"
        parameters = {}

    return StrategyInfo(
        strategy_name=strategy_name,
        parameters=parameters,
    )


# ============================================================================
# Strategy Management Endpoints
# ============================================================================


@app.get("/strategy/get", response_model=StrategyGetResponse)
async def get_strategy_endpoint():
    """
    Get the current scheduling strategy and its parameters.

    Returns:
        StrategyGetResponse with current strategy information
    """
    strategy_info = get_current_strategy_info()

    return StrategyGetResponse(
        success=True,
        strategy_info=strategy_info,
    )


@app.post("/strategy/set", response_model=StrategySetResponse)
async def set_strategy_endpoint(request: StrategySetRequest):
    """
    Set the scheduling strategy for the scheduler.

    This endpoint:
    1. Validates that no tasks are currently running
    2. Clears all tasks from the task queue
    3. Reinitializes instance queue info to match the new strategy
    4. Switches to the new scheduling strategy

    Args:
        request: Strategy configuration including name and parameters

    Returns:
        StrategySetResponse with operation details and new strategy info

    Raises:
        HTTPException 400: If tasks are currently running
        HTTPException 500: If strategy initialization fails
    """
    global scheduling_strategy

    # Check if there are any running tasks
    running_count = task_registry.get_count_by_status(TaskStatus.RUNNING)
    if running_count > 0:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": f"Cannot switch strategy while {running_count} task(s) are running. Please wait for tasks to complete or fail them first.",
            },
        )

    # Clear all tasks from the task queue
    cleared_count = task_registry.clear_all()
    logger.info(f"Cleared {cleared_count} tasks before switching strategy")

    # Reinitialize instance queues to match the new strategy
    reinitialized_count = reinitialize_instance_queues(request.strategy_name.value)
    logger.info(f"Reinitialized {reinitialized_count} instance queues for strategy '{request.strategy_name.value}'")

    # Create new scheduling strategy instance
    try:
        new_strategy = get_strategy(
            strategy_name=request.strategy_name.value,
            predictor_client=predictor_client,
            instance_registry=instance_registry,
            target_quantile=request.target_quantile,
        )
        logger.info(f"Created new scheduling strategy: {request.strategy_name.value}")
    except Exception as e:
        logger.error(f"Failed to initialize strategy '{request.strategy_name.value}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Failed to initialize strategy: {str(e)}",
            },
        )

    # Update global scheduling strategy
    scheduling_strategy = new_strategy

    # Update config (in-memory only, not persisted)
    config.scheduling.default_strategy = request.strategy_name.value
    if request.strategy_name == StrategyType.PROBABILISTIC:
        config.scheduling.probabilistic_quantile = request.target_quantile

    # Get the new strategy info
    strategy_info = get_current_strategy_info()

    logger.success(f"Successfully switched to strategy '{request.strategy_name.value}'")

    return StrategySetResponse(
        success=True,
        message=f"Successfully switched to '{request.strategy_name.value}' strategy",
        cleared_tasks=cleared_count,
        reinitialized_instances=reinitialized_count,
        strategy_info=strategy_info,
    )


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
        # Log the health check failure
        logger.error(f"Health check failed: {e}", exc_info=True)
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
    logger.info("Scheduler service starting up...")
    logger.info(f"Configuration: {config}")
    logger.info(f"Scheduling strategy: {config.scheduling.default_strategy}")
    logger.info(f"Auto-training enabled: {config.training.enable_auto_training}")
    # TODO: Load persisted state if needed
    # TODO: Initialize background tasks
    # TODO: Connect to external services
    logger.success("Scheduler service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    logger.info("Scheduler service shutting down...")

    # Close HTTP clients
    await task_dispatcher.close()
    logger.debug("Task dispatcher closed")

    if training_client:
        await training_client.close()
        logger.debug("Training client closed")

    await predictor_client.close()
    logger.debug("Predictor client closed")

    # TODO: Persist state if needed
    # TODO: Cancel background tasks
    logger.info("Scheduler service shutdown complete")
