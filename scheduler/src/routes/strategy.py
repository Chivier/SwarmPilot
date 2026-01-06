"""Strategy management endpoints.

Provides endpoints for getting and setting the scheduling strategy.
"""

import random

import numpy as np
from fastapi import APIRouter, HTTPException
from loguru import logger

from ..config import config
from ..model import (
    InstanceQueueExpectError,
    InstanceQueueProbabilistic,
    StrategyGetResponse,
    StrategyInfo,
    StrategySetRequest,
    StrategySetResponse,
    TaskStatus,
)
from ..scheduler import get_strategy
from .deps import (
    get_background_scheduler,
    get_central_queue,
    get_instance_registry,
    get_predictor_client,
    get_scheduling_strategy,
    get_task_registry,
)

router = APIRouter(prefix="/strategy", tags=["strategy"])


async def reinitialize_instance_queues(
    strategy_name: str, quantiles: list[float] | None = None
) -> int:
    """Reinitialize all instance queue info to match the new strategy type.

    Args:
        strategy_name: Name of the new scheduling strategy
        quantiles: Custom quantiles for probabilistic strategy (optional)

    Returns:
        Number of instances whose queue info was reinitialized
    """
    instance_registry = get_instance_registry()

    # Determine the queue info type for the new strategy
    if (
        strategy_name == "min_time"
        or strategy_name == "po2"
        or strategy_name == "severless"
    ):
        queue_info_type = "expect_error"
    elif strategy_name == "probabilistic":
        queue_info_type = "probabilistic"
    else:  # round_robin
        queue_info_type = "probabilistic"

    # Update the global queue_info_type FIRST
    instance_registry._queue_info_type = queue_info_type

    # Update stored quantiles configuration if custom quantiles provided
    if quantiles:
        instance_registry._quantiles = sorted(set(quantiles))

    # Get all registered instances
    all_instances = await instance_registry.list_all()

    # Reinitialize queue info for each instance
    for instance in all_instances:
        if queue_info_type == "expect_error":
            new_queue_info = InstanceQueueExpectError(
                instance_id=instance.instance_id,
                expected_time_ms=0.0,
                error_margin_ms=0.0,
            )
        else:
            if quantiles:
                sorted_quantiles = sorted(set(quantiles))
                values = [0.0] * len(sorted_quantiles)
            else:
                sorted_quantiles = [0.5, 0.9, 0.95, 0.99]
                values = [0.0, 0.0, 0.0, 0.0]

            new_queue_info = InstanceQueueProbabilistic(
                instance_id=instance.instance_id,
                quantiles=sorted_quantiles,
                values=values,
            )

        await instance_registry.update_queue_info(instance.instance_id, new_queue_info)

    return len(all_instances)


def get_current_strategy_info() -> StrategyInfo:
    """Get information about the current scheduling strategy.

    Returns:
        StrategyInfo with strategy name and parameters
    """
    scheduling_strategy = get_scheduling_strategy()
    strategy_class_name = scheduling_strategy.__class__.__name__

    if strategy_class_name == "MinimumExpectedTimeStrategy":
        strategy_name = "min_time"
        parameters = {}
    elif strategy_class_name == "ProbabilisticSchedulingStrategy":
        strategy_name = "probabilistic"
        target_quantile = getattr(scheduling_strategy, "target_quantile", 0.9)
        parameters = {"target_quantile": target_quantile}
    elif strategy_class_name == "RoundRobinStrategy":
        strategy_name = "round_robin"
        parameters = {}
    elif strategy_class_name == "PowerOfTwoStrategy":
        strategy_name = "po2"
        parameters = {}
    else:
        strategy_name = "unknown"
        parameters = {}

    return StrategyInfo(
        strategy_name=strategy_name,
        parameters=parameters,
    )


@router.get("/get", response_model=StrategyGetResponse)
async def get_strategy_endpoint():
    """Get the current scheduling strategy and its parameters.

    Returns:
        StrategyGetResponse with current strategy information
    """
    strategy_info = get_current_strategy_info()

    return StrategyGetResponse(
        success=True,
        strategy_info=strategy_info,
    )


@router.post("/set", response_model=StrategySetResponse)
async def set_strategy_endpoint(request: StrategySetRequest):
    """Set the scheduling strategy for the scheduler.

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
    # Import here to avoid circular imports - we need to modify the global
    import src.api as api_module

    task_registry = get_task_registry()
    instance_registry = get_instance_registry()
    predictor_client = get_predictor_client()
    background_scheduler = get_background_scheduler()
    central_queue = get_central_queue()

    random.seed(42)

    # Check if there are any running tasks
    running_count = await task_registry.get_count_by_status(TaskStatus.RUNNING)
    if running_count > 0:
        error_msg = f"Cannot switch strategy while {running_count} task(s) are running. Please wait for tasks to complete or fail them first."
        logger.error(
            f"[set_strategy] {error_msg} | strategy={request.strategy_name.value} | running_count={running_count}"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": error_msg,
            },
        )

    # Clear all tasks from the task queue
    cleared_count = await task_registry.clear_all()
    logger.info(f"Cleared {cleared_count} tasks before switching strategy")

    logger.info(
        f"Switching the scheduling strategy to {request.strategy_name.value}, with the quantiles: {request.quantiles}"
    )

    # Reinitialize instance queues to match the new strategy
    reinitialized_count = await reinitialize_instance_queues(
        request.strategy_name.value, quantiles=request.quantiles
    )
    logger.info(
        f"Reinitialized {reinitialized_count} instance queues for strategy '{request.strategy_name.value}'"
    )

    # Create new scheduling strategy instance
    try:
        target_quantile = (
            request.target_quantile if request.target_quantile is not None else 0.9
        )

        new_strategy = get_strategy(
            strategy_name=request.strategy_name.value,
            predictor_client=predictor_client,
            instance_registry=instance_registry,
            target_quantile=target_quantile,
        )
        logger.info(f"Created new scheduling strategy: {request.strategy_name.value}")
    except Exception as e:
        logger.error(
            f"Failed to initialize strategy '{request.strategy_name.value}': {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Failed to initialize strategy: {e!s}",
            },
        ) from e

    # Update global scheduling strategy
    api_module.scheduling_strategy = new_strategy

    # Update BackgroundScheduler to use the new strategy
    background_scheduler.scheduling_strategy = new_strategy

    central_queue.set_scheduling_strategy(new_strategy)

    # Update config (in-memory only, not persisted)
    config.scheduling.default_strategy = request.strategy_name.value

    # Get the new strategy info
    strategy_info = get_current_strategy_info()

    random.seed(42)
    np.random.seed(42)

    logger.success(f"Successfully switched to strategy '{request.strategy_name.value}'")

    return StrategySetResponse(
        success=True,
        message=f"Successfully switched to '{request.strategy_name.value}' strategy",
        cleared_tasks=cleared_count,
        reinitialized_instances=reinitialized_count,
        strategy_info=strategy_info,
    )
