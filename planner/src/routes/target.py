"""Target and throughput submission endpoints.

Note: These endpoints access global state from the api module.
The state is modified in place via the api module reference.
"""

from fastapi import APIRouter
from loguru import logger

from ..models import (
    SubmitTargetRequest,
    SubmitTargetResponse,
    SubmitThroughputRequest,
    SubmitThroughputResponse,
)

router = APIRouter(tags=["target"])


@router.post("/submit_target", response_model=SubmitTargetResponse)
async def submit_target(request: SubmitTargetRequest):
    """Submit queue length from scheduler to update target distribution.

    This endpoint receives queue length data from each scheduler and accumulates
    it into the target distribution array. Only effective after /deploy or
    /deploy/migration has been called to establish the model mapping.

    Args:
        request: Queue length data with model_id and value

    Returns:
        SubmitTargetResponse with update status and current target
    """
    # Import api module to access and modify global state
    from .. import api as api_module

    # No error if mapping doesn't exist, just no effect
    if api_module._stored_model_mapping is None:
        logger.info("submit_target called but no mapping exists yet")
        return SubmitTargetResponse(
            success=True,
            message="No active mapping. Call /deploy or /deploy/migration first.",
            current_target=None,
        )

    # Check if model_id exists in mapping
    if request.model_id not in api_module._stored_model_mapping:
        logger.info(
            f"submit_target: model_id {request.model_id} not in mapping"
        )
        return SubmitTargetResponse(
            success=True,
            message=f"Model {request.model_id} not in current mapping",
            current_target=api_module._current_target,
        )

    # Update target at corresponding position
    idx = api_module._stored_model_mapping[request.model_id]
    api_module._current_target[idx] = request.value

    # Track submitted models and mark first data received
    api_module._submitted_models.add(request.model_id)

    # Mark that first data has been received in this cycle
    if not api_module._first_data_received:
        api_module._first_data_received = True
        logger.info(
            f"First data received in this cycle from model {request.model_id}"
        )

    logger.info(
        f"Updated target[{idx}] for model {request.model_id} to {request.value} ({len(api_module._submitted_models)}/{len(api_module._stored_model_mapping)} models submitted)"
    )

    return SubmitTargetResponse(
        success=True,
        message=f"Updated target[{idx}] for {request.model_id} ({len(api_module._submitted_models)}/{len(api_module._stored_model_mapping)} submitted)",
        current_target=api_module._current_target,
    )


@router.get("/target")
async def get_target():
    """Get current accumulated target distribution.

    Returns the current target array and model mapping set by /deploy or
    /deploy/migration, updated by /submit_target calls.

    Returns:
        Dictionary with target array and model mapping
    """
    from .. import api as api_module

    return {
        "target": api_module._current_target,
        "model_mapping": api_module._stored_model_mapping,
        "reverse_mapping": api_module._stored_reverse_mapping,
    }


@router.post("/submit_throughput", response_model=SubmitThroughputResponse)
async def submit_throughput(request: SubmitThroughputRequest):
    """Submit throughput data from an instance to update the B matrix.

    This endpoint receives average execution time from instances and converts
    it to processing capacity (1/avg_execution_time). The capacity is stored
    and applied to the B matrix before each auto-reconfiguration cycle.

    The model ID is automatically determined by looking up the instance's
    current deployment state.

    Args:
        request: Throughput data with instance_url and avg_execution_time

    Returns:
        SubmitThroughputResponse with computed capacity and status
    """
    from .. import api as api_module

    # Compute capacity from execution time
    capacity = 1.0 / request.avg_execution_time

    # Look up instance to determine model
    model_id = None
    if api_module._stored_deployment_input is not None:
        for inst in api_module._stored_deployment_input.instances:
            if inst.endpoint == request.instance_url:
                model_id = inst.current_model
                break

    if model_id is None:
        # Instance not found in current deployment
        logger.warning(
            f"submit_throughput: instance {request.instance_url} not in current deployment"
        )
        return SubmitThroughputResponse(
            success=True,
            message=f"Instance {request.instance_url} not found in current deployment. Data not stored.",
            instance_url=request.instance_url,
            model_id=None,
            computed_capacity=capacity,
        )

    # Store/update throughput data with EMA
    # TODO: Temporarily disabled - only accept data without side effects
    # _update_throughput_entry(request.instance_url, model_id, capacity)

    logger.info(
        f"Throughput received (not stored): {request.instance_url} -> {model_id}, capacity={capacity:.4f}"
    )

    return SubmitThroughputResponse(
        success=True,
        message=f"Throughput recorded for {model_id} on {request.instance_url}",
        instance_url=request.instance_url,
        model_id=model_id,
        computed_capacity=capacity,
    )
