"""Instance management endpoints."""

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from ..available_instance_store import AvailableInstance, get_available_instance_store
from ..models import InstanceRegisterRequest, InstanceRegisterResponse

router = APIRouter(tags=["instance"])


@router.post("/instance/register", response_model=InstanceRegisterResponse)
async def register_available_instance(request: InstanceRegisterRequest):
    """Register an available instance to the planner's available instance store.

    This endpoint has the same parameters as the scheduler's /instance/register
    but stores instances for migration-based redeployment instead of task scheduling.

    Args:
        request: Instance registration details (instance_id, model_id, endpoint, platform_info)

    Returns:
        InstanceRegisterResponse with registration status

    Raises:
        HTTPException: If registration fails
    """
    try:
        logger.info(
            f"Registering available instance: {request.instance_id} "
            f"for model {request.model_id} at {request.endpoint}"
        )

        # Get the available instance store
        instance_store = get_available_instance_store()

        # Create AvailableInstance and add to store
        available_instance = AvailableInstance(
            model_id=request.model_id, endpoint=request.endpoint
        )

        await instance_store.add_available_instance(available_instance)

        logger.info(
            f"Successfully registered instance {request.instance_id} "
            f"for model {request.model_id}"
        )

        return InstanceRegisterResponse(
            success=True,
            message=f"Instance {request.instance_id} registered successfully for model {request.model_id}",
        )

    except Exception as e:
        client_msg = f"Failed to register instance: {str(e)}"
        logger.error(
            f"/instance/register failed - Error: {e}. Returning HTTP 500. Client will receive: {client_msg}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=client_msg
        )


@router.get("/migration/info")
def get_migration_info() -> dict[str, AvailableInstance]:
    """Get current available instances for migration."""
    instance_store = get_available_instance_store()

    return instance_store.available_instances
