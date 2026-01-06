"""Model Info API routes.

Provides endpoint for querying currently serving model information.
"""

from fastapi import APIRouter

from src.services.inference_manager import InferenceManagerService

router = APIRouter(tags=["model-info"])

# Service instance (set during app startup)
_inference_manager_service: InferenceManagerService | None = None


def set_inference_manager_service(service: InferenceManagerService) -> None:
    """Set the inference manager service instance.

    Args:
        service: InferenceManagerService instance to use.
    """
    global _inference_manager_service
    _inference_manager_service = service


def get_inference_manager_service() -> InferenceManagerService:
    """Get the inference manager service instance.

    Returns:
        The InferenceManagerService instance.

    Raises:
        RuntimeError: If inference manager service not initialized.
    """
    if _inference_manager_service is None:
        raise RuntimeError("Inference manager service not initialized")
    return _inference_manager_service


@router.get("/info")
async def get_model_info() -> dict:
    """Get information about the currently serving model.

    Returns detailed information about the model currently loaded
    on the inference server, including resources, parameters, and stats.

    Returns:
        Model info response with serving status and details.
    """
    inference = get_inference_manager_service()
    return await inference.get_model_info()
