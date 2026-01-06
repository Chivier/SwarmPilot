"""Model listing endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from src.api import dependencies
from src.api.routes.helpers import handle_library_exception
from src.models import ModelListResponse
from src.models import ModelMetadata

router = APIRouter()


@router.get("/list", response_model=ModelListResponse, tags=["Models"])
async def list_models():
    """List all trained models with their metadata.

    Returns information about all models stored in the system.

    Returns:
        ModelListResponse containing list of all stored models.
    """
    try:
        # Use library API for model listing
        models = dependencies.predictor_api.list_models()

        # Convert ModelInfo to ModelMetadata for backwards compatibility
        model_metadata = [
            ModelMetadata(
                model_id=m.model_id,
                platform_info=m.platform_info,
                prediction_type=m.prediction_type,
                samples_count=m.samples_count,
                last_trained=m.last_trained,
            )
            for m in models
        ]

        return ModelListResponse(models=model_metadata)

    except Exception as e:
        raise handle_library_exception(e, "list models")
