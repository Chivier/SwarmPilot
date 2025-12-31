"""Model listing endpoints."""

from __future__ import annotations

import traceback

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import status

from src.api import dependencies
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
        models_data = dependencies.storage.list_models()

        # Convert to ModelMetadata objects
        models = [
            ModelMetadata(
                model_id=m["model_id"],
                platform_info=m["platform_info"],
                prediction_type=m["prediction_type"],
                samples_count=m["samples_count"],
                last_trained=m["last_trained"],
            )
            for m in models_data
        ]

        return ModelListResponse(models=models)

    except Exception as e:
        error_detail = {
            "error": "Failed to list models",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        dependencies._log_error(
            error_context="List models operation failed",
            error_detail=error_detail,
            exception=e,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail,
        )
