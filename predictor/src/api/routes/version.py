"""Version check endpoint for model version management."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone

from fastapi import APIRouter

from src.api import dependencies
from src.api.routes.helpers import handle_library_exception
from src.models import VersionCheckRequest
from src.models import VersionCheckResponse

router = APIRouter()


@router.post("/version/check", response_model=VersionCheckResponse, tags=["Version"])
async def check_version(request: VersionCheckRequest):
    """Check model version information.

    Returns version information for a model configuration without loading
    the model or making predictions. Use this endpoint to:

    - Check if a model exists
    - Get the latest version timestamp
    - List all available versions
    - Verify version consistency across distributed predictors

    Args:
        request: VersionCheckRequest with model_id, platform_info, prediction_type.

    Returns:
        VersionCheckResponse with version information.
    """
    try:
        version_info = dependencies.predictor_api.get_version_info(
            model_id=request.model_id,
            platform_info=request.platform_info,
            prediction_type=request.prediction_type,
        )

        latest = version_info.get("latest_version")
        latest_iso = None
        if latest is not None and latest > 0:
            latest_iso = datetime.fromtimestamp(latest, tz=timezone.utc).isoformat()
        elif latest == 0:
            latest_iso = "legacy"

        return VersionCheckResponse(
            model_id=request.model_id,
            platform_info=request.platform_info,
            prediction_type=request.prediction_type,
            exists=latest is not None,
            latest_version=latest,
            latest_version_iso=latest_iso,
            available_versions=version_info.get("available_versions", []),
            version_count=version_info.get("version_count", 0),
        )

    except Exception as e:
        raise handle_library_exception(e, f"version check model_id={request.model_id}")
