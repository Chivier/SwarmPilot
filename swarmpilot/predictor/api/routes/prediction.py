"""HTTP prediction endpoint."""

from __future__ import annotations

import traceback

from fastapi import APIRouter, HTTPException, status

from swarmpilot.predictor.api import dependencies
from swarmpilot.predictor.api.services.prediction_service import (
    PredictionServiceError,
    execute_prediction,
)
from swarmpilot.predictor.models import PredictionRequest, PredictionResponse

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """Make a runtime prediction.

    Returns prediction based on trained model or experiment mode.
    Supports both expect_error and quantile prediction types.

    Args:
        request: PredictionRequest with model_id, features, and platform_info.

    Returns:
        PredictionResponse with prediction result.

    Raises:
        HTTPException: If model not found or prediction fails.
    """
    try:
        return execute_prediction(request)
    except PredictionServiceError as exc:
        dependencies._log_error(
            error_context=f"Prediction service error for model_id={request.model_id}",
            error_detail=exc.error_detail,
            exception=exc,
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.error_detail,
        ) from exc
    except ValueError as exc:
        error_detail = {
            "error": "Invalid features",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        dependencies._log_error(
            error_context=f"Feature validation error in /predict for model_id={request.model_id}",
            error_detail=error_detail,
            exception=exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_detail,
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        error_detail = {
            "error": "Unexpected error",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        dependencies._log_error(
            error_context=(
                f"Unexpected error in /predict endpoint for model_id={request.model_id}"
            ),
            error_detail=error_detail,
            exception=exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail,
        ) from exc
