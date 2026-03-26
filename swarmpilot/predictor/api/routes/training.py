"""Training endpoint for model training."""

from __future__ import annotations

import traceback

from fastapi import APIRouter, HTTPException, status

from swarmpilot.predictor.api import dependencies
from swarmpilot.predictor.api.services.prediction_service import (
    apply_v1_preprocessors,
)
from swarmpilot.predictor.models import TrainingRequest, TrainingResponse
from swarmpilot.predictor.predictor.registry import create_predictor
from swarmpilot.predictor.utils.logging import get_logger

logger = get_logger()

router = APIRouter()


@router.post("/train", response_model=TrainingResponse, tags=["Training"])
async def train_model(request: TrainingRequest):
    """Train or update a model.

    Trains an MLP model on the provided features and runtime data.
    Supports both expect_error and quantile prediction types.

    Args:
        request: TrainingRequest with model_id, features, and config.

    Returns:
        TrainingResponse with training status and model key.

    Raises:
        HTTPException: If validation fails or training errors occur.
    """
    try:
        if len(request.features_list) < 10:
            error_detail = {
                "error": "Insufficient training data",
                "message": (
                    f"Need at least 10 samples, got {len(request.features_list)}"
                ),
                "samples_provided": len(request.features_list),
                "minimum_required": 10,
            }
            dependencies._log_error(
                error_context=(
                    f"Training validation failed for model_id={request.model_id}"
                ),
                error_detail=error_detail,
                include_traceback=False,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_detail,
            )

        try:
            predictor = create_predictor(request.prediction_type)
        except ValueError as e:
            error_detail = {
                "error": "Invalid prediction type",
                "message": str(e),
            }
            dependencies._log_error(
                error_context=(
                    f"Training failed - invalid prediction_type "
                    f"for model_id={request.model_id}"
                ),
                error_detail=error_detail,
                include_traceback=False,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_detail,
            )

        try:
            if request.enable_preprocessors:
                processed_features_list = []
                for features in request.features_list:
                    processed = dict(features)
                    apply_v1_preprocessors(
                        processed,
                        request.enable_preprocessors,
                        request.preprocessor_mappings,
                    )
                    processed_features_list.append(processed)
            else:
                processed_features_list = request.features_list
        except Exception as e:
            error_detail = {
                "error": "Preprocessor error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            dependencies._log_error(
                error_context=(
                    f"Preprocessing failed for model_id={request.model_id}"
                ),
                error_detail=error_detail,
                exception=e,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_detail,
            )

        try:
            predictor.train(
                features_list=processed_features_list,
                config=request.training_config or {},
            )
        except ValueError as e:
            error_detail = {
                "error": "Training validation error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            dependencies._log_error(
                error_context=(
                    f"Training validation error for model_id={request.model_id}"
                ),
                error_detail=error_detail,
                exception=e,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_detail,
            )
        except Exception as e:
            error_detail = {
                "error": "Training failed",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            dependencies._log_error(
                error_context=f"Training failed for model_id={request.model_id}",
                error_detail=error_detail,
                exception=e,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_detail,
            )

        model_key = dependencies.storage.generate_model_key(
            model_id=request.model_id,
            platform_info=request.platform_info.model_dump(),
            prediction_type=request.prediction_type,
        )

        try:
            predictor_state = predictor.get_model_state()
            metadata = {
                "model_id": request.model_id,
                "platform_info": request.platform_info.model_dump(),
                "prediction_type": request.prediction_type,
                "samples_count": len(request.features_list),
                "training_config": request.training_config,
            }
            dependencies.storage.save_model(
                model_key, predictor_state, metadata
            )
            dependencies.model_cache.invalidate(model_key)
            logger.info(f"Invalidated cache for retrained model: {model_key}")
        except Exception as e:
            error_detail = {
                "error": "Model save failed",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            dependencies._log_error(
                error_context=(
                    f"Model save failed for model_id={request.model_id}, "
                    f"model_key={model_key}"
                ),
                error_detail=error_detail,
                exception=e,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_detail,
            )

        return TrainingResponse(
            status="success",
            message=(
                f"Model trained successfully with "
                f"{len(request.features_list)} samples"
            ),
            model_key=model_key,
            samples_trained=len(request.features_list),
        )

    except HTTPException:
        raise
    except Exception as e:
        error_detail = {
            "error": "Unexpected error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        dependencies._log_error(
            error_context=(
                f"Unexpected error in /train endpoint "
                f"for model_id={request.model_id}"
            ),
            error_detail=error_detail,
            exception=e,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail,
        )
