"""HTTP prediction endpoint."""

from __future__ import annotations

import random
import traceback

import numpy as np
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import status

from swarmpilot.predictor.api import dependencies
from swarmpilot.predictor.models import PredictionRequest
from swarmpilot.predictor.models import PredictionResponse
from swarmpilot.predictor.predictor.expect_error import ExpectErrorPredictor
from swarmpilot.predictor.predictor.quantile import QuantilePredictor
from swarmpilot.predictor.utils.experiment import generate_experiment_prediction
from swarmpilot.predictor.utils.experiment import is_experiment_mode
from swarmpilot.predictor.utils.logging import get_logger

logger = get_logger()

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
    random.seed(42)
    np.random.seed(42)
    try:
        # Check if experiment mode
        if is_experiment_mode(request.features, request.platform_info.model_dump()):
            # Generate synthetic prediction
            logger.debug(f"Got experiment mode request, raw request: {request}")
            try:
                # Pass custom quantiles to experiment mode if provided
                config = {}
                if request.quantiles is not None:
                    config["quantiles"] = request.quantiles

                result = generate_experiment_prediction(
                    prediction_type=request.prediction_type,
                    features=request.features,
                    config=config,
                )

                return PredictionResponse(
                    model_id=request.model_id,
                    platform_info=request.platform_info,
                    prediction_type=request.prediction_type,
                    result=result,
                )

            except ValueError as e:
                error_detail = {
                    "error": "Experiment mode error",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
                dependencies._log_error(
                    error_context=(
                        f"Experiment mode error for model_id={request.model_id}"
                    ),
                    error_detail=error_detail,
                    exception=e,
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_detail,
                )

        # Normal mode: load model and predict
        model_key = dependencies.storage.generate_model_key(
            model_id=request.model_id,
            platform_info=request.platform_info.model_dump(),
            prediction_type=request.prediction_type,
        )

        # Try to get predictor from cache
        cached_result = dependencies.model_cache.get(model_key)

        if cached_result is not None:
            # Cache hit - use cached predictor
            predictor, stored_prediction_type = cached_result

            # Validate prediction type matches
            if stored_prediction_type != request.prediction_type:
                error_detail = {
                    "error": "Prediction type mismatch",
                    "message": (
                        f"Model was trained with prediction_type="
                        f"'{stored_prediction_type}', but request has "
                        f"'{request.prediction_type}'"
                    ),
                    "model_prediction_type": stored_prediction_type,
                    "request_prediction_type": request.prediction_type,
                }
                dependencies._log_error(
                    error_context=(
                        f"Prediction type mismatch (cached) "
                        f"for model_id={request.model_id}"
                    ),
                    error_detail=error_detail,
                    include_traceback=False,
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_detail,
                )
        else:
            # Cache miss - load model from storage
            model_data = dependencies.storage.load_model(model_key)
            if model_data is None:
                error_detail = {
                    "error": "Model not found",
                    "message": (
                        f"No trained model found for model_id='{request.model_id}' "
                        f"with given platform_info"
                    ),
                    "model_id": request.model_id,
                    "platform_info": request.platform_info.model_dump(),
                    "model_key": model_key,
                }
                dependencies._log_error(
                    error_context=f"Model not found for model_id={request.model_id}",
                    error_detail=error_detail,
                    include_traceback=False,
                )
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=error_detail,
                )

            # Validate prediction type matches
            stored_prediction_type = model_data["metadata"].get("prediction_type")
            if stored_prediction_type != request.prediction_type:
                error_detail = {
                    "error": "Prediction type mismatch",
                    "message": (
                        f"Model was trained with prediction_type="
                        f"'{stored_prediction_type}', but request has "
                        f"'{request.prediction_type}'"
                    ),
                    "model_prediction_type": stored_prediction_type,
                    "request_prediction_type": request.prediction_type,
                }
                dependencies._log_error(
                    error_context=(
                        f"Prediction type mismatch (storage) "
                        f"for model_id={request.model_id}"
                    ),
                    error_detail=error_detail,
                    include_traceback=False,
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_detail,
                )

            # Create predictor and load state
            if request.prediction_type == "expect_error":
                predictor = ExpectErrorPredictor()
            elif request.prediction_type == "quantile":
                predictor = QuantilePredictor()
            else:
                error_detail = {
                    "error": "Invalid prediction type",
                    "message": (
                        f"prediction_type must be 'expect_error' or 'quantile', "
                        f"got '{request.prediction_type}'"
                    ),
                }
                dependencies._log_error(
                    error_context=(
                        f"Invalid prediction_type in /predict "
                        f"for model_id={request.model_id}"
                    ),
                    error_detail=error_detail,
                    include_traceback=False,
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_detail,
                )

            predictor.load_model_state(model_data["predictor_state"])

            # Cache the loaded predictor
            dependencies.model_cache.put(model_key, predictor, stored_prediction_type)
            logger.info(f"Loaded and cached model: {model_key}")

        # Make prediction
        try:
            # Append hardware info
            hardware_features = request.platform_info.extract_gpu_specs()
            all_features = request.features.copy()
            if hardware_features:
                for key, value in hardware_features.items():
                    all_features[key] = value

            # Start preprocessing if enabled
            if request.enable_preprocessors:
                for preprocessor_name in request.enable_preprocessors:
                    preprocessor = (
                        dependencies.preprocessors_registry.get_preprocessor(
                            preprocessor_name
                        )
                    )
                    target_feature_keys = request.preprocessor_mappings[
                        preprocessor_name
                    ]

                    # Validate all required features exist
                    assert all(
                        key in all_features for key in target_feature_keys
                    ), f"Feature keys {target_feature_keys} not all found in features"

                    # Extract target feature values
                    target_feature_values = [
                        all_features[key] for key in target_feature_keys
                    ]

                    # Apply preprocessor
                    processed_features, remove_origin = preprocessor(
                        target_feature_values
                    )

                    # Add processed features
                    for k, v in processed_features.items():
                        all_features[k] = v

                    # Remove original features if requested
                    if remove_origin:
                        for key in target_feature_keys:
                            del all_features[key]

            result = predictor.predict(all_features)

            return PredictionResponse(
                model_id=request.model_id,
                platform_info=request.platform_info,
                prediction_type=request.prediction_type,
                result=result,
            )

        except ValueError as e:
            # Feature validation errors
            error_detail = {
                "error": "Invalid features",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            dependencies._log_error(
                error_context=(
                    f"Feature validation error in /predict "
                    f"for model_id={request.model_id}"
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
                "error": "Prediction failed",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            dependencies._log_error(
                error_context=f"Prediction failed for model_id={request.model_id}",
                error_detail=error_detail,
                exception=e,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_detail,
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
                f"Unexpected error in /predict endpoint "
                f"for model_id={request.model_id}"
            ),
            error_detail=error_detail,
            exception=e,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail,
        )
