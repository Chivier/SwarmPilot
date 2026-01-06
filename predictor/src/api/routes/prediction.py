"""HTTP prediction endpoint.

Provides the /predict endpoint for making runtime predictions using trained models.
Supports both normal mode (using trained models) and experiment mode (synthetic data).
"""

from __future__ import annotations

import random
import traceback

import numpy as np
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import status

from src.api import dependencies
from src.api.core import ModelNotFoundError, PredictionError, ValidationError
from src.api.routes.helpers import handle_library_exception
from src.models import PredictionRequest
from src.models import PredictionResponse
from src.utils.experiment import generate_experiment_prediction
from src.utils.experiment import is_experiment_mode
from src.utils.logging import get_logger

logger = get_logger()

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """Make a runtime prediction.

    Returns prediction based on trained model or experiment mode.
    Supports all prediction types: expect_error, quantile, linear_regression, decision_tree.

    If preprocess_config is provided, applies per-feature preprocessing chains
    before making the prediction.

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
        # Check experiment mode first (special case - doesn't use real models)
        if is_experiment_mode(request.features, request.platform_info.model_dump()):
            return _handle_experiment_mode(request)

        # Normal mode: use library API for prediction
        # Append hardware info to features
        features = request.features.copy()
        hardware_features = request.platform_info.extract_gpu_specs()
        if hardware_features:
            features.update(hardware_features)

        # Determine preprocessor config to use
        # Support both new preprocess_config and deprecated fields
        preprocess_config = request.preprocess_config
        if preprocess_config is None and request.enable_preprocessors:
            # Convert deprecated format to new format (backwards compatibility)
            if request.preprocessor_mappings:
                preprocess_config = {}
                for prep_name, feature_keys in request.preprocessor_mappings.items():
                    for feature_key in feature_keys:
                        if feature_key not in preprocess_config:
                            preprocess_config[feature_key] = []
                        preprocess_config[feature_key].append(prep_name)

        # Use inference_pipeline for combined load + preprocess + predict
        result = dependencies.predictor_core.inference_pipeline(
            model_id=request.model_id,
            platform_info=request.platform_info,
            prediction_type=request.prediction_type,
            features=features,
            preprocess_config=preprocess_config,
        )

        return PredictionResponse(
            model_id=result.model_id,
            platform_info=result.platform_info,
            prediction_type=result.prediction_type,
            result=result.result,
        )

    except ModelNotFoundError as e:
        raise handle_library_exception(e, f"predict model_id={request.model_id}")
    except ValidationError as e:
        raise handle_library_exception(e, f"predict model_id={request.model_id}")
    except PredictionError as e:
        raise handle_library_exception(e, f"predict model_id={request.model_id}")
    except HTTPException:
        raise
    except Exception as e:
        raise handle_library_exception(e, f"predict model_id={request.model_id}")


def _handle_experiment_mode(request: PredictionRequest) -> PredictionResponse:
    """Handle experiment mode prediction (synthetic data).

    Args:
        request: PredictionRequest with experiment mode features.

    Returns:
        PredictionResponse with synthetic prediction.

    Raises:
        HTTPException: If experiment mode prediction fails.
    """
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
            error_context=f"Experiment mode error for model_id={request.model_id}",
            error_detail=error_detail,
            exception=e,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_detail,
        )
