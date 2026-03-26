"""Transport-independent prediction business logic."""

from __future__ import annotations

import random
import traceback
from typing import Any

import numpy as np

from swarmpilot.predictor.api import dependencies
from swarmpilot.predictor.models import PredictionRequest, PredictionResponse
from swarmpilot.predictor.predictor.base import BasePredictor
from swarmpilot.predictor.predictor.registry import create_predictor
from swarmpilot.predictor.utils.experiment import (
    generate_experiment_prediction,
    is_experiment_mode,
)
from swarmpilot.predictor.utils.logging import get_logger

logger = get_logger()


class PredictionServiceError(Exception):
    """Raised when prediction business logic fails.

    Carries a structured error_detail dict and an HTTP-equivalent status_code
    so that transport layers (HTTP/WebSocket) can translate appropriately.
    """

    def __init__(self, error_detail: dict[str, Any], status_code: int = 400):
        self.error_detail = error_detail
        self.status_code = status_code
        super().__init__(error_detail.get("message", ""))


def try_experiment_mode(
    request: PredictionRequest,
) -> PredictionResponse | None:
    """Return synthetic prediction response when request is in experiment mode."""
    random.seed(42)
    np.random.seed(42)

    if not is_experiment_mode(
        request.features, request.platform_info.model_dump()
    ):
        return None

    try:
        config: dict[str, Any] = {}
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
    except ValueError as exc:
        raise PredictionServiceError(
            error_detail={
                "error": "Experiment mode error",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            status_code=400,
        ) from exc


def resolve_predictor(request: PredictionRequest) -> BasePredictor:
    """Resolve predictor from cache or storage with validation and caching."""
    model_key = dependencies.storage.generate_model_key(
        model_id=request.model_id,
        platform_info=request.platform_info.model_dump(),
        prediction_type=request.prediction_type,
    )

    cached_result = dependencies.model_cache.get(model_key)
    if cached_result is not None:
        predictor, stored_prediction_type = cached_result
        if stored_prediction_type != request.prediction_type:
            raise PredictionServiceError(
                error_detail={
                    "error": "Prediction type mismatch",
                    "message": (
                        f"Model was trained with prediction_type="
                        f"'{stored_prediction_type}', but request has "
                        f"'{request.prediction_type}'"
                    ),
                    "model_prediction_type": stored_prediction_type,
                    "request_prediction_type": request.prediction_type,
                },
                status_code=400,
            )
        return predictor

    model_data = dependencies.storage.load_model(model_key)
    if model_data is None:
        raise PredictionServiceError(
            error_detail={
                "error": "Model not found",
                "message": (
                    f"No trained model found for model_id='{request.model_id}' "
                    "with given platform_info"
                ),
                "model_id": request.model_id,
                "platform_info": request.platform_info.model_dump(),
                "model_key": model_key,
            },
            status_code=404,
        )

    stored_prediction_type = model_data["metadata"].get("prediction_type")
    if stored_prediction_type != request.prediction_type:
        raise PredictionServiceError(
            error_detail={
                "error": "Prediction type mismatch",
                "message": (
                    f"Model was trained with prediction_type="
                    f"'{stored_prediction_type}', but request has "
                    f"'{request.prediction_type}'"
                ),
                "model_prediction_type": stored_prediction_type,
                "request_prediction_type": request.prediction_type,
            },
            status_code=400,
        )

    try:
        predictor = create_predictor(request.prediction_type)
    except ValueError:
        raise PredictionServiceError(
            error_detail={
                "error": "Invalid prediction type",
                "message": (
                    "prediction_type must be 'expect_error' or 'quantile', "
                    f"got '{request.prediction_type}'"
                ),
            },
            status_code=400,
        )
    predictor.load_model_state(model_data["predictor_state"])
    dependencies.model_cache.put(model_key, predictor, stored_prediction_type)
    logger.info(f"Loaded and cached model: {model_key}")
    return predictor


def apply_v1_preprocessors(
    features: dict[str, Any],
    enable_preprocessors: list[str],
    preprocessor_mappings: dict[str, list[str]] | None,
) -> dict[str, Any]:
    """Apply V1 preprocessor chain to a feature dict (in-place copy).

    Args:
        features: Mutable feature dict (will be modified).
        enable_preprocessors: Ordered list of preprocessor names to apply.
        preprocessor_mappings: Maps preprocessor name → list of feature keys.

    Returns:
        The mutated features dict.

    Raises:
        ValueError: If mappings are missing or required feature keys absent.
    """
    if preprocessor_mappings is None:
        raise ValueError(
            "preprocessor_mappings is required when preprocessors enabled"
        )

    for preprocessor_name in enable_preprocessors:
        preprocessor = dependencies.preprocessors_registry.get_preprocessor(
            preprocessor_name
        )
        target_feature_keys = preprocessor_mappings[preprocessor_name]

        if not all(key in features for key in target_feature_keys):
            raise ValueError(
                f"Feature keys {target_feature_keys} not all found in features"
            )

        target_feature_values = [features[key] for key in target_feature_keys]
        processed_features, remove_origin = preprocessor(target_feature_values)

        for key, value in processed_features.items():
            features[key] = value

        if remove_origin:
            for key in target_feature_keys:
                del features[key]

    return features


def prepare_features(request: PredictionRequest) -> dict[str, Any]:
    """Prepare and preprocess request features before prediction."""
    all_features = request.features.copy()
    hardware_features = request.platform_info.extract_gpu_specs()
    if hardware_features:
        all_features.update(hardware_features)

    if request.enable_preprocessors:
        apply_v1_preprocessors(
            all_features,
            request.enable_preprocessors,
            request.preprocessor_mappings,
        )

    return all_features


def execute_prediction(request: PredictionRequest) -> PredictionResponse:
    """Execute complete prediction workflow for any transport layer."""
    experiment_result = try_experiment_mode(request)
    if experiment_result is not None:
        return experiment_result

    predictor = resolve_predictor(request)
    features = prepare_features(request)
    result = predictor.predict(features)

    return PredictionResponse(
        model_id=request.model_id,
        platform_info=request.platform_info,
        prediction_type=request.prediction_type,
        result=result,
    )
