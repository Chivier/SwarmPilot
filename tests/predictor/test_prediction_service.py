"""Tests for the prediction service layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from swarmpilot.predictor.api.services.prediction_service import (
    PredictionServiceError,
    execute_prediction,
    prepare_features,
    resolve_predictor,
    try_experiment_mode,
)
from swarmpilot.predictor.models import PlatformInfo, PredictionRequest


def _make_request(**overrides):
    """Build a valid PredictionRequest and apply optional overrides."""
    defaults = {
        "model_id": "test-model",
        "platform_info": {
            "software_name": "PyTorch",
            "software_version": "2.0",
            "hardware_name": "NVIDIA A100",
        },
        "prediction_type": "quantile",
        "features": {"x": 1.0, "y": 2.0},
    }
    defaults.update(overrides)
    return PredictionRequest(**defaults)


def test_try_experiment_mode_returns_none_for_normal_request():
    """Experiment mode helper should return None for standard requests."""
    request = _make_request()

    with patch(
        "swarmpilot.predictor.api.services.prediction_service.is_experiment_mode",
        return_value=False,
    ):
        assert try_experiment_mode(request) is None


def test_try_experiment_mode_returns_response_for_experiment():
    """Experiment mode helper should return PredictionResponse in exp mode."""
    request = _make_request(quantiles=[0.5, 0.9])
    experiment_result = {"quantiles": {"0.5": 100.0, "0.9": 120.0}}

    with (
        patch(
            "swarmpilot.predictor.api.services.prediction_service.is_experiment_mode",
            return_value=True,
        ),
        patch(
            "swarmpilot.predictor.api.services.prediction_service.generate_experiment_prediction",
            return_value=experiment_result,
        ),
    ):
        response = try_experiment_mode(request)

    assert response is not None
    assert response.model_id == request.model_id
    assert response.prediction_type == request.prediction_type
    assert response.result == experiment_result


def test_resolve_predictor_cache_hit():
    """Predictor resolution should return cached predictor on cache hit."""
    request = _make_request(prediction_type="expect_error")
    predictor = MagicMock()

    with (
        patch(
            "swarmpilot.predictor.api.dependencies.storage.generate_model_key",
            return_value="key-1",
        ),
        patch(
            "swarmpilot.predictor.api.dependencies.model_cache.get",
            return_value=(predictor, "expect_error"),
        ),
    ):
        resolved = resolve_predictor(request)

    assert resolved is predictor


def test_resolve_predictor_cache_miss_loads_from_storage():
    """Predictor resolution should load from storage and cache on miss."""
    request = _make_request(prediction_type="quantile")
    predictor_instance = MagicMock()
    model_data = {
        "metadata": {"prediction_type": "quantile"},
        "predictor_state": {"weights": [1, 2, 3]},
    }

    with (
        patch(
            "swarmpilot.predictor.api.dependencies.storage.generate_model_key",
            return_value="key-2",
        ),
        patch(
            "swarmpilot.predictor.api.dependencies.model_cache.get",
            return_value=None,
        ),
        patch(
            "swarmpilot.predictor.api.dependencies.storage.load_model",
            return_value=model_data,
        ),
        patch(
            "swarmpilot.predictor.api.dependencies.model_cache.put",
        ) as mock_cache_put,
        patch(
            "swarmpilot.predictor.api.services.prediction_service.create_predictor",
            return_value=predictor_instance,
        ),
    ):
        resolved = resolve_predictor(request)

    assert resolved is predictor_instance
    predictor_instance.load_model_state.assert_called_once_with(
        model_data["predictor_state"]
    )
    mock_cache_put.assert_called_once_with(
        "key-2", predictor_instance, "quantile"
    )


def test_resolve_predictor_model_not_found_raises():
    """Predictor resolution should raise 404 service error when model is missing."""
    request = _make_request(prediction_type="expect_error")

    with (
        patch(
            "swarmpilot.predictor.api.dependencies.storage.generate_model_key",
            return_value="missing-key",
        ),
        patch(
            "swarmpilot.predictor.api.dependencies.model_cache.get",
            return_value=None,
        ),
        patch(
            "swarmpilot.predictor.api.dependencies.storage.load_model",
            return_value=None,
        ),
        pytest.raises(PredictionServiceError) as exc_info,
    ):
        resolve_predictor(request)

    assert exc_info.value.status_code == 404


def test_resolve_predictor_type_mismatch_raises():
    """Predictor resolution should raise 400 service error on type mismatch."""
    request = _make_request(prediction_type="quantile")
    predictor = MagicMock()

    with (
        patch(
            "swarmpilot.predictor.api.dependencies.storage.generate_model_key",
            return_value="key-3",
        ),
        patch(
            "swarmpilot.predictor.api.dependencies.model_cache.get",
            return_value=(predictor, "expect_error"),
        ),
        pytest.raises(PredictionServiceError) as exc_info,
    ):
        resolve_predictor(request)

    assert exc_info.value.status_code == 400


def test_prepare_features_merges_hardware_specs():
    """Feature preparation should merge GPU specs into feature payload."""
    request = _make_request(features={"x": 1.0})

    with patch.object(
        PlatformInfo,
        "extract_gpu_specs",
        return_value={"cuda_cores": 6912, "memory_gb": 80},
    ):
        features = prepare_features(request)

    assert features["x"] == 1.0
    assert features["cuda_cores"] == 6912
    assert features["memory_gb"] == 80


def test_prepare_features_without_preprocessors():
    """Feature preparation should keep original features when preprocessing disabled."""
    request = _make_request(enable_preprocessors=None, features={"x": 2.0})

    with patch.object(
        PlatformInfo,
        "extract_gpu_specs",
        return_value={"fp32_tflops": 19.5},
    ):
        features = prepare_features(request)

    assert features == {"x": 2.0, "fp32_tflops": 19.5}


def test_prepare_features_missing_key_raises_valueerror():
    """Feature preparation should raise ValueError when mapped keys are missing."""
    request = _make_request(
        enable_preprocessors=["semantic"],
        preprocessor_mappings={"semantic": ["sentence"]},
        features={"x": 1.0},
    )

    with (
        patch.object(PlatformInfo, "extract_gpu_specs", return_value=None),
        patch(
            "swarmpilot.predictor.api.dependencies.preprocessors_registry.get_preprocessor",
            return_value=MagicMock(),
        ),
        pytest.raises(ValueError, match="not all found in features"),
    ):
        prepare_features(request)


def test_execute_prediction_full_flow():
    """Execution should compose experiment check, resolve, preprocess, and predict."""
    request = _make_request(prediction_type="expect_error")
    predictor = MagicMock()
    predictor.predict.return_value = {
        "expected_runtime_ms": 123.0,
        "error_margin_ms": 5.0,
    }

    with (
        patch(
            "swarmpilot.predictor.api.services.prediction_service.try_experiment_mode",
            return_value=None,
        ),
        patch(
            "swarmpilot.predictor.api.services.prediction_service.resolve_predictor",
            return_value=predictor,
        ),
        patch(
            "swarmpilot.predictor.api.services.prediction_service.prepare_features",
            return_value={"x": 1.0, "y": 2.0},
        ),
    ):
        response = execute_prediction(request)

    assert response.model_id == request.model_id
    assert response.platform_info == request.platform_info
    assert response.prediction_type == request.prediction_type
    assert response.result == predictor.predict.return_value
