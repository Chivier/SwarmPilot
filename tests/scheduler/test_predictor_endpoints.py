"""Tests for predictor management endpoints.

Verifies the ``/v1/predictor/*`` routes exposed by the scheduler,
mocking the underlying predictor and training library clients.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from swarmpilot.scheduler.clients.models import Prediction

# ====================================================================
# Fixtures
# ====================================================================


@pytest.fixture()
def client() -> TestClient:
    """Create a synchronous test client for the scheduler app."""
    from swarmpilot.scheduler.api import app

    return TestClient(app)


# ====================================================================
# POST /v1/predictor/train
# ====================================================================


class TestTrainEndpoint:
    """Tests for POST /v1/predictor/train."""

    @patch("swarmpilot.scheduler.routes.predictor._get_clients")
    def test_train_success(
        self, mock_get_clients: MagicMock, client: TestClient
    ) -> None:
        """Training returns success when flush succeeds."""
        mock_predictor = MagicMock()
        mock_training = MagicMock()
        mock_training.flush = MagicMock(return_value=True)
        mock_training.get_buffer_size = MagicMock(return_value=0)

        async def _flush(force: bool = False) -> bool:
            return True

        mock_training.flush = _flush
        mock_get_clients.return_value = (
            mock_predictor,
            mock_training,
        )

        resp = client.post(
            "/v1/predictor/train",
            json={
                "model_id": "test-model",
                "prediction_type": "expect_error",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["model_id"] == "test-model"

    @patch("swarmpilot.scheduler.routes.predictor._get_clients")
    def test_train_no_training_client(
        self, mock_get_clients: MagicMock, client: TestClient
    ) -> None:
        """Returns 503 when training client is None."""
        mock_get_clients.return_value = (MagicMock(), None)

        resp = client.post(
            "/v1/predictor/train",
            json={"model_id": "test-model"},
        )

        assert resp.status_code == 503

    @patch("swarmpilot.scheduler.routes.predictor._get_clients")
    def test_train_no_samples(
        self, mock_get_clients: MagicMock, client: TestClient
    ) -> None:
        """Returns success=False when buffer is empty."""
        mock_predictor = MagicMock()
        mock_training = MagicMock()
        mock_training.get_buffer_size = MagicMock(return_value=0)

        async def _flush(force: bool = False) -> bool:
            return False

        mock_training.flush = _flush
        mock_get_clients.return_value = (
            mock_predictor,
            mock_training,
        )

        resp = client.post(
            "/v1/predictor/train",
            json={"model_id": "test-model"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert "No samples" in body["message"]


# ====================================================================
# POST /v1/predictor/retrain
# ====================================================================


class TestRetrainEndpoint:
    """Tests for POST /v1/predictor/retrain."""

    @patch("swarmpilot.scheduler.routes.predictor._get_clients")
    def test_retrain_delegates_to_train(
        self, mock_get_clients: MagicMock, client: TestClient
    ) -> None:
        """Retrain delegates to the same logic as train."""
        mock_predictor = MagicMock()
        mock_training = MagicMock()
        mock_training.get_buffer_size = MagicMock(return_value=5)

        async def _flush(force: bool = False) -> bool:
            return True

        mock_training.flush = _flush
        mock_get_clients.return_value = (
            mock_predictor,
            mock_training,
        )

        resp = client.post(
            "/v1/predictor/retrain",
            json={
                "model_id": "test-model",
                "prediction_type": "quantile",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["model_id"] == "test-model"


# ====================================================================
# GET /v1/predictor/status/{model_id}
# ====================================================================


class TestStatusEndpoint:
    """Tests for GET /v1/predictor/status/{model_id}."""

    @patch("swarmpilot.scheduler.routes.predictor._get_clients")
    def test_status_found(
        self, mock_get_clients: MagicMock, client: TestClient
    ) -> None:
        """Returns status when model exists in storage."""
        mock_predictor = MagicMock()
        mock_storage = MagicMock()
        mock_storage.list_models.return_value = [
            {
                "model_id": "my-model",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "gpu-1",
                },
                "prediction_type": "expect_error",
                "samples_count": 50,
                "last_trained": "2025-01-01T00:00:00",
            },
        ]
        mock_predictor._low_level._storage = mock_storage

        mock_get_clients.return_value = (mock_predictor, None)

        resp = client.get("/v1/predictor/status/my-model")

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["model_id"] == "my-model"
        assert len(body["models"]) == 1
        assert body["samples_collected"] == 0

    @patch("swarmpilot.scheduler.routes.predictor._get_clients")
    def test_status_not_found(
        self, mock_get_clients: MagicMock, client: TestClient
    ) -> None:
        """Returns 404 when model has never been seen."""
        mock_predictor = MagicMock()
        mock_storage = MagicMock()
        mock_storage.list_models.return_value = []
        mock_predictor._low_level._storage = mock_storage

        mock_get_clients.return_value = (mock_predictor, None)

        resp = client.get("/v1/predictor/status/no-such-model")

        assert resp.status_code == 404

    @patch("swarmpilot.scheduler.routes.predictor._get_clients")
    def test_status_with_buffered_samples(
        self, mock_get_clients: MagicMock, client: TestClient
    ) -> None:
        """Counts buffered training samples for the model."""
        mock_predictor = MagicMock()
        mock_storage = MagicMock()
        mock_storage.list_models.return_value = []
        mock_predictor._low_level._storage = mock_storage

        # Build a fake training client with samples
        mock_training = MagicMock()
        sample = MagicMock()
        sample.model_id = "buf-model"
        mock_training._samples_buffer = [sample, sample]

        mock_get_clients.return_value = (
            mock_predictor,
            mock_training,
        )

        resp = client.get("/v1/predictor/status/buf-model")

        assert resp.status_code == 200
        body = resp.json()
        assert body["samples_collected"] == 2


# ====================================================================
# POST /v1/predictor/predict
# ====================================================================


class TestPredictEndpoint:
    """Tests for POST /v1/predictor/predict."""

    @patch("swarmpilot.scheduler.routes.predictor._get_clients")
    def test_predict_expect_error(
        self, mock_get_clients: MagicMock, client: TestClient
    ) -> None:
        """Returns prediction for expect_error type."""
        mock_predictor = MagicMock()
        mock_predictor._predict_single_platform.return_value = [
            Prediction(
                instance_id="manual",
                predicted_time_ms=123.4,
                confidence=None,
                quantiles=None,
                error_margin_ms=10.0,
            ),
        ]

        mock_get_clients.return_value = (mock_predictor, None)

        resp = client.post(
            "/v1/predictor/predict",
            json={
                "model_id": "test-model",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "gpu-1",
                },
                "features": {"input_tokens": 100},
                "prediction_type": "expect_error",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["expected_runtime_ms"] == 123.4
        assert body["error_margin_ms"] == 10.0
        assert body["quantiles"] is None

    @patch("swarmpilot.scheduler.routes.predictor._get_clients")
    def test_predict_quantile(
        self, mock_get_clients: MagicMock, client: TestClient
    ) -> None:
        """Returns prediction for quantile type."""
        mock_predictor = MagicMock()
        mock_predictor._predict_single_platform.return_value = [
            Prediction(
                instance_id="manual",
                predicted_time_ms=150.0,
                confidence=None,
                quantiles={0.5: 100.0, 0.9: 200.0},
            ),
        ]

        mock_get_clients.return_value = (mock_predictor, None)

        resp = client.post(
            "/v1/predictor/predict",
            json={
                "model_id": "test-model",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "gpu-1",
                },
                "features": {"input_tokens": 100},
                "prediction_type": "quantile",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["quantiles"] == {"0.5": 100.0, "0.9": 200.0}

    @patch("swarmpilot.scheduler.routes.predictor._get_clients")
    def test_predict_model_not_found(
        self, mock_get_clients: MagicMock, client: TestClient
    ) -> None:
        """Returns 404 when model is not trained."""
        mock_predictor = MagicMock()
        mock_predictor._predict_single_platform.side_effect = (
            ValueError("Model not found: test-model")
        )

        mock_get_clients.return_value = (mock_predictor, None)

        resp = client.post(
            "/v1/predictor/predict",
            json={
                "model_id": "test-model",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "gpu-1",
                },
                "features": {"input_tokens": 100},
                "prediction_type": "expect_error",
            },
        )

        assert resp.status_code == 404

    @patch("swarmpilot.scheduler.routes.predictor._get_clients")
    def test_predict_bad_features(
        self, mock_get_clients: MagicMock, client: TestClient
    ) -> None:
        """Returns 400 on feature validation error."""
        mock_predictor = MagicMock()
        mock_predictor._predict_single_platform.side_effect = (
            ValueError("Invalid features")
        )

        mock_get_clients.return_value = (mock_predictor, None)

        resp = client.post(
            "/v1/predictor/predict",
            json={
                "model_id": "test-model",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "gpu-1",
                },
                "features": {},
                "prediction_type": "expect_error",
            },
        )

        assert resp.status_code == 400


# ====================================================================
# GET /v1/predictor/models
# ====================================================================


class TestModelsEndpoint:
    """Tests for GET /v1/predictor/models."""

    @patch("swarmpilot.scheduler.routes.predictor._get_clients")
    def test_list_models_empty(
        self, mock_get_clients: MagicMock, client: TestClient
    ) -> None:
        """Returns empty list when no models are trained."""
        mock_predictor = MagicMock()
        mock_storage = MagicMock()
        mock_storage.list_models.return_value = []
        mock_predictor._low_level._storage = mock_storage

        mock_get_clients.return_value = (mock_predictor, None)

        resp = client.get("/v1/predictor/models")

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["models"] == []

    @patch("swarmpilot.scheduler.routes.predictor._get_clients")
    def test_list_models_populated(
        self, mock_get_clients: MagicMock, client: TestClient
    ) -> None:
        """Returns all trained models from storage."""
        entries = [
            {
                "model_id": "model-a",
                "prediction_type": "expect_error",
                "samples_count": 100,
            },
            {
                "model_id": "model-b",
                "prediction_type": "quantile",
                "samples_count": 50,
            },
        ]
        mock_predictor = MagicMock()
        mock_storage = MagicMock()
        mock_storage.list_models.return_value = entries
        mock_predictor._low_level._storage = mock_storage

        mock_get_clients.return_value = (mock_predictor, None)

        resp = client.get("/v1/predictor/models")

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert len(body["models"]) == 2
        assert body["models"][0]["model_id"] == "model-a"
        assert body["models"][1]["model_id"] == "model-b"


# ====================================================================
# Strategy auto-switch on successful training
# ====================================================================


def _make_training_client(
    flush_result: bool,
    buffer_size: int,
) -> MagicMock:
    """Build a mock training client.

    Args:
        flush_result: Value returned by ``flush()``.
        buffer_size: Value returned by ``get_buffer_size()``.

    Returns:
        Configured MagicMock.
    """
    mock = MagicMock()

    async def _flush(force: bool = False) -> bool:
        return flush_result

    mock.flush = _flush
    mock.get_buffer_size = MagicMock(return_value=buffer_size)
    return mock


class TestTrainStrategySwitching:
    """Strategy auto-switch after successful predictor training."""

    @patch(
        "swarmpilot.scheduler.routes.predictor"
        "._maybe_switch_to_probabilistic",
        return_value="probabilistic",
    )
    @patch("swarmpilot.scheduler.routes.predictor._get_clients")
    def test_train_switches_strategy_on_enough_samples(
        self,
        mock_get_clients: MagicMock,
        mock_switch: MagicMock,
        client: TestClient,
    ) -> None:
        """Successful train with enough samples triggers switch."""
        mock_get_clients.return_value = (
            MagicMock(),
            _make_training_client(
                flush_result=True, buffer_size=20
            ),
        )

        resp = client.post(
            "/v1/predictor/train",
            json={"model_id": "m1"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["strategy"] == "probabilistic"
        mock_switch.assert_called_once_with(20)

    @patch(
        "swarmpilot.scheduler.routes.predictor"
        "._maybe_switch_to_probabilistic",
        return_value=None,
    )
    @patch("swarmpilot.scheduler.routes.predictor._get_clients")
    def test_train_no_switch_on_insufficient_samples(
        self,
        mock_get_clients: MagicMock,
        mock_switch: MagicMock,
        client: TestClient,
    ) -> None:
        """Train with too few samples does NOT switch strategy."""
        mock_get_clients.return_value = (
            MagicMock(),
            _make_training_client(
                flush_result=True, buffer_size=5
            ),
        )

        resp = client.post(
            "/v1/predictor/train",
            json={"model_id": "m1"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["strategy"] is None
        mock_switch.assert_called_once_with(5)

    @patch(
        "swarmpilot.scheduler.routes.predictor"
        "._maybe_switch_to_probabilistic",
    )
    @patch("swarmpilot.scheduler.routes.predictor._get_clients")
    def test_failed_training_keeps_strategy(
        self,
        mock_get_clients: MagicMock,
        mock_switch: MagicMock,
        client: TestClient,
    ) -> None:
        """Failed training (success=False) must not attempt switch."""
        mock_get_clients.return_value = (
            MagicMock(),
            _make_training_client(
                flush_result=False, buffer_size=0
            ),
        )

        resp = client.post(
            "/v1/predictor/train",
            json={"model_id": "m1"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["strategy"] is None
        mock_switch.assert_not_called()

    @patch(
        "swarmpilot.scheduler.routes.predictor"
        "._maybe_switch_to_probabilistic",
        return_value="probabilistic",
    )
    @patch("swarmpilot.scheduler.routes.predictor._get_clients")
    def test_retrain_switches_strategy_on_success(
        self,
        mock_get_clients: MagicMock,
        mock_switch: MagicMock,
        client: TestClient,
    ) -> None:
        """Retrain also switches strategy on success."""
        mock_get_clients.return_value = (
            MagicMock(),
            _make_training_client(
                flush_result=True, buffer_size=15
            ),
        )

        resp = client.post(
            "/v1/predictor/retrain",
            json={"model_id": "m1"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["strategy"] == "probabilistic"
        mock_switch.assert_called_once_with(15)


class TestMaybeSwitchToProbabilistic:
    """Unit tests for ``_maybe_switch_to_probabilistic``."""

    def test_returns_none_when_below_threshold(self) -> None:
        """Below MIN_TRAINING_SAMPLES returns None, no switch."""
        from swarmpilot.scheduler.routes.predictor import (
            _maybe_switch_to_probabilistic,
        )

        result = _maybe_switch_to_probabilistic(9)
        assert result is None

    @patch("swarmpilot.scheduler.algorithms.get_strategy")
    def test_switches_strategy_above_threshold(
        self,
        mock_get_strategy: MagicMock,
    ) -> None:
        """Above threshold creates new strategy and assigns it."""
        from swarmpilot.scheduler.routes.predictor import (
            _maybe_switch_to_probabilistic,
        )

        # Fake the current strategy as non-probabilistic.
        mock_old_strategy = MagicMock()
        mock_old_strategy.__class__ = type(
            "RoundRobinStrategy", (), {}
        )
        mock_old_strategy._worker_queue_manager = MagicMock()

        mock_new_strategy = MagicMock()
        mock_get_strategy.return_value = mock_new_strategy

        with patch(
            "swarmpilot.scheduler.api.scheduling_strategy",
            mock_old_strategy,
        ):
            result = _maybe_switch_to_probabilistic(20)

        assert result == "probabilistic"
        mock_get_strategy.assert_called_once()
        mock_new_strategy.set_worker_queue_manager.assert_called_once_with(
            mock_old_strategy._worker_queue_manager,
        )

    def test_already_probabilistic_skips_switch(
        self,
    ) -> None:
        """If already probabilistic, returns name without reset."""
        from swarmpilot.scheduler.routes.predictor import (
            _maybe_switch_to_probabilistic,
        )

        mock_strategy = MagicMock(
            spec=["__class__", "_worker_queue_manager"],
        )
        mock_strategy.__class__ = type(
            "ProbabilisticSchedulingStrategy", (), {}
        )

        with patch(
            "swarmpilot.scheduler.api.scheduling_strategy",
            mock_strategy,
        ):
            result = _maybe_switch_to_probabilistic(50)

        assert result == "probabilistic"

    @patch(
        "swarmpilot.scheduler.algorithms.get_strategy",
        side_effect=RuntimeError("boom"),
    )
    def test_returns_none_on_exception(
        self,
        mock_get_strategy: MagicMock,
    ) -> None:
        """Returns None and logs error on exception."""
        from swarmpilot.scheduler.routes.predictor import (
            _maybe_switch_to_probabilistic,
        )

        mock_old = MagicMock()
        mock_old.__class__ = type("RoundRobinStrategy", (), {})

        with patch(
            "swarmpilot.scheduler.api.scheduling_strategy",
            mock_old,
        ):
            result = _maybe_switch_to_probabilistic(20)

        assert result is None
