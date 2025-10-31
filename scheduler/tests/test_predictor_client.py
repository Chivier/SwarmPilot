"""
Unit tests for PredictorClient.

Tests HTTP communication, predictions, and health checks with mocking.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from src.predictor_client import PredictorClient, Prediction


# ============================================================================
# Prediction Dataclass Tests
# ============================================================================

class TestPrediction:
    """Tests for Prediction dataclass."""

    def test_prediction_initialization(self):
        """Test creating a Prediction."""
        pred = Prediction(
            instance_id="inst-1",
            predicted_time_ms=150.5,
            confidence=0.95,
            quantiles={0.5: 100.0, 0.9: 200.0}
        )

        assert pred.instance_id == "inst-1"
        assert pred.predicted_time_ms == 150.5
        assert pred.confidence == 0.95
        assert pred.quantiles == {0.5: 100.0, 0.9: 200.0}

    def test_prediction_optional_fields(self):
        """Test Prediction with optional fields as None."""
        pred = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0
        )

        assert pred.confidence is None
        assert pred.quantiles is None


# ============================================================================
# PredictorClient Initialization Tests
# ============================================================================

class TestPredictorClientInit:
    """Tests for PredictorClient initialization."""

    def test_init_with_trailing_slash(self):
        """Test that trailing slash is removed from URL."""
        client = PredictorClient("http://localhost:8080/", timeout=10.0)

        assert client.predictor_url == "http://localhost:8080"
        assert client.timeout == 10.0

    def test_init_without_trailing_slash(self):
        """Test initialization with URL without trailing slash."""
        client = PredictorClient("http://localhost:8080", timeout=5.0)

        assert client.predictor_url == "http://localhost:8080"
        assert client.timeout == 5.0

    def test_init_default_timeout(self):
        """Test default timeout value."""
        client = PredictorClient("http://localhost:8080")

        assert client.timeout == 5.0


# ============================================================================
# Prediction Tests (Current Dummy Implementation)
# ============================================================================

class TestPredictMethod:
    """Tests for predict method (current dummy implementation)."""

    @pytest.mark.asyncio
    async def test_predict_single_instance(self):
        """Test prediction for single instance."""
        client = PredictorClient("http://localhost:8080")

        predictions = await client.predict(
            model_id="model-1",
            metadata={"size": "large"},
            instance_ids=["inst-1"]
        )

        assert len(predictions) == 1
        assert predictions[0].instance_id == "inst-1"
        assert predictions[0].predicted_time_ms == 100.0
        assert predictions[0].confidence == 0.8

    @pytest.mark.asyncio
    async def test_predict_multiple_instances(self):
        """Test prediction for multiple instances."""
        client = PredictorClient("http://localhost:8080")

        predictions = await client.predict(
            model_id="model-1",
            metadata={},
            instance_ids=["inst-1", "inst-2", "inst-3"]
        )

        assert len(predictions) == 3
        instance_ids = [p.instance_id for p in predictions]
        assert "inst-1" in instance_ids
        assert "inst-2" in instance_ids
        assert "inst-3" in instance_ids

    @pytest.mark.asyncio
    async def test_predict_empty_instance_list(self):
        """Test prediction with empty instance list."""
        client = PredictorClient("http://localhost:8080")

        predictions = await client.predict(
            model_id="model-1",
            metadata={},
            instance_ids=[]
        )

        assert predictions == []

    @pytest.mark.asyncio
    async def test_predict_returns_quantiles(self):
        """Test that predictions include quantile information."""
        client = PredictorClient("http://localhost:8080")

        predictions = await client.predict(
            model_id="model-1",
            metadata={},
            instance_ids=["inst-1"]
        )

        assert predictions[0].quantiles is not None
        assert 0.5 in predictions[0].quantiles
        assert 0.9 in predictions[0].quantiles
        assert 0.95 in predictions[0].quantiles
        assert 0.99 in predictions[0].quantiles


# ============================================================================
# Health Check Tests
# ============================================================================

class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check when service is healthy."""
        client = PredictorClient("http://localhost:8080")

        # Mock httpx.AsyncClient
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await client.health_check()

            assert result is True
            mock_client.get.assert_called_once_with("http://localhost:8080/health")

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_status(self):
        """Test health check when service returns non-200 status."""
        client = PredictorClient("http://localhost:8080")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await client.health_check()

            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self):
        """Test health check when connection fails."""
        client = PredictorClient("http://localhost:8080")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await client.health_check()

            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_timeout(self):
        """Test health check when request times out."""
        client = PredictorClient("http://localhost:8080", timeout=1.0)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await client.health_check()

            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_generic_exception(self):
        """Test health check with generic exception."""
        client = PredictorClient("http://localhost:8080")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=Exception("Unknown error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await client.health_check()

            assert result is False
