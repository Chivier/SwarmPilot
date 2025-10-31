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
    @patch("src.predictor_client.httpx.AsyncClient")
    async def test_predict_single_instance(self, mock_client_class, sample_instance):
        """Test prediction for single instance."""
        # Mock the HTTP response with correct format
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "quantiles": {
                    "0.5": 100.0,
                    "0.9": 150.0,
                    "0.95": 200.0,
                    "0.99": 300.0
                }
            }
        }
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = AsyncMock()
        mock_client_class.return_value = mock_client

        client = PredictorClient("http://localhost:8080")

        predictions = await client.predict(
            model_id="model-1",
            metadata={"size": "large"},
            instances=[sample_instance]
        )

        assert len(predictions) == 1
        assert predictions[0].instance_id == sample_instance.instance_id
        assert predictions[0].predicted_time_ms == 100.0  # Median (P50)

    @pytest.mark.asyncio
    @patch("src.predictor_client.httpx.AsyncClient")
    async def test_predict_multiple_instances(self, mock_client_class, sample_instances):
        """Test prediction for multiple instances."""
        # Mock the HTTP response with correct format
        # Note: Since sample_instances have different hardware names, they need separate calls
        # But they share same software, so we'll mock based on first call
        mock_response = MagicMock()

        # Return different responses for different platform_info
        def json_side_effect():
            return {
                "result": {
                    "quantiles": {
                        "0.5": 120.0,
                        "0.9": 180.0,
                        "0.95": 220.0,
                        "0.99": 350.0
                    }
                }
            }

        mock_response.json.side_effect = [json_side_effect() for _ in range(3)]
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = AsyncMock()
        mock_client_class.return_value = mock_client

        client = PredictorClient("http://localhost:8080")

        predictions = await client.predict(
            model_id="model-1",
            metadata={},
            instances=sample_instances
        )

        assert len(predictions) == 3
        instance_ids = [p.instance_id for p in predictions]
        for instance in sample_instances:
            assert instance.instance_id in instance_ids

    @pytest.mark.asyncio
    async def test_predict_empty_instance_list(self):
        """Test prediction with empty instance list."""
        client = PredictorClient("http://localhost:8080")

        predictions = await client.predict(
            model_id="model-1",
            metadata={},
            instances=[]
        )

        assert predictions == []

    @pytest.mark.asyncio
    @patch("src.predictor_client.httpx.AsyncClient")
    async def test_predict_returns_quantiles(self, mock_client_class, sample_instance):
        """Test that predictions include quantile information."""
        # Mock the HTTP response with quantiles in correct format
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "quantiles": {
                    "0.5": 90.0,
                    "0.9": 160.0,
                    "0.95": 210.0,
                    "0.99": 320.0
                }
            }
        }
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = AsyncMock()
        mock_client_class.return_value = mock_client

        client = PredictorClient("http://localhost:8080")

        predictions = await client.predict(
            model_id="model-1",
            metadata={},
            instances=[sample_instance]
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
