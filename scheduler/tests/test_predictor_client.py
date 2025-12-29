"""Unit tests for PredictorClient with HTTP API support.

Tests HTTP API communication, predictions, retry logic, and health checks.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.model import Instance
from src.predictor_client import Prediction, PredictorClient

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
            quantiles={0.5: 100.0, 0.9: 200.0},
        )

        assert pred.instance_id == "inst-1"
        assert pred.predicted_time_ms == 150.5
        assert pred.confidence == 0.95
        assert pred.quantiles == {0.5: 100.0, 0.9: 200.0}

    def test_prediction_optional_fields(self):
        """Test Prediction with optional fields as None."""
        pred = Prediction(instance_id="inst-1", predicted_time_ms=100.0)

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

    def test_init_default_values(self):
        """Test default initialization values."""
        client = PredictorClient("http://localhost:8080")

        assert client.timeout == 5.0
        assert client.max_retries == 3
        assert client.retry_delay == 1.0
        assert client._client is None

    def test_init_custom_retry_params(self):
        """Test initialization with custom retry parameters."""
        client = PredictorClient(
            "http://localhost:8080", max_retries=5, retry_delay=2.0
        )

        assert client.max_retries == 5
        assert client.retry_delay == 2.0


# ============================================================================
# HTTP Client Management Tests
# ============================================================================


class TestHTTPClientConnection:
    """Tests for HTTP client management."""

    @pytest.mark.asyncio
    async def test_ensure_client_establishes_new(self):
        """Test that _ensure_client establishes a new HTTP client."""
        client = PredictorClient("http://localhost:8080")

        mock_client = AsyncMock()
        mock_client.is_closed = False

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await client._ensure_client()

            assert result == mock_client
            assert client._client == mock_client

    @pytest.mark.asyncio
    async def test_ensure_client_reuses_existing(self):
        """Test that existing open client is reused."""
        client = PredictorClient("http://localhost:8080")

        mock_client = AsyncMock()
        mock_client.is_closed = False
        client._client = mock_client

        result = await client._ensure_client()

        assert result == mock_client
        # Should return same client without creating new one

    @pytest.mark.asyncio
    async def test_ensure_client_reconnects_if_closed(self):
        """Test that closed client triggers reconnection."""
        client = PredictorClient("http://localhost:8080")

        # Old closed client
        old_client = AsyncMock()
        old_client.is_closed = True
        client._client = old_client

        # New client
        new_client = AsyncMock()
        new_client.is_closed = False

        with patch("httpx.AsyncClient", return_value=new_client):
            result = await client._ensure_client()

            assert result == new_client
            assert client._client == new_client

    @pytest.mark.asyncio
    async def test_ensure_client_failure(self):
        """Test client initialization failure handling."""
        client = PredictorClient("http://localhost:8080")

        with patch(
            "httpx.AsyncClient", side_effect=Exception("Initialization failed")
        ):
            with pytest.raises(ConnectionError) as exc_info:
                await client._ensure_client()

            assert "Failed to initialize" in str(exc_info.value)
            assert client._client is None

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test closing HTTP client."""
        client = PredictorClient("http://localhost:8080")

        mock_client = AsyncMock()
        mock_client.is_closed = False
        client._client = mock_client

        await client._close_client()

        mock_client.aclose.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_client_with_error(self):
        """Test closing client handles errors gracefully."""
        client = PredictorClient("http://localhost:8080")

        mock_client = AsyncMock()
        mock_client.aclose.side_effect = Exception("Close error")
        mock_client.is_closed = False
        client._client = mock_client

        # Should not raise exception
        await client._close_client()

        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_predictor_client(self):
        """Test closing the predictor client."""
        client = PredictorClient("http://localhost:8080")

        mock_client = AsyncMock()
        mock_client.is_closed = False
        client._client = mock_client

        await client.close()

        mock_client.aclose.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager support."""
        client = PredictorClient("http://localhost:8080")

        async with client as c:
            assert c == client

        # Client should be closed after exiting context
        # (if there was one, it would be closed)


# ============================================================================
# HTTP Request with Retry Tests
# ============================================================================


class TestMakeRequestWithRetry:
    """Tests for _make_request_with_retry method."""

    @pytest.fixture
    def predictor_client(self):
        """Create a PredictorClient with short retry delays for testing."""
        return PredictorClient(
            "http://localhost:8080",
            timeout=1.0,
            max_retries=3,
            retry_delay=0.01,  # Very short for testing
        )

    @pytest.mark.asyncio
    async def test_successful_request(self, predictor_client):
        """Test successful HTTP request returns response data."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {"quantiles": {"0.5": 100.0}}
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await predictor_client._make_request_with_retry(
                {"model_id": "test", "features": {}}
            )

        assert result == {"result": {"quantiles": {"0.5": 100.0}}}

    @pytest.mark.asyncio
    async def test_400_bad_request_no_retry(self, predictor_client):
        """Test 400 Bad Request raises ValueError without retry."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "detail": {"error": "Invalid request", "message": "Bad input"}
        }
        mock_response.text = "Bad Request"

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ValueError) as exc_info:
                await predictor_client._make_request_with_retry(
                    {"model_id": "test", "features": {}}
                )

        assert "Invalid request" in str(exc_info.value)
        # Should only be called once (no retries for 400)
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_400_with_string_detail(self, predictor_client):
        """Test 400 with string detail instead of dict."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"detail": "Simple error message"}
        mock_response.text = "Bad Request"

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ValueError) as exc_info:
                await predictor_client._make_request_with_retry(
                    {"model_id": "test", "features": {}}
                )

        assert "Simple error message" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_400_json_parse_error(self, predictor_client):
        """Test 400 when JSON parsing fails."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.side_effect = Exception("JSON parse error")
        mock_response.text = "Plain text error"

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ValueError) as exc_info:
                await predictor_client._make_request_with_retry(
                    {"model_id": "test", "features": {}}
                )

        assert "Plain text error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_404_model_not_found(self, predictor_client):
        """Test 404 raises ValueError without retry."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            "detail": {
                "error": "Model not found",
                "message": "No model for test",
            }
        }
        mock_response.text = "Not Found"

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ValueError) as exc_info:
                await predictor_client._make_request_with_retry(
                    {"model_id": "test", "features": {}}
                )

        assert "Model not found" in str(exc_info.value)
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_500_server_error_retries(self, predictor_client):
        """Test 500 Server Error retries and eventually fails."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(RuntimeError) as exc_info:
                await predictor_client._make_request_with_retry(
                    {"model_id": "test", "features": {}}
                )

        assert "Server error" in str(exc_info.value)
        assert mock_client.post.call_count == 3  # Retried max_retries times

    @pytest.mark.asyncio
    async def test_other_http_error(self, predictor_client):
        """Test other HTTP status codes (e.g., 403) raise ValueError."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ValueError) as exc_info:
                await predictor_client._make_request_with_retry(
                    {"model_id": "test", "features": {}}
                )

        assert "HTTP 403" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_connection_error_retries(self, predictor_client):
        """Test connection errors trigger retries."""
        import httpx

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock_client.aclose = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ConnectionError) as exc_info:
                await predictor_client._make_request_with_retry(
                    {"model_id": "test", "features": {}}
                )

        assert "unavailable after" in str(exc_info.value)
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_error_retries(self, predictor_client):
        """Test timeout errors trigger retries."""
        import httpx

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(
            side_effect=httpx.TimeoutException("Request timeout")
        )
        mock_client.aclose = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(TimeoutError) as exc_info:
                await predictor_client._make_request_with_retry(
                    {"model_id": "test", "features": {}}
                )

        assert "timeout after" in str(exc_info.value)
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_json_decode_error_no_retry(self, predictor_client):
        """Test JSON decode errors don't trigger retries."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError(
            "Invalid JSON", "", 0
        )

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(json.JSONDecodeError):
                await predictor_client._make_request_with_retry(
                    {"model_id": "test", "features": {}}
                )

        assert mock_client.post.call_count == 1


# ============================================================================
# Predict Method Tests
# ============================================================================


class TestPredictMethod:
    """Tests for predict method with platform batching."""

    @pytest.fixture
    def predictor_client(self):
        """Create a PredictorClient for testing."""
        return PredictorClient(
            "http://localhost:8080",
            timeout=1.0,
            max_retries=1,
            retry_delay=0.01,
        )

    @pytest.fixture
    def sample_instances(self):
        """Create sample instances for testing."""
        return [
            Instance(
                instance_id="inst-1",
                model_id="test-model",
                endpoint="http://localhost:9001",
                platform_info={
                    "software_name": "Linux",
                    "software_version": "5.15",
                    "hardware_name": "x86_64",
                },
            ),
            Instance(
                instance_id="inst-2",
                model_id="test-model",
                endpoint="http://localhost:9002",
                platform_info={
                    "software_name": "Linux",
                    "software_version": "5.15",
                    "hardware_name": "x86_64",
                },
            ),
            Instance(
                instance_id="inst-3",
                model_id="test-model",
                endpoint="http://localhost:9003",
                platform_info={
                    "software_name": "Windows",
                    "software_version": "10.0",
                    "hardware_name": "x86_64",
                },
            ),
        ]

    @pytest.mark.asyncio
    async def test_predict_expect_error_type(
        self, predictor_client, sample_instances
    ):
        """Test predict with expect_error prediction type."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {"expected_runtime_ms": 100.0, "error_margin_ms": 10.0}
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            predictions = await predictor_client.predict(
                model_id="test-model",
                metadata={"image_size": 1024},
                instances=sample_instances[:2],  # Same platform
                prediction_type="expect_error",
            )

        assert len(predictions) == 2
        for pred in predictions:
            assert pred.predicted_time_ms == 100.0
            assert pred.error_margin_ms == 10.0
            assert pred.quantiles is None

    @pytest.mark.asyncio
    async def test_predict_quantile_type(
        self, predictor_client, sample_instances
    ):
        """Test predict with quantile prediction type."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {"quantiles": {"0.5": 100.0, "0.9": 150.0, "0.95": 180.0}}
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            predictions = await predictor_client.predict(
                model_id="test-model",
                metadata={"image_size": 1024},
                instances=sample_instances[:2],
                prediction_type="quantile",
            )

        assert len(predictions) == 2
        for pred in predictions:
            assert pred.predicted_time_ms == 100.0  # Median (0.5 quantile)
            assert pred.quantiles == {0.5: 100.0, 0.9: 150.0, 0.95: 180.0}

    @pytest.mark.asyncio
    async def test_predict_platform_batching(
        self, predictor_client, sample_instances
    ):
        """Test that predictions are batched by platform."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {"quantiles": {"0.5": 100.0}}
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            predictions = await predictor_client.predict(
                model_id="test-model",
                metadata={"image_size": 1024},
                instances=sample_instances,  # 2 Linux, 1 Windows
                prediction_type="quantile",
            )

        # Should have 3 predictions (one per instance)
        assert len(predictions) == 3
        # But only 2 HTTP calls (one per unique platform)
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_predict_unknown_type_raises(
        self, predictor_client, sample_instances
    ):
        """Test predict with unknown prediction type raises ValueError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {"quantiles": {"0.5": 100.0}}
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ValueError) as exc_info:
                await predictor_client.predict(
                    model_id="test-model",
                    metadata={},
                    instances=sample_instances[:1],
                    prediction_type="unknown_type",
                )

        assert "Unknown prediction type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_predict_model_not_found_error(
        self, predictor_client, sample_instances
    ):
        """Test predict when model is not found."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            "detail": {
                "error": "Model not found",
                "message": "No model trained",
            }
        }
        mock_response.text = "Not Found"

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ValueError) as exc_info:
                await predictor_client.predict(
                    model_id="test-model",
                    metadata={},
                    instances=sample_instances[:1],
                    prediction_type="quantile",
                )

        assert "No trained model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_predict_invalid_features_error(
        self, predictor_client, sample_instances
    ):
        """Test predict when features are invalid."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "detail": {
                "error": "Invalid features",
                "message": "Missing required field",
            }
        }
        mock_response.text = "Bad Request"

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ValueError) as exc_info:
                await predictor_client.predict(
                    model_id="test-model",
                    metadata={},
                    instances=sample_instances[:1],
                    prediction_type="quantile",
                )

        assert "Invalid task metadata" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_predict_connection_error(
        self, predictor_client, sample_instances
    ):
        """Test predict handles connection errors."""
        import httpx

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock_client.aclose = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ConnectionError):
                await predictor_client.predict(
                    model_id="test-model",
                    metadata={},
                    instances=sample_instances[:1],
                    prediction_type="quantile",
                )

    @pytest.mark.asyncio
    async def test_predict_with_custom_quantiles(
        self, predictor_client, sample_instances
    ):
        """Test predict with custom quantiles parameter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {"quantiles": {"0.5": 100.0, "0.75": 130.0}}
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            _predictions = await predictor_client.predict(
                model_id="test-model",
                metadata={},
                instances=sample_instances[:1],
                prediction_type="quantile",
                quantiles=[0.5, 0.75],
            )

        # Verify quantiles was passed in the request
        call_args = mock_client.post.call_args
        request_json = call_args.kwargs.get(
            "json", call_args[1].get("json", {})
        )
        assert request_json.get("quantiles") == [0.5, 0.75]

    @pytest.mark.asyncio
    async def test_predict_llm_service_enables_semantic_preprocessor(
        self, predictor_client, sample_instances
    ):
        """Test that LLM service models enable semantic preprocessor."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {"quantiles": {"0.5": 100.0}}
        }

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await predictor_client.predict(
                model_id="llm_service_model",
                metadata={"sentence": "test"},
                instances=sample_instances[:1],
                prediction_type="quantile",
            )

        # Verify preprocessors were enabled
        call_args = mock_client.post.call_args
        request_json = call_args.kwargs.get(
            "json", call_args[1].get("json", {})
        )
        assert request_json.get("enable_preprocessors") == ["semantic"]


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    @pytest.fixture
    def predictor_client(self):
        """Create a PredictorClient for testing."""
        return PredictorClient("http://localhost:8080", timeout=1.0)

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, predictor_client):
        """Test health_check returns True when healthy."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await predictor_client.health_check()

        assert result is True
        mock_client.get.assert_called_once_with("/health", timeout=1.0)

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_status(self, predictor_client):
        """Test health_check returns False when status is not healthy."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "unhealthy"}

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await predictor_client.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_non_200_status(self, predictor_client):
        """Test health_check returns False for non-200 status code."""
        mock_response = MagicMock()
        mock_response.status_code = 503

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await predictor_client.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self, predictor_client):
        """Test health_check returns False on connection error."""
        import httpx

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock_client.aclose = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await predictor_client.health_check()

        assert result is False
        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_missing_status_field(self, predictor_client):
        """Test health_check returns False when status field is missing."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}  # No status field

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await predictor_client.health_check()

        assert result is False  # Default status is "unhealthy"
