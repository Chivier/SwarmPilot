"""
Unit tests for PredictorClient with WebSocket support.

Tests WebSocket communication, predictions, retry logic, and health checks.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock, call
import asyncio
import json

from src.predictor_client import PredictorClient, Prediction
from src.model import Instance


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
        client = PredictorClient("ws://localhost:8080/", timeout=10.0)

        assert client.ws_url == "ws://localhost:8080"
        assert client.timeout == 10.0

    def test_init_without_trailing_slash(self):
        """Test initialization with URL without trailing slash."""
        client = PredictorClient("ws://localhost:8080", timeout=5.0)

        assert client.ws_url == "ws://localhost:8080"
        assert client.timeout == 5.0

    def test_init_default_values(self):
        """Test default initialization values."""
        client = PredictorClient("ws://localhost:8080")

        assert client.timeout == 5.0
        assert client.max_retries == 3
        assert client.retry_delay == 1.0
        assert client._websocket is None

    def test_init_custom_retry_params(self):
        """Test initialization with custom retry parameters."""
        client = PredictorClient(
            "ws://localhost:8080",
            max_retries=5,
            retry_delay=2.0
        )

        assert client.max_retries == 5
        assert client.retry_delay == 2.0


# ============================================================================
# WebSocket Connection Management Tests
# ============================================================================

class TestWebSocketConnection:
    """Tests for WebSocket connection management."""

    @pytest.mark.asyncio
    async def test_ensure_connection_establishes_new(self):
        """Test that _ensure_connection establishes a new connection."""
        client = PredictorClient("ws://localhost:8080")

        mock_ws = AsyncMock()
        mock_ws.close_code = None  # Connection is open

        with patch('websockets.connect', new=AsyncMock(return_value=mock_ws)):
            result = await client._ensure_connection()

            assert result == mock_ws
            assert client._websocket == mock_ws

    @pytest.mark.asyncio
    async def test_ensure_connection_reuses_existing(self):
        """Test that existing open connection is reused."""
        client = PredictorClient("ws://localhost:8080")

        mock_ws = AsyncMock()
        mock_ws.close_code = None  # Connection is open
        client._websocket = mock_ws

        result = await client._ensure_connection()

        assert result == mock_ws
        # Should return same websocket without creating new connection

    @pytest.mark.asyncio
    async def test_ensure_connection_reconnects_if_closed(self):
        """Test that closed connection triggers reconnection."""
        client = PredictorClient("ws://localhost:8080")

        # Old closed connection
        old_ws = AsyncMock()
        old_ws.close_code = 1000  # Connection was closed
        client._websocket = old_ws

        # New connection
        new_ws = AsyncMock()
        new_ws.close_code = None

        with patch('websockets.connect', new=AsyncMock(return_value=new_ws)):
            result = await client._ensure_connection()

            assert result == new_ws
            assert client._websocket == new_ws

    @pytest.mark.asyncio
    async def test_ensure_connection_timeout(self):
        """Test connection timeout handling."""
        client = PredictorClient("ws://localhost:8080", timeout=0.1)

        async def slow_connect(*args, **kwargs):
            await asyncio.sleep(1.0)  # Longer than timeout

        with patch('websockets.connect', new=slow_connect):
            with pytest.raises(ConnectionError):
                await client._ensure_connection()

        assert client._websocket is None

    @pytest.mark.asyncio
    async def test_ensure_connection_failure(self):
        """Test connection failure handling."""
        client = PredictorClient("ws://localhost:8080")

        with patch('websockets.connect', side_effect=Exception("Connection refused")):
            with pytest.raises(ConnectionError) as exc_info:
                await client._ensure_connection()

            assert "Failed to connect" in str(exc_info.value)
            assert client._websocket is None

    @pytest.mark.asyncio
    async def test_close_connection(self):
        """Test closing WebSocket connection."""
        client = PredictorClient("ws://localhost:8080")

        mock_ws = AsyncMock()
        client._websocket = mock_ws

        await client._close_connection()

        mock_ws.close.assert_called_once()
        assert client._websocket is None

    @pytest.mark.asyncio
    async def test_close_connection_with_error(self):
        """Test closing connection handles errors gracefully."""
        client = PredictorClient("ws://localhost:8080")

        mock_ws = AsyncMock()
        mock_ws.close.side_effect = Exception("Close error")
        client._websocket = mock_ws

        # Should not raise exception
        await client._close_connection()

        assert client._websocket is None

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test closing the client."""
        client = PredictorClient("ws://localhost:8080")

        mock_ws = AsyncMock()
        client._websocket = mock_ws

        await client.close()

        mock_ws.close.assert_called_once()
        assert client._websocket is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager support."""
        client = PredictorClient("ws://localhost:8080")

        async with client as c:
            assert c == client

        # Connection should be closed after exiting context
        # (if there was one, it would be closed)


# ============================================================================
# Retry Logic Tests
# ============================================================================

class TestRetryLogic:
    """Tests for retry logic in _make_request_with_retry."""

    @pytest.mark.asyncio
    async def test_successful_request_first_attempt(self):
        """Test successful request on first attempt."""
        client = PredictorClient("ws://localhost:8080")

        mock_ws = AsyncMock()
        mock_ws.close_code = None
        mock_ws.recv = AsyncMock(return_value=json.dumps({"result": {"value": 123}}))

        client._websocket = mock_ws

        request_data = {"model_id": "model-1", "features": {}}

        result = await client._make_request_with_retry(
            ws_url="ws://localhost:8080/ws/predict",
            json_data=request_data
        )

        assert result == {"result": {"value": 123}}
        mock_ws.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """Test retry logic on connection errors."""
        client = PredictorClient("ws://localhost:8080", retry_delay=0.01)

        mock_ws = AsyncMock()
        mock_ws.close_code = None
        # First two attempts fail, third succeeds
        mock_ws.send.side_effect = [
            ConnectionError("Connection lost"),
            ConnectionError("Connection lost"),
            None  # Success on third attempt
        ]
        mock_ws.recv = AsyncMock(return_value=json.dumps({"result": {"value": 123}}))

        with patch.object(client, '_ensure_connection', return_value=mock_ws):
            result = await client._make_request_with_retry(
                ws_url="ws://localhost:8080/ws/predict",
                json_data={"test": "data"}
            )

            assert result == {"result": {"value": 123}}
            assert mock_ws.send.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Test retry logic on timeout errors."""
        client = PredictorClient("ws://localhost:8080", timeout=0.1, retry_delay=0.01)

        mock_ws = AsyncMock()
        mock_ws.close_code = None

        async def slow_recv():
            await asyncio.sleep(1.0)  # Longer than timeout

        # First two attempts timeout, third succeeds
        mock_ws.recv.side_effect = [
            slow_recv(),
            slow_recv(),
            json.dumps({"result": {"value": 123}})
        ]

        with patch.object(client, '_ensure_connection', return_value=mock_ws):
            result = await client._make_request_with_retry(
                ws_url="ws://localhost:8080/ws/predict",
                json_data={"test": "data"}
            )

            assert result == {"result": {"value": 123}}

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """Test that max retries are respected."""
        client = PredictorClient("ws://localhost:8080", max_retries=3, retry_delay=0.01)

        mock_ws = AsyncMock()
        mock_ws.close_code = None
        mock_ws.send.side_effect = ConnectionError("Connection lost")

        with patch.object(client, '_ensure_connection', return_value=mock_ws):
            with pytest.raises(ConnectionError):
                await client._make_request_with_retry(
                    ws_url="ws://localhost:8080/ws/predict",
                    json_data={"test": "data"}
                )

            # Should have tried max_retries times
            assert mock_ws.send.call_count == 3

    @pytest.mark.asyncio
    async def test_client_error_no_retry(self):
        """Test that client errors (4xx) are not retried."""
        client = PredictorClient("ws://localhost:8080")

        mock_ws = AsyncMock()
        mock_ws.close_code = None
        mock_ws.recv = AsyncMock(
            return_value=json.dumps({"error": "Invalid request", "code": 400})
        )

        with patch.object(client, '_ensure_connection', return_value=mock_ws):
            with pytest.raises(ValueError) as exc_info:
                await client._make_request_with_retry(
                    ws_url="ws://localhost:8080/ws/predict",
                    json_data={"test": "data"}
                )

            assert "error" in str(exc_info.value).lower()
            # Should only try once (no retries for client errors)
            mock_ws.send.assert_called_once()


# ============================================================================
# Predict Method Tests
# ============================================================================

class TestPredictMethod:
    """Tests for the predict method."""

    @pytest.fixture
    def sample_instances(self):
        """Create sample instances for testing."""
        platform1 = {
            "software_name": "docker",
            "software_version": "20.10",
            "hardware_name": "hw1"
        }
        platform2 = {
            "software_name": "docker",
            "software_version": "20.10",
            "hardware_name": "hw2"
        }

        return [
            Instance(
                instance_id="inst-1",
                model_id="model-1",
                endpoint="http://inst1:8000",
                platform_info=platform1
            ),
            Instance(
                instance_id="inst-2",
                model_id="model-1",
                endpoint="http://inst2:8000",
                platform_info=platform2
            ),
        ]

    @pytest.mark.asyncio
    async def test_predict_single_instance_quantile(self, sample_instances):
        """Test quantile prediction for single instance."""
        client = PredictorClient("ws://localhost:8080")

        mock_response = {
            "result": {
                "quantiles": {
                    "0.5": 100.0,
                    "0.9": 150.0,
                    "0.95": 200.0
                }
            }
        }

        with patch.object(
            client, '_make_request_with_retry',
            return_value=mock_response
        ):
            predictions = await client.predict(
                model_id="model-1",
                metadata={"input_size": 1000},
                instances=[sample_instances[0]],
                prediction_type="quantile"
            )

            assert len(predictions) == 1
            assert predictions[0].instance_id == "inst-1"
            assert predictions[0].quantiles == {0.5: 100.0, 0.9: 150.0, 0.95: 200.0}

    @pytest.mark.asyncio
    async def test_predict_single_instance_expect_error(self, sample_instances):
        """Test expect_error prediction for single instance."""
        client = PredictorClient("ws://localhost:8080")

        mock_response = {
            "result": {
                "expected_runtime_ms": 125.0,
                "error_margin_ms": 25.0
            }
        }

        with patch.object(
            client, '_make_request_with_retry',
            return_value=mock_response
        ):
            predictions = await client.predict(
                model_id="model-1",
                metadata={"input_size": 1000},
                instances=[sample_instances[0]],
                prediction_type="expect_error"
            )

            assert len(predictions) == 1
            assert predictions[0].instance_id == "inst-1"
            assert predictions[0].predicted_time_ms == 125.0

    @pytest.mark.asyncio
    async def test_predict_multiple_instances(self, sample_instances):
        """Test prediction for multiple instances with different platforms."""
        client = PredictorClient("ws://localhost:8080")

        # Different responses for different platforms
        responses = [
            {"result": {"quantiles": {"0.5": 100.0, "0.9": 150.0}}},
            {"result": {"quantiles": {"0.5": 200.0, "0.9": 250.0}}}
        ]

        with patch.object(
            client, '_make_request_with_retry',
            side_effect=responses
        ):
            predictions = await client.predict(
                model_id="model-1",
                metadata={"input_size": 1000},
                instances=sample_instances,
                prediction_type="quantile"
            )

            assert len(predictions) == 2
            assert predictions[0].instance_id == "inst-1"
            assert predictions[1].instance_id == "inst-2"

    @pytest.mark.asyncio
    async def test_predict_platform_batching(self):
        """Test that instances with same platform are batched."""
        client = PredictorClient("ws://localhost:8080")

        # Create 3 instances with same platform
        platform = {
            "software_name": "docker",
            "software_version": "20.10",
            "hardware_name": "hw1"
        }

        instances = [
            Instance(
                instance_id=f"inst-{i}",
                model_id="model-1",
                endpoint=f"http://inst{i}:8000",
                platform_info=platform
            )
            for i in range(3)
        ]

        mock_response = {
            "result": {"quantiles": {"0.5": 100.0, "0.9": 150.0}}
        }

        with patch.object(
            client, '_make_request_with_retry',
            return_value=mock_response
        ) as mock_request:
            predictions = await client.predict(
                model_id="model-1",
                metadata={"input_size": 1000},
                instances=instances,
                prediction_type="quantile"
            )

            # Should only make one request (batching)
            assert mock_request.call_count == 1

            # But should return 3 predictions (one per instance)
            assert len(predictions) == 3
            assert all(p.quantiles == {0.5: 100.0, 0.9: 150.0} for p in predictions)

    @pytest.mark.asyncio
    async def test_predict_model_not_found(self, sample_instances):
        """Test handling of model not found error."""
        client = PredictorClient("ws://localhost:8080")

        with patch.object(
            client, '_make_request_with_retry',
            side_effect=ValueError("Model not found: unknown-model")
        ):
            with pytest.raises(ValueError) as exc_info:
                await client.predict(
                    model_id="unknown-model",
                    metadata={},
                    instances=[sample_instances[0]],
                    prediction_type="quantile"
                )

            assert "model not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_predict_connection_error(self, sample_instances):
        """Test handling of connection errors during prediction."""
        client = PredictorClient("ws://localhost:8080")

        with patch.object(
            client, '_make_request_with_retry',
            side_effect=ConnectionError("Connection failed")
        ):
            with pytest.raises(ConnectionError):
                await client.predict(
                    model_id="model-1",
                    metadata={},
                    instances=[sample_instances[0]],
                    prediction_type="quantile"
                )

    @pytest.mark.asyncio
    async def test_predict_empty_instances(self):
        """Test prediction with empty instance list."""
        client = PredictorClient("ws://localhost:8080")

        # This should return empty list without making any requests
        predictions = await client.predict(
            model_id="model-1",
            metadata={},
            instances=[],
            prediction_type="quantile"
        )

        assert predictions == []


# ============================================================================
# Health Check Tests
# ============================================================================

class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        client = PredictorClient("ws://localhost:8080")

        mock_ws = AsyncMock()
        mock_ws.close_code = None
        mock_ws.recv = AsyncMock(
            return_value=json.dumps({"status": "healthy"})
        )

        with patch.object(client, '_ensure_connection', return_value=mock_ws):
            result = await client.health_check()

            assert result is True
            mock_ws.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_connection_failure(self):
        """Test health check with connection failure."""
        client = PredictorClient("ws://localhost:8080")

        with patch.object(
            client, '_ensure_connection',
            side_effect=ConnectionError("Cannot connect")
        ):
            result = await client.health_check()

            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_timeout(self):
        """Test health check timeout."""
        client = PredictorClient("ws://localhost:8080", timeout=0.1)

        mock_ws = AsyncMock()
        mock_ws.close_code = None

        async def slow_recv():
            await asyncio.sleep(1.0)

        mock_ws.recv = slow_recv

        with patch.object(client, '_ensure_connection', return_value=mock_ws):
            result = await client.health_check()

            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_generic_exception(self):
        """Test health check with generic exception."""
        client = PredictorClient("ws://localhost:8080")

        mock_ws = AsyncMock()
        mock_ws.close_code = None
        mock_ws.send.side_effect = Exception("Unexpected error")

        with patch.object(client, '_ensure_connection', return_value=mock_ws):
            result = await client.health_check()

            assert result is False
