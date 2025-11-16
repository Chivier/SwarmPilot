"""
Unit tests for PredictorClient with HTTP API support.

Tests HTTP API communication, predictions, retry logic, and health checks.
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
            "http://localhost:8080",
            max_retries=5,
            retry_delay=2.0
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

        with patch('httpx.AsyncClient', return_value=mock_client):
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

        with patch('httpx.AsyncClient', return_value=new_client):
            result = await client._ensure_client()

            assert result == new_client
            assert client._client == new_client

    @pytest.mark.asyncio
    async def test_ensure_client_failure(self):
        """Test client initialization failure handling."""
        client = PredictorClient("http://localhost:8080")

        with patch('httpx.AsyncClient', side_effect=Exception("Initialization failed")):
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


