"""
Tests for the TrainingClient module.

Covers training data collection, buffering, and submission to the predictor service.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import httpx

from src.training_client import TrainingClient, TrainingSample


@pytest.fixture
def predictor_url():
    """Predictor service URL."""
    return "http://predictor:8000"


@pytest.fixture
def platform_info():
    """Sample platform info."""
    return {
        "software_name": "docker",
        "software_version": "20.10",
        "hardware_name": "test-hardware",
    }


@pytest.fixture
def features():
    """Sample task features."""
    return {
        "input_size": 1000,
        "complexity": "medium",
    }


@pytest.fixture
def training_client(predictor_url):
    """Create a TrainingClient instance."""
    return TrainingClient(
        predictor_url=predictor_url,
        timeout=10.0,
        batch_size=100,
        min_samples=10,
    )


class TestInitialization:
    """Tests for TrainingClient initialization."""

    def test_init_with_defaults(self, predictor_url):
        """Test initialization with default parameters."""
        client = TrainingClient(predictor_url=predictor_url)

        assert client.predictor_url == predictor_url
        assert client.timeout == 10.0
        assert client.batch_size == 100
        assert client.min_samples == 10
        assert client._samples_buffer == []
        assert client._http_client is not None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        client = TrainingClient(
            predictor_url="http://custom:9000/",
            timeout=5.0,
            batch_size=50,
            min_samples=5,
        )

        assert client.predictor_url == "http://custom:9000"
        assert client.timeout == 5.0
        assert client.batch_size == 50
        assert client.min_samples == 5

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is removed from URL."""
        client = TrainingClient(predictor_url="http://predictor:8000/")
        assert client.predictor_url == "http://predictor:8000"


class TestAddSample:
    """Tests for adding training samples."""

    @patch("src.training_client.datetime")
    def test_add_single_sample(
        self, mock_datetime, training_client, platform_info, features
    ):
        """Test adding a single training sample."""
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T00:00:00"

        training_client.add_sample(
            model_id="model-1",
            platform_info=platform_info,
            features=features,
            actual_runtime_ms=1234.56,
        )

        assert training_client.get_buffer_size() == 1

        sample = training_client._samples_buffer[0]
        assert sample.model_id == "model-1"
        assert sample.platform_info == platform_info
        assert sample.features == features
        assert sample.actual_runtime_ms == 1234.56
        assert sample.timestamp == "2024-01-01T00:00:00"

    @patch("src.training_client.datetime")
    def test_add_multiple_samples(
        self, mock_datetime, training_client, platform_info, features
    ):
        """Test adding multiple training samples."""
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T00:00:00"

        for i in range(5):
            training_client.add_sample(
                model_id=f"model-{i}",
                platform_info=platform_info,
                features=features,
                actual_runtime_ms=float(i * 100),
            )

        assert training_client.get_buffer_size() == 5

        # Verify samples are stored correctly
        for i, sample in enumerate(training_client._samples_buffer):
            assert sample.model_id == f"model-{i}"
            assert sample.actual_runtime_ms == float(i * 100)

    @patch("src.training_client.datetime")
    def test_add_sample_different_platforms(self, mock_datetime, training_client, features):
        """Test adding samples from different platforms."""
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T00:00:00"

        platform1 = {"software_name": "docker", "software_version": "20.10", "hardware_name": "hw1"}
        platform2 = {"software_name": "docker", "software_version": "20.10", "hardware_name": "hw2"}

        training_client.add_sample("model-1", platform1, features, 100.0)
        training_client.add_sample("model-1", platform2, features, 200.0)

        assert training_client.get_buffer_size() == 2
        assert training_client._samples_buffer[0].platform_info == platform1
        assert training_client._samples_buffer[1].platform_info == platform2


class TestFlushIfReady:
    """Tests for automatic flush when batch size reached."""

    @pytest.mark.asyncio
    async def test_flush_when_batch_size_reached(self, training_client, platform_info, features):
        """Test that flush is triggered when batch size is reached."""
        # Add samples up to batch size
        for i in range(training_client.batch_size):
            training_client.add_sample("model-1", platform_info, features, float(i))

        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        training_client._http_client.post = AsyncMock(return_value=mock_response)

        result = await training_client.flush_if_ready()

        assert result is True
        training_client._http_client.post.assert_called_once()
        assert training_client.get_buffer_size() == 0

    @pytest.mark.asyncio
    async def test_no_flush_below_batch_size(self, training_client, platform_info, features):
        """Test that flush is not triggered below batch size."""
        # Add samples below batch size
        for i in range(training_client.batch_size - 1):
            training_client.add_sample("model-1", platform_info, features, float(i))

        result = await training_client.flush_if_ready()

        assert result is False
        assert training_client.get_buffer_size() == training_client.batch_size - 1


class TestFlush:
    """Tests for manual flush operations."""

    @pytest.mark.asyncio
    async def test_flush_empty_buffer(self, training_client):
        """Test flushing an empty buffer."""
        result = await training_client.flush()

        assert result is False

    @pytest.mark.asyncio
    async def test_flush_below_minimum_without_force(self, training_client, platform_info, features):
        """Test that flush skips training when below min_samples without force."""
        # Add samples below minimum
        for i in range(training_client.min_samples - 1):
            training_client.add_sample("model-1", platform_info, features, float(i))

        result = await training_client.flush(force=False)

        assert result is False
        # Buffer should not be cleared
        assert training_client.get_buffer_size() == training_client.min_samples - 1

    @pytest.mark.asyncio
    async def test_flush_with_force_flag(self, training_client, platform_info, features):
        """Test that force flag bypasses min_samples check."""
        # Add just 1 sample (below minimum)
        training_client.add_sample("model-1", platform_info, features, 100.0)

        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        training_client._http_client.post = AsyncMock(return_value=mock_response)

        result = await training_client.flush(force=True)

        assert result is True
        training_client._http_client.post.assert_called_once()
        assert training_client.get_buffer_size() == 0

    @pytest.mark.asyncio
    async def test_flush_single_model_platform(self, training_client, platform_info, features):
        """Test flushing samples for a single model-platform combination."""
        # Add enough samples
        for i in range(training_client.min_samples):
            training_client.add_sample("model-1", platform_info, features, float(i * 100))

        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        training_client._http_client.post = AsyncMock(return_value=mock_response)

        result = await training_client.flush()

        assert result is True
        assert training_client.get_buffer_size() == 0

        # Verify the POST request
        training_client._http_client.post.assert_called_once()
        call_args = training_client._http_client.post.call_args

        assert call_args[0][0] == f"{training_client.predictor_url}/train"
        training_data = call_args[1]["json"]

        assert training_data["model_id"] == "model-1"
        assert training_data["platform_info"] == platform_info
        assert len(training_data["samples"]) == training_client.min_samples

        # Verify sample structure
        for i, sample in enumerate(training_data["samples"]):
            assert "features" in sample
            assert "actual_runtime_ms" in sample
            assert sample["actual_runtime_ms"] == float(i * 100)

    @pytest.mark.asyncio
    async def test_flush_multiple_model_platform_groups(self, training_client, features):
        """Test flushing samples grouped by model-platform combination."""
        platform1 = {"software_name": "docker", "software_version": "20.10", "hardware_name": "hw1"}
        platform2 = {"software_name": "docker", "software_version": "20.10", "hardware_name": "hw2"}

        # Add samples for different combinations
        for i in range(5):
            training_client.add_sample("model-1", platform1, features, 100.0)
            training_client.add_sample("model-1", platform2, features, 200.0)
            training_client.add_sample("model-2", platform1, features, 300.0)

        assert training_client.get_buffer_size() == 15

        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        training_client._http_client.post = AsyncMock(return_value=mock_response)

        result = await training_client.flush(force=True)

        assert result is True
        assert training_client.get_buffer_size() == 0

        # Should make 3 separate POST requests (one per model-platform combination)
        assert training_client._http_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_flush_http_success(self, training_client, platform_info, features):
        """Test successful HTTP request during flush."""
        # Add samples
        for i in range(training_client.min_samples):
            training_client.add_sample("model-1", platform_info, features, float(i))

        # Mock successful HTTP response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        training_client._http_client.post = AsyncMock(return_value=mock_response)

        result = await training_client.flush()

        assert result is True
        mock_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_http_error(self, training_client, platform_info, features):
        """Test HTTP error handling during flush."""
        # Add samples
        for i in range(training_client.min_samples):
            training_client.add_sample("model-1", platform_info, features, float(i))

        # Mock HTTP error
        training_client._http_client.post = AsyncMock(
            side_effect=httpx.HTTPError("Connection failed")
        )

        result = await training_client.flush()

        # Should return False due to failure
        assert result is False
        # Buffer should still be cleared
        assert training_client.get_buffer_size() == 0

    @pytest.mark.asyncio
    async def test_flush_partial_failures(self, training_client, features):
        """Test handling partial failures across multiple groups."""
        platform1 = {"software_name": "docker", "software_version": "20.10", "hardware_name": "hw1"}
        platform2 = {"software_name": "docker", "software_version": "20.10", "hardware_name": "hw2"}

        # Add samples for different platforms
        for i in range(5):
            training_client.add_sample("model-1", platform1, features, 100.0)
            training_client.add_sample("model-1", platform2, features, 200.0)

        # Mock HTTP client with mixed success/failure
        mock_success = MagicMock()
        mock_success.raise_for_status = MagicMock()

        training_client._http_client.post = AsyncMock(
            side_effect=[mock_success, httpx.HTTPError("Failed")]
        )

        result = await training_client.flush(force=True)

        # Should return False because one failed
        assert result is False
        # Buffer should still be cleared
        assert training_client.get_buffer_size() == 0

    @pytest.mark.asyncio
    async def test_flush_clears_buffer_after_attempt(self, training_client, platform_info, features):
        """Test that buffer is cleared even if training fails."""
        training_client.add_sample("model-1", platform_info, features, 100.0)

        # Mock HTTP failure
        training_client._http_client.post = AsyncMock(
            side_effect=httpx.HTTPError("Connection failed")
        )

        await training_client.flush(force=True)

        # Buffer should be cleared regardless of success
        assert training_client.get_buffer_size() == 0


class TestBufferManagement:
    """Tests for buffer management operations."""

    def test_get_buffer_size_empty(self, training_client):
        """Test getting buffer size when empty."""
        assert training_client.get_buffer_size() == 0

    def test_get_buffer_size_with_samples(self, training_client, platform_info, features):
        """Test getting buffer size with samples."""
        for i in range(10):
            training_client.add_sample("model-1", platform_info, features, float(i))

        assert training_client.get_buffer_size() == 10

    def test_clear_buffer_empty(self, training_client):
        """Test clearing an empty buffer."""
        training_client.clear_buffer()
        assert training_client.get_buffer_size() == 0

    def test_clear_buffer_with_samples(self, training_client, platform_info, features):
        """Test clearing buffer with samples."""
        for i in range(10):
            training_client.add_sample("model-1", platform_info, features, float(i))

        assert training_client.get_buffer_size() == 10

        training_client.clear_buffer()

        assert training_client.get_buffer_size() == 0

    def test_clear_buffer_does_not_send_data(self, training_client, platform_info, features):
        """Test that clear_buffer does not trigger any HTTP requests."""
        for i in range(10):
            training_client.add_sample("model-1", platform_info, features, float(i))

        # Mock HTTP client to verify no calls are made
        training_client._http_client.post = AsyncMock()

        training_client.clear_buffer()

        training_client._http_client.post.assert_not_called()


class TestClose:
    """Tests for resource cleanup."""

    @pytest.mark.asyncio
    async def test_close_http_client(self, training_client):
        """Test that close properly cleans up HTTP client."""
        # Mock aclose method
        training_client._http_client.aclose = AsyncMock()

        await training_client.close()

        training_client._http_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_multiple_times(self, training_client):
        """Test that close can be called multiple times safely."""
        training_client._http_client.aclose = AsyncMock()

        await training_client.close()
        await training_client.close()

        # Should be called twice
        assert training_client._http_client.aclose.call_count == 2


class TestTrainingSampleDataclass:
    """Tests for TrainingSample dataclass."""

    def test_training_sample_creation(self, platform_info, features):
        """Test creating a TrainingSample instance."""
        sample = TrainingSample(
            model_id="model-1",
            platform_info=platform_info,
            features=features,
            actual_runtime_ms=1234.56,
            timestamp="2024-01-01T00:00:00",
        )

        assert sample.model_id == "model-1"
        assert sample.platform_info == platform_info
        assert sample.features == features
        assert sample.actual_runtime_ms == 1234.56
        assert sample.timestamp == "2024-01-01T00:00:00"
