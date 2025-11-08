"""
Unit tests for PredictorClient auto-start functionality.

Tests cover port management, process lifecycle, and exception handling.
"""

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from src.clients.predictor_client import PredictorClient


class TestPredictorClientPortManagement:
    """Test suite for port management methods."""

    def test_is_port_in_use_free_port(self):
        """Test _is_port_in_use returns False for available port."""
        client = PredictorClient()

        with patch("socket.socket") as mock_socket:
            mock_socket_instance = MagicMock()
            mock_socket.return_value.__enter__.return_value = mock_socket_instance
            mock_socket_instance.bind.return_value = None  # Success

            result = client._is_port_in_use(8001)

            assert result is False
            mock_socket_instance.bind.assert_called_once_with(("127.0.0.1", 8001))

    def test_is_port_in_use_occupied_port(self):
        """Test _is_port_in_use returns True for occupied port."""
        client = PredictorClient()

        with patch("socket.socket") as mock_socket:
            mock_socket_instance = MagicMock()
            mock_socket.return_value.__enter__.return_value = mock_socket_instance
            mock_socket_instance.bind.side_effect = OSError("Port in use")

            result = client._is_port_in_use(8001)

            assert result is True

    def test_find_available_port_first_available(self):
        """Test _find_available_port finds first available port."""
        client = PredictorClient()

        with patch.object(client, "_is_port_in_use") as mock_check:
            mock_check.return_value = False

            port = client._find_available_port(8001)

            assert port == 8001
            mock_check.assert_called_once_with(8001)

    def test_find_available_port_after_occupied(self):
        """Test _find_available_port skips occupied ports."""
        client = PredictorClient()

        with patch.object(client, "_is_port_in_use") as mock_check:
            # First 3 ports occupied, 4th available
            mock_check.side_effect = [True, True, True, False]

            port = client._find_available_port(8001)

            assert port == 8004
            assert mock_check.call_count == 4

    def test_find_available_port_none_available(self):
        """Test _find_available_port returns None when no ports available."""
        client = PredictorClient()

        with patch.object(client, "_is_port_in_use") as mock_check:
            mock_check.return_value = True

            port = client._find_available_port(8001, max_attempts=10)

            assert port is None
            assert mock_check.call_count == 10

    def test_extract_port_from_url_with_port(self):
        """Test _extract_port_from_url with explicit port."""
        client = PredictorClient()

        port = client._extract_port_from_url("http://localhost:8001")
        assert port == 8001

        port = client._extract_port_from_url("https://example.com:9000")
        assert port == 9000

    def test_extract_port_from_url_default_http(self):
        """Test _extract_port_from_url defaults to 80 for http."""
        client = PredictorClient()

        port = client._extract_port_from_url("http://localhost")
        assert port == 80

    def test_extract_port_from_url_default_https(self):
        """Test _extract_port_from_url defaults to 443 for https."""
        client = PredictorClient()

        port = client._extract_port_from_url("https://localhost")
        assert port == 443

    def test_update_predictor_url_port(self):
        """Test _update_predictor_url_port updates URL correctly."""
        client = PredictorClient(predictor_url="http://localhost:8001")

        client._update_predictor_url_port(9000)

        assert client.predictor_url == "http://localhost:9000"


class TestPredictorClientModulePath:
    """Test suite for module path detection."""

    def test_find_predictor_module_relative_path(self):
        """Test _find_predictor_module finds module relative to file."""
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True

            client = PredictorClient()

            assert "predictor" in client.predictor_module_path

    def test_find_predictor_module_env_variable(self):
        """Test _find_predictor_module uses environment variable."""
        with patch("pathlib.Path.exists", return_value=False), \
             patch.dict("os.environ", {"PREDICTOR_MODULE_PATH": "/custom/predictor"}):

            client = PredictorClient()

            assert client.predictor_module_path == "/custom/predictor"

    def test_find_predictor_module_fallback(self):
        """Test _find_predictor_module fallback to current directory."""
        with patch("pathlib.Path.exists", return_value=False), \
             patch.dict("os.environ", {}, clear=True):

            client = PredictorClient()

            assert "predictor" in client.predictor_module_path


class TestPredictorClientProcessManagement:
    """Test suite for process lifecycle management."""

    def test_is_predictor_running_no_process(self):
        """Test is_predictor_running returns False when no process."""
        client = PredictorClient()

        assert client.is_predictor_running is False

    def test_is_predictor_running_with_running_process(self):
        """Test is_predictor_running returns True for running process."""
        client = PredictorClient()
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Still running
        client._predictor_process = mock_process

        assert client.is_predictor_running is True

    def test_is_predictor_running_with_stopped_process(self):
        """Test is_predictor_running returns False for stopped process."""
        client = PredictorClient()
        mock_process = MagicMock()
        mock_process.poll.return_value = 0  # Exited
        client._predictor_process = mock_process

        assert client.is_predictor_running is False

    @pytest.mark.asyncio
    async def test_wait_for_predictor_ready_success(self):
        """Test _wait_for_predictor_ready succeeds when service becomes ready."""
        client = PredictorClient()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await client._wait_for_predictor_ready(timeout=5.0)

            assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_predictor_ready_timeout(self):
        """Test _wait_for_predictor_ready times out."""
        client = PredictorClient()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await client._wait_for_predictor_ready(timeout=0.5, check_interval=0.1)

            assert result is False


class TestPredictorClientAutoStart:
    """Test suite for start_predictor functionality."""

    @pytest.mark.asyncio
    async def test_start_predictor_already_running(self):
        """Test start_predictor detects already running service."""
        client = PredictorClient()

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client, "_is_port_in_use", return_value=True), \
             patch("httpx.AsyncClient") as mock_client_class:

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await client.start_predictor()

            assert result is True

    @pytest.mark.asyncio
    async def test_start_predictor_port_occupied_auto_find(self):
        """Test start_predictor finds new port when occupied."""
        client = PredictorClient()

        with patch.object(client, "_is_port_in_use") as mock_port_check, \
             patch.object(client, "_find_available_port", return_value=8002), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen") as mock_popen, \
             patch.object(client, "_wait_for_predictor_ready", return_value=True):

            # First check: port occupied, health check fails
            mock_port_check.return_value = True

            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.get.side_effect = Exception("Connection refused")
                mock_client_class.return_value.__aenter__.return_value = mock_client

                result = await client.start_predictor(auto_find_port=True)

            assert result is True
            assert client.predictor_url == "http://localhost:8002"

    @pytest.mark.asyncio
    async def test_start_predictor_port_occupied_no_auto_find(self):
        """Test start_predictor raises error when port occupied and auto_find=False."""
        client = PredictorClient()

        with patch.object(client, "_is_port_in_use", return_value=True), \
             patch("httpx.AsyncClient") as mock_client_class:

            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(ConnectionError, match="Port .* is already in use"):
                await client.start_predictor(auto_find_port=False)

    @pytest.mark.asyncio
    async def test_start_predictor_module_not_found(self):
        """Test start_predictor raises RuntimeError when module not found."""
        client = PredictorClient(predictor_module_path="/nonexistent/path")

        with patch.object(client, "_is_port_in_use", return_value=False), \
             patch("os.path.exists", return_value=False):

            with pytest.raises(RuntimeError, match="Predictor module not found"):
                await client.start_predictor()

    @pytest.mark.asyncio
    async def test_start_predictor_success(self):
        """Test start_predictor successfully starts service."""
        client = PredictorClient()

        mock_process = MagicMock()
        mock_process.pid = 12345

        with patch.object(client, "_is_port_in_use", return_value=False), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", return_value=mock_process) as mock_popen, \
             patch.object(client, "_wait_for_predictor_ready", return_value=True):

            result = await client.start_predictor()

            assert result is True
            assert client._predictor_process == mock_process
            mock_popen.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_predictor_fails_to_become_ready(self):
        """Test start_predictor cleans up when service fails to become ready."""
        client = PredictorClient()

        mock_process = MagicMock()

        with patch.object(client, "_is_port_in_use", return_value=False), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", return_value=mock_process), \
             patch.object(client, "_wait_for_predictor_ready", return_value=False), \
             patch.object(client, "stop_predictor") as mock_stop:

            with pytest.raises(ConnectionError, match="failed to become ready"):
                await client.start_predictor()

            mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_predictor_no_available_port(self):
        """Test start_predictor raises error when no port available."""
        client = PredictorClient()

        with patch.object(client, "_is_port_in_use", return_value=True), \
             patch.object(client, "_find_available_port", return_value=None), \
             patch("httpx.AsyncClient") as mock_client_class:

            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(ConnectionError, match="No available port"):
                await client.start_predictor(auto_find_port=True)


class TestPredictorClientStop:
    """Test suite for stop_predictor functionality."""

    def test_stop_predictor_no_process(self):
        """Test stop_predictor returns False when no process."""
        client = PredictorClient()

        result = client.stop_predictor()

        assert result is False

    def test_stop_predictor_already_stopped(self):
        """Test stop_predictor handles already stopped process."""
        client = PredictorClient()
        mock_process = MagicMock()
        mock_process.poll.return_value = 0  # Exited
        client._predictor_process = mock_process

        result = client.stop_predictor()

        assert result is True
        assert client._predictor_process is None

    def test_stop_predictor_graceful(self):
        """Test stop_predictor terminates gracefully."""
        client = PredictorClient()
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Running
        mock_process.pid = 12345
        mock_process.wait.return_value = None
        client._predictor_process = mock_process

        result = client.stop_predictor()

        assert result is True
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=10.0)
        assert client._predictor_process is None

    def test_stop_predictor_force_kill(self):
        """Test stop_predictor kills when terminate times out."""
        client = PredictorClient()
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_process.wait.side_effect = [subprocess.TimeoutExpired(cmd="", timeout=10), None]
        client._predictor_process = mock_process

        result = client.stop_predictor()

        assert result is True
        mock_process.kill.assert_called_once()
        assert client._predictor_process is None


class TestPredictorClientAPIs:
    """Test suite for Predictor API methods."""

    @pytest.mark.asyncio
    async def test_predict_quantile_success(self):
        """Test predict with quantile type."""
        from src.clients.predictor_client import PlatformInfo

        async with PredictorClient() as client:
            platform = PlatformInfo("vllm", "0.2.5", "nvidia-a100")

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "model_id": "gpt-4",
                "platform_info": {"software_name": "vllm", "software_version": "0.2.5", "hardware_name": "nvidia-a100"},
                "prediction_type": "quantile",
                "result": {"quantiles": {"0.5": 100.0, "0.9": 150.0}}
            }

            with patch.object(client._http_client, "post", return_value=mock_response):
                result = await client.predict(
                    model_id="gpt-4",
                    platform_info=platform,
                    features={"prompt_length": 100},
                    prediction_type="quantile"
                )

                assert result.model_id == "gpt-4"
                assert result.expected_runtime_ms == 100.0
                assert result.quantiles == {0.5: 100.0, 0.9: 150.0}

    @pytest.mark.asyncio
    async def test_predict_expect_error_success(self):
        """Test predict with expect_error type."""
        from src.clients.predictor_client import PlatformInfo

        async with PredictorClient() as client:
            platform = PlatformInfo("vllm", "0.2.5", "nvidia-a100")

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "model_id": "gpt-4",
                "platform_info": {"software_name": "vllm", "software_version": "0.2.5", "hardware_name": "nvidia-a100"},
                "prediction_type": "expect_error",
                "result": {"expected_runtime_ms": 100.0, "error_margin_ms": 10.0}
            }

            with patch.object(client._http_client, "post", return_value=mock_response):
                result = await client.predict(
                    model_id="gpt-4",
                    platform_info=platform,
                    features={"prompt_length": 100},
                    prediction_type="expect_error"
                )

                assert result.model_id == "gpt-4"
                assert result.expected_runtime_ms == 100.0
                assert result.error_margin_ms == 10.0

    @pytest.mark.asyncio
    async def test_predict_invalid_type(self):
        """Test predict with invalid prediction type."""
        from src.clients.predictor_client import PlatformInfo

        async with PredictorClient() as client:
            platform = PlatformInfo("vllm", "0.2.5", "nvidia-a100")

            with pytest.raises(ValueError, match="Invalid prediction_type"):
                await client.predict(
                    model_id="gpt-4",
                    platform_info=platform,
                    features={"prompt_length": 100},
                    prediction_type="invalid"
                )

    @pytest.mark.asyncio
    async def test_predict_4xx_error(self):
        """Test predict with 4xx HTTP error."""
        from src.clients.predictor_client import PlatformInfo

        async with PredictorClient() as client:
            platform = PlatformInfo("vllm", "0.2.5", "nvidia-a100")

            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Bad request"

            error = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=mock_response)

            with patch.object(client._http_client, "post", side_effect=error):
                with pytest.raises(ValueError, match="Prediction request failed"):
                    await client.predict(
                        model_id="gpt-4",
                        platform_info=platform,
                        features={"prompt_length": 100}
                    )

    @pytest.mark.asyncio
    async def test_predict_retry_and_fail(self):
        """Test predict retries and eventually fails."""
        from src.clients.predictor_client import PlatformInfo

        async with PredictorClient(max_retries=2, retry_delay=0.01) as client:
            platform = PlatformInfo("vllm", "0.2.5", "nvidia-a100")

            with patch.object(client._http_client, "post", side_effect=httpx.ConnectError("Connection failed")):
                with pytest.raises(ConnectionError, match="Prediction failed after 2 retries"):
                    await client.predict(
                        model_id="gpt-4",
                        platform_info=platform,
                        features={"prompt_length": 100}
                    )

    @pytest.mark.asyncio
    async def test_train_success(self):
        """Test train method."""
        from src.clients.predictor_client import PlatformInfo

        async with PredictorClient() as client:
            platform = PlatformInfo("vllm", "0.2.5", "nvidia-a100")

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "status": "success",
                "message": "Training completed",
                "model_key": "gpt-4_vllm_0.2.5_nvidia-a100",
                "samples_trained": 100
            }

            with patch.object(client._http_client, "post", return_value=mock_response):
                result = await client.train(
                    model_id="gpt-4",
                    platform_info=platform,
                    prediction_type="quantile",
                    features_list=[{"prompt_length": 100, "runtime_ms": 120.0}]
                )

                assert result.status == "success"
                assert result.samples_trained == 100

    @pytest.mark.asyncio
    async def test_train_with_config(self):
        """Test train with training config."""
        from src.clients.predictor_client import PlatformInfo

        async with PredictorClient() as client:
            platform = PlatformInfo("vllm", "0.2.5", "nvidia-a100")

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "status": "success",
                "message": "Training completed",
                "model_key": "gpt-4_vllm_0.2.5_nvidia-a100",
                "samples_trained": 100
            }

            with patch.object(client._http_client, "post", return_value=mock_response) as mock_post:
                await client.train(
                    model_id="gpt-4",
                    platform_info=platform,
                    prediction_type="quantile",
                    features_list=[{"prompt_length": 100, "runtime_ms": 120.0}],
                    training_config={"quantiles": [0.5, 0.9]}
                )

                # Verify training_config was passed
                call_args = mock_post.call_args
                assert call_args.kwargs["json"]["training_config"] == {"quantiles": [0.5, 0.9]}

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        """Test list_models method."""
        async with PredictorClient() as client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "models": [
                    {
                        "model_id": "gpt-4",
                        "platform_info": {"software_name": "vllm", "software_version": "0.2.5", "hardware_name": "nvidia-a100"},
                        "prediction_type": "quantile",
                        "samples_count": 100,
                        "last_trained": "2025-01-01T00:00:00"
                    }
                ]
            }

            with patch.object(client._http_client, "get", return_value=mock_response):
                models = await client.list_models()

                assert len(models) == 1
                assert models[0].model_id == "gpt-4"
                assert models[0].samples_count == 100

    @pytest.mark.asyncio
    async def test_list_models_retry(self):
        """Test list_models with retry."""
        async with PredictorClient(max_retries=2, retry_delay=0.01) as client:
            with patch.object(client._http_client, "get", side_effect=httpx.ConnectError("Connection failed")):
                with pytest.raises(ConnectionError, match="List models failed"):
                    await client.list_models()

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health_check when service is healthy."""
        async with PredictorClient() as client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "healthy"}

            with patch.object(client._http_client, "get", return_value=mock_response):
                result = await client.health_check()

                assert result is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """Test health_check when service is unhealthy."""
        async with PredictorClient() as client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "unhealthy"}

            with patch.object(client._http_client, "get", return_value=mock_response):
                result = await client.health_check()

                assert result is False

    @pytest.mark.asyncio
    async def test_health_check_error(self):
        """Test health_check with connection error."""
        async with PredictorClient() as client:
            with patch.object(client._http_client, "get", side_effect=httpx.ConnectError("Connection failed")):
                result = await client.health_check()

                assert result is False

    @pytest.mark.asyncio
    async def test_ensure_client_not_initialized(self):
        """Test _ensure_client raises error when client not initialized."""
        client = PredictorClient()

        with pytest.raises(RuntimeError, match="must be used as an async context manager"):
            client._ensure_client()

    @pytest.mark.asyncio
    async def test_close_method(self):
        """Test close method."""
        client = PredictorClient()
        client._http_client = AsyncMock()

        await client.close()

        assert client._http_client is None
