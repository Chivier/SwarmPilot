"""
Comprehensive unit tests for scheduler_client module.

Tests cover:
- Configuration classes (PredictorConfig, SchedulingConfig, etc.)
- SchedulerClient initialization and properties
- Port detection and management
- Predictor dependency checking
- Scheduler auto-start functionality
- Instance registration and deregistration
- Draining operations
- Task result callbacks
- Process management
- Utility methods

Target coverage: >90%
"""

import asyncio
import os
import socket
import subprocess
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from src.scheduler_client import (
    PredictorConfig,
    SchedulerClient,
    SchedulingConfig,
    TrainingConfig,
    WebSocketConfig,
    get_scheduler_client,
    initialize_scheduler_client,
)


# ==================== Configuration Class Tests ====================


class TestPredictorConfig:
    """Test PredictorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PredictorConfig()
        assert config.url == "http://localhost:8001"
        assert config.timeout == 5.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PredictorConfig(
            url="http://custom:9000", timeout=10.0, max_retries=5, retry_delay=2.0
        )
        assert config.url == "http://custom:9000"
        assert config.timeout == 10.0
        assert config.max_retries == 5
        assert config.retry_delay == 2.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = PredictorConfig(url="http://test:8001", timeout=7.5)
        result = config.to_dict()
        assert result == {
            "url": "http://test:8001",
            "timeout": 7.5,
            "max_retries": 3,
            "retry_delay": 1.0,
        }

    def test_env_var_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("PREDICTOR_URL", "http://env:8002")
        monkeypatch.setenv("PREDICTOR_TIMEOUT", "15.0")
        config = PredictorConfig()
        assert config.url == "http://env:8002"
        assert config.timeout == 15.0


class TestSchedulingConfig:
    """Test SchedulingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SchedulingConfig()
        assert config.default_strategy == "probabilistic"
        assert config.probabilistic_quantile == 0.9

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SchedulingConfig(default_strategy="min_time", probabilistic_quantile=0.95)
        assert config.default_strategy == "min_time"
        assert config.probabilistic_quantile == 0.95

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = SchedulingConfig(default_strategy="round_robin", probabilistic_quantile=0.99)
        result = config.to_dict()
        assert result == {
            "default_strategy": "round_robin",
            "probabilistic_quantile": 0.99,
        }


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.enable_auto_training is False
        assert config.batch_size == 100
        assert config.frequency_seconds == 3600
        assert config.min_samples == 10

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TrainingConfig(enable_auto_training=True, batch_size=200)
        result = config.to_dict()
        assert result["enable_auto_training"] is True
        assert result["batch_size"] == 200


class TestWebSocketConfig:
    """Test WebSocketConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = WebSocketConfig()
        assert config.heartbeat_interval == 30
        assert config.heartbeat_timeout_threshold == 3
        assert config.ack_timeout == 10.0
        assert config.max_message_size == 16 * 1024 * 1024

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = WebSocketConfig(heartbeat_interval=60)
        result = config.to_dict()
        assert result["heartbeat_interval"] == 60


# ==================== SchedulerClient Initialization Tests ====================


class TestSchedulerClientInit:
    """Test SchedulerClient initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        client = SchedulerClient()
        assert client.scheduler_url == "http://localhost:8000"
        assert client.instance_id == "instance-default"
        assert client.timeout == 10.0
        assert client.max_retries == 3
        assert client.retry_delay == 2.0
        assert isinstance(client.predictor_config, PredictorConfig)
        assert isinstance(client.scheduling_config, SchedulingConfig)
        assert client._registered is False
        assert client._scheduler_process is None

    def test_custom_initialization(self):
        """Test custom initialization."""
        predictor_cfg = PredictorConfig(url="http://predictor:8001")
        scheduling_cfg = SchedulingConfig(default_strategy="min_time")

        client = SchedulerClient(
            scheduler_url="http://scheduler:9000",
            instance_id="test-instance",
            instance_endpoint="http://instance:5000",
            timeout=20.0,
            predictor_config=predictor_cfg,
            scheduling_config=scheduling_cfg,
        )

        assert client.scheduler_url == "http://scheduler:9000"
        assert client.instance_id == "test-instance"
        assert client.instance_endpoint == "http://instance:5000"
        assert client.timeout == 20.0
        assert client.predictor_config == predictor_cfg
        assert client.scheduling_config == scheduling_cfg

    def test_env_var_initialization(self, monkeypatch):
        """Test initialization from environment variables."""
        monkeypatch.setenv("SCHEDULER_URL", "http://env-scheduler:8000")
        monkeypatch.setenv("INSTANCE_ID", "env-instance")
        monkeypatch.setenv("INSTANCE_PORT", "6000")

        client = SchedulerClient()
        assert client.scheduler_url == "http://env-scheduler:8000"
        assert client.instance_id == "env-instance"
        assert "6000" in client.instance_endpoint


class TestSchedulerClientProperties:
    """Test SchedulerClient properties."""

    def test_is_enabled_true(self):
        """Test is_enabled when scheduler URL is set."""
        client = SchedulerClient(scheduler_url="http://localhost:8000")
        assert client.is_enabled is True

    def test_is_enabled_false(self):
        """Test is_enabled when scheduler URL is None."""
        # Directly set scheduler_url to None after initialization
        client = SchedulerClient()
        client.scheduler_url = None
        assert client.is_enabled is False

    def test_is_registered_initially_false(self):
        """Test is_registered is initially False."""
        client = SchedulerClient()
        assert client.is_registered is False

    def test_is_registered_after_setting(self):
        """Test is_registered after setting."""
        client = SchedulerClient()
        client._registered = True
        assert client.is_registered is True

    def test_is_scheduler_running_no_process(self):
        """Test is_scheduler_running with no process."""
        client = SchedulerClient()
        assert client.is_scheduler_running is False

    def test_is_scheduler_running_with_process(self):
        """Test is_scheduler_running with active process."""
        client = SchedulerClient()
        mock_process = Mock()
        mock_process.poll.return_value = None  # Still running
        client._scheduler_process = mock_process
        assert client.is_scheduler_running is True

    def test_is_scheduler_running_process_stopped(self):
        """Test is_scheduler_running with stopped process."""
        client = SchedulerClient()
        mock_process = Mock()
        mock_process.poll.return_value = 0  # Exited
        client._scheduler_process = mock_process
        assert client.is_scheduler_running is False

    def test_get_config_dict(self):
        """Test get_config_dict method."""
        client = SchedulerClient()
        config_dict = client.get_config_dict()
        assert "predictor" in config_dict
        assert "scheduling" in config_dict
        assert "training" in config_dict
        assert "websocket" in config_dict


# ==================== Port Management Tests ====================


class TestPortManagement:
    """Test port detection and management methods."""

    def test_is_port_in_use_free_port(self):
        """Test _is_port_in_use with a free port."""
        client = SchedulerClient()
        # Find a truly free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]
        assert client._is_port_in_use(free_port) is False

    def test_is_port_in_use_occupied_port(self):
        """Test _is_port_in_use with an occupied port."""
        client = SchedulerClient()
        # Create a socket to occupy a port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            occupied_port = s.getsockname()[1]
            # Port is in use while socket is open
            assert client._is_port_in_use(occupied_port) is True

    def test_find_available_port_success(self):
        """Test _find_available_port finds an available port."""
        client = SchedulerClient()
        # Find a base port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            base_port = s.getsockname()[1]

        available = client._find_available_port(base_port + 10, max_attempts=10)
        assert available is not None
        assert available >= base_port + 10

    def test_find_available_port_no_port_found(self):
        """Test _find_available_port when no port is available."""
        client = SchedulerClient()
        # Mock to always return port in use
        with patch.object(client, "_is_port_in_use", return_value=True):
            available = client._find_available_port(8000, max_attempts=5)
            assert available is None

    def test_extract_port_from_url_with_port(self):
        """Test _extract_port_from_url with explicit port."""
        client = SchedulerClient()
        port = client._extract_port_from_url("http://localhost:8000")
        assert port == 8000

    def test_extract_port_from_url_https_with_port(self):
        """Test _extract_port_from_url with HTTPS and port."""
        client = SchedulerClient()
        port = client._extract_port_from_url("https://example.com:9000")
        assert port == 9000

    def test_extract_port_from_url_default_http(self):
        """Test _extract_port_from_url defaults to 80 for HTTP."""
        client = SchedulerClient()
        port = client._extract_port_from_url("http://localhost")
        assert port == 80

    def test_extract_port_from_url_default_https(self):
        """Test _extract_port_from_url defaults to 443 for HTTPS."""
        client = SchedulerClient()
        port = client._extract_port_from_url("https://localhost")
        assert port == 443

    def test_update_scheduler_url_port(self):
        """Test _update_scheduler_url_port updates URL correctly."""
        client = SchedulerClient(scheduler_url="http://localhost:8000/path")
        client._update_scheduler_url_port(9000)
        assert client.scheduler_url == "http://localhost:9000/path"


# ==================== Predictor Dependency Tests ====================


class TestPredictorDependency:
    """Test predictor dependency checking."""

    @pytest.mark.asyncio
    async def test_check_predictor_available_healthy(self):
        """Test _check_predictor_available when predictor is healthy."""
        client = SchedulerClient()

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"status": "healthy"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client._check_predictor_available()
            assert result is True

    @pytest.mark.asyncio
    async def test_check_predictor_available_unhealthy(self):
        """Test _check_predictor_available when predictor is unhealthy."""
        client = SchedulerClient()

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"status": "unhealthy"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client._check_predictor_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_check_predictor_available_connection_error(self):
        """Test _check_predictor_available when connection fails."""
        client = SchedulerClient()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
            mock_client_class.return_value = mock_client

            result = await client._check_predictor_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_scheduler_ready_success(self):
        """Test _wait_for_scheduler_ready when scheduler becomes ready."""
        client = SchedulerClient()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client._wait_for_scheduler_ready(timeout=5.0, check_interval=0.1)
            assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_scheduler_ready_timeout(self):
        """Test _wait_for_scheduler_ready when timeout occurs."""
        client = SchedulerClient()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Not ready"))
            mock_client_class.return_value = mock_client

            result = await client._wait_for_scheduler_ready(timeout=0.5, check_interval=0.1)
            assert result is False


# ==================== Scheduler Auto-Start Tests ====================


class TestSchedulerAutoStart:
    """Test scheduler auto-start functionality."""

    @pytest.mark.asyncio
    async def test_start_scheduler_predictor_not_available(self):
        """Test start_scheduler fails when predictor is not available."""
        client = SchedulerClient()

        with patch.object(client, "_check_predictor_available", return_value=False):
            result = await client.start_scheduler(check_predictor=True)
            assert result is False

    @pytest.mark.asyncio
    async def test_start_scheduler_already_running(self):
        """Test start_scheduler when scheduler is already running."""
        client = SchedulerClient()

        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(client, "_check_predictor_available", return_value=True), patch.object(
            client, "_is_port_in_use", return_value=True
        ), patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.start_scheduler()
            assert result is True

    @pytest.mark.asyncio
    async def test_start_scheduler_port_occupied_no_auto_find(self):
        """Test start_scheduler fails when port occupied and auto_find_port=False."""
        client = SchedulerClient()

        with patch.object(client, "_check_predictor_available", return_value=True), patch.object(
            client, "_is_port_in_use", return_value=True
        ), patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Not scheduler"))
            mock_client_class.return_value = mock_client

            result = await client.start_scheduler(auto_find_port=False)
            assert result is False

    @pytest.mark.asyncio
    async def test_start_scheduler_module_not_found(self):
        """Test start_scheduler raises error when module not found."""
        client = SchedulerClient(scheduler_module_path="/non/existent/path.py")

        with patch.object(client, "_check_predictor_available", return_value=True), patch.object(
            client, "_is_port_in_use", return_value=False
        ):
            with pytest.raises(RuntimeError, match="Scheduler module not found"):
                await client.start_scheduler()

    @pytest.mark.asyncio
    async def test_start_scheduler_success(self, tmp_path):
        """Test successful scheduler start."""
        # Create temporary module file
        module_path = tmp_path / "main.py"
        module_path.write_text("# Dummy module")

        client = SchedulerClient(scheduler_module_path=str(module_path))

        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None

        with patch.object(client, "_check_predictor_available", return_value=True), patch.object(
            client, "_is_port_in_use", return_value=False
        ), patch("subprocess.Popen", return_value=mock_process), patch.object(
            client, "_wait_for_scheduler_ready", return_value=True
        ):
            result = await client.start_scheduler()
            assert result is True
            assert client._scheduler_process == mock_process

    @pytest.mark.asyncio
    async def test_start_scheduler_fails_to_start_process(self, tmp_path):
        """Test start_scheduler when process fails to start."""
        module_path = tmp_path / "main.py"
        module_path.write_text("# Dummy module")

        client = SchedulerClient(scheduler_module_path=str(module_path))

        with patch.object(client, "_check_predictor_available", return_value=True), patch.object(
            client, "_is_port_in_use", return_value=False
        ), patch("subprocess.Popen", side_effect=Exception("Failed to start")):
            result = await client.start_scheduler()
            assert result is False

    @pytest.mark.asyncio
    async def test_start_scheduler_not_ready_after_start(self, tmp_path):
        """Test start_scheduler when scheduler doesn't become ready."""
        module_path = tmp_path / "main.py"
        module_path.write_text("# Dummy module")

        client = SchedulerClient(scheduler_module_path=str(module_path))

        mock_process = Mock()
        mock_process.pid = 12345

        with patch.object(client, "_check_predictor_available", return_value=True), patch.object(
            client, "_is_port_in_use", return_value=False
        ), patch("subprocess.Popen", return_value=mock_process), patch.object(
            client, "_wait_for_scheduler_ready", return_value=False
        ), patch.object(client, "stop_scheduler", return_value=True):
            result = await client.start_scheduler()
            assert result is False

    def test_stop_scheduler_no_process(self):
        """Test stop_scheduler when no process exists."""
        client = SchedulerClient()
        result = client.stop_scheduler()
        assert result is False

    def test_stop_scheduler_already_stopped(self):
        """Test stop_scheduler when process already stopped."""
        client = SchedulerClient()
        mock_process = Mock()
        mock_process.poll.return_value = 0  # Already stopped
        client._scheduler_process = mock_process

        result = client.stop_scheduler()
        assert result is True
        assert client._scheduler_process is None

    def test_stop_scheduler_graceful(self):
        """Test stop_scheduler with graceful termination."""
        client = SchedulerClient()
        mock_process = Mock()
        mock_process.poll.return_value = None  # Running
        mock_process.pid = 12345
        mock_process.terminate = Mock()
        mock_process.wait = Mock(return_value=0)
        client._scheduler_process = mock_process

        result = client.stop_scheduler()
        assert result is True
        assert client._scheduler_process is None
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()

    def test_stop_scheduler_force_kill(self):
        """Test stop_scheduler with force kill after timeout."""
        client = SchedulerClient()
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_process.terminate = Mock()
        mock_process.wait = Mock(side_effect=[subprocess.TimeoutExpired("cmd", 10), None])
        mock_process.kill = Mock()
        client._scheduler_process = mock_process

        result = client.stop_scheduler()
        assert result is True
        assert client._scheduler_process is None
        mock_process.kill.assert_called_once()

    def test_stop_scheduler_error_handling(self):
        """Test stop_scheduler handles errors gracefully."""
        client = SchedulerClient()
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_process.terminate = Mock(side_effect=Exception("Termination failed"))
        client._scheduler_process = mock_process

        result = client.stop_scheduler()
        assert result is False


# ==================== Instance Registration Tests ====================


class TestInstanceRegistration:
    """Test instance registration and deregistration."""

    @pytest.mark.asyncio
    async def test_register_instance_disabled(self):
        """Test register_instance when scheduler is disabled."""
        client = SchedulerClient(scheduler_url=None)
        result = await client.register_instance(model_id="test-model")
        assert result is False

    @pytest.mark.asyncio
    async def test_register_instance_success(self):
        """Test successful instance registration."""
        client = SchedulerClient()

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"success": True}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.register_instance(model_id="test-model")
            assert result is True
            assert client._registered is True

    @pytest.mark.asyncio
    async def test_register_instance_with_platform_info(self):
        """Test registration with custom platform info."""
        client = SchedulerClient()

        platform_info = {"software_name": "custom", "software_version": "1.0", "hardware_name": "gpu"}

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"success": True}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.register_instance(model_id="test-model", platform_info=platform_info)
            assert result is True

            # Verify platform_info was included in request
            call_args = mock_client.post.call_args
            request_data = call_args[1]["json"]
            assert request_data["platform_info"] == platform_info

    @pytest.mark.asyncio
    async def test_register_instance_failure(self):
        """Test failed instance registration."""
        client = SchedulerClient()

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"success": False, "error": "Already exists"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.register_instance(model_id="test-model")
            assert result is False
            assert client._registered is False

    @pytest.mark.asyncio
    async def test_register_instance_retry_on_error(self):
        """Test registration retries on HTTP error."""
        client = SchedulerClient(max_retries=2)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
            mock_client_class.return_value = mock_client

            with patch("asyncio.sleep"):
                result = await client.register_instance(model_id="test-model")
                assert result is False
                assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_deregister_instance_not_registered(self):
        """Test deregister when not registered."""
        client = SchedulerClient()
        result = await client.deregister_instance()
        assert result is False

    @pytest.mark.asyncio
    async def test_deregister_instance_success(self):
        """Test successful deregistration."""
        client = SchedulerClient()
        client._registered = True

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"success": True}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.deregister_instance()
            assert result is True
            assert client._registered is False


# ==================== Draining Tests ====================


class TestDraining:
    """Test instance draining operations."""

    @pytest.mark.asyncio
    async def test_drain_instance_disabled(self):
        """Test drain_instance when scheduler is disabled."""
        client = SchedulerClient()
        # Disable scheduler by setting URL to None
        client.scheduler_url = None

        with pytest.raises(Exception, match="Scheduler integration disabled"):
            await client.drain_instance()

    @pytest.mark.asyncio
    async def test_drain_instance_success(self):
        """Test successful draining."""
        client = SchedulerClient()

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "success": True,
            "pending_tasks": 5,
            "running_tasks": 2,
            "estimated_completion_time_ms": 5000.0,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.drain_instance()
            assert result["success"] is True
            assert result["pending_tasks"] == 5

    @pytest.mark.asyncio
    async def test_drain_instance_failure(self):
        """Test failed draining."""
        client = SchedulerClient()

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"success": False, "error": "Instance not found"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            with pytest.raises(Exception, match="Drain request failed"):
                await client.drain_instance()


# ==================== Task Callback Tests ====================


class TestTaskCallbacks:
    """Test task result callback functionality."""

    @pytest.mark.asyncio
    async def test_send_task_result_disabled(self):
        """Test send_task_result when scheduler is disabled."""
        client = SchedulerClient(scheduler_url=None)
        result = await client.send_task_result(task_id="task-1", status="completed")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_task_result_success(self):
        """Test successful task result callback."""
        client = SchedulerClient()

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"success": True}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.send_task_result(
                task_id="task-1",
                status="completed",
                result={"output": "data"},
                execution_time_ms=100.5,
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_send_task_result_failed_task(self):
        """Test callback for failed task."""
        client = SchedulerClient()

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"success": True}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.send_task_result(
                task_id="task-2", status="failed", error="Out of memory", execution_time_ms=50.0
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_send_task_result_retry(self):
        """Test callback retry on failure."""
        client = SchedulerClient(max_retries=3)

        mock_success_response = Mock()
        mock_success_response.raise_for_status = Mock()
        mock_success_response.json.return_value = {"success": True}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            # Fail twice, succeed on third attempt
            mock_client.post = AsyncMock(
                side_effect=[
                    httpx.ConnectError("Failed"),
                    httpx.ConnectError("Failed"),
                    mock_success_response,
                ]
            )
            mock_client_class.return_value = mock_client

            with patch("asyncio.sleep"):
                result = await client.send_task_result(task_id="task-3", status="completed")
                assert result is True
                assert mock_client.post.call_count == 3


# ==================== Utility Methods Tests ====================


class TestUtilityMethods:
    """Test utility methods."""

    def test_get_platform_info(self):
        """Test _get_platform_info returns correct structure."""
        client = SchedulerClient()
        info = client._get_platform_info()

        assert "software_name" in info
        assert "software_version" in info
        assert "hardware_name" in info
        assert isinstance(info["software_name"], str)
        assert isinstance(info["software_version"], str)
        assert isinstance(info["hardware_name"], str)

    def test_update_predictor_config(self):
        """Test update_predictor_config."""
        client = SchedulerClient()
        new_config = PredictorConfig(url="http://new:8001")

        client.update_predictor_config(new_config)
        assert client.predictor_config == new_config
        assert client.predictor_config.url == "http://new:8001"

    def test_update_scheduling_config(self):
        """Test update_scheduling_config."""
        client = SchedulerClient()
        new_config = SchedulingConfig(default_strategy="min_time", probabilistic_quantile=0.95)

        client.update_scheduling_config(new_config)
        assert client.scheduling_config == new_config
        assert client.scheduling_config.default_strategy == "min_time"


# ==================== Global Functions Tests ====================


class TestGlobalFunctions:
    """Test global helper functions."""

    def test_get_scheduler_client_singleton(self):
        """Test get_scheduler_client returns singleton."""
        # Reset global
        import src.scheduler_client as sc

        sc._scheduler_client = None

        client1 = get_scheduler_client()
        client2 = get_scheduler_client()
        assert client1 is client2

    def test_initialize_scheduler_client(self):
        """Test initialize_scheduler_client creates configured client."""
        predictor_cfg = PredictorConfig(url="http://predictor:8001")

        client = initialize_scheduler_client(
            scheduler_url="http://scheduler:9000",
            instance_id="test-instance",
            predictor_config=predictor_cfg,
        )

        assert client.scheduler_url == "http://scheduler:9000"
        assert client.instance_id == "test-instance"
        assert client.predictor_config.url == "http://predictor:8001"

    def test_initialize_scheduler_client_replaces_global(self):
        """Test initialize_scheduler_client replaces global instance."""
        client1 = initialize_scheduler_client(instance_id="client-1")
        client2 = initialize_scheduler_client(instance_id="client-2")

        assert client1 is not client2
        assert get_scheduler_client() is client2


# ==================== Edge Cases and Error Handling Tests ====================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_register_instance_auto_detect_platform(self):
        """Test registration auto-detects platform info when not provided."""
        client = SchedulerClient()

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"success": True}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.register_instance(model_id="test-model")
            assert result is True

            # Verify platform_info was auto-detected
            call_args = mock_client.post.call_args
            request_data = call_args[1]["json"]
            assert "platform_info" in request_data
            assert "software_name" in request_data["platform_info"]

    @pytest.mark.asyncio
    async def test_send_task_result_custom_callback_url(self):
        """Test send_task_result with custom callback URL."""
        client = SchedulerClient()

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"success": True}

        custom_url = "http://custom-callback:8000/callback"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.send_task_result(
                task_id="task-1", status="completed", callback_url=custom_url
            )
            assert result is True

            # Verify custom URL was used
            call_args = mock_client.post.call_args
            assert call_args[0][0] == custom_url

    @pytest.mark.asyncio
    async def test_register_instance_exclude_config(self):
        """Test registration without including configuration."""
        client = SchedulerClient()

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"success": True}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.register_instance(model_id="test-model", include_config=False)
            assert result is True

            # Verify config was not included
            call_args = mock_client.post.call_args
            request_data = call_args[1]["json"]
            assert "config" not in request_data

    @pytest.mark.asyncio
    async def test_check_predictor_available_exception(self):
        """Test _check_predictor_available with general exception."""
        client = SchedulerClient()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(side_effect=Exception("Unexpected error"))
            mock_client_class.return_value = mock_client

            result = await client._check_predictor_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_start_scheduler_auto_find_port_no_port_available(self, tmp_path):
        """Test start_scheduler when no available port can be found."""
        module_path = tmp_path / "main.py"
        module_path.write_text("# Dummy module")

        client = SchedulerClient(scheduler_module_path=str(module_path))

        with patch.object(client, "_check_predictor_available", return_value=True), patch.object(
            client, "_is_port_in_use", return_value=True
        ), patch.object(client, "_find_available_port", return_value=None), patch(
            "httpx.AsyncClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Not scheduler"))
            mock_client_class.return_value = mock_client

            result = await client.start_scheduler(auto_find_port=True)
            assert result is False

    @pytest.mark.asyncio
    async def test_register_instance_http_exception(self):
        """Test registration with unexpected HTTP exception."""
        client = SchedulerClient()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=Exception("Unexpected HTTP exception"))
            mock_client_class.return_value = mock_client

            result = await client.register_instance(model_id="test-model")
            assert result is False

    @pytest.mark.asyncio
    async def test_deregister_instance_http_error(self):
        """Test deregister with HTTP error."""
        client = SchedulerClient()
        client._registered = True

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=httpx.HTTPError("HTTP error"))
            mock_client_class.return_value = mock_client

            result = await client.deregister_instance()
            assert result is False

    @pytest.mark.asyncio
    async def test_deregister_instance_unexpected_exception(self):
        """Test deregister with unexpected exception."""
        client = SchedulerClient()
        client._registered = True

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=Exception("Unexpected error"))
            mock_client_class.return_value = mock_client

            result = await client.deregister_instance()
            assert result is False

    @pytest.mark.asyncio
    async def test_drain_instance_http_error(self):
        """Test drain with HTTP error."""
        client = SchedulerClient()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=httpx.HTTPError("HTTP error"))
            mock_client_class.return_value = mock_client

            with pytest.raises(Exception, match="Failed to drain instance"):
                await client.drain_instance()

    @pytest.mark.asyncio
    async def test_send_task_result_callback_failure_max_retries(self):
        """Test callback failure response after max retries."""
        client = SchedulerClient(max_retries=2)

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"success": False, "error": "Callback error"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            with patch("asyncio.sleep"):
                result = await client.send_task_result(task_id="task-1", status="completed")
                assert result is False
                # Should have tried max_retries times
                assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_send_task_result_unexpected_exception(self):
        """Test callback with unexpected exception."""
        client = SchedulerClient()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=Exception("Unexpected error"))
            mock_client_class.return_value = mock_client

            result = await client.send_task_result(task_id="task-1", status="completed")
            assert result is False
