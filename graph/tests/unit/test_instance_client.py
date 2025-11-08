"""
Unit tests for InstanceClient auto-start functionality.

Tests cover port management, process lifecycle, and exception handling.
"""

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from src.clients.instance_client import InstanceClient


class TestInstanceClientPortManagement:
    """Test suite for port management methods."""

    def test_is_port_in_use_free_port(self):
        """Test _is_port_in_use returns False for available port."""
        client = InstanceClient(base_url="http://localhost:5000")

        with patch("socket.socket") as mock_socket:
            mock_socket_instance = MagicMock()
            mock_socket.return_value.__enter__.return_value = mock_socket_instance
            mock_socket_instance.bind.return_value = None

            result = client._is_port_in_use(5000)

            assert result is False
            mock_socket_instance.bind.assert_called_once_with(("127.0.0.1", 5000))

    def test_is_port_in_use_occupied_port(self):
        """Test _is_port_in_use returns True for occupied port."""
        client = InstanceClient(base_url="http://localhost:5000")

        with patch("socket.socket") as mock_socket:
            mock_socket_instance = MagicMock()
            mock_socket.return_value.__enter__.return_value = mock_socket_instance
            mock_socket_instance.bind.side_effect = OSError("Port in use")

            result = client._is_port_in_use(5000)

            assert result is True

    def test_find_available_port_first_available(self):
        """Test _find_available_port finds first available port."""
        client = InstanceClient(base_url="http://localhost:5000")

        with patch.object(client, "_is_port_in_use") as mock_check:
            mock_check.return_value = False

            port = client._find_available_port(5000)

            assert port == 5000
            mock_check.assert_called_once_with(5000)

    def test_find_available_port_after_occupied(self):
        """Test _find_available_port skips occupied ports."""
        client = InstanceClient(base_url="http://localhost:5000")

        with patch.object(client, "_is_port_in_use") as mock_check:
            # First 2 ports occupied, 3rd available
            mock_check.side_effect = [True, True, False]

            port = client._find_available_port(5000)

            assert port == 5002
            assert mock_check.call_count == 3

    def test_find_available_port_none_available(self):
        """Test _find_available_port returns None when no ports available."""
        client = InstanceClient(base_url="http://localhost:5000")

        with patch.object(client, "_is_port_in_use") as mock_check:
            mock_check.return_value = True

            port = client._find_available_port(5000, max_attempts=10)

            assert port is None
            assert mock_check.call_count == 10

    def test_extract_port_from_url_with_port(self):
        """Test _extract_port_from_url with explicit port."""
        client = InstanceClient(base_url="http://localhost:5000")

        port = client._extract_port_from_url("http://localhost:5000")
        assert port == 5000

        port = client._extract_port_from_url("https://example.com:6000")
        assert port == 6000

    def test_extract_port_from_url_default_http(self):
        """Test _extract_port_from_url defaults to 80 for http."""
        client = InstanceClient(base_url="http://localhost")

        port = client._extract_port_from_url("http://localhost")
        assert port == 80

    def test_extract_port_from_url_default_https(self):
        """Test _extract_port_from_url defaults to 443 for https."""
        client = InstanceClient(base_url="https://localhost")

        port = client._extract_port_from_url("https://localhost")
        assert port == 443

    def test_update_instance_url_port(self):
        """Test _update_instance_url_port updates both HTTP and WebSocket URLs."""
        client = InstanceClient(base_url="http://localhost:5000")

        client._update_instance_url_port(6000)

        assert client.base_url == "http://localhost:6000"
        assert client.websocket_url == "ws://localhost:6000/ws"

    def test_update_instance_url_port_https(self):
        """Test _update_instance_url_port handles HTTPS correctly."""
        client = InstanceClient(base_url="https://localhost:5000")

        client._update_instance_url_port(6000)

        assert client.base_url == "https://localhost:6000"
        assert client.websocket_url == "wss://localhost:6000/ws"


class TestInstanceClientModulePath:
    """Test suite for module path detection."""

    def test_find_instance_module_relative_path(self):
        """Test _find_instance_module finds module relative to file."""
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True

            client = InstanceClient(base_url="http://localhost:5000")

            assert "instance" in client.instance_module_path

    def test_find_instance_module_env_variable(self):
        """Test _find_instance_module uses environment variable."""
        with patch("pathlib.Path.exists", return_value=False), \
             patch.dict("os.environ", {"INSTANCE_MODULE_PATH": "/custom/instance"}):

            client = InstanceClient(base_url="http://localhost:5000")

            assert client.instance_module_path == "/custom/instance"

    def test_find_instance_module_fallback(self):
        """Test _find_instance_module fallback to current directory."""
        with patch("pathlib.Path.exists", return_value=False), \
             patch.dict("os.environ", {}, clear=True):

            client = InstanceClient(base_url="http://localhost:5000")

            assert "instance" in client.instance_module_path


class TestInstanceClientProcessManagement:
    """Test suite for process lifecycle management."""

    def test_is_instance_running_no_process(self):
        """Test is_instance_running returns False when no process."""
        client = InstanceClient(base_url="http://localhost:5000")

        assert client.is_instance_running is False

    def test_is_instance_running_with_running_process(self):
        """Test is_instance_running returns True for running process."""
        client = InstanceClient(base_url="http://localhost:5000")
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Still running
        client._instance_process = mock_process

        assert client.is_instance_running is True

    def test_is_instance_running_with_stopped_process(self):
        """Test is_instance_running returns False for stopped process."""
        client = InstanceClient(base_url="http://localhost:5000")
        mock_process = MagicMock()
        mock_process.poll.return_value = 0  # Exited
        client._instance_process = mock_process

        assert client.is_instance_running is False

    @pytest.mark.asyncio
    async def test_wait_for_instance_ready_success(self):
        """Test _wait_for_instance_ready succeeds when service becomes ready."""
        client = InstanceClient(base_url="http://localhost:5000")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await client._wait_for_instance_ready(timeout=5.0)

            assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_instance_ready_timeout(self):
        """Test _wait_for_instance_ready times out."""
        client = InstanceClient(base_url="http://localhost:5000")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await client._wait_for_instance_ready(timeout=0.5, check_interval=0.1)

            assert result is False


class TestInstanceClientAutoStart:
    """Test suite for start_instance functionality."""

    @pytest.mark.asyncio
    async def test_start_instance_already_running(self):
        """Test start_instance detects already running service."""
        client = InstanceClient(base_url="http://localhost:5000")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client, "_is_port_in_use", return_value=True), \
             patch("httpx.AsyncClient") as mock_client_class:

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await client.start_instance()

            assert result is True

    @pytest.mark.asyncio
    async def test_start_instance_port_occupied_auto_find(self):
        """Test start_instance finds new port when occupied."""
        client = InstanceClient(base_url="http://localhost:5000")

        with patch.object(client, "_is_port_in_use") as mock_port_check, \
             patch.object(client, "_find_available_port", return_value=5002), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen") as mock_popen, \
             patch.object(client, "_wait_for_instance_ready", return_value=True):

            mock_port_check.return_value = True

            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.get.side_effect = Exception("Connection refused")
                mock_client_class.return_value.__aenter__.return_value = mock_client

                result = await client.start_instance(auto_find_port=True)

            assert result is True
            assert client.base_url == "http://localhost:5002"
            assert client.websocket_url == "ws://localhost:5002/ws"

    @pytest.mark.asyncio
    async def test_start_instance_port_occupied_no_auto_find(self):
        """Test start_instance raises error when port occupied and auto_find=False."""
        client = InstanceClient(base_url="http://localhost:5000")

        with patch.object(client, "_is_port_in_use", return_value=True), \
             patch("httpx.AsyncClient") as mock_client_class:

            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(ConnectionError, match="Port .* is already in use"):
                await client.start_instance(auto_find_port=False)

    @pytest.mark.asyncio
    async def test_start_instance_module_not_found(self):
        """Test start_instance raises RuntimeError when module not found."""
        client = InstanceClient(
            base_url="http://localhost:5000",
            instance_module_path="/nonexistent/path"
        )

        with patch.object(client, "_is_port_in_use", return_value=False), \
             patch("os.path.exists", return_value=False):

            with pytest.raises(RuntimeError, match="Instance module not found"):
                await client.start_instance()

    @pytest.mark.asyncio
    async def test_start_instance_success(self):
        """Test start_instance successfully starts service."""
        client = InstanceClient(base_url="http://localhost:5000")

        mock_process = MagicMock()
        mock_process.pid = 12345

        with patch.object(client, "_is_port_in_use", return_value=False), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", return_value=mock_process) as mock_popen, \
             patch.object(client, "_wait_for_instance_ready", return_value=True):

            result = await client.start_instance()

            assert result is True
            assert client._instance_process == mock_process
            mock_popen.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_instance_with_instance_id(self):
        """Test start_instance sets instance ID environment variable."""
        client = InstanceClient(base_url="http://localhost:5000")

        mock_process = MagicMock()

        with patch.object(client, "_is_port_in_use", return_value=False), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", return_value=mock_process) as mock_popen, \
             patch.object(client, "_wait_for_instance_ready", return_value=True):

            await client.start_instance(instance_id="test-instance")

            # Check that INSTANCE_ID was set in environment
            call_args = mock_popen.call_args
            env = call_args.kwargs["env"]
            assert env["INSTANCE_ID"] == "test-instance"

    @pytest.mark.asyncio
    async def test_start_instance_fails_to_become_ready(self):
        """Test start_instance cleans up when service fails to become ready."""
        client = InstanceClient(base_url="http://localhost:5000")

        mock_process = MagicMock()

        with patch.object(client, "_is_port_in_use", return_value=False), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", return_value=mock_process), \
             patch.object(client, "_wait_for_instance_ready", return_value=False), \
             patch.object(client, "stop_instance") as mock_stop:

            with pytest.raises(ConnectionError, match="failed to become ready"):
                await client.start_instance()

            mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_instance_no_available_port(self):
        """Test start_instance raises error when no port available."""
        client = InstanceClient(base_url="http://localhost:5000")

        with patch.object(client, "_is_port_in_use", return_value=True), \
             patch.object(client, "_find_available_port", return_value=None), \
             patch("httpx.AsyncClient") as mock_client_class:

            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(ConnectionError, match="No available port"):
                await client.start_instance(auto_find_port=True)


class TestInstanceClientStop:
    """Test suite for stop_instance functionality."""

    def test_stop_instance_no_process(self):
        """Test stop_instance returns False when no process."""
        client = InstanceClient(base_url="http://localhost:5000")

        result = client.stop_instance()

        assert result is False

    def test_stop_instance_already_stopped(self):
        """Test stop_instance handles already stopped process."""
        client = InstanceClient(base_url="http://localhost:5000")
        mock_process = MagicMock()
        mock_process.poll.return_value = 0  # Exited
        client._instance_process = mock_process

        result = client.stop_instance()

        assert result is True
        assert client._instance_process is None

    def test_stop_instance_graceful(self):
        """Test stop_instance terminates gracefully."""
        client = InstanceClient(base_url="http://localhost:5000")
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Running
        mock_process.pid = 12345
        mock_process.wait.return_value = None
        client._instance_process = mock_process

        result = client.stop_instance()

        assert result is True
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=10.0)
        assert client._instance_process is None

    def test_stop_instance_force_kill(self):
        """Test stop_instance kills when terminate times out."""
        client = InstanceClient(base_url="http://localhost:5000")
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_process.wait.side_effect = [subprocess.TimeoutExpired(cmd="", timeout=10), None]
        client._instance_process = mock_process

        result = client.stop_instance()

        assert result is True
        mock_process.kill.assert_called_once()
        assert client._instance_process is None
