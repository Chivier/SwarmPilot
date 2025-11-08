"""
Integration tests for client auto-start functionality.

These tests verify end-to-end auto-start workflows for both
PredictorClient and InstanceClient.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clients.instance_client import InstanceClient
from src.clients.predictor_client import PredictorClient


class TestPredictorClientIntegration:
    """Integration tests for PredictorClient auto-start."""

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        """Test complete start/stop lifecycle of predictor."""
        client = PredictorClient()

        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None

        with patch.object(client, "_is_port_in_use", return_value=False), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", return_value=mock_process), \
             patch.object(client, "_wait_for_predictor_ready", return_value=True):

            # Start predictor
            result = await client.start_predictor()
            assert result is True
            assert client.is_predictor_running is True

            # Stop predictor
            with patch.object(mock_process, "wait"):
                stop_result = client.stop_predictor()
                assert stop_result is True
                assert client.is_predictor_running is False

    @pytest.mark.asyncio
    async def test_port_conflict_resolution(self):
        """Test automatic port resolution when default port is occupied."""
        client = PredictorClient(predictor_url="http://localhost:8001")

        mock_process = MagicMock()
        mock_process.pid = 12345

        # Simulate: port 8001 occupied, 8002 available
        def port_check(port):
            return port == 8001

        with patch.object(client, "_is_port_in_use", side_effect=port_check), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", return_value=mock_process), \
             patch.object(client, "_wait_for_predictor_ready", return_value=True), \
             patch("httpx.AsyncClient") as mock_http:

            # Mock failed health check for port 8001
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_http.return_value.__aenter__.return_value = mock_client

            result = await client.start_predictor(auto_find_port=True)

            assert result is True
            # Should have found port 8002
            assert "8002" in client.predictor_url

    @pytest.mark.asyncio
    async def test_startup_failure_cleanup(self):
        """Test that failed startup properly cleans up resources."""
        client = PredictorClient()

        mock_process = MagicMock()
        mock_process.pid = 12345

        with patch.object(client, "_is_port_in_use", return_value=False), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", return_value=mock_process), \
             patch.object(client, "_wait_for_predictor_ready", return_value=False), \
             patch.object(client, "stop_predictor") as mock_stop:

            with pytest.raises(ConnectionError):
                await client.start_predictor()

            # Verify cleanup was called
            mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_custom_port_override(self):
        """Test starting predictor with custom port override."""
        client = PredictorClient()

        mock_process = MagicMock()
        mock_process.pid = 12345

        with patch.object(client, "_is_port_in_use", return_value=False), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", return_value=mock_process), \
             patch.object(client, "_wait_for_predictor_ready", return_value=True):

            result = await client.start_predictor(port=9000)

            assert result is True
            assert "9000" in client.predictor_url

    @pytest.mark.asyncio
    async def test_concurrent_start_attempts(self):
        """Test that concurrent start attempts are handled correctly."""
        client = PredictorClient()

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client, "_is_port_in_use", return_value=True), \
             patch("httpx.AsyncClient") as mock_http:

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_http.return_value.__aenter__.return_value = mock_client

            # First call should detect service already running
            result1 = await client.start_predictor()
            result2 = await client.start_predictor()

            assert result1 is True
            assert result2 is True


class TestInstanceClientIntegration:
    """Integration tests for InstanceClient auto-start."""

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        """Test complete start/stop lifecycle of instance."""
        client = InstanceClient(base_url="http://localhost:5000")

        mock_process = MagicMock()
        mock_process.pid = 54321
        mock_process.poll.return_value = None

        with patch.object(client, "_is_port_in_use", return_value=False), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", return_value=mock_process), \
             patch.object(client, "_wait_for_instance_ready", return_value=True):

            # Start instance
            result = await client.start_instance()
            assert result is True
            assert client.is_instance_running is True

            # Stop instance
            with patch.object(mock_process, "wait"):
                stop_result = client.stop_instance()
                assert stop_result is True
                assert client.is_instance_running is False

    @pytest.mark.asyncio
    async def test_port_conflict_with_websocket_update(self):
        """Test that WebSocket URL is updated when port changes."""
        client = InstanceClient(base_url="http://localhost:5000")

        mock_process = MagicMock()
        mock_process.pid = 54321

        # Simulate: port 5000 occupied, 5001 available
        def port_check(port):
            return port == 5000

        with patch.object(client, "_is_port_in_use", side_effect=port_check), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", return_value=mock_process), \
             patch.object(client, "_wait_for_instance_ready", return_value=True), \
             patch("httpx.AsyncClient") as mock_http:

            # Mock failed health check for port 5000
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_http.return_value.__aenter__.return_value = mock_client

            result = await client.start_instance(auto_find_port=True)

            assert result is True
            # Verify both HTTP and WebSocket URLs updated
            assert "5001" in client.base_url
            assert "5001" in client.websocket_url

    @pytest.mark.asyncio
    async def test_instance_with_custom_id(self):
        """Test starting instance with custom instance ID."""
        client = InstanceClient(base_url="http://localhost:5000")

        mock_process = MagicMock()
        mock_process.pid = 54321

        with patch.object(client, "_is_port_in_use", return_value=False), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", return_value=mock_process) as mock_popen, \
             patch.object(client, "_wait_for_instance_ready", return_value=True):

            result = await client.start_instance(instance_id="custom-instance-123")

            assert result is True

            # Verify environment variable was set
            call_args = mock_popen.call_args
            env = call_args.kwargs["env"]
            assert env["INSTANCE_ID"] == "custom-instance-123"

    @pytest.mark.asyncio
    async def test_https_websocket_url_sync(self):
        """Test that HTTPS URLs generate WSS WebSocket URLs."""
        client = InstanceClient(base_url="https://localhost:5000")

        mock_process = MagicMock()

        with patch.object(client, "_is_port_in_use", return_value=False), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", return_value=mock_process), \
             patch.object(client, "_wait_for_instance_ready", return_value=True):

            result = await client.start_instance(port=6000)

            assert result is True
            assert "https://localhost:6000" in client.base_url
            assert client.websocket_url == "wss://localhost:6000/ws"


class TestMultiClientScenarios:
    """Integration tests involving multiple clients."""

    @pytest.mark.asyncio
    async def test_sequential_predictor_then_instance(self):
        """Test starting predictor followed by instance."""
        predictor = PredictorClient()
        instance = InstanceClient(base_url="http://localhost:5000")

        mock_predictor_process = MagicMock()
        mock_predictor_process.pid = 11111
        mock_predictor_process.poll.return_value = None  # Running

        mock_instance_process = MagicMock()
        mock_instance_process.pid = 22222
        mock_instance_process.poll.return_value = None  # Running

        # Start predictor
        with patch.object(predictor, "_is_port_in_use", return_value=False), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", return_value=mock_predictor_process), \
             patch.object(predictor, "_wait_for_predictor_ready", return_value=True):

            result1 = await predictor.start_predictor()
            assert result1 is True

        # Start instance
        with patch.object(instance, "_is_port_in_use", return_value=False), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", return_value=mock_instance_process), \
             patch.object(instance, "_wait_for_instance_ready", return_value=True):

            result2 = await instance.start_instance()
            assert result2 is True

        # Both should be running
        assert predictor.is_predictor_running is True
        assert instance.is_instance_running is True

    @pytest.mark.asyncio
    async def test_parallel_client_starts(self):
        """Test starting multiple clients in parallel."""
        predictor = PredictorClient()
        instance1 = InstanceClient(base_url="http://localhost:5000")
        instance2 = InstanceClient(base_url="http://localhost:5001")

        async def start_predictor():
            with patch.object(predictor, "_is_port_in_use", return_value=False), \
                 patch("os.path.exists", return_value=True), \
                 patch("subprocess.Popen", return_value=MagicMock()), \
                 patch.object(predictor, "_wait_for_predictor_ready", return_value=True):
                return await predictor.start_predictor()

        async def start_instance(client):
            with patch.object(client, "_is_port_in_use", return_value=False), \
                 patch("os.path.exists", return_value=True), \
                 patch("subprocess.Popen", return_value=MagicMock()), \
                 patch.object(client, "_wait_for_instance_ready", return_value=True):
                return await client.start_instance()

        # Start all clients in parallel
        results = await asyncio.gather(
            start_predictor(),
            start_instance(instance1),
            start_instance(instance2)
        )

        assert all(results)


class TestErrorRecovery:
    """Integration tests for error handling and recovery."""

    @pytest.mark.asyncio
    async def test_predictor_restart_after_failure(self):
        """Test restarting predictor after a failure."""
        client = PredictorClient()

        mock_process = MagicMock()

        # First attempt fails
        with patch.object(client, "_is_port_in_use", return_value=False), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", return_value=mock_process), \
             patch.object(client, "_wait_for_predictor_ready", return_value=False), \
             patch.object(client, "stop_predictor"):

            with pytest.raises(ConnectionError):
                await client.start_predictor()

        # Second attempt succeeds
        with patch.object(client, "_is_port_in_use", return_value=False), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", return_value=mock_process), \
             patch.object(client, "_wait_for_predictor_ready", return_value=True):

            result = await client.start_predictor()
            assert result is True

    @pytest.mark.asyncio
    async def test_instance_recovery_from_module_not_found(self):
        """Test proper error when module not found."""
        client = InstanceClient(
            base_url="http://localhost:5000",
            instance_module_path="/nonexistent"
        )

        with patch.object(client, "_is_port_in_use", return_value=False), \
             patch("os.path.exists", return_value=False):

            with pytest.raises(RuntimeError, match="Instance module not found"):
                await client.start_instance()

        # Fix the path and try again
        client.instance_module_path = "/valid/path"

        with patch.object(client, "_is_port_in_use", return_value=False), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.Popen", return_value=MagicMock()), \
             patch.object(client, "_wait_for_instance_ready", return_value=True):

            result = await client.start_instance()
            assert result is True
