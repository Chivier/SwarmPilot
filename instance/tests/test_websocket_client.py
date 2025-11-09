"""
Unit tests for Instance WebSocket client.

NOTE: Tests in this file are temporarily disabled because WebSocket
communication with scheduler has been disabled. All Instance-Scheduler
communication now uses HTTP API only.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.websocket_client import WebSocketClient

# Skip all tests since WebSocket functionality is temporarily disabled
pytestmark = pytest.mark.skip(reason="WebSocket functionality temporarily disabled")


class TestWebSocketClient:
    """Test suite for WebSocketClient."""

    @pytest.fixture
    def client(self):
        """Create WebSocketClient instance."""
        return WebSocketClient(
            scheduler_url="ws://localhost:8001/instance/ws",
            instance_id="test-instance",
            model_id="test-model",
            platform_info={"software_name": "test"},
            reconnect_delay_max=4,
            heartbeat_interval=1,
        )

    def test_initialization(self, client):
        """Test client initializes correctly."""
        assert client.scheduler_url == "ws://localhost:8001/instance/ws"
        assert client.instance_id == "test-instance"
        assert client.model_id == "test-model"
        assert client.reconnect_delay == 1
        assert client.reconnect_delay_max == 4

    @pytest.mark.asyncio
    async def test_start_stop(self, client):
        """Test client start and stop."""
        # Start client
        await client.start()
        assert client.running is True

        # Stop client
        await client.stop()
        assert client.running is False

    @pytest.mark.asyncio
    async def test_register_handler(self, client):
        """Test message handler registration."""
        handler = AsyncMock()
        client.register_handler("test_type", handler)

        assert "test_type" in client.message_handlers
        assert client.message_handlers["test_type"] == handler

    @pytest.mark.asyncio
    async def test_send_message_not_connected(self, client):
        """Test sending message when not connected."""
        with pytest.raises(ConnectionError, match="Not connected"):
            await client.send_message({"type": "test"})

    @pytest.mark.asyncio
    @patch("websockets.connect")
    async def test_connection_establishes(self, mock_connect, client):
        """Test WebSocket connection establishment."""
        # Mock WebSocket
        mock_ws = AsyncMock()
        mock_ws.__aiter__.return_value = iter([])  # No messages
        mock_connect.return_value = mock_ws

        # Start connection loop
        connection_task = asyncio.create_task(client._connection_loop())

        # Wait a bit
        await asyncio.sleep(0.2)

        # Stop
        client.running = False
        await asyncio.sleep(0.1)

        # Verify connection was attempted
        mock_connect.assert_called()

    @pytest.mark.asyncio
    async def test_handle_register_ack(self, client):
        """Test handling REGISTER_ACK message."""
        message = {
            "type": "register_ack",
            "message_id": "msg-1",
            "success": True,
            "message": "Registration successful"
        }

        await client._handle_message(message)

        assert client.connected is True

    @pytest.mark.asyncio
    async def test_handle_register_ack_failure(self, client):
        """Test handling failed REGISTER_ACK."""
        message = {
            "type": "register_ack",
            "message_id": "msg-1",
            "success": False,
            "message": "Registration failed"
        }

        await client._handle_message(message)

        assert client.connected is False

    @pytest.mark.asyncio
    async def test_message_handler_routing(self, client):
        """Test message routing to registered handlers."""
        handler = AsyncMock()
        client.register_handler("task_submit", handler)

        message = {
            "type": "task_submit",
            "message_id": "msg-2",
            "task_id": "task-001",
            "data": "test"
        }

        await client._handle_message(message)

        handler.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_ack_handling(self, client):
        """Test ACK response handling."""
        # Create pending ACK
        message_id = "test-msg-id"
        future = asyncio.Future()
        client._pending_acks[message_id] = future

        # Simulate ACK response
        ack_message = {
            "type": "task_ack",
            "reply_to": message_id,
            "success": True
        }

        await client._handle_message(ack_message)

        # Future should be completed
        assert future.done()
        assert future.result()["success"] is True

    @pytest.mark.asyncio
    async def test_reconnection_backoff(self, client):
        """Test exponential backoff on reconnection."""
        assert client.reconnect_delay == 1

        # Simulate failed connections
        for expected_delay in [1, 2, 4]:
            assert client.reconnect_delay == expected_delay
            await client._handle_reconnect()

        # Should cap at max
        assert client.reconnect_delay == 4

    @pytest.mark.asyncio
    async def test_send_task_result(self, client):
        """Test sending task result."""
        # Mock connected state
        client.connected = True
        client.websocket = AsyncMock()

        result = await client.send_task_result(
            task_id="task-001",
            status="completed",
            result={"output": "test"},
            execution_time_ms=1000.0
        )

        # Should fail since no ACK handler
        assert result is False

    @pytest.mark.asyncio
    async def test_unregister(self, client):
        """Test unregister message sending."""
        client.connected = True
        client.websocket = AsyncMock()

        await client.send_unregister()

        # Verify message was sent
        client.websocket.send.assert_called_once()
        sent_data = json.loads(client.websocket.send.call_args[0][0])
        assert sent_data["type"] == "unregister"
        assert sent_data["instance_id"] == "test-instance"

    def test_is_connected(self, client):
        """Test connection status check."""
        assert client.is_connected() is False

        client.connected = True
        client.websocket = Mock()
        assert client.is_connected() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
