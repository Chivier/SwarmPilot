"""
Unit tests for Instance WebSocket server and connection management.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, UTC

from src.instance_websocket_server import InstanceWebSocketServer
from src.instance_connection_manager import InstanceConnectionManager, ConnectionState


class TestInstanceWebSocketServer:
    """Test suite for InstanceWebSocketServer."""

    @pytest.fixture
    def mock_instance_registry(self):
        """Create mock instance registry."""
        registry = AsyncMock()
        registry.register = AsyncMock(return_value=True)
        registry.get = AsyncMock(return_value=None)
        return registry

    @pytest.fixture
    def mock_connection_manager(self):
        """Create mock connection manager."""
        manager = AsyncMock()
        manager.register_connection = AsyncMock()
        manager.handle_disconnect = AsyncMock()
        manager.handle_task_result = AsyncMock()
        manager.handle_ack = AsyncMock()
        manager.get_instance_id_by_websocket = AsyncMock(return_value="test-instance")
        manager.update_heartbeat = AsyncMock()
        return manager

    @pytest.fixture
    def server(self, mock_instance_registry, mock_connection_manager):
        """Create InstanceWebSocketServer instance."""
        return InstanceWebSocketServer(
            host="localhost",
            port=8001,
            instance_registry=mock_instance_registry,
            instance_connection_manager=mock_connection_manager,
        )

    @pytest.mark.asyncio
    async def test_server_initialization(self, server):
        """Test server initializes correctly."""
        assert server.host == "localhost"
        assert server.port == 8001
        assert server.instance_registry is not None
        assert server.connection_manager is not None

    @pytest.mark.asyncio
    async def test_register_message_handling(self, server, mock_connection_manager):
        """Test REGISTER message processing."""
        mock_ws = AsyncMock()
        message = {
            "type": "register",
            "message_id": "test-msg-1",
            "instance_id": "test-instance",
            "model_id": "test-model",
            "endpoint": "ws://test",
            "platform_info": {
                "software_name": "test",
                "software_version": "1.0",
                "hardware_name": "test-hw"
            }
        }

        await server._handle_register(mock_ws, message)

        # Verify connection was registered
        mock_connection_manager.register_connection.assert_called_once()

        # Verify ACK was sent
        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "register_ack"
        assert sent_data["success"] is True

    @pytest.mark.asyncio
    async def test_register_missing_fields(self, server):
        """Test REGISTER with missing required fields."""
        mock_ws = AsyncMock()
        message = {
            "type": "register",
            "message_id": "test-msg-1",
            # Missing instance_id and model_id
        }

        await server._handle_register(mock_ws, message)

        # Verify NACK was sent
        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "register_ack"
        assert sent_data["success"] is False

    @pytest.mark.asyncio
    async def test_task_result_handling(self, server, mock_connection_manager):
        """Test TASK_RESULT message processing."""
        mock_ws = AsyncMock()
        message = {
            "type": "task_result",
            "message_id": "test-msg-2",
            "task_id": "task-001",
            "status": "completed",
            "result": {"output": "test"},
            "error": None,
            "execution_time_ms": 1000.0
        }

        await server._handle_task_result(mock_ws, message)

        # Verify task result was processed
        mock_connection_manager.handle_task_result.assert_called_once_with(
            task_id="task-001",
            status="completed",
            result={"output": "test"},
            error=None,
            execution_time_ms=1000.0
        )

        # Verify ACK was sent
        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "result_ack"
        assert sent_data["task_id"] == "task-001"

    @pytest.mark.asyncio
    async def test_ping_pong(self, server, mock_connection_manager):
        """Test PING/PONG exchange."""
        mock_ws = AsyncMock()
        message = {
            "type": "ping",
            "message_id": "test-msg-3",
        }

        await server._handle_ping(mock_ws, message)

        # Verify PONG was sent
        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "pong"
        assert sent_data["reply_to"] == "test-msg-3"

    @pytest.mark.asyncio
    async def test_unknown_message_type(self, server):
        """Test handling of unknown message type."""
        mock_ws = AsyncMock()
        message = {
            "type": "unknown_type",
            "message_id": "test-msg-4",
        }

        # Should not raise exception
        await server._route_message(mock_ws, message)

        # Should send error
        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "error"


class TestInstanceConnectionManager:
    """Test suite for InstanceConnectionManager."""

    @pytest.fixture
    def mock_instance_registry(self):
        """Create mock instance registry."""
        registry = AsyncMock()
        registry.update_status = AsyncMock()
        registry.get_queue_info = AsyncMock()
        return registry

    @pytest.fixture
    def mock_task_dispatcher(self):
        """Create mock task dispatcher."""
        dispatcher = AsyncMock()
        dispatcher.handle_task_result = AsyncMock()
        return dispatcher

    @pytest.fixture
    def manager(self, mock_instance_registry, mock_task_dispatcher):
        """Create InstanceConnectionManager instance."""
        return InstanceConnectionManager(
            instance_registry=mock_instance_registry,
            task_dispatcher=mock_task_dispatcher,
            heartbeat_interval=1,  # Short for testing
            heartbeat_timeout_threshold=2,
            ack_timeout=2.0,
        )

    @pytest.mark.asyncio
    async def test_register_connection(self, manager):
        """Test connection registration."""
        mock_ws = AsyncMock()
        mock_ws.remote_address = ("127.0.0.1", 12345)

        await manager.register_connection("test-instance", mock_ws)

        assert "test-instance" in manager.connections
        assert manager.connections["test-instance"].instance_id == "test-instance"
        assert manager.connections["test-instance"].state == ConnectionState.CONNECTED

    @pytest.mark.asyncio
    async def test_send_message_with_ack(self, manager):
        """Test sending message with ACK requirement."""
        mock_ws = AsyncMock()
        await manager.register_connection("test-instance", mock_ws)

        message = {"type": "test", "data": "value"}

        # Start send in background
        send_task = asyncio.create_task(
            manager.send_message("test-instance", message, require_ack=True, timeout=2.0)
        )

        # Give it time to send
        await asyncio.sleep(0.1)

        # Simulate ACK
        message_id = json.loads(mock_ws.send.call_args[0][0])["message_id"]
        await manager.handle_ack(message_id, {"success": True})

        # Wait for send to complete
        ack_data = await send_task

        assert ack_data["success"] is True
        mock_ws.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_timeout(self, manager):
        """Test message send timeout."""
        mock_ws = AsyncMock()
        await manager.register_connection("test-instance", mock_ws)

        message = {"type": "test"}

        # Should timeout after 1 second
        with pytest.raises(TimeoutError):
            await manager.send_message(
                "test-instance", message, require_ack=True, timeout=1.0
            )

    @pytest.mark.asyncio
    async def test_heartbeat_update(self, manager):
        """Test heartbeat timestamp update."""
        mock_ws = AsyncMock()
        await manager.register_connection("test-instance", mock_ws)

        import time
        initial_time = manager.connections["test-instance"].last_heartbeat
        await asyncio.sleep(0.1)
        await manager.update_heartbeat("test-instance")

        assert manager.connections["test-instance"].last_heartbeat > initial_time

    @pytest.mark.asyncio
    async def test_disconnect_cleanup(self, manager, mock_instance_registry):
        """Test connection cleanup on disconnect."""
        mock_ws = AsyncMock()
        await manager.register_connection("test-instance", mock_ws)

        await manager.handle_disconnect(mock_ws)

        assert "test-instance" not in manager.connections
        mock_instance_registry.update_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_not_found(self, manager):
        """Test sending to non-existent connection."""
        with pytest.raises(ConnectionError):
            await manager.send_message("non-existent", {"type": "test"})

    @pytest.mark.asyncio
    async def test_get_connection_info(self, manager):
        """Test retrieving connection information."""
        mock_ws = AsyncMock()
        await manager.register_connection("test-instance", mock_ws)

        info = await manager.get_connection_info("test-instance")

        assert info is not None
        assert info["instance_id"] == "test-instance"
        assert info["state"] == "connected"
        assert "message_count" in info
        assert "error_count" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
