"""
Unit tests for Instance WebSocket server and connection management.
"""

import pytest
import asyncio
import json
import time
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

    @pytest.mark.asyncio
    async def test_start_server(self, server):
        """Test server start."""
        await server.start()
        assert server.server is not None

        # Clean up
        await server.stop()

    @pytest.mark.asyncio
    async def test_start_server_already_running(self, server):
        """Test starting server when already running."""
        await server.start()

        # Try to start again
        await server.start()  # Should log warning

        # Clean up
        await server.stop()

    @pytest.mark.asyncio
    async def test_stop_server_not_running(self, server):
        """Test stopping server when not running."""
        # Should log warning but not crash
        await server.stop()

    @pytest.mark.asyncio
    async def test_handle_task_ack(self, server, mock_connection_manager):
        """Test TASK_ACK message handling."""
        mock_ws = AsyncMock()
        message = {
            "type": "task_ack",
            "message_id": "test-msg-5",
            "reply_to": "original-msg-id",
            "task_id": "task-001",
            "success": True,
        }

        await server._handle_task_ack(mock_ws, message)

        # Verify ACK was forwarded to connection manager
        mock_connection_manager.handle_ack.assert_called_once_with(
            message_id="original-msg-id",
            ack_data=message,
        )

    @pytest.mark.asyncio
    async def test_handle_pong(self, server, mock_connection_manager):
        """Test PONG message handling."""
        mock_ws = AsyncMock()
        message = {
            "type": "pong",
            "message_id": "test-msg-6",
            "reply_to": "ping-msg-id",
        }

        mock_connection_manager.get_instance_id_by_websocket.return_value = "test-instance"

        await server._handle_pong(mock_ws, message)

        # Verify heartbeat was updated
        mock_connection_manager.update_heartbeat.assert_called_once_with("test-instance")

    @pytest.mark.asyncio
    async def test_handle_unregister(self, server, mock_connection_manager):
        """Test UNREGISTER message handling."""
        mock_ws = AsyncMock()
        message = {
            "type": "unregister",
            "message_id": "test-msg-7",
            "instance_id": "test-instance",
            "reason": "Shutting down",
        }

        await server._handle_unregister(mock_ws, message)

        # Verify ACK was sent
        mock_ws.send.assert_called()
        sent_data = json.loads(mock_ws.send.call_args_list[0][0][0])
        assert sent_data["type"] == "unregister_ack"

        # Verify disconnect was handled
        mock_connection_manager.handle_disconnect.assert_called_once_with(mock_ws)

    @pytest.mark.asyncio
    async def test_handle_unregister_error(self, server, mock_connection_manager):
        """Test UNREGISTER with error during handling."""
        mock_ws = AsyncMock()
        mock_connection_manager.handle_disconnect.side_effect = Exception("Disconnect error")

        message = {
            "type": "unregister",
            "message_id": "test-msg-8",
            "instance_id": "test-instance",
        }

        await server._handle_unregister(mock_ws, message)

        # Should send error ACK
        sent_data = json.loads(mock_ws.send.call_args_list[-1][0][0])
        assert sent_data["type"] == "unregister_ack"
        assert sent_data["success"] is False

    @pytest.mark.asyncio
    async def test_register_missing_model_id(self, server):
        """Test REGISTER with missing model_id."""
        mock_ws = AsyncMock()
        message = {
            "type": "register",
            "message_id": "test-msg-9",
            "instance_id": "test-instance",
            # Missing model_id
        }

        await server._handle_register(mock_ws, message)

        # Verify NACK was sent
        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "register_ack"
        assert sent_data["success"] is False
        assert "model_id" in sent_data["message"]

    @pytest.mark.asyncio
    async def test_register_already_registered(self, server, mock_instance_registry):
        """Test REGISTER for already registered instance."""
        mock_ws = AsyncMock()
        message = {
            "type": "register",
            "message_id": "test-msg-10",
            "instance_id": "test-instance",
            "model_id": "test-model",
            "endpoint": "ws://test",
        }

        # Simulate existing instance
        mock_instance_registry.get.return_value = {"instance_id": "test-instance"}

        await server._handle_register(mock_ws, message)

        # Should still succeed
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "register_ack"
        assert sent_data["success"] is True

    @pytest.mark.asyncio
    async def test_register_registry_failure(self, server, mock_instance_registry):
        """Test REGISTER when registry fails."""
        mock_ws = AsyncMock()
        message = {
            "type": "register",
            "message_id": "test-msg-11",
            "instance_id": "test-instance",
            "model_id": "test-model",
        }

        mock_instance_registry.register.return_value = False

        await server._handle_register(mock_ws, message)

        # Should send failure ACK
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "register_ack"
        assert sent_data["success"] is False

    @pytest.mark.asyncio
    async def test_register_exception(self, server, mock_instance_registry):
        """Test REGISTER with exception."""
        mock_ws = AsyncMock()
        message = {
            "type": "register",
            "message_id": "test-msg-12",
            "instance_id": "test-instance",
            "model_id": "test-model",
        }

        mock_instance_registry.register.side_effect = Exception("Registry error")

        await server._handle_register(mock_ws, message)

        # Should send error ACK
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "register_ack"
        assert sent_data["success"] is False

    @pytest.mark.asyncio
    async def test_task_result_missing_task_id(self, server):
        """Test TASK_RESULT with missing task_id."""
        mock_ws = AsyncMock()
        message = {
            "type": "task_result",
            "message_id": "test-msg-13",
            "status": "completed",
            # Missing task_id
        }

        await server._handle_task_result(mock_ws, message)

        # Should send error
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "error"

    @pytest.mark.asyncio
    async def test_task_result_missing_status(self, server):
        """Test TASK_RESULT with missing status."""
        mock_ws = AsyncMock()
        message = {
            "type": "task_result",
            "message_id": "test-msg-14",
            "task_id": "task-001",
            # Missing status
        }

        await server._handle_task_result(mock_ws, message)

        # Should send error
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "error"

    @pytest.mark.asyncio
    async def test_task_result_exception(self, server, mock_connection_manager):
        """Test TASK_RESULT with exception."""
        mock_ws = AsyncMock()
        message = {
            "type": "task_result",
            "message_id": "test-msg-15",
            "task_id": "task-001",
            "status": "completed",
        }

        mock_connection_manager.handle_task_result.side_effect = Exception("Handler error")

        await server._handle_task_result(mock_ws, message)

        # Should send ACK with failure
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "result_ack"
        assert sent_data["success"] is False

    @pytest.mark.asyncio
    async def test_send_message_helper(self, server):
        """Test _send_message helper method."""
        mock_ws = AsyncMock()
        message = {"type": "test", "data": "value"}

        await server._send_message(mock_ws, message)

        # Verify message was sent
        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "test"
        assert "timestamp" in sent_data

    @pytest.mark.asyncio
    async def test_send_message_with_timestamp(self, server):
        """Test _send_message preserves existing timestamp."""
        mock_ws = AsyncMock()
        timestamp = "2024-01-01T00:00:00Z"
        message = {"type": "test", "timestamp": timestamp}

        await server._send_message(mock_ws, message)

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["timestamp"] == timestamp

    @pytest.mark.asyncio
    async def test_send_error_message(self, server):
        """Test sending error message."""
        mock_ws = AsyncMock()

        await server._send_error(
            mock_ws,
            error="Test error",
            error_code="TEST_ERROR",
            reply_to="msg-123",
        )

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "error"
        assert sent_data["error"] == "Test error"
        assert sent_data["error_code"] == "TEST_ERROR"
        assert sent_data["reply_to"] == "msg-123"

    @pytest.mark.asyncio
    async def test_send_error_without_reply_to(self, server):
        """Test sending error message without reply_to."""
        mock_ws = AsyncMock()

        await server._send_error(
            mock_ws,
            error="Test error",
            error_code="TEST_ERROR",
        )

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "error"
        assert "reply_to" not in sent_data

    @pytest.mark.asyncio
    async def test_start_server_error(self, server):
        """Test server start with error."""
        # Mock websockets.serve to raise an exception
        with patch('src.instance_websocket_server.websockets.serve') as mock_serve:
            mock_serve.side_effect = Exception("Failed to bind port")

            with pytest.raises(Exception):
                await server.start()

    @pytest.mark.asyncio
    async def test_stop_server_error(self, server, mock_connection_manager):
        """Test server stop with error."""
        await server.start()

        # Make close_all_connections raise an error
        mock_connection_manager.close_all_connections.side_effect = Exception("Close error")

        with pytest.raises(Exception):
            await server.stop()

    @pytest.mark.asyncio
    async def test_route_message_handler_error(self, server):
        """Test routing message when handler raises error."""
        mock_ws = AsyncMock()
        message = {
            "type": "register",
            "message_id": "test-msg-16",
            "instance_id": "test-instance",
            "model_id": "test-model",
        }

        # Make the handler raise an error
        original_handler = server._handlers["register"]
        async def failing_handler(ws, msg):
            raise Exception("Handler failed")

        server._handlers["register"] = failing_handler

        try:
            await server._route_message(mock_ws, message)

            # Should send error
            sent_data = json.loads(mock_ws.send.call_args[0][0])
            assert sent_data["type"] == "error"
            assert "Handler" in sent_data["error"]
        finally:
            # Restore original handler
            server._handlers["register"] = original_handler

    @pytest.mark.asyncio
    async def test_task_ack_handler_error(self, server, mock_connection_manager):
        """Test TASK_ACK handling with error."""
        mock_ws = AsyncMock()
        message = {
            "type": "task_ack",
            "message_id": "test-msg-17",
            "reply_to": "original-msg-id",
            "task_id": "task-001",
            "success": True,
        }

        # Make handle_ack raise an error
        mock_connection_manager.handle_ack.side_effect = Exception("ACK error")

        # Should not raise exception (error is logged)
        await server._handle_task_ack(mock_ws, message)

    @pytest.mark.asyncio
    async def test_send_message_error(self, server):
        """Test _send_message with send error."""
        mock_ws = AsyncMock()
        mock_ws.send.side_effect = Exception("Send failed")

        message = {"type": "test"}

        with pytest.raises(Exception):
            await server._send_message(mock_ws, message)

    @pytest.mark.asyncio
    async def test_handle_disconnect(self, server, mock_connection_manager):
        """Test _handle_disconnect method."""
        mock_ws = AsyncMock()

        await server._handle_disconnect(mock_ws)

        # Verify connection manager was called
        mock_connection_manager.handle_disconnect.assert_called_once_with(mock_ws)

    @pytest.mark.asyncio
    async def test_send_register_ack_with_error_code(self, server):
        """Test sending register ACK with error code."""
        mock_ws = AsyncMock()

        await server._send_register_ack(
            mock_ws,
            reply_to="msg-123",
            success=False,
            message="Registration failed",
            error_code="REG_ERROR",
        )

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "register_ack"
        assert sent_data["success"] is False
        assert sent_data["error_code"] == "REG_ERROR"

    @pytest.mark.asyncio
    async def test_send_result_ack(self, server):
        """Test sending result ACK."""
        mock_ws = AsyncMock()

        await server._send_result_ack(
            mock_ws,
            reply_to="msg-123",
            task_id="task-001",
            success=True,
        )

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "result_ack"
        assert sent_data["task_id"] == "task-001"
        assert sent_data["success"] is True

    @pytest.mark.asyncio
    async def test_send_pong(self, server):
        """Test sending pong message."""
        mock_ws = AsyncMock()

        await server._send_pong(mock_ws, reply_to="ping-123")

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "pong"
        assert sent_data["reply_to"] == "ping-123"

    @pytest.mark.asyncio
    async def test_send_unregister_ack(self, server):
        """Test sending unregister ACK."""
        mock_ws = AsyncMock()

        await server._send_unregister_ack(
            mock_ws,
            reply_to="msg-123",
            success=True,
            message="Unregistered successfully",
        )

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "unregister_ack"
        assert sent_data["success"] is True



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

    @pytest.mark.asyncio
    async def test_start_manager(self, manager):
        """Test starting the connection manager."""
        await manager.start()
        assert manager._heartbeat_task is not None
        assert not manager._heartbeat_task.done()

        # Clean up
        manager._heartbeat_task.cancel()
        try:
            await manager._heartbeat_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_stop_manager(self, manager):
        """Test stopping the connection manager."""
        mock_ws = AsyncMock()
        await manager.register_connection("test-instance", mock_ws)
        await manager.start()

        # Stop with grace period
        await manager.stop(grace_period=1)

        # Verify connections closed
        assert len(manager.connections) == 0

    @pytest.mark.asyncio
    async def test_reconnection_closes_old_connection(self, manager):
        """Test that reconnecting closes old connection."""
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()

        # First connection
        await manager.register_connection("test-instance", mock_ws1)

        # Second connection (should close first)
        await manager.register_connection("test-instance", mock_ws2)

        # Verify old connection was closed
        mock_ws1.close.assert_called_once()

        # Verify new connection is registered
        assert manager.connections["test-instance"].websocket == mock_ws2

    @pytest.mark.asyncio
    async def test_disconnect_unknown_websocket(self, manager):
        """Test disconnect with unknown websocket."""
        mock_ws = AsyncMock()

        # Should not raise exception
        await manager.handle_disconnect(mock_ws)

    @pytest.mark.asyncio
    async def test_send_message_without_ack(self, manager):
        """Test sending message without ACK requirement."""
        mock_ws = AsyncMock()
        await manager.register_connection("test-instance", mock_ws)

        message = {"type": "test", "data": "value"}

        result = await manager.send_message(
            "test-instance", message, require_ack=False
        )

        assert result is None
        mock_ws.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_error(self, manager):
        """Test send message with error."""
        mock_ws = AsyncMock()
        mock_ws.send.side_effect = Exception("Send failed")

        await manager.register_connection("test-instance", mock_ws)

        message = {"type": "test"}

        with pytest.raises(Exception):
            await manager.send_message(
                "test-instance", message, require_ack=False
            )

        # Verify error count incremented
        assert manager.connections["test-instance"].error_count == 1

    @pytest.mark.asyncio
    async def test_send_message_disconnected_state(self, manager):
        """Test sending to disconnected instance."""
        mock_ws = AsyncMock()
        await manager.register_connection("test-instance", mock_ws)

        # Set to disconnected state
        manager.connections["test-instance"].state = ConnectionState.DISCONNECTED

        with pytest.raises(ConnectionError):
            await manager.send_message("test-instance", {"type": "test"})

    @pytest.mark.asyncio
    async def test_handle_task_result_no_dispatcher(self, manager):
        """Test handling task result without dispatcher."""
        # Set dispatcher to None
        manager.task_dispatcher = None

        # Should not raise exception
        await manager.handle_task_result(
            task_id="task-1",
            status="completed",
            result={"output": "test"},
        )

    @pytest.mark.asyncio
    async def test_handle_task_result_with_error(self, manager, mock_task_dispatcher):
        """Test handling task result when dispatcher raises error."""
        mock_task_dispatcher.handle_task_result.side_effect = Exception("Dispatcher error")

        # Should not raise exception (error is logged)
        await manager.handle_task_result(
            task_id="task-1",
            status="failed",
            error="Task failed",
        )

    @pytest.mark.asyncio
    async def test_send_task_to_instance_success(self, manager):
        """Test sending task to instance successfully."""
        mock_ws = AsyncMock()
        await manager.register_connection("test-instance", mock_ws)

        # Mock successful ACK
        async def mock_send_message(instance_id, message, require_ack=True, timeout=None):
            return {"success": True}

        manager.send_message = mock_send_message

        result = await manager.send_task_to_instance(
            instance_id="test-instance",
            task_id="task-1",
            model_id="model-1",
            task_input={"prompt": "test"},
            metadata={"key": "value"},
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_send_task_to_instance_rejected(self, manager):
        """Test task rejected by instance."""
        mock_ws = AsyncMock()
        await manager.register_connection("test-instance", mock_ws)

        # Mock rejection ACK
        async def mock_send_message(instance_id, message, require_ack=True, timeout=None):
            return {"success": False, "error": "Queue full"}

        manager.send_message = mock_send_message

        result = await manager.send_task_to_instance(
            instance_id="test-instance",
            task_id="task-1",
            model_id="model-1",
            task_input={"prompt": "test"},
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_send_task_to_instance_error(self, manager):
        """Test send task with connection error."""
        # Don't register connection

        with pytest.raises(ConnectionError):
            await manager.send_task_to_instance(
                instance_id="non-existent",
                task_id="task-1",
                model_id="model-1",
                task_input={"prompt": "test"},
            )

    @pytest.mark.asyncio
    async def test_send_ping(self, manager):
        """Test sending ping."""
        mock_ws = AsyncMock()
        await manager.register_connection("test-instance", mock_ws)

        result = await manager.send_ping("test-instance")

        assert result is True
        mock_ws.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_ping_error(self, manager):
        """Test send ping with error."""
        # Don't register connection

        result = await manager.send_ping("non-existent")

        assert result is False

    @pytest.mark.asyncio
    async def test_broadcast_shutdown(self, manager):
        """Test broadcasting shutdown notification."""
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()

        await manager.register_connection("instance-1", mock_ws1)
        await manager.register_connection("instance-2", mock_ws2)

        await manager.broadcast_shutdown(grace_period=5)

        # Verify both instances received shutdown notification
        assert mock_ws1.send.call_count >= 1
        assert mock_ws2.send.call_count >= 1

    @pytest.mark.asyncio
    async def test_broadcast_shutdown_no_connections(self, manager):
        """Test broadcasting shutdown with no connections."""
        # Should not raise exception
        await manager.broadcast_shutdown(grace_period=5)

    @pytest.mark.asyncio
    async def test_close_all_connections(self, manager):
        """Test closing all connections."""
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()

        await manager.register_connection("instance-1", mock_ws1)
        await manager.register_connection("instance-2", mock_ws2)

        await manager.close_all_connections()

        # Verify connections closed
        mock_ws1.close.assert_called_once()
        mock_ws2.close.assert_called_once()

        # Verify mappings cleared
        assert len(manager.connections) == 0
        assert len(manager._websocket_to_instance) == 0

    @pytest.mark.asyncio
    async def test_heartbeat_monitor_timeout(self, manager):
        """Test heartbeat monitor detecting timeout."""
        mock_ws = AsyncMock()
        await manager.register_connection("test-instance", mock_ws)

        # Set last heartbeat to old time
        manager.connections["test-instance"].last_heartbeat = time.time() - 1000

        # Start heartbeat monitor
        await manager.start()

        # Wait for one heartbeat cycle
        await asyncio.sleep(1.5)

        # Connection should be disconnected
        assert "test-instance" not in manager.connections

        # Clean up
        if manager._heartbeat_task:
            manager._heartbeat_task.cancel()
            try:
                await manager._heartbeat_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_get_connection_count(self, manager):
        """Test getting connection count."""
        assert await manager.get_connection_count() == 0

        mock_ws = AsyncMock()
        await manager.register_connection("test-instance", mock_ws)

        assert await manager.get_connection_count() == 1

    @pytest.mark.asyncio
    async def test_get_connection_info_not_found(self, manager):
        """Test getting info for non-existent connection."""
        info = await manager.get_connection_info("non-existent")

        assert info is None

    @pytest.mark.asyncio
    async def test_get_all_connections_info(self, manager):
        """Test getting info for all connections."""
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()

        await manager.register_connection("instance-1", mock_ws1)
        await manager.register_connection("instance-2", mock_ws2)

        all_info = await manager.get_all_connections_info()

        assert len(all_info) == 2
        assert "instance-1" in all_info
        assert "instance-2" in all_info

    @pytest.mark.asyncio
    async def test_handle_ack_unknown_message(self, manager):
        """Test handling ACK for unknown message."""
        # Should not raise exception
        await manager.handle_ack("unknown-message-id", {"success": True})

    @pytest.mark.asyncio
    async def test_disconnect_with_pending_acks(self, manager):
        """Test disconnection cancels pending ACKs."""
        mock_ws = AsyncMock()
        await manager.register_connection("test-instance", mock_ws)

        message = {"type": "test"}

        # Start send that will wait for ACK
        send_task = asyncio.create_task(
            manager.send_message("test-instance", message, require_ack=True, timeout=5.0)
        )

        # Give it time to set up
        await asyncio.sleep(0.1)

        # Disconnect
        await manager.handle_disconnect(mock_ws)

        # Send should raise ConnectionError
        with pytest.raises(ConnectionError):
            await send_task


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
