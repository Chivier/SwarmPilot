"""
WebSocket server for Instance connections.

This module implements a WebSocket server specifically for Instance-to-Scheduler
communication, separate from the client WebSocket API.

The server handles:
- Instance registration and unregistration
- Task submission to instances
- Task result callbacks from instances
- Connection health monitoring (heartbeat/ping-pong)
"""

import asyncio
import json
import uuid
from datetime import datetime, UTC
from typing import Dict, Optional, Callable, Any
import websockets
from websockets.asyncio.server import ServerConnection
from loguru import logger

from .model import InstanceStatus


class InstanceWebSocketServer:
    """
    WebSocket server for managing Instance connections.

    This server runs on a separate port from the main HTTP API and handles
    bidirectional communication with Instance services.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8001,
        instance_registry=None,
        instance_connection_manager=None,
    ):
        """
        Initialize the WebSocket server.

        Args:
            host: Server host address
            port: Server port number
            instance_registry: Instance registry for managing instance metadata
            instance_connection_manager: Connection manager for WebSocket lifecycle
        """
        self.host = host
        self.port = port
        self.instance_registry = instance_registry
        self.connection_manager = instance_connection_manager

        # WebSocket server instance
        self.server: Optional[websockets.WebSocketServer] = None
        self._server_task: Optional[asyncio.Task] = None

        # Message handlers
        self._handlers: Dict[str, Callable] = {
            "register": self._handle_register,
            "task_result": self._handle_task_result,
            "ping": self._handle_ping,
            "pong": self._handle_pong,
            "unregister": self._handle_unregister,
            "task_ack": self._handle_task_ack,
        }

        logger.info(f"InstanceWebSocketServer initialized on {host}:{port}")

    async def start(self) -> None:
        """
        Start the WebSocket server.

        This method starts the server in the background and returns immediately.
        """
        if self.server is not None:
            logger.warning("WebSocket server is already running")
            return

        logger.info(f"Starting Instance WebSocket server on ws://{self.host}:{self.port}/instance/ws")

        try:
            # Start WebSocket server
            self.server = await websockets.serve(
                self._handle_connection,
                self.host,
                self.port,
                max_size=16 * 1024 * 1024,  # 16MB max message size
                ping_interval=None,  # We handle pings manually
                ping_timeout=None,
                close_timeout=10,
            )

            logger.success(f"Instance WebSocket server started on ws://{self.host}:{self.port}/instance/ws")

        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise

    async def stop(self) -> None:
        """
        Stop the WebSocket server gracefully.

        This method closes all active connections and shuts down the server.
        """
        if self.server is None:
            logger.warning("WebSocket server is not running")
            return

        logger.info("Stopping Instance WebSocket server...")

        try:
            # Close all connections via connection manager
            if self.connection_manager:
                await self.connection_manager.close_all_connections()

            # Close the server
            self.server.close()
            await self.server.wait_closed()
            self.server = None

            logger.success("Instance WebSocket server stopped")

        except Exception as e:
            logger.error(f"Error stopping WebSocket server: {e}")
            raise

    async def _handle_connection(self, websocket: ServerConnection, path: str) -> None:
        """
        Handle a new WebSocket connection.

        Args:
            websocket: WebSocket connection
            path: Request path (should be /instance/ws)
        """
        remote_address = websocket.remote_address
        logger.info(f"New WebSocket connection from {remote_address}")

        try:
            # Process messages from this connection
            async for raw_message in websocket:
                try:
                    # Parse message
                    if isinstance(raw_message, bytes):
                        raw_message = raw_message.decode("utf-8")

                    message = json.loads(raw_message)

                    # Validate basic message structure
                    if not isinstance(message, dict):
                        await self._send_error(
                            websocket,
                            error="Invalid message format: expected JSON object",
                            error_code="INVALID_MESSAGE",
                        )
                        continue

                    if "type" not in message:
                        await self._send_error(
                            websocket,
                            error="Invalid message format: missing 'type' field",
                            error_code="INVALID_MESSAGE",
                        )
                        continue

                    # Route message to appropriate handler
                    await self._route_message(websocket, message)

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message: {e}")
                    await self._send_error(
                        websocket,
                        error=f"Invalid JSON format: {str(e)}",
                        error_code="INVALID_MESSAGE",
                    )

                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    await self._send_error(
                        websocket,
                        error=f"Internal error processing message: {str(e)}",
                        error_code="INTERNAL_ERROR",
                    )

        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"Connection closed normally from {remote_address}")

        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"Connection closed with error from {remote_address}: {e}")

        except Exception as e:
            logger.error(f"Unexpected error in connection handler: {e}", exc_info=True)

        finally:
            # Clean up connection
            await self._handle_disconnect(websocket)

    async def _route_message(self, websocket: ServerConnection, message: Dict[str, Any]) -> None:
        """
        Route message to appropriate handler based on type.

        Args:
            websocket: WebSocket connection
            message: Parsed message dictionary
        """
        msg_type = message.get("type")
        message_id = message.get("message_id", "unknown")

        logger.debug(f"Routing message type={msg_type} id={message_id}")

        # Get handler for message type
        handler = self._handlers.get(msg_type)

        if handler is None:
            logger.warning(f"Unknown message type: {msg_type}")
            await self._send_error(
                websocket,
                error=f"Unknown message type: {msg_type}",
                error_code="UNKNOWN_MESSAGE_TYPE",
                reply_to=message_id,
            )
            return

        try:
            # Call handler
            await handler(websocket, message)

        except Exception as e:
            logger.error(f"Handler error for {msg_type}: {e}", exc_info=True)
            await self._send_error(
                websocket,
                error=f"Error handling {msg_type}: {str(e)}",
                error_code="INTERNAL_ERROR",
                reply_to=message_id,
            )

    # ========================================================================
    # Message Handlers
    # ========================================================================

    async def _handle_register(self, websocket: ServerConnection, message: Dict[str, Any]) -> None:
        """
        Handle REGISTER message from Instance.

        Args:
            websocket: WebSocket connection
            message: REGISTER message
        """
        try:
            # Extract registration data
            instance_id = message.get("instance_id")
            model_id = message.get("model_id")
            endpoint = message.get("endpoint")
            platform_info = message.get("platform_info", {})

            # Validate required fields
            if not instance_id:
                await self._send_register_ack(
                    websocket,
                    reply_to=message.get("message_id"),
                    success=False,
                    message="Missing required field: instance_id",
                    error_code="INVALID_MESSAGE",
                )
                return

            if not model_id:
                await self._send_register_ack(
                    websocket,
                    reply_to=message.get("message_id"),
                    success=False,
                    message="Missing required field: model_id",
                    error_code="INVALID_MESSAGE",
                )
                return

            logger.info(f"Registering instance {instance_id} with model {model_id}")

            # Register with instance registry
            if self.instance_registry:
                # Check if instance already exists
                existing = await self.instance_registry.get(instance_id)
                if existing:
                    logger.warning(f"Instance {instance_id} already registered, updating...")

                # Register instance
                success = await self.instance_registry.register(
                    instance_id=instance_id,
                    model_id=model_id,
                    endpoint=endpoint or f"ws://{websocket.remote_address[0]}",
                    platform_info=platform_info,
                )

                if not success:
                    await self._send_register_ack(
                        websocket,
                        reply_to=message.get("message_id"),
                        success=False,
                        message="Failed to register instance",
                        error_code="REGISTRATION_FAILED",
                    )
                    return

            # Register connection with connection manager
            if self.connection_manager:
                await self.connection_manager.register_connection(
                    instance_id=instance_id,
                    websocket=websocket,
                )

            # Send success acknowledgment
            await self._send_register_ack(
                websocket,
                reply_to=message.get("message_id"),
                success=True,
                message="Instance registered successfully",
            )

            logger.success(f"Instance {instance_id} registered successfully")

        except Exception as e:
            logger.error(f"Error handling REGISTER: {e}", exc_info=True)
            await self._send_register_ack(
                websocket,
                reply_to=message.get("message_id"),
                success=False,
                message=f"Registration error: {str(e)}",
                error_code="INTERNAL_ERROR",
            )

    async def _handle_task_result(self, websocket: ServerConnection, message: Dict[str, Any]) -> None:
        """
        Handle TASK_RESULT message from Instance.

        This is called when an Instance sends back task results.

        Args:
            websocket: WebSocket connection
            message: TASK_RESULT message
        """
        try:
            task_id = message.get("task_id")
            status = message.get("status")
            result = message.get("result")
            error = message.get("error")
            execution_time_ms = message.get("execution_time_ms")

            if not task_id:
                await self._send_error(
                    websocket,
                    error="Missing required field: task_id",
                    error_code="INVALID_MESSAGE",
                    reply_to=message.get("message_id"),
                )
                return

            if not status:
                await self._send_error(
                    websocket,
                    error="Missing required field: status",
                    error_code="INVALID_MESSAGE",
                    reply_to=message.get("message_id"),
                )
                return

            logger.info(f"Received TASK_RESULT for task {task_id}, status={status}")

            # Forward to task dispatcher via connection manager
            if self.connection_manager:
                await self.connection_manager.handle_task_result(
                    task_id=task_id,
                    status=status,
                    result=result,
                    error=error,
                    execution_time_ms=execution_time_ms,
                )

            # Send acknowledgment
            await self._send_result_ack(
                websocket,
                reply_to=message.get("message_id"),
                task_id=task_id,
                success=True,
            )

            logger.debug(f"Sent RESULT_ACK for task {task_id}")

        except Exception as e:
            logger.error(f"Error handling TASK_RESULT: {e}", exc_info=True)
            await self._send_result_ack(
                websocket,
                reply_to=message.get("message_id"),
                task_id=message.get("task_id", "unknown"),
                success=False,
            )

    async def _handle_task_ack(self, websocket: ServerConnection, message: Dict[str, Any]) -> None:
        """
        Handle TASK_ACK message from Instance.

        This acknowledges that the Instance received a task submission.

        Args:
            websocket: WebSocket connection
            message: TASK_ACK message
        """
        try:
            task_id = message.get("task_id")
            success = message.get("success", False)
            reply_to = message.get("reply_to")

            logger.debug(f"Received TASK_ACK for task {task_id}, success={success}")

            # Notify connection manager about ACK
            if self.connection_manager and reply_to:
                await self.connection_manager.handle_ack(
                    message_id=reply_to,
                    ack_data=message,
                )

        except Exception as e:
            logger.error(f"Error handling TASK_ACK: {e}", exc_info=True)

    async def _handle_ping(self, websocket: ServerConnection, message: Dict[str, Any]) -> None:
        """
        Handle PING message and respond with PONG.

        Args:
            websocket: WebSocket connection
            message: PING message
        """
        # Send PONG response
        await self._send_pong(websocket, reply_to=message.get("message_id"))

        # Update last heartbeat time in connection manager
        if self.connection_manager:
            # Get instance_id from connection manager
            instance_id = await self.connection_manager.get_instance_id_by_websocket(websocket)
            if instance_id:
                await self.connection_manager.update_heartbeat(instance_id)

    async def _handle_pong(self, websocket: ServerConnection, message: Dict[str, Any]) -> None:
        """
        Handle PONG message (response to our PING).

        Args:
            websocket: WebSocket connection
            message: PONG message
        """
        # Update last heartbeat time in connection manager
        if self.connection_manager:
            instance_id = await self.connection_manager.get_instance_id_by_websocket(websocket)
            if instance_id:
                await self.connection_manager.update_heartbeat(instance_id)
                logger.debug(f"Received PONG from instance {instance_id}")

    async def _handle_unregister(self, websocket: ServerConnection, message: Dict[str, Any]) -> None:
        """
        Handle UNREGISTER message from Instance.

        Args:
            websocket: WebSocket connection
            message: UNREGISTER message
        """
        try:
            instance_id = message.get("instance_id")
            reason = message.get("reason", "No reason provided")

            logger.info(f"Instance {instance_id} requesting unregistration: {reason}")

            # Send acknowledgment first
            await self._send_unregister_ack(
                websocket,
                reply_to=message.get("message_id"),
                success=True,
                message="Unregistration acknowledged",
            )

            # Handle disconnection (will clean up resources)
            await self._handle_disconnect(websocket)

            logger.info(f"Instance {instance_id} unregistered successfully")

        except Exception as e:
            logger.error(f"Error handling UNREGISTER: {e}", exc_info=True)
            await self._send_unregister_ack(
                websocket,
                reply_to=message.get("message_id"),
                success=False,
                message=f"Unregistration error: {str(e)}",
            )

    async def _handle_disconnect(self, websocket: ServerConnection) -> None:
        """
        Handle WebSocket disconnection.

        Args:
            websocket: WebSocket connection that disconnected
        """
        if self.connection_manager:
            await self.connection_manager.handle_disconnect(websocket)

    # ========================================================================
    # Message Sending Helpers
    # ========================================================================

    async def _send_message(self, websocket: ServerConnection, message: Dict[str, Any]) -> None:
        """
        Send a message to a WebSocket connection.

        Args:
            websocket: WebSocket connection
            message: Message dictionary to send
        """
        try:
            # Add timestamp if not present
            if "timestamp" not in message:
                message["timestamp"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")

            # Send as JSON
            await websocket.send(json.dumps(message))

        except Exception as e:
            logger.error(f"Error sending message: {e}", exc_info=True)
            raise

    async def _send_register_ack(
        self,
        websocket: ServerConnection,
        reply_to: str,
        success: bool,
        message: str,
        error_code: Optional[str] = None,
    ) -> None:
        """Send REGISTER_ACK message."""
        ack = {
            "type": "register_ack",
            "message_id": str(uuid.uuid4()),
            "reply_to": reply_to,
            "success": success,
            "message": message,
        }
        if error_code:
            ack["error_code"] = error_code

        await self._send_message(websocket, ack)

    async def _send_result_ack(
        self,
        websocket: ServerConnection,
        reply_to: str,
        task_id: str,
        success: bool,
    ) -> None:
        """Send RESULT_ACK message."""
        ack = {
            "type": "result_ack",
            "message_id": str(uuid.uuid4()),
            "reply_to": reply_to,
            "task_id": task_id,
            "success": success,
        }
        await self._send_message(websocket, ack)

    async def _send_pong(self, websocket: ServerConnection, reply_to: str) -> None:
        """Send PONG message."""
        pong = {
            "type": "pong",
            "message_id": str(uuid.uuid4()),
            "reply_to": reply_to,
        }
        await self._send_message(websocket, pong)

    async def _send_unregister_ack(
        self,
        websocket: ServerConnection,
        reply_to: str,
        success: bool,
        message: str,
    ) -> None:
        """Send UNREGISTER_ACK message."""
        ack = {
            "type": "unregister_ack",
            "message_id": str(uuid.uuid4()),
            "reply_to": reply_to,
            "success": success,
            "message": message,
        }
        await self._send_message(websocket, ack)

    async def _send_error(
        self,
        websocket: ServerConnection,
        error: str,
        error_code: str,
        reply_to: Optional[str] = None,
    ) -> None:
        """Send ERROR message."""
        error_msg = {
            "type": "error",
            "message_id": str(uuid.uuid4()),
            "error": error,
            "error_code": error_code,
        }
        if reply_to:
            error_msg["reply_to"] = reply_to

        await self._send_message(websocket, error_msg)
