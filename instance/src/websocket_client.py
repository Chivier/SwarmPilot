"""
WebSocket client for Instance-to-Scheduler communication.

This module provides WebSocket client functionality for Instance services
to communicate with the Scheduler.
"""

import asyncio
import json
import uuid
from datetime import datetime, UTC
from typing import Dict, Optional, Callable, Any
import websockets
from websockets import ClientConnection
from loguru import logger


class WebSocketClient:
    """
    WebSocket client for connecting Instance to Scheduler.

    Handles connection lifecycle, automatic reconnection, and message exchange.
    """

    def __init__(
        self,
        scheduler_url: str,
        instance_id: str,
        model_id: str,
        platform_info: Optional[Dict[str, Any]] = None,
        reconnect_delay_max: int = 32,
        heartbeat_interval: int = 30,
    ):
        """
        Initialize WebSocket client.

        Args:
            scheduler_url: Scheduler WebSocket URL (ws://host:port/instance/ws)
            instance_id: Unique instance identifier
            model_id: Model being served
            platform_info: Platform information dictionary
            reconnect_delay_max: Maximum reconnection delay in seconds
            heartbeat_interval: Heartbeat interval in seconds
        """
        self.scheduler_url = scheduler_url
        self.instance_id = instance_id
        self.model_id = model_id
        self.platform_info = platform_info or {}

        # Connection state
        self.websocket: Optional[ClientConnection] = None
        self.running = False
        self.connected = False

        # Reconnection configuration
        self.reconnect_delay = 1
        self.reconnect_delay_max = reconnect_delay_max

        # Heartbeat configuration
        self.heartbeat_interval = heartbeat_interval
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Message handlers
        self.message_handlers: Dict[str, Callable] = {}
        self._pending_acks: Dict[str, asyncio.Future] = {}

        # Background tasks
        self._connection_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None

        logger.info(
            f"WebSocketClient initialized for instance {instance_id}, "
            f"scheduler: {scheduler_url}"
        )

    async def start(self) -> None:
        """Start the WebSocket client and begin connection loop."""
        if self.running:
            logger.warning("WebSocket client already running")
            return

        self.running = True
        self._connection_task = asyncio.create_task(self._connection_loop())
        logger.info("WebSocket client started")

    async def stop(self) -> None:
        """Stop the WebSocket client and close connection."""
        if not self.running:
            return

        logger.info("Stopping WebSocket client...")
        self.running = False

        # Stop heartbeat
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()

        # Stop receive task
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()

        # Send unregister message if connected
        if self.connected and self.websocket:
            try:
                await self.send_unregister()
                await asyncio.sleep(0.5)  # Give time for message to send
            except:
                pass

        # Close WebSocket
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass

        # Cancel connection task
        if self._connection_task and not self._connection_task.done():
            self._connection_task.cancel()

        logger.info("WebSocket client stopped")

    def register_handler(self, message_type: str, handler: Callable) -> None:
        """
        Register a message handler for a specific message type.

        Args:
            message_type: Message type to handle
            handler: Async callable to handle the message
        """
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for message type: {message_type}")

    async def send_message(
        self,
        message: Dict[str, Any],
        require_ack: bool = False,
        timeout: float = 10.0,
        skip_connection_check: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Send a message to the Scheduler.

        Args:
            message: Message dictionary
            require_ack: Whether to wait for ACK
            timeout: ACK timeout in seconds
            skip_connection_check: Skip connection state check (for registration)

        Returns:
            ACK response if require_ack=True, None otherwise

        Raises:
            ConnectionError: If not connected
            TimeoutError: If ACK timeout
        """
        if not skip_connection_check and (not self.connected or not self.websocket):
            raise ConnectionError("Not connected to Scheduler")

        if not self.websocket:
            raise ConnectionError("WebSocket not initialized")

        # Add message_id and timestamp if not present
        if "message_id" not in message:
            message["message_id"] = str(uuid.uuid4())
        if "timestamp" not in message:
            message["timestamp"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")

        message_id = message["message_id"]

        # Create future for ACK if required
        ack_future = None
        if require_ack:
            ack_future = asyncio.Future()
            self._pending_acks[message_id] = ack_future

        try:
            # Send message
            await self.websocket.send(json.dumps(message))
            logger.debug(f"Sent message type={message.get('type')} id={message_id}")

            # Wait for ACK if required
            if require_ack:
                try:
                    ack_data = await asyncio.wait_for(ack_future, timeout=timeout)
                    return ack_data
                except asyncio.TimeoutError:
                    logger.warning(f"ACK timeout for message {message_id}")
                    raise TimeoutError(f"ACK timeout for message {message_id}")
                finally:
                    self._pending_acks.pop(message_id, None)

            return None

        except Exception as e:
            if ack_future:
                self._pending_acks.pop(message_id, None)
                if not ack_future.done():
                    ack_future.set_exception(e)
            raise

    async def send_task_result(
        self,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
    ) -> bool:
        """
        Send task result to Scheduler.

        Args:
            task_id: Task identifier
            status: Task status ("completed" or "failed")
            result: Task result data
            error: Error message
            execution_time_ms: Execution time

        Returns:
            True if acknowledged, False otherwise
        """
        message = {
            "type": "task_result",
            "task_id": task_id,
            "status": status,
            "result": result,
            "error": error,
            "execution_time_ms": execution_time_ms,
        }

        try:
            ack = await self.send_message(message, require_ack=True)
            success = ack.get("success", False) if ack else False
            if success:
                logger.info(f"Task result for {task_id} acknowledged by Scheduler")
            return success
        except Exception as e:
            logger.error(f"Failed to send task result for {task_id}: {e}")
            return False

    async def send_unregister(self) -> None:
        """Send unregister message to Scheduler."""
        message = {
            "type": "unregister",
            "instance_id": self.instance_id,
            "reason": "shutdown",
        }
        try:
            await self.send_message(message, require_ack=False)
            logger.info("Sent unregister message to Scheduler")
        except Exception as e:
            logger.error(f"Failed to send unregister message: {e}")

    def is_connected(self) -> bool:
        """Check if connected to Scheduler."""
        return self.connected and self.websocket is not None

    # ========================================================================
    # Internal Methods
    # ========================================================================

    async def _connection_loop(self) -> None:
        """Main connection loop with automatic reconnection."""
        logger.info("Connection loop started")

        while self.running:
            try:
                # Attempt to connect
                logger.info(f"Connecting to Scheduler at {self.scheduler_url}...")
                self.websocket = await websockets.connect(
                    self.scheduler_url,
                    max_size=16 * 1024 * 1024,  # 16MB
                    ping_interval=None,  # We handle pings manually
                    ping_timeout=None,
                )

                logger.success("WebSocket connection established")

                # Send registration
                await self._send_register()

                # Reset reconnection delay on successful connection
                self.reconnect_delay = 1

                # Start heartbeat
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

                # Start receiving messages
                self._receive_task = asyncio.create_task(self._receive_loop())

                # Wait for receive task to complete (connection lost)
                await self._receive_task

            except websockets.exceptions.WebSocketException as e:
                logger.error(f"WebSocket error: {e}")
                self.connected = False

            except Exception as e:
                logger.error(f"Connection error: {e}", exc_info=True)
                self.connected = False

            finally:
                # Clean up
                if self._heartbeat_task and not self._heartbeat_task.done():
                    self._heartbeat_task.cancel()

                if self.websocket:
                    try:
                        await self.websocket.close()
                    except:
                        pass
                    self.websocket = None

            # Reconnect with exponential backoff
            if self.running:
                logger.info(f"Reconnecting in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(
                    self.reconnect_delay * 2,
                    self.reconnect_delay_max
                )

        logger.info("Connection loop stopped")

    async def _send_register(self) -> None:
        """Send registration message to Scheduler."""
        register_msg = {
            "type": "register",
            "instance_id": self.instance_id,
            "model_id": self.model_id,
            "endpoint": f"ws://{self.instance_id}",  # Placeholder
            "platform_info": self.platform_info,
        }

        try:
            # Skip connection check for registration message since we're establishing the connection
            await self.send_message(register_msg, require_ack=False, skip_connection_check=True)
            logger.info("Sent registration message to Scheduler")
        except Exception as e:
            logger.error(f"Failed to send registration: {e}")
            raise

    async def _receive_loop(self) -> None:
        """Receive and process messages from Scheduler."""
        try:
            async for raw_message in self.websocket:
                try:
                    # Parse message
                    if isinstance(raw_message, bytes):
                        raw_message = raw_message.decode("utf-8")

                    message = json.loads(raw_message)

                    # Handle message
                    await self._handle_message(message)

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message: {e}")

                except Exception as e:
                    logger.error(f"Error handling message: {e}", exc_info=True)

        except websockets.exceptions.ConnectionClosedOK:
            logger.info("Connection closed normally")

        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"Connection closed with error: {e}")

        except Exception as e:
            logger.error(f"Receive loop error: {e}", exc_info=True)

        finally:
            self.connected = False

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """
        Handle received message.

        Args:
            message: Parsed message dictionary
        """
        msg_type = message.get("type")
        message_id = message.get("message_id", "unknown")

        logger.debug(f"Received message type={msg_type} id={message_id}")

        # Handle ACKs for pending messages
        reply_to = message.get("reply_to")
        if reply_to and reply_to in self._pending_acks:
            future = self._pending_acks.pop(reply_to)
            if not future.done():
                future.set_result(message)
            return

        # Special handling for register_ack
        if msg_type == "register_ack":
            success = message.get("success", False)
            if success:
                self.connected = True
                logger.success("Registration acknowledged by Scheduler")
            else:
                logger.error(f"Registration failed: {message.get('message')}")
            return

        # Route to custom handlers
        handler = self.message_handlers.get(msg_type)
        if handler:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Handler error for {msg_type}: {e}", exc_info=True)
        else:
            logger.warning(f"No handler for message type: {msg_type}")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat pings to Scheduler."""
        try:
            while self.connected and self.running:
                await asyncio.sleep(self.heartbeat_interval)

                if self.connected and self.websocket:
                    try:
                        ping_msg = {
                            "type": "ping",
                        }
                        await self.send_message(ping_msg, require_ack=False)
                        logger.debug("Sent heartbeat ping to Scheduler")
                    except Exception as e:
                        logger.warning(f"Failed to send heartbeat: {e}")

        except asyncio.CancelledError:
            logger.debug("Heartbeat loop cancelled")
        except Exception as e:
            logger.error(f"Heartbeat loop error: {e}", exc_info=True)
