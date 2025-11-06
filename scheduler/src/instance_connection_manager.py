"""
Instance connection manager for WebSocket connections.

This module manages WebSocket connections to Instance services, including
connection lifecycle, heartbeat monitoring, and message delivery.
"""

import asyncio
import time
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import datetime, UTC
import json

from websockets.server import WebSocketServerProtocol
from loguru import logger

from .model import InstanceStatus


class ConnectionState(Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class InstanceConnection:
    """Represents a WebSocket connection to an Instance."""
    instance_id: str
    websocket: WebSocketServerProtocol
    state: ConnectionState
    last_heartbeat: float
    registered_at: float
    pending_acks: Dict[str, asyncio.Future] = field(default_factory=dict)
    message_count: int = 0
    error_count: int = 0


class InstanceConnectionManager:
    """
    Manages WebSocket connections to Instance services.

    Responsibilities:
    - Track active connections per instance
    - Monitor connection health via heartbeat
    - Handle message ACKs and timeouts
    - Send messages to instances
    - Clean up disconnected instances
    """

    def __init__(
        self,
        instance_registry=None,
        task_dispatcher=None,
        heartbeat_interval: int = 30,
        heartbeat_timeout_threshold: int = 3,
        ack_timeout: float = 10.0,
        max_retries: int = 3,
    ):
        """
        Initialize connection manager.

        Args:
            instance_registry: Instance registry for metadata
            task_dispatcher: Task dispatcher for result processing
            heartbeat_interval: Seconds between heartbeat pings
            heartbeat_timeout_threshold: Number of missed pings before disconnect
            ack_timeout: Seconds to wait for ACK before timeout
            max_retries: Maximum message send retries
        """
        self.instance_registry = instance_registry
        self.task_dispatcher = task_dispatcher

        # Connection tracking
        self.connections: Dict[str, InstanceConnection] = {}
        self._websocket_to_instance: Dict[WebSocketServerProtocol, str] = {}
        self._lock = asyncio.Lock()

        # Heartbeat configuration
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout_threshold = heartbeat_timeout_threshold
        self.heartbeat_timeout = heartbeat_interval * heartbeat_timeout_threshold
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Message sending configuration
        self.ack_timeout = ack_timeout
        self.max_retries = max_retries

        logger.info(
            f"InstanceConnectionManager initialized: "
            f"heartbeat={heartbeat_interval}s, timeout={self.heartbeat_timeout}s, "
            f"ack_timeout={ack_timeout}s"
        )

    async def start(self) -> None:
        """Start background tasks (heartbeat monitoring)."""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            logger.info("Heartbeat monitor started")

    async def stop(self, grace_period: int = 5) -> None:
        """Stop background tasks and close all connections gracefully.

        Args:
            grace_period: Seconds to wait for graceful shutdown
        """
        logger.info(f"Stopping connection manager (grace period: {grace_period}s)...")

        # Broadcast shutdown notification to all instances
        await self.broadcast_shutdown(grace_period)

        # Wait grace period for instances to handle shutdown
        await asyncio.sleep(min(grace_period, 5))

        # Close all connections
        await self.close_all_connections()

        # Stop heartbeat monitor
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        logger.info("Connection manager stopped")

    async def register_connection(
        self,
        instance_id: str,
        websocket: WebSocketServerProtocol,
    ) -> None:
        """
        Register a new WebSocket connection.

        Args:
            instance_id: Instance identifier
            websocket: WebSocket connection
        """
        async with self._lock:
            # Check if instance already has a connection
            if instance_id in self.connections:
                old_connection = self.connections[instance_id]
                logger.warning(
                    f"Instance {instance_id} reconnecting, closing old connection"
                )
                # Close old connection
                try:
                    await old_connection.websocket.close()
                except:
                    pass
                # Clean up old mapping
                if old_connection.websocket in self._websocket_to_instance:
                    del self._websocket_to_instance[old_connection.websocket]

            # Create new connection
            current_time = time.time()
            connection = InstanceConnection(
                instance_id=instance_id,
                websocket=websocket,
                state=ConnectionState.CONNECTED,
                last_heartbeat=current_time,
                registered_at=current_time,
            )

            self.connections[instance_id] = connection
            self._websocket_to_instance[websocket] = instance_id

            logger.info(f"Connection registered for instance {instance_id}")

    async def handle_disconnect(self, websocket: WebSocketServerProtocol) -> None:
        """
        Handle WebSocket disconnection.

        Args:
            websocket: Disconnected WebSocket
        """
        async with self._lock:
            # Find instance_id from websocket
            instance_id = self._websocket_to_instance.get(websocket)
            if not instance_id:
                logger.warning("Disconnect from unknown websocket")
                return

            logger.info(f"Handling disconnect for instance {instance_id}")

            # Get connection
            connection = self.connections.get(instance_id)
            if not connection:
                return

            # Update state
            connection.state = ConnectionState.DISCONNECTED

            # Cancel pending ACKs with exception
            for future in connection.pending_acks.values():
                if not future.done():
                    future.set_exception(
                        ConnectionError(f"Instance {instance_id} disconnected")
                    )

            # Remove from mappings
            del self.connections[instance_id]
            del self._websocket_to_instance[websocket]

            # Update instance registry status
            if self.instance_registry:
                try:
                    await self.instance_registry.update_status(
                        instance_id, InstanceStatus.DISCONNECTED
                    )
                except Exception as e:
                    logger.error(f"Failed to update instance status: {e}")

            logger.info(f"Instance {instance_id} disconnected and cleaned up")

    async def update_heartbeat(self, instance_id: str) -> None:
        """
        Update last heartbeat time for an instance.

        Args:
            instance_id: Instance identifier
        """
        connection = self.connections.get(instance_id)
        if connection:
            connection.last_heartbeat = time.time()
            logger.debug(f"Heartbeat updated for instance {instance_id}")

    async def get_instance_id_by_websocket(
        self, websocket: WebSocketServerProtocol
    ) -> Optional[str]:
        """
        Get instance ID from WebSocket.

        Args:
            websocket: WebSocket connection

        Returns:
            Instance ID or None
        """
        return self._websocket_to_instance.get(websocket)

    async def send_message(
        self,
        instance_id: str,
        message: Dict[str, Any],
        require_ack: bool = True,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send a message to an instance and optionally wait for ACK.

        Args:
            instance_id: Instance identifier
            message: Message dictionary to send
            require_ack: Whether to wait for ACK
            timeout: ACK timeout (uses default if None)

        Returns:
            ACK response if require_ack=True, None otherwise

        Raises:
            ConnectionError: If instance not connected
            TimeoutError: If ACK timeout
        """
        connection = self.connections.get(instance_id)
        if not connection:
            raise ConnectionError(f"Instance {instance_id} not connected")

        if connection.state != ConnectionState.CONNECTED:
            raise ConnectionError(
                f"Instance {instance_id} is not in CONNECTED state: {connection.state}"
            )

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
            connection.pending_acks[message_id] = ack_future

        try:
            # Send message
            await connection.websocket.send(json.dumps(message))
            connection.message_count += 1

            logger.debug(
                f"Sent message type={message.get('type')} id={message_id} to {instance_id}"
            )

            # Wait for ACK if required
            if require_ack:
                ack_timeout = timeout if timeout is not None else self.ack_timeout
                try:
                    ack_data = await asyncio.wait_for(ack_future, timeout=ack_timeout)
                    logger.debug(f"Received ACK for message {message_id}")
                    return ack_data

                except asyncio.TimeoutError:
                    logger.warning(
                        f"ACK timeout for message {message_id} to {instance_id}"
                    )
                    raise TimeoutError(
                        f"ACK timeout for message {message_id} after {ack_timeout}s"
                    )

                finally:
                    # Clean up pending ACK
                    connection.pending_acks.pop(message_id, None)

            return None

        except Exception as e:
            connection.error_count += 1
            logger.error(f"Error sending message to {instance_id}: {e}")
            # Clean up pending ACK if exists
            if ack_future:
                connection.pending_acks.pop(message_id, None)
                if not ack_future.done():
                    ack_future.set_exception(e)
            raise

    async def handle_ack(self, message_id: str, ack_data: Dict[str, Any]) -> None:
        """
        Handle ACK message for a pending request.

        Args:
            message_id: Original message ID being acknowledged
            ack_data: ACK message data
        """
        # Find connection with pending ACK for this message_id
        for connection in self.connections.values():
            future = connection.pending_acks.get(message_id)
            if future and not future.done():
                future.set_result(ack_data)
                logger.debug(f"ACK handled for message {message_id}")
                return

        logger.debug(f"Received ACK for unknown/completed message {message_id}")

    async def handle_task_result(
        self,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
    ) -> None:
        """
        Handle task result from instance and forward to task dispatcher.

        Args:
            task_id: Task identifier
            status: Task status ("completed" or "failed")
            result: Task result data
            error: Error message
            execution_time_ms: Execution time
        """
        if not self.task_dispatcher:
            logger.warning("No task dispatcher configured, dropping task result")
            return

        try:
            await self.task_dispatcher.handle_task_result(
                task_id=task_id,
                status=status,
                result=result,
                error=error,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            logger.error(f"Error handling task result for {task_id}: {e}", exc_info=True)

    async def send_task_to_instance(
        self,
        instance_id: str,
        task_id: str,
        model_id: str,
        task_input: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send TASK_SUBMIT message to an instance.

        Args:
            instance_id: Instance identifier
            task_id: Task identifier
            model_id: Model identifier
            task_input: Task input data
            metadata: Optional task metadata

        Returns:
            True if task was accepted, False otherwise

        Raises:
            ConnectionError: If instance not connected
            TimeoutError: If ACK timeout
        """
        message = {
            "type": "task_submit",
            "task_id": task_id,
            "model_id": model_id,
            "task_input": task_input,
        }

        if metadata:
            message["metadata"] = metadata

        logger.info(f"Sending TASK_SUBMIT for task {task_id} to instance {instance_id}")

        try:
            # Send with ACK required
            ack_data = await self.send_message(
                instance_id=instance_id,
                message=message,
                require_ack=True,
            )

            # Check if task was accepted
            success = ack_data.get("success", False) if ack_data else False

            if not success:
                error = ack_data.get("error", "Unknown error") if ack_data else "No ACK received"
                logger.warning(f"Task {task_id} rejected by instance {instance_id}: {error}")

            return success

        except Exception as e:
            logger.error(f"Failed to send task {task_id} to instance {instance_id}: {e}")
            raise

    async def send_ping(self, instance_id: str) -> bool:
        """
        Send PING to an instance.

        Args:
            instance_id: Instance identifier

        Returns:
            True if ping sent successfully, False otherwise
        """
        try:
            await self.send_message(
                instance_id=instance_id,
                message={"type": "ping"},
                require_ack=False,  # PONG is handled separately
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to send ping to instance {instance_id}: {e}")
            return False

    async def broadcast_shutdown(self, grace_period: int = 5) -> None:
        """Broadcast shutdown notification to all connected Instances.

        Args:
            grace_period: Seconds before actual shutdown

        This notifies all Instances that Scheduler is shutting down,
        giving them time to complete ongoing tasks and clean up.
        """
        async with self._lock:
            if not self.connections:
                logger.debug("No active connections to notify of shutdown")
                return

            logger.info(f"Broadcasting shutdown notification to {len(self.connections)} instances...")

            shutdown_message = {
                "type": "scheduler_shutdown",
                "message": "Scheduler is shutting down",
                "grace_period": grace_period,
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }

            # Send to all connections (fire-and-forget, no ACK required)
            tasks = []
            for instance_id, connection in self.connections.items():
                try:
                    task = connection.websocket.send(json.dumps(shutdown_message))
                    tasks.append(task)
                    logger.debug(f"Sent shutdown notification to instance {instance_id}")
                except Exception as e:
                    logger.warning(f"Failed to send shutdown notification to {instance_id}: {e}")

            # Wait for all sends to complete (with timeout)
            if tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Timeout sending shutdown notifications")

            logger.info("Shutdown notification broadcast complete")

    async def close_all_connections(self) -> None:
        """Close all active WebSocket connections."""
        async with self._lock:
            logger.info(f"Closing {len(self.connections)} WebSocket connections...")

            for instance_id, connection in list(self.connections.items()):
                try:
                    # Send UNREGISTER notification if possible
                    try:
                        await connection.websocket.send(
                            json.dumps({
                                "type": "scheduler_shutdown",
                                "message": "Scheduler is shutting down",
                                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                            })
                        )
                    except:
                        pass

                    # Close connection
                    await connection.websocket.close()
                    logger.debug(f"Closed connection to instance {instance_id}")

                except Exception as e:
                    logger.error(f"Error closing connection to {instance_id}: {e}")

            # Clear all mappings
            self.connections.clear()
            self._websocket_to_instance.clear()

            logger.info("All connections closed")

    # ========================================================================
    # Background Tasks
    # ========================================================================

    async def _heartbeat_monitor(self) -> None:
        """
        Monitor connection health and send periodic pings.

        This task runs in the background and:
        1. Sends PING to all instances every heartbeat_interval seconds
        2. Checks for timed-out connections (no PONG received)
        3. Disconnects timed-out instances
        """
        logger.info("Heartbeat monitor task started")

        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval)

                current_time = time.time()
                disconnected_instances = []

                # Check all connections
                for instance_id, connection in list(self.connections.items()):
                    try:
                        # Check if connection timed out
                        time_since_heartbeat = current_time - connection.last_heartbeat

                        if time_since_heartbeat > self.heartbeat_timeout:
                            logger.warning(
                                f"Instance {instance_id} heartbeat timeout "
                                f"({time_since_heartbeat:.1f}s > {self.heartbeat_timeout}s)"
                            )
                            disconnected_instances.append(instance_id)
                            continue

                        # Send ping
                        success = await self.send_ping(instance_id)
                        if not success:
                            logger.warning(f"Failed to ping instance {instance_id}")

                    except Exception as e:
                        logger.error(
                            f"Error in heartbeat check for {instance_id}: {e}",
                            exc_info=True,
                        )

                # Handle timed-out connections
                for instance_id in disconnected_instances:
                    connection = self.connections.get(instance_id)
                    if connection:
                        logger.info(f"Disconnecting timed-out instance {instance_id}")
                        try:
                            await self.handle_disconnect(connection.websocket)
                        except Exception as e:
                            logger.error(
                                f"Error disconnecting {instance_id}: {e}",
                                exc_info=True,
                            )

        except asyncio.CancelledError:
            logger.info("Heartbeat monitor task cancelled")
            raise

        except Exception as e:
            logger.error(f"Heartbeat monitor task crashed: {e}", exc_info=True)
            raise

    # ========================================================================
    # Status and Diagnostics
    # ========================================================================

    async def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.connections)

    async def get_connection_info(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """
        Get connection information for an instance.

        Args:
            instance_id: Instance identifier

        Returns:
            Connection info dictionary or None
        """
        connection = self.connections.get(instance_id)
        if not connection:
            return None

        current_time = time.time()
        return {
            "instance_id": instance_id,
            "state": connection.state.value,
            "connected_duration": current_time - connection.registered_at,
            "last_heartbeat": current_time - connection.last_heartbeat,
            "message_count": connection.message_count,
            "error_count": connection.error_count,
            "pending_acks": len(connection.pending_acks),
        }

    async def get_all_connections_info(self) -> Dict[str, Dict[str, Any]]:
        """Get connection info for all instances."""
        return {
            instance_id: await self.get_connection_info(instance_id)
            for instance_id in self.connections.keys()
        }
