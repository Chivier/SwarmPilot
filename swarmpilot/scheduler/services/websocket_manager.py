"""WebSocket connection manager for real-time task result delivery.

This module manages WebSocket connections and task result subscriptions.
"""

import asyncio

from fastapi import WebSocket

from swarmpilot.scheduler.models import (
    TaskStatus,
    TaskTimestamps,
    WSTaskResultMessage,
)


class ConnectionManager:
    """Manager for WebSocket connections and task subscriptions."""

    def __init__(self):
        # Map of task_id -> set of websockets subscribed to that task
        self._subscriptions: dict[str, set[WebSocket]] = {}
        # Map of websocket -> set of task_ids it's subscribed to
        self._connections: dict[WebSocket, set[str]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Register a new WebSocket connection.

        Args:
            websocket: WebSocket connection to register
        """
        async with self._lock:
            self._connections[websocket] = set()

    async def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a WebSocket connection and clean up subscriptions.

        Args:
            websocket: WebSocket connection to unregister
        """
        async with self._lock:
            # Get all task IDs this connection was subscribed to
            task_ids = self._connections.get(websocket, set())

            # Remove this websocket from all task subscriptions
            for task_id in task_ids:
                if task_id in self._subscriptions:
                    self._subscriptions[task_id].discard(websocket)
                    # Clean up empty subscription sets
                    if not self._subscriptions[task_id]:
                        del self._subscriptions[task_id]

            # Remove connection record
            if websocket in self._connections:
                del self._connections[websocket]

    async def subscribe(self, websocket: WebSocket, task_ids: list[str]) -> None:
        """Subscribe a WebSocket connection to task result updates.

        Args:
            websocket: WebSocket connection
            task_ids: List of task IDs to subscribe to
        """
        async with self._lock:
            # Ensure connection is registered
            if websocket not in self._connections:
                self._connections[websocket] = set()

            # Add subscriptions
            for task_id in task_ids:
                # Add to task -> websocket mapping
                if task_id not in self._subscriptions:
                    self._subscriptions[task_id] = set()
                self._subscriptions[task_id].add(websocket)

                # Add to websocket -> task mapping
                self._connections[websocket].add(task_id)

    async def unsubscribe(self, websocket: WebSocket, task_ids: list[str]) -> None:
        """Unsubscribe a WebSocket connection from task result updates.

        Args:
            websocket: WebSocket connection
            task_ids: List of task IDs to unsubscribe from
        """
        async with self._lock:
            for task_id in task_ids:
                # Remove from task -> websocket mapping
                if task_id in self._subscriptions:
                    self._subscriptions[task_id].discard(websocket)
                    if not self._subscriptions[task_id]:
                        del self._subscriptions[task_id]

                # Remove from websocket -> task mapping
                if websocket in self._connections:
                    self._connections[websocket].discard(task_id)

    async def get_subscribed_tasks(self, websocket: WebSocket) -> list[str]:
        """Get list of task IDs a WebSocket is subscribed to.

        Args:
            websocket: WebSocket connection

        Returns:
            List of subscribed task IDs
        """
        async with self._lock:
            return list(self._connections.get(websocket, set()))

    async def get_subscribers(self, task_id: str) -> list[WebSocket]:
        """Get list of WebSockets subscribed to a task.

        Args:
            task_id: Task ID

        Returns:
            List of WebSocket connections
        """
        async with self._lock:
            return list(self._subscriptions.get(task_id, set()))

    async def broadcast_task_result(
        self,
        task_id: str,
        status: TaskStatus,
        result: dict | None = None,
        error: str | None = None,
        timestamps: TaskTimestamps = None,
        execution_time_ms: int | None = None,
    ) -> None:
        """Broadcast task result to all subscribed WebSocket connections.

        Args:
            task_id: Task ID
            status: Task status (completed or failed)
            result: Task result data (if completed)
            error: Error message (if failed)
            timestamps: Task timestamps
            execution_time_ms: Execution time in milliseconds
        """
        # Get subscribers (make a copy to avoid holding lock during I/O)
        subscribers = await self.get_subscribers(task_id)

        if not subscribers:
            # FIX: Add logging for no subscribers case (may indicate issue)
            from loguru import logger

            logger.warning(
                f"No subscribers for task {task_id} when broadcasting result"
            )
            return

        # FIX: Add logging for broadcast start
        from loguru import logger

        logger.info(
            f"Broadcasting result for task {task_id} to {len(subscribers)} subscriber(s)"
        )

        # Create result message
        message = WSTaskResultMessage(
            task_id=task_id,
            status=status,
            result=result,
            error=error,
            timestamps=timestamps,
            execution_time_ms=execution_time_ms,
        )

        # FIX: Track send statistics
        sent_count = 0
        failed_count = 0
        disconnected = []

        # Send to all subscribers
        for websocket in subscribers:
            try:
                await websocket.send_json(message.model_dump())
                sent_count += 1
                logger.debug(
                    f"Successfully sent {task_id} result to websocket {id(websocket)}"
                )
            except Exception as e:
                # FIX: Log the specific error instead of silent catch
                failed_count += 1
                logger.error(
                    f"Failed to send {task_id} result to websocket {id(websocket)}: {type(e).__name__}: {e}"
                )
                # Mark for cleanup if sending fails
                disconnected.append(websocket)

        # FIX: Log broadcast summary
        logger.info(
            f"Broadcast complete for {task_id}: {sent_count} sent, {failed_count} failed"
        )

        # Clean up disconnected websockets
        for websocket in disconnected:
            await self.disconnect(websocket)

        # Clean up all subscriptions for this task immediately after broadcast
        async with self._lock:
            if task_id in self._subscriptions:
                subscriber_count = len(self._subscriptions[task_id])
                logger.debug(
                    f"Cleaning up subscriptions for {task_id} (had {subscriber_count} subscribers)"
                )

                for ws in self._subscriptions[task_id].copy():
                    if ws in self._connections:
                        self._connections[ws].discard(task_id)
                del self._subscriptions[task_id]
