"""WebSocket endpoint for real-time task result delivery.

Provides WebSocket connections for clients to receive task results in real-time.
"""

import asyncio
import json
from contextlib import suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from ..model import (
    TaskStatus,
    WSAckMessage,
    WSErrorMessage,
    WSMessageType,
    WSPingMessage,
    WSPongMessage,
    WSTaskResultMessage,
)
from .deps import get_task_registry, get_websocket_manager

router = APIRouter(tags=["websocket"])


@router.websocket("/task/get_result")
async def websocket_get_result(websocket: WebSocket):
    """WebSocket endpoint for real-time task result delivery.

    Clients can subscribe to multiple task IDs and receive results
    as soon as tasks complete or fail.

    Includes keepalive ping mechanism to prevent connection timeout.

    Args:
        websocket: WebSocket connection
    """
    websocket_manager = get_websocket_manager()
    task_registry = get_task_registry()

    await websocket.accept()

    # Register connection
    await websocket_manager.connect(websocket)

    # Keepalive configuration
    ping_interval = 10  # Send ping every 10 seconds

    async def send_keepalive():
        """Send periodic ping messages to keep connection alive."""
        try:
            while True:
                await asyncio.sleep(ping_interval)
                ping_msg = WSPingMessage(timestamp=asyncio.get_event_loop().time())
                await websocket.send_json(ping_msg.model_dump())
                logger.debug("Sent keepalive ping to WebSocket client")
        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully
            pass
        except Exception as e:
            logger.warning(f"Keepalive task error: {e}")

    # Start keepalive task
    keepalive_task = asyncio.create_task(send_keepalive())

    try:
        while True:
            # Receive message from client (handle both text and ping/pong frames)
            message = await websocket.receive()

            # Handle WebSocket protocol-level ping/pong
            if message["type"] == "websocket.ping":
                # Respond to protocol-level ping with pong
                await websocket.send(
                    {
                        "type": "websocket.pong",
                        "bytes": message.get("bytes", b""),
                    }
                )
                logger.debug("Responded to protocol-level ping")
                continue

            elif message["type"] == "websocket.pong":
                # Received protocol-level pong, just log
                logger.debug("Received protocol-level pong")
                continue

            elif message["type"] == "websocket.disconnect":
                # Client disconnected
                logger.debug("Client initiated disconnect")
                break

            elif message["type"] != "websocket.receive":
                # Unknown message type
                logger.warning(f"Unknown WebSocket message type: {message['type']}")
                continue

            # Parse JSON data from text message
            try:
                data = json.loads(message.get("text", "{}"))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON message: {e}")
                error_msg = WSErrorMessage(error="Invalid JSON format")
                await websocket.send_json(error_msg.model_dump())
                continue

            # Parse message type
            message_type = data.get("type")

            if message_type == WSMessageType.PONG:
                # Client responded to our application-level ping, just log and continue
                logger.debug("Received application-level pong from WebSocket client")
                continue

            elif message_type == WSMessageType.PING:
                # Client sent application-level ping, respond with pong
                pong_msg = WSPongMessage(timestamp=asyncio.get_event_loop().time())
                await websocket.send_json(pong_msg.model_dump())
                logger.debug("Sent application-level pong response to WebSocket client")
                continue

            elif message_type == WSMessageType.SUBSCRIBE:
                # Parse task_ids
                task_ids = data.get("task_ids", [])

                if not isinstance(task_ids, list):
                    error_msg = WSErrorMessage(
                        error="task_ids must be a list of strings"
                    )
                    await websocket.send_json(error_msg.model_dump())
                    continue

                # Subscribe to tasks
                await websocket_manager.subscribe(websocket, task_ids)

                # For each task_id, check if already completed and send result immediately
                for task_id in task_ids:
                    task = await task_registry.get(task_id)
                    if task and task.status in (
                        TaskStatus.COMPLETED,
                        TaskStatus.FAILED,
                    ):
                        # Send result immediately
                        result_msg = WSTaskResultMessage(
                            task_id=task.task_id,
                            status=task.status,
                            result=task.result,
                            error=task.error,
                            timestamps=task.get_timestamps(),
                            execution_time_ms=task.execution_time_ms,
                        )
                        await websocket.send_json(result_msg.model_dump())

                # Send acknowledgment
                subscribed = await websocket_manager.get_subscribed_tasks(websocket)
                ack_msg = WSAckMessage(
                    message=f"Subscribed to {len(task_ids)} tasks",
                    subscribed_tasks=subscribed,
                )
                await websocket.send_json(ack_msg.model_dump())

            elif message_type == WSMessageType.UNSUBSCRIBE:
                # Parse task_ids
                task_ids = data.get("task_ids", [])

                if not isinstance(task_ids, list):
                    error_msg = WSErrorMessage(
                        error="task_ids must be a list of strings"
                    )
                    await websocket.send_json(error_msg.model_dump())
                    continue

                # Unsubscribe from tasks
                await websocket_manager.unsubscribe(websocket, task_ids)

                # Send acknowledgment
                subscribed = await websocket_manager.get_subscribed_tasks(websocket)
                ack_msg = WSAckMessage(
                    message=f"Unsubscribed from {len(task_ids)} tasks",
                    subscribed_tasks=subscribed,
                )
                await websocket.send_json(ack_msg.model_dump())

            else:
                # Unknown message type
                error_msg = WSErrorMessage(error=f"Unknown message type: {message_type}")
                await websocket.send_json(error_msg.model_dump())

    except WebSocketDisconnect:
        # Clean up subscriptions
        await websocket_manager.disconnect(websocket)
        logger.debug("WebSocket client disconnected")

    except Exception as e:
        # Log the error
        logger.error(f"WebSocket error: {e}", exc_info=True)
        # Send error message to client
        with suppress(Exception):
            error_msg = WSErrorMessage(error=f"Server error: {e!s}")
            await websocket.send_json(error_msg.model_dump())

        # Clean up
        await websocket_manager.disconnect(websocket)

    finally:
        # Cancel keepalive task
        keepalive_task.cancel()
        with suppress(asyncio.CancelledError):
            await keepalive_task
        logger.debug("WebSocket keepalive task cancelled")
