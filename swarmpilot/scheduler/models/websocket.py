"""WebSocket message models for the scheduler.

This module defines all Pydantic models used for WebSocket communication.
"""

from typing import Any

from pydantic import BaseModel

from swarmpilot.scheduler.models.core import TaskTimestamps
from swarmpilot.scheduler.models.status import TaskStatus, WSMessageType


class WSAckMessage(BaseModel):
    """WebSocket acknowledgment message."""

    type: WSMessageType = WSMessageType.ACK
    message: str
    subscribed_tasks: list[str]


class WSTaskResultMessage(BaseModel):
    """WebSocket message for task result notification."""

    type: WSMessageType = WSMessageType.RESULT
    task_id: str
    status: TaskStatus
    result: dict[str, Any] | None = None
    error: str | None = None
    timestamps: TaskTimestamps
    execution_time_ms: float | None = None


class WSErrorMessage(BaseModel):
    """WebSocket error message."""

    type: WSMessageType = WSMessageType.ERROR
    error: str
    task_id: str | None = None


class WSPingMessage(BaseModel):
    """WebSocket ping message for keepalive."""

    type: WSMessageType = WSMessageType.PING
    timestamp: float | None = None


class WSPongMessage(BaseModel):
    """WebSocket pong message for keepalive response."""

    type: WSMessageType = WSMessageType.PONG
    timestamp: float | None = None
