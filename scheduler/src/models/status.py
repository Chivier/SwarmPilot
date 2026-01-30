"""Status enumerations for the scheduler.

This module defines all enum types used throughout the scheduler system.
"""

from enum import Enum


class InstanceStatus(str, Enum):
    """Enumeration of possible instance statuses."""

    INITIALIZING = "initializing"  # Instance registered, work stealing in progress, no new tasks yet
    ACTIVE = "active"  # Normal operation, accepts new tasks
    DRAINING = "draining"  # No new tasks, waiting for existing tasks to complete
    REMOVING = "removing"  # All tasks complete, safe to remove
    REDEPLOYING = "redeploying"  # Instance is being redeployed (no new tasks accepted, pending tasks returned)


class TaskStatus(str, Enum):
    """Enumeration of possible task statuses."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class WSMessageType(str, Enum):
    """Enumeration of WebSocket message types."""

    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    RESULT = "result"
    ERROR = "error"
    ACK = "ack"
    PING = "ping"
    PONG = "pong"


class StrategyType(str, Enum):
    """Enumeration of available scheduling strategies."""

    MIN_TIME = "min_time"
    PROBABILISTIC = "probabilistic"
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    POWEROFTWO = "po2"
    SERVERLESS = "serverless"
    ADAPTIVE_BOOTSTRAP = "adaptive_bootstrap"
