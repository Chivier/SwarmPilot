"""
Client modules for interacting with SwarmPilot services.

This package provides clients for:
- Instance Service: Model lifecycle and task execution
- Scheduler Service: Task routing and instance management
- Predictor Service: Inference time prediction
"""

from .instance_client import (
    InstanceClient,
    InstanceClientError,
    InstanceConnectionError,
    InstanceAPIError,
    InstanceTimeoutError,
)

from .models import (
    # Task models
    TaskInfo,
    TaskSubmitResponse,
    TaskListResponse,
    TaskCancelResponse,
    TaskClearResponse,
    # Model models
    ModelInfo,
    ModelStartResponse,
    ModelStopResponse,
    RestartInitResponse,
    RestartOperation,
    # Instance models
    InstanceInfo,
    HealthResponse,
    QueueStats,
    ErrorResponse,
    # WebSocket models
    WSTaskSubmit,
    WSTaskResult,
    WSTaskStatus,
    WSError,
    WSPing,
    WSPong,
)

__all__ = [
    # Instance Client
    "InstanceClient",
    "InstanceClientError",
    "InstanceConnectionError",
    "InstanceAPIError",
    "InstanceTimeoutError",
    # Task models
    "TaskInfo",
    "TaskSubmitResponse",
    "TaskListResponse",
    "TaskCancelResponse",
    "TaskClearResponse",
    # Model models
    "ModelInfo",
    "ModelStartResponse",
    "ModelStopResponse",
    "RestartInitResponse",
    "RestartOperation",
    # Instance models
    "InstanceInfo",
    "HealthResponse",
    "QueueStats",
    "ErrorResponse",
    # WebSocket models
    "WSTaskSubmit",
    "WSTaskResult",
    "WSTaskStatus",
    "WSError",
    "WSPing",
    "WSPong",
]
