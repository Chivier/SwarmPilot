"""Data models for the scheduler API.

This module provides backward compatibility by re-exporting all
models from the src.models package.

For new code, prefer importing directly from src.models:
    from src.models import Task, TaskStatus, Instance
"""

# Re-export all models from the models package
from src.models import (
    # Common responses
    ErrorResponse,
    # Health models
    HealthErrorResponse,
    HealthResponse,
    HealthStats,
    # Core models
    Instance,
    # Instance requests
    InstanceDrainRequest,
    # Instance responses
    InstanceDrainResponse,
    InstanceDrainStatusResponse,
    InstanceInfoResponse,
    InstanceListResponse,
    # Queue models
    InstanceQueueBase,
    InstanceQueueExpectError,
    InstanceQueueProbabilistic,
    InstanceRedeployRequest,
    InstanceRedeployResponse,
    InstanceRegisterRequest,
    InstanceRegisterResponse,
    InstanceRemoveRequest,
    InstanceRemoveResponse,
    InstanceStats,
    # Status enums
    InstanceStatus,
    # Strategy models
    StrategyGetResponse,
    StrategyInfo,
    StrategySetRequest,
    StrategySetResponse,
    StrategyType,
    SuccessResponse,
    Task,
    # Task responses
    TaskClearResponse,
    TaskDetailInfo,
    TaskDetailResponse,
    TaskInfo,
    TaskListResponse,
    # Task requests
    TaskMetadataUpdate,
    TaskRepredictResponse,
    TaskResubmitRequest,
    TaskResubmitResponse,
    # Callback models
    TaskResultCallbackRequest,
    TaskResultCallbackResponse,
    TaskScheduleInfo,
    TaskScheduleInfoResponse,
    TaskStatus,
    TaskSubmitRequest,
    TaskSubmitResponse,
    TaskSummary,
    TaskTimestamps,
    TaskUpdateMetadataRequest,
    TaskUpdateMetadataResponse,
    TaskUpdateMetadataResult,
    # WebSocket models
    WSAckMessage,
    WSErrorMessage,
    WSMessageType,
    WSPingMessage,
    WSPongMessage,
    WSSubscribeMessage,
    WSTaskResultMessage,
    WSUnsubscribeMessage,
)

__all__ = [
    # Status enums
    "InstanceStatus",
    "TaskStatus",
    "WSMessageType",
    "StrategyType",
    # Core models
    "Task",
    "Instance",
    "InstanceStats",
    "TaskTimestamps",
    # Queue models
    "InstanceQueueBase",
    "InstanceQueueProbabilistic",
    "InstanceQueueExpectError",
    # Common responses
    "SuccessResponse",
    "ErrorResponse",
    # Instance requests
    "InstanceRegisterRequest",
    "InstanceRemoveRequest",
    "InstanceDrainRequest",
    "InstanceRedeployRequest",
    # Instance responses
    "InstanceRegisterResponse",
    "InstanceRemoveResponse",
    "InstanceListResponse",
    "InstanceInfoResponse",
    "InstanceDrainResponse",
    "InstanceDrainStatusResponse",
    "InstanceRedeployResponse",
    # Task requests
    "TaskSubmitRequest",
    "TaskResubmitRequest",
    "TaskMetadataUpdate",
    "TaskUpdateMetadataRequest",
    # Task responses
    "TaskInfo",
    "TaskSubmitResponse",
    "TaskSummary",
    "TaskListResponse",
    "TaskDetailInfo",
    "TaskDetailResponse",
    "TaskClearResponse",
    "TaskResubmitResponse",
    "TaskUpdateMetadataResult",
    "TaskUpdateMetadataResponse",
    "TaskRepredictResponse",
    "TaskScheduleInfo",
    "TaskScheduleInfoResponse",
    # Health models
    "HealthStats",
    "HealthResponse",
    "HealthErrorResponse",
    # Strategy models
    "StrategySetRequest",
    "StrategyInfo",
    "StrategySetResponse",
    "StrategyGetResponse",
    # Callback models
    "TaskResultCallbackRequest",
    "TaskResultCallbackResponse",
    # WebSocket models
    "WSSubscribeMessage",
    "WSUnsubscribeMessage",
    "WSAckMessage",
    "WSTaskResultMessage",
    "WSErrorMessage",
    "WSPingMessage",
    "WSPongMessage",
]
