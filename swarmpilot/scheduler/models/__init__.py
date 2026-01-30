"""Data models package.

This package contains all Pydantic models and enums for the scheduler.
Models are organized by domain for maintainability.
"""

# Re-export all models from submodules
from swarmpilot.scheduler.models.core import Instance, InstanceStats, Task, TaskTimestamps
from swarmpilot.scheduler.models.queue import (
    InstanceQueueBase,
    InstanceQueueExpectError,
    InstanceQueueProbabilistic,
)
from swarmpilot.scheduler.models.requests import (
    InstanceDrainRequest,
    InstanceRedeployRequest,
    InstanceRegisterRequest,
    InstanceRemoveRequest,
    StrategySetRequest,
    TaskMetadataUpdate,
    TaskResubmitRequest,
    TaskResultCallbackRequest,
    TaskSubmitRequest,
    TaskUpdateMetadataRequest,
)
from swarmpilot.scheduler.models.responses import (
    ErrorResponse,
    HealthErrorResponse,
    HealthResponse,
    HealthStats,
    InstanceDrainResponse,
    InstanceDrainStatusResponse,
    InstanceInfoResponse,
    InstanceListResponse,
    InstanceRedeployResponse,
    InstanceRegisterResponse,
    InstanceRemoveResponse,
    StrategyGetResponse,
    StrategyInfo,
    StrategySetResponse,
    SuccessResponse,
    TaskClearResponse,
    TaskDetailInfo,
    TaskDetailResponse,
    TaskInfo,
    TaskListResponse,
    TaskRepredictResponse,
    TaskResubmitResponse,
    TaskResultCallbackResponse,
    TaskScheduleInfo,
    TaskScheduleInfoResponse,
    TaskSubmitResponse,
    TaskSummary,
    TaskUpdateMetadataResponse,
    TaskUpdateMetadataResult,
)
from swarmpilot.scheduler.models.status import (
    InstanceStatus,
    StrategyType,
    TaskStatus,
    WSMessageType,
)
from swarmpilot.scheduler.models.websocket import (
    WSAckMessage,
    WSErrorMessage,
    WSPingMessage,
    WSPongMessage,
    WSTaskResultMessage,
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
    "WSAckMessage",
    "WSTaskResultMessage",
    "WSErrorMessage",
    "WSPingMessage",
    "WSPongMessage",
]
