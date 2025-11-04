"""
Data models for the scheduler API.

This module defines all Pydantic models used for request/response validation
and data structures throughout the scheduler system.
"""

from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field


# ============================================================================
# Base Models
# ============================================================================

class Task(BaseModel):
    """Task definition for scheduling."""
    task_id: str
    model_id: str
    task_input: Dict[str, Any]
    metadata: Dict[str, Any]


class InstanceStatus(str, Enum):
    """Enumeration of possible instance statuses."""
    ACTIVE = "active"        # Normal operation, accepts new tasks
    DRAINING = "draining"    # No new tasks, waiting for existing tasks to complete
    REMOVING = "removing"    # All tasks complete, safe to remove


class Instance(BaseModel):
    """Instance definition for model execution."""
    instance_id: str
    model_id: str
    endpoint: str
    platform_info: Dict[str, str]  # Required: software_name, software_version, hardware_name
    status: InstanceStatus = InstanceStatus.ACTIVE  # Instance lifecycle status
    drain_initiated_at: Optional[str] = None  # ISO timestamp when draining started


class InstanceQueueBase(BaseModel):
    """Base class for instance queue information."""
    instance_id: str


class InstanceQueueProbabilistic(InstanceQueueBase):
    """Queue information for probabilistic scheduling strategy."""
    quantiles: List[float]
    values: List[float]


class InstanceQueueExpectError(InstanceQueueBase):
    """Queue information for minimum expected time strategy."""
    expected_time_ms: float
    error_margin_ms: float


# ============================================================================
# Common Response Models
# ============================================================================

class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Generic error response."""
    success: bool
    error: str


# ============================================================================
# Instance Management Models
# ============================================================================

class InstanceRegisterRequest(BaseModel):
    """Request model for instance registration."""
    instance_id: str
    model_id: str
    endpoint: str
    platform_info: Dict[str, str]  # Required: software_name, software_version, hardware_name


class InstanceRegisterResponse(BaseModel):
    """Response model for instance registration."""
    success: bool
    message: str
    instance: Instance


class InstanceRemoveRequest(BaseModel):
    """Request model for instance removal."""
    instance_id: str


class InstanceRemoveResponse(BaseModel):
    """Response model for instance removal."""
    success: bool
    message: str
    instance_id: str


class InstanceListResponse(BaseModel):
    """Response model for instance listing."""
    success: bool
    count: int
    instances: List[Instance]


class InstanceStats(BaseModel):
    """Statistics for an instance."""
    pending_tasks: int
    completed_tasks: int
    failed_tasks: int


class InstanceInfoResponse(BaseModel):
    """Response model for detailed instance information."""
    success: bool
    instance: Instance
    queue_info: Union[InstanceQueueProbabilistic, InstanceQueueExpectError]
    stats: InstanceStats


class InstanceDrainRequest(BaseModel):
    """Request model for starting instance draining."""
    instance_id: str


class InstanceDrainResponse(BaseModel):
    """Response model for starting instance draining."""
    success: bool
    message: str
    instance_id: str
    status: InstanceStatus
    pending_tasks: int
    running_tasks: int
    estimated_completion_time_ms: Optional[float] = None


class InstanceDrainStatusResponse(BaseModel):
    """Response model for checking instance drain status."""
    success: bool
    instance_id: str
    status: InstanceStatus
    pending_tasks: int
    running_tasks: int
    can_remove: bool
    drain_initiated_at: Optional[str] = None


# ============================================================================
# Task Management Models
# ============================================================================

class TaskStatus(str, Enum):
    """Enumeration of possible task statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskSubmitRequest(BaseModel):
    """Request model for task submission."""
    task_id: str
    model_id: str
    task_input: Dict[str, Any]
    metadata: Dict[str, Any]


class TaskInfo(BaseModel):
    """Basic task information returned after submission."""
    task_id: str
    status: TaskStatus
    assigned_instance: str
    submitted_at: str


class TaskSubmitResponse(BaseModel):
    """Response model for task submission."""
    success: bool
    message: str
    task: TaskInfo


class TaskSummary(BaseModel):
    """Summary information for a task."""
    task_id: str
    model_id: str
    status: TaskStatus
    assigned_instance: str
    submitted_at: str
    completed_at: Optional[str] = None


class TaskListResponse(BaseModel):
    """Response model for task listing with pagination."""
    success: bool
    count: int
    total: int
    offset: int
    limit: int
    tasks: List[TaskSummary]


class TaskTimestamps(BaseModel):
    """Timestamp information for a task."""
    submitted_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class TaskDetailInfo(BaseModel):
    """Detailed information for a specific task."""
    task_id: str
    model_id: str
    status: TaskStatus
    assigned_instance: str
    task_input: Dict[str, Any]
    metadata: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamps: TaskTimestamps
    execution_time_ms: Optional[float] = None


class TaskDetailResponse(BaseModel):
    """Response model for detailed task information."""
    success: bool
    task: TaskDetailInfo


class TaskClearResponse(BaseModel):
    """Response model for clearing all tasks."""
    success: bool
    message: str
    cleared_count: int


# ============================================================================
# Health Check Models
# ============================================================================

class HealthStats(BaseModel):
    """Statistics for health check."""
    total_instances: int
    active_instances: int
    total_tasks: int
    pending_tasks: int
    running_tasks: int
    completed_tasks: int
    failed_tasks: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    success: bool
    status: str
    timestamp: str
    version: str
    stats: HealthStats


class HealthErrorResponse(BaseModel):
    """Error response model for health check."""
    success: bool
    status: str
    error: str
    timestamp: str


# ============================================================================
# Strategy Management Models
# ============================================================================

class StrategyType(str, Enum):
    """Enumeration of available scheduling strategies."""
    MIN_TIME = "min_time"
    PROBABILISTIC = "probabilistic"
    ROUND_ROBIN = "round_robin"


class StrategySetRequest(BaseModel):
    """Request model for setting scheduling strategy."""
    strategy_name: StrategyType = Field(..., description="Name of the scheduling strategy to use")
    quantiles: Optional[List[float]] = Field(
        None,
        description="Custom quantiles for probabilistic strategy (default: [0.5, 0.9, 0.95, 0.99])",
        min_length=1,
        max_length=100
    )


class StrategyInfo(BaseModel):
    """Information about the current scheduling strategy."""
    strategy_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy-specific parameters")


class StrategySetResponse(BaseModel):
    """Response model for setting scheduling strategy."""
    success: bool
    message: str
    cleared_tasks: int
    reinitialized_instances: int
    strategy_info: StrategyInfo


class StrategyGetResponse(BaseModel):
    """Response model for getting current scheduling strategy."""
    success: bool
    strategy_info: StrategyInfo


# ============================================================================
# Callback Models (Instance -> Scheduler)
# ============================================================================

class TaskResultCallbackRequest(BaseModel):
    """Request model for task result callback from instance to scheduler."""
    task_id: str = Field(..., description="ID of the completed task")
    status: str = Field(..., description="Task status: 'completed' or 'failed'")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result data (if completed)")
    error: Optional[str] = Field(None, description="Error message (if failed)")
    execution_time_ms: Optional[float] = Field(None, description="Execution time in milliseconds")


class TaskResultCallbackResponse(BaseModel):
    """Response model for task result callback."""
    success: bool
    message: str


# ============================================================================
# WebSocket Models
# ============================================================================

class WSMessageType(str, Enum):
    """Enumeration of WebSocket message types."""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    RESULT = "result"
    ERROR = "error"
    ACK = "ack"
    PING = "ping"
    PONG = "pong"


class WSSubscribeMessage(BaseModel):
    """WebSocket message for subscribing to task results."""
    type: WSMessageType = WSMessageType.SUBSCRIBE
    task_ids: List[str]


class WSUnsubscribeMessage(BaseModel):
    """WebSocket message for unsubscribing from task results."""
    type: WSMessageType = WSMessageType.UNSUBSCRIBE
    task_ids: List[str]


class WSAckMessage(BaseModel):
    """WebSocket acknowledgment message."""
    type: WSMessageType = WSMessageType.ACK
    message: str
    subscribed_tasks: List[str]


class WSTaskResultMessage(BaseModel):
    """WebSocket message for task result notification."""
    type: WSMessageType = WSMessageType.RESULT
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamps: TaskTimestamps
    execution_time_ms: Optional[float] = None


class WSErrorMessage(BaseModel):
    """WebSocket error message."""
    type: WSMessageType = WSMessageType.ERROR
    error: str
    task_id: Optional[str] = None


class WSPingMessage(BaseModel):
    """WebSocket ping message for keepalive."""
    type: WSMessageType = WSMessageType.PING
    timestamp: Optional[float] = None


class WSPongMessage(BaseModel):
    """WebSocket pong message for keepalive response."""
    type: WSMessageType = WSMessageType.PONG
    timestamp: Optional[float] = None
