"""API response models for the scheduler.

This module defines all Pydantic models used for API response validation.
"""

from typing import Any

from pydantic import BaseModel, Field

from src.models.core import Instance, InstanceStats, TaskTimestamps
from src.models.queue import (
    InstanceQueueExpectError,
    InstanceQueueProbabilistic,
)
from src.models.status import InstanceStatus, TaskStatus

# ============================================================================
# Common Response Models
# ============================================================================


class SuccessResponse(BaseModel):
    """Generic success response."""

    success: bool
    message: str | None = None


class ErrorResponse(BaseModel):
    """Generic error response."""

    success: bool
    error: str


# ============================================================================
# Instance Management Responses
# ============================================================================


class InstanceRegisterResponse(BaseModel):
    """Response model for instance registration."""

    success: bool
    message: str
    instance: Instance


class InstanceRemoveResponse(BaseModel):
    """Response model for instance removal."""

    success: bool
    message: str
    instance_id: str


class InstanceListResponse(BaseModel):
    """Response model for instance listing."""

    success: bool
    count: int
    instances: list[Instance]


class InstanceInfoResponse(BaseModel):
    """Response model for detailed instance information."""

    success: bool
    instance: Instance
    queue_info: InstanceQueueProbabilistic | InstanceQueueExpectError
    stats: InstanceStats


class InstanceDrainResponse(BaseModel):
    """Response model for starting instance draining."""

    success: bool
    message: str
    instance_id: str
    status: InstanceStatus
    pending_tasks: int
    running_tasks: int
    estimated_completion_time_ms: float | None = None


class InstanceDrainStatusResponse(BaseModel):
    """Response model for checking instance drain status."""

    success: bool
    instance_id: str
    status: InstanceStatus
    pending_tasks: int
    running_tasks: int
    can_remove: bool
    drain_initiated_at: str | None = None


class InstanceRedeployResponse(BaseModel):
    """Response model for instance redeployment."""

    success: bool
    message: str
    returned_tasks: list[dict[str, Any]] = Field(
        default_factory=list, description="Tasks returned from the instance"
    )
    redistributed_tasks: list[str] = Field(
        default_factory=list,
        description="Task IDs that were successfully redistributed",
    )
    failed_redistributions: list[str] = Field(
        default_factory=list, description="Task IDs that failed to redistribute"
    )
    current_task: dict[str, Any] | None = Field(
        None, description="Currently executing task on the instance"
    )
    estimated_redeploy_time_ms: float | None = Field(
        None, description="Estimated time for redeployment in milliseconds"
    )


# ============================================================================
# Task Management Responses
# ============================================================================


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
    completed_at: str | None = None


class TaskListResponse(BaseModel):
    """Response model for task listing with pagination."""

    success: bool
    count: int
    total: int
    offset: int
    limit: int
    tasks: list[TaskSummary]


class TaskDetailInfo(BaseModel):
    """Detailed information for a specific task."""

    task_id: str
    model_id: str
    status: TaskStatus
    assigned_instance: str
    task_input: dict[str, Any]
    metadata: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    timestamps: TaskTimestamps
    execution_time_ms: float | None = None


class TaskDetailResponse(BaseModel):
    """Response model for detailed task information."""

    success: bool
    task: TaskDetailInfo


class TaskClearResponse(BaseModel):
    """Response model for clearing all tasks."""

    success: bool
    message: str
    cleared_count: int


class TaskResubmitResponse(BaseModel):
    """Response model for task resubmission."""

    success: bool
    message: str


class TaskUpdateMetadataResult(BaseModel):
    """Result for a single task metadata update."""

    task_id: str = Field(..., description="ID of the task")
    success: bool = Field(..., description="Whether the update was successful")
    message: str = Field(..., description="Status message")
    queue_updated: bool = Field(
        default=False, description="Whether queue info was updated"
    )
    old_prediction_ms: float | None = Field(
        None, description="Previous predicted time in ms"
    )
    new_prediction_ms: float | None = Field(
        None, description="New predicted time in ms"
    )


class TaskUpdateMetadataResponse(BaseModel):
    """Response model for batch task metadata update."""

    success: bool = Field(..., description="Overall success (true if no failures)")
    message: str = Field(..., description="Summary message")
    total: int = Field(..., description="Total number of updates requested")
    succeeded: int = Field(..., description="Number of successful updates")
    failed: int = Field(..., description="Number of failed updates")
    skipped: int = Field(
        default=0,
        description="Number of skipped updates (COMPLETED/FAILED tasks)",
    )
    results: list[TaskUpdateMetadataResult] = Field(
        default_factory=list, description="Results for each task update"
    )


class TaskRepredictResponse(BaseModel):
    """Response model for batch task re-prediction (summary only, no per-task details)."""

    success: bool = Field(..., description="Overall success (true if no failures)")
    message: str = Field(..., description="Summary message")
    total_tasks: int = Field(..., description="Total tasks in registry")
    eligible_tasks: int = Field(
        ...,
        description="Tasks eligible for re-prediction (PENDING/RUNNING in queue)",
    )
    repredicted: int = Field(
        ..., description="Number of successfully re-predicted tasks"
    )
    failed: int = Field(..., description="Number of tasks that failed re-prediction")
    skipped: int = Field(
        ...,
        description="Number of skipped tasks (COMPLETED/FAILED/not in queue)",
    )


class TaskScheduleInfo(BaseModel):
    """Schedule information for a single task."""

    task_id: str = Field(..., description="Unique task identifier")
    model_id: str = Field(..., description="Model ID the task is associated with")
    status: TaskStatus = Field(..., description="Current task status")
    assigned_instance: str = Field(
        ..., description="Instance ID the task was scheduled to"
    )
    submitted_at: str = Field(..., description="ISO timestamp when task was submitted")


class TaskScheduleInfoResponse(BaseModel):
    """Response model for task scheduling information."""

    success: bool = Field(
        default=True, description="Whether the request was successful"
    )
    count: int = Field(..., description="Number of tasks in this response")
    total: int = Field(..., description="Total number of tasks matching the filter")
    tasks: list[TaskScheduleInfo] = Field(
        default_factory=list, description="List of task scheduling information"
    )


# ============================================================================
# Health Check Responses
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
# Strategy Management Responses
# ============================================================================


class StrategyInfo(BaseModel):
    """Information about the current scheduling strategy."""

    strategy_name: str
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific parameters"
    )


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
# Callback Responses
# ============================================================================


class TaskResultCallbackResponse(BaseModel):
    """Response model for task result callback."""

    success: bool
    message: str
