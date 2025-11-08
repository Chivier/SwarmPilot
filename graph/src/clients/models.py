"""
Pydantic models for Instance Service API responses.

These models provide strict validation for all API responses from the Instance Service.
"""

from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field


# ============================================================================
# Task-related Models
# ============================================================================

TaskStatusType = Literal["queued", "running", "completed", "failed", "cancelled"]


class TaskInfo(BaseModel):
    """Information about a task."""

    task_id: str = Field(..., description="Unique task identifier")
    model_id: str = Field(..., description="Model ID this task is for")
    status: TaskStatusType = Field(..., description="Current task status")
    queue_position: Optional[int] = Field(None, description="Position in queue (if queued)")

    # Timestamps
    submitted_at: Optional[datetime] = Field(None, description="When task was submitted")
    started_at: Optional[datetime] = Field(None, description="When task started processing")
    completed_at: Optional[datetime] = Field(None, description="When task completed")

    # Results and errors
    result: Optional[Dict[str, Any]] = Field(None, description="Task result (if completed)")
    error: Optional[str] = Field(None, description="Error message (if failed)")

    # Input data
    task_input: Optional[Dict[str, Any]] = Field(None, description="Original task input")


class TaskSubmitResponse(BaseModel):
    """Response from task submission."""

    message: str = Field(..., description="Success message")
    task_id: str = Field(..., description="Task ID")
    status: TaskStatusType = Field(..., description="Initial task status")
    queue_position: Optional[int] = Field(None, description="Position in queue")


class TaskListResponse(BaseModel):
    """Response from listing tasks."""

    tasks: List[TaskInfo] = Field(..., description="List of tasks")
    total: int = Field(..., description="Total number of tasks")
    filtered: Optional[int] = Field(None, description="Number after filtering")


class TaskCancelResponse(BaseModel):
    """Response from cancelling a task."""

    message: str = Field(..., description="Success message")
    task_id: str = Field(..., description="Cancelled task ID")


class TaskClearResponse(BaseModel):
    """Response from clearing tasks."""

    message: str = Field(..., description="Success message")
    cleared: int = Field(..., description="Number of tasks cleared")


# ============================================================================
# Model-related Models
# ============================================================================

class ModelInfo(BaseModel):
    """Information about the currently loaded model."""

    model_id: str = Field(..., description="Model identifier")
    status: Literal["idle", "busy", "loading", "error"] = Field(..., description="Model status")
    loaded_at: Optional[datetime] = Field(None, description="When model was loaded")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")


class ModelStartResponse(BaseModel):
    """Response from starting a model."""

    message: str = Field(..., description="Success message")
    model_id: str = Field(..., description="Started model ID")
    scheduler_url: Optional[str] = Field(None, description="Scheduler URL if registered")


class ModelStopResponse(BaseModel):
    """Response from stopping a model."""

    message: str = Field(..., description="Success message")
    model_id: str = Field(..., description="Stopped model ID")


class RestartOperation(BaseModel):
    """Information about a restart operation."""

    operation_id: str = Field(..., description="Unique operation identifier")
    status: Literal["draining", "stopping", "starting", "completed", "failed"] = Field(
        ..., description="Current operation status"
    )
    old_model_id: str = Field(..., description="Model being replaced")
    new_model_id: str = Field(..., description="New model to start")

    started_at: datetime = Field(..., description="When operation started")
    completed_at: Optional[datetime] = Field(None, description="When operation completed")

    error: Optional[str] = Field(None, description="Error message if failed")


class RestartInitResponse(BaseModel):
    """Response from initiating a restart."""

    message: str = Field(..., description="Success message")
    operation_id: str = Field(..., description="Operation ID for tracking")
    status: str = Field(..., description="Initial operation status")


# ============================================================================
# Instance-related Models
# ============================================================================

class QueueStats(BaseModel):
    """Queue statistics."""

    queued: int = Field(0, description="Number of queued tasks")
    running: int = Field(0, description="Number of running tasks")
    completed: int = Field(0, description="Number of completed tasks")
    failed: int = Field(0, description="Number of failed tasks")
    cancelled: int = Field(0, description="Number of cancelled tasks")
    total: int = Field(0, description="Total tasks")


class InstanceInfo(BaseModel):
    """Complete instance information."""

    instance_id: str = Field(..., description="Instance identifier")
    status: Literal["healthy", "busy", "unhealthy", "starting"] = Field(
        ..., description="Instance status"
    )

    # Model information
    current_model: Optional[ModelInfo] = Field(None, description="Currently loaded model")

    # Queue statistics
    queue: QueueStats = Field(default_factory=QueueStats, description="Queue statistics")

    # Capacity and performance
    max_queue_size: int = Field(100, description="Maximum queue size")
    average_task_duration: Optional[float] = Field(None, description="Average task duration (seconds)")

    # Uptime
    started_at: datetime = Field(..., description="When instance started")
    uptime_seconds: float = Field(..., description="Instance uptime in seconds")


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy"] = Field(..., description="Health status")
    instance_id: str = Field(..., description="Instance identifier")
    model_loaded: bool = Field(..., description="Whether a model is loaded")
    queue_size: int = Field(..., description="Current queue size")
    uptime_seconds: float = Field(..., description="Uptime in seconds")


# ============================================================================
# Error Models
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response from API."""

    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    status_code: Optional[int] = Field(None, description="HTTP status code")


# ============================================================================
# WebSocket Message Models
# ============================================================================

class WSTaskSubmit(BaseModel):
    """WebSocket task submission message."""

    type: Literal["task_submit"] = "task_submit"
    task_id: str
    model_id: str
    task_input: Dict[str, Any]


class WSTaskResult(BaseModel):
    """WebSocket task result message."""

    type: Literal["task_result"] = "task_result"
    task_id: str
    status: TaskStatusType
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    completed_at: datetime


class WSTaskStatus(BaseModel):
    """WebSocket task status update."""

    type: Literal["task_status"] = "task_status"
    task_id: str
    status: TaskStatusType
    queue_position: Optional[int] = None
    started_at: Optional[datetime] = None


class WSError(BaseModel):
    """WebSocket error message."""

    type: Literal["error"] = "error"
    error: str
    task_id: Optional[str] = None


class WSPing(BaseModel):
    """WebSocket ping message."""

    type: Literal["ping"] = "ping"
    timestamp: datetime


class WSPong(BaseModel):
    """WebSocket pong response."""

    type: Literal["pong"] = "pong"
    timestamp: datetime
