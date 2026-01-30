"""Scheduler-compatible dummy endpoint models for the Planner service."""

from pydantic import BaseModel, Field

from .base import InstanceStatus


class InstanceDrainRequest(BaseModel):
    """Request model for instance drain (compatible with Scheduler)."""

    instance_id: str = Field(..., description="Instance ID to drain")


class InstanceDrainResponse(BaseModel):
    """Response model for instance drain (compatible with Scheduler)."""

    success: bool = Field(
        ..., description="Whether drain was initiated successfully"
    )
    message: str = Field(..., description="Status message")
    instance_id: str = Field(..., description="Instance ID")
    status: InstanceStatus = Field(..., description="Current instance status")
    pending_tasks: int = Field(..., description="Number of pending tasks")
    running_tasks: int = Field(..., description="Number of running tasks")
    estimated_completion_time_ms: float | None = Field(
        None, description="Estimated completion time"
    )


class InstanceDrainStatusResponse(BaseModel):
    """Response model for instance drain status check (compatible with Scheduler)."""

    success: bool = Field(
        ..., description="Whether status check was successful"
    )
    instance_id: str = Field(..., description="Instance ID")
    status: InstanceStatus = Field(..., description="Current instance status")
    pending_tasks: int = Field(..., description="Number of pending tasks")
    running_tasks: int = Field(..., description="Number of running tasks")
    can_remove: bool = Field(
        ..., description="Whether instance can be safely removed"
    )
    drain_initiated_at: str | None = Field(
        None, description="ISO timestamp when drain started"
    )


class InstanceRemoveRequest(BaseModel):
    """Request model for instance removal (compatible with Scheduler)."""

    instance_id: str = Field(..., description="Instance ID to remove")


class InstanceRemoveResponse(BaseModel):
    """Response model for instance removal (compatible with Scheduler)."""

    success: bool = Field(..., description="Whether removal was successful")
    message: str = Field(..., description="Status message")
    instance_id: str = Field(..., description="Instance ID that was removed")


class TaskResubmitRequest(BaseModel):
    """Request model for task resubmission (compatible with Scheduler)."""

    task_id: str = Field(..., description="ID of the task to resubmit")
    original_instance_id: str = Field(
        ..., description="ID of the original instance"
    )


class TaskResubmitResponse(BaseModel):
    """Response model for task resubmission (compatible with Scheduler)."""

    success: bool = Field(
        ..., description="Whether resubmission was successful"
    )
    message: str = Field(..., description="Status message")
