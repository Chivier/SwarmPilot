"""Core entity models for the scheduler.

This module defines the fundamental data models used throughout the system.
"""

from typing import Any

from pydantic import BaseModel

from swarmpilot.scheduler.models.status import InstanceStatus


class Task(BaseModel):
    """Task definition for scheduling."""

    task_id: str
    model_id: str
    task_input: dict[str, Any]
    metadata: dict[str, Any]


class Instance(BaseModel):
    """Instance definition for model execution."""

    instance_id: str
    model_id: str
    endpoint: str
    platform_info: dict[
        str, str
    ]  # Required: software_name, software_version, hardware_name
    status: InstanceStatus = InstanceStatus.ACTIVE  # Instance lifecycle status
    drain_initiated_at: str | None = None  # ISO timestamp when draining started


class InstanceStats(BaseModel):
    """Statistics for an instance."""

    pending_tasks: int
    completed_tasks: int
    failed_tasks: int


class TaskTimestamps(BaseModel):
    """Timestamp information for a task."""

    submitted_at: str
    started_at: str | None = None
    completed_at: str | None = None
