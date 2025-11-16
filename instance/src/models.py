"""
Core data models for Instance Service
"""

import time
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task status enumeration"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class InstanceStatus(str, Enum):
    """Instance status enumeration"""
    IDLE = "idle"
    RUNNING = "running"
    BUSY = "busy"
    ERROR = "error"


class RestartStatus(str, Enum):
    """Restart operation status enumeration"""
    PENDING = "pending"
    DRAINING = "draining"
    EXTRACTING_TASKS = "extracting_tasks"
    WAITING_RUNNING_TASK = "waiting_running_task"
    STOPPING_MODEL = "stopping_model"
    DEREGISTERING = "deregistering"
    STARTING_MODEL = "starting_model"
    REGISTERING = "registering"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    """Task data model"""
    task_id: str = Field(..., description="Unique identifier for this task")
    model_id: str = Field(..., description="Model/tool ID to use for this task")
    task_input: Dict[str, Any] = Field(..., description="Model-specific input data")
    status: TaskStatus = Field(default=TaskStatus.QUEUED, description="Current task status")
    callback_url: Optional[str] = Field(None, description="Optional callback URL for task result")

    # Timestamps
    submitted_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z"))
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    enqueue_time: float = Field(
        default_factory=time.time,
        description="Unix timestamp when task was enqueued, used for priority queue ordering"
    )

    # Results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def mark_started(self):
        """Mark task as started"""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    def mark_completed(self, result: Dict[str, Any]):
        """Mark task as completed with result"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        self.result = result

    def mark_failed(self, error: str):
        """Mark task as failed with error"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        self.error = error


class ModelInfo(BaseModel):
    """Information about a running model"""
    model_id: str
    started_at: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    container_name: Optional[str] = None


class ModelRegistryEntry(BaseModel):
    """Model registry entry"""
    model_id: str
    name: str
    directory: str
    resource_requirements: Dict[str, Any]


class RestartOperation(BaseModel):
    """Tracks the state of a model restart operation"""
    operation_id: str = Field(..., description="Unique identifier for this restart operation")
    status: RestartStatus = Field(default=RestartStatus.PENDING, description="Current restart status")
    old_model_id: Optional[str] = None
    new_model_id: str
    new_parameters: Dict[str, Any] = Field(default_factory=dict)
    new_scheduler_url: Optional[str] = None

    # Timestamps
    initiated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z"))
    completed_at: Optional[str] = None

    # Progress tracking
    pending_tasks_at_start: int = 0
    pending_tasks_completed: int = 0
    redistributed_tasks_count: int = 0
    error: Optional[str] = None

    def update_status(self, new_status: RestartStatus, error: Optional[str] = None):
        """Update the operation status"""
        self.status = new_status
        if error:
            self.error = error
        if new_status in (RestartStatus.COMPLETED, RestartStatus.FAILED):
            self.completed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
