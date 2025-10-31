"""
Core data models for Instance Service
"""

from datetime import datetime
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


class Task(BaseModel):
    """Task data model"""
    task_id: str = Field(..., description="Unique identifier for this task")
    model_id: str = Field(..., description="Model/tool ID to use for this task")
    task_input: Dict[str, Any] = Field(..., description="Model-specific input data")
    status: TaskStatus = Field(default=TaskStatus.QUEUED, description="Current task status")

    # Timestamps
    submitted_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def mark_started(self):
        """Mark task as started"""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.utcnow().isoformat() + "Z"

    def mark_completed(self, result: Dict[str, Any]):
        """Mark task as completed with result"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow().isoformat() + "Z"
        self.result = result

    def mark_failed(self, error: str):
        """Mark task as failed with error"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow().isoformat() + "Z"
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
