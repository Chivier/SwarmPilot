"""Scheduler service package.

This package provides a FastAPI-based scheduler service for managing
task distribution across multiple compute instances.
"""

from .api import app
from .model import (
    Instance,
    InstanceStatus,
    Task,
    TaskStatus,
)

__version__ = "1.0.0"

__all__ = [
    "Instance",
    "InstanceStatus",
    "Task",
    "TaskStatus",
    "app",
]
