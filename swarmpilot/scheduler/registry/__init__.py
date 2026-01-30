"""Registry package.

This package contains instance and task registries.
"""

from swarmpilot.scheduler.registry.instance_registry import InstanceRegistry
from swarmpilot.scheduler.registry.task_registry import TaskRecord, TaskRegistry

__all__ = [
    "InstanceRegistry",
    "TaskRecord",
    "TaskRegistry",
]
