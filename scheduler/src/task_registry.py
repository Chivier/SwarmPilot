"""Task registry for managing task state.

This module provides backward compatibility by re-exporting
from src.registry.task_registry.
"""

from src.registry.task_registry import TaskRecord, TaskRegistry

__all__ = ["TaskRegistry", "TaskRecord"]
