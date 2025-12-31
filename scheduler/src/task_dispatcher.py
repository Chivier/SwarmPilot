"""Task dispatcher for sending tasks to instances.

This module provides backward compatibility by re-exporting
from src.services.task_dispatcher.
"""

from src.services.task_dispatcher import TaskDispatcher

__all__ = ["TaskDispatcher"]
