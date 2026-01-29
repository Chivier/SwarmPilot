"""Background services package.

This package contains background services like the scheduler,
task dispatcher, and WebSocket manager.
"""

from src.services.shutdown_handler import ShutdownHandler, ShutdownResult
from src.services.task_result_callback import TaskResultCallback
from src.services.websocket_manager import ConnectionManager
from src.services.worker_queue_manager import WorkerQueueManager
from src.services.worker_queue_thread import (
    QueuedTask as WorkerQueuedTask,
    TaskResult,
    WorkerQueueThread,
)

__all__ = [
    "ConnectionManager",
    "ShutdownHandler",
    "ShutdownResult",
    "TaskResult",
    "TaskResultCallback",
    "WorkerQueueManager",
    "WorkerQueueThread",
    "WorkerQueuedTask",
]
