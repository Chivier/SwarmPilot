"""Background services package.

This package contains background services like the scheduler,
task dispatcher, and WebSocket manager.
"""

from src.services.background_scheduler import BackgroundScheduler
from src.services.central_queue import CentralTaskQueue, QueuedTask
from src.services.task_dispatcher import TaskDispatcher
from src.services.websocket_manager import ConnectionManager
from src.services.worker_queue_thread import (
    QueuedTask as WorkerQueuedTask,
    TaskResult,
    WorkerQueueThread,
)
from src.services.task_result_callback import TaskResultCallback
from src.services.worker_queue_manager import WorkerQueueManager
from src.services.shutdown_handler import ShutdownHandler, ShutdownResult

__all__ = [
    "BackgroundScheduler",
    "CentralTaskQueue",
    "QueuedTask",
    "TaskDispatcher",
    "ConnectionManager",
    "WorkerQueuedTask",
    "TaskResult",
    "WorkerQueueThread",
    "TaskResultCallback",
    "WorkerQueueManager",
    "ShutdownHandler",
    "ShutdownResult",
]
