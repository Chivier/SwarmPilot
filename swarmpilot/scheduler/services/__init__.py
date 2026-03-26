"""Background services package.

This package contains background services like the scheduler,
task dispatcher, and WebSocket manager.
"""

from swarmpilot.scheduler.services.task_result_callback import (
    TaskResultCallback,
)
from swarmpilot.scheduler.services.websocket_manager import ConnectionManager
from swarmpilot.scheduler.services.worker_queue_manager import (
    WorkerQueueManager,
)
from swarmpilot.scheduler.services.worker_queue_thread import (
    QueuedTask as WorkerQueuedTask,
    TaskResult,
    WorkerQueueThread,
)

__all__ = [
    "ConnectionManager",
    "TaskResult",
    "TaskResultCallback",
    "WorkerQueueManager",
    "WorkerQueueThread",
    "WorkerQueuedTask",
]
