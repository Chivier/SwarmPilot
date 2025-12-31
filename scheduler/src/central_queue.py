"""Central task queue for FIFO dispatch.

This module provides backward compatibility by re-exporting
from src.services.central_queue.
"""

from loguru import logger

from src.services.central_queue import CentralTaskQueue, QueuedTask

__all__ = ["CentralTaskQueue", "QueuedTask", "logger"]
