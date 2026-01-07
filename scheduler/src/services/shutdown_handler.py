"""Graceful shutdown handler for PYLET-020.

This module implements graceful shutdown handling for the scheduler-side
task queue system, ensuring tasks are properly handled during scheduler
shutdown.

Key differences from instance removal (PYLET-019):
- Instance removal: Tasks are rescheduled to other workers
- Scheduler shutdown: Tasks are dropped (no workers left to reschedule to)

Example:
    ```python
    handler = ShutdownHandler(
        worker_queue_manager=manager,
        instance_registry=registry,
    )

    # During scheduler shutdown
    result = await handler.shutdown_all(timeout=60.0)
    print(f"Stopped {result.workers_stopped} workers, dropped {result.tasks_dropped} tasks")
    ```
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Awaitable

from loguru import logger

if TYPE_CHECKING:
    from src.registry.instance_registry import InstanceRegistry
    from src.services.worker_queue_manager import WorkerQueueManager


@dataclass
class ShutdownResult:
    """Result from shutdown operation.

    Attributes:
        workers_stopped: Number of workers successfully stopped.
        tasks_dropped: Number of tasks dropped during shutdown.
        timeout_occurred: Whether timeout was exceeded.
    """

    workers_stopped: int = 0
    tasks_dropped: int = 0
    timeout_occurred: bool = False


class ShutdownHandler:
    """Handles graceful shutdown of worker queues.

    This handler is used during scheduler shutdown. Unlike instance removal
    (which reschedules tasks), shutdown drops all pending tasks since there
    are no workers left to reschedule to.

    Attributes:
        worker_queue_manager: Manager for worker queue threads.
        instance_registry: Registry for instance management.
        on_shutdown_complete: Optional callback invoked when shutdown completes.
    """

    def __init__(
        self,
        worker_queue_manager: "WorkerQueueManager",
        instance_registry: "InstanceRegistry",
        on_shutdown_complete: Callable[[ShutdownResult], Awaitable[None]] | None = None,
    ):
        """Initialize shutdown handler.

        Args:
            worker_queue_manager: Manager for worker queue threads.
            instance_registry: Registry for instance management.
            on_shutdown_complete: Optional async callback invoked when shutdown
                completes with the ShutdownResult.
        """
        self.worker_queue_manager = worker_queue_manager
        self.instance_registry = instance_registry
        self.on_shutdown_complete = on_shutdown_complete

    async def shutdown_all(
        self,
        timeout: float = 60.0,
    ) -> ShutdownResult:
        """Gracefully shutdown all workers.

        This method:
        1. Gets all registered worker IDs
        2. Stops each worker's queue thread
        3. Counts dropped tasks from each worker
        4. Removes workers from registry

        Note: Unlike instance removal, tasks are NOT rescheduled during
        scheduler shutdown. They are simply dropped since there are no
        workers left to handle them.

        Args:
            timeout: Total timeout for shutdown in seconds. Timeout is
                distributed across workers proportionally.

        Returns:
            ShutdownResult with summary of shutdown operation.
        """
        logger.info("Starting graceful shutdown of all workers...")

        result = ShutdownResult()
        start_time = time.time()

        worker_ids = self.worker_queue_manager.get_worker_ids()
        total_workers = len(worker_ids)

        if total_workers == 0:
            logger.info("No workers to shutdown")
            if self.on_shutdown_complete:
                await self.on_shutdown_complete(result)
            return result

        for i, worker_id in enumerate(worker_ids):
            # Calculate remaining time
            elapsed = time.time() - start_time
            remaining_time = timeout - elapsed

            if remaining_time <= 0:
                result.timeout_occurred = True
                logger.warning(
                    f"Shutdown timeout exceeded after {result.workers_stopped} workers"
                )
                break

            # Distribute remaining time across remaining workers
            remaining_workers = total_workers - i
            per_worker_timeout = remaining_time / remaining_workers

            try:
                # Stop worker and get pending tasks
                pending_tasks = self.worker_queue_manager.deregister_worker(
                    worker_id,
                    stop_timeout=per_worker_timeout,
                )

                result.tasks_dropped += len(pending_tasks)

                if pending_tasks:
                    logger.info(
                        f"Worker {worker_id}: dropped {len(pending_tasks)} pending tasks"
                    )

                # Remove from registry
                try:
                    await self.instance_registry.remove(worker_id)
                except Exception as e:
                    logger.error(f"Failed to remove {worker_id} from registry: {e}")

                result.workers_stopped += 1

            except Exception as e:
                logger.error(f"Error stopping worker {worker_id}: {e}")

        logger.info(
            f"Shutdown complete: {result.workers_stopped}/{total_workers} workers stopped, "
            f"{result.tasks_dropped} tasks dropped"
        )

        if self.on_shutdown_complete:
            await self.on_shutdown_complete(result)

        return result
