"""Worker queue manager for coordinating all worker threads.

This module implements PYLET-017: Worker Queue Manager, providing central
coordination for all WorkerQueueThread instances.

Key features:
- Creates threads when workers register
- Destroys threads when workers deregister
- Routes tasks to the correct worker thread
- Provides queue depth information for scheduling decisions
- Handles task redistribution on worker removal
"""

import threading
from collections.abc import Callable

from loguru import logger

from swarmpilot.scheduler.services.worker_queue_thread import (
    QueuedTask,
    WorkerQueueThread,
)


class WorkerQueueManager:
    """Manages WorkerQueueThreads for all registered workers.

    This class is the central coordinator for all worker queue threads:
    1. Creates threads when workers register
    2. Destroys threads when workers deregister
    3. Routes tasks to the correct worker thread
    4. Provides queue depth information for scheduling decisions
    5. Handles task redistribution on worker removal

    Thread Safety:
    - Uses a threading.Lock for worker dictionary access
    - Individual WorkerQueueThread instances are thread-safe

    Example:
        ```python
        manager = WorkerQueueManager(
            callback=task_result_callback.create_thread_callback(loop),
            http_timeout=300.0,
        )

        # Register workers
        manager.register_worker("worker-1", "http://localhost:8001", "gpt-4")

        # Enqueue tasks
        manager.enqueue_task("worker-1", task)

        # Query queue state
        depths = manager.get_all_queue_depths()

        # Shutdown
        pending = manager.shutdown()
        ```
    """

    def __init__(
        self,
        callback: Callable,
        http_timeout: float = 300.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        on_task_started: Callable[[str], None] | None = None,
    ):
        """Initialize worker queue manager.

        Args:
            callback: Thread-safe callback for task results.
            http_timeout: HTTP timeout for worker requests.
            max_retries: Max retries for transient errors.
            retry_delay: Initial retry delay (exponential backoff).
            on_task_started: Optional callback when task begins execution.
        """
        self._callback = callback
        self._http_timeout = http_timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._on_task_started = on_task_started

        # Worker ID -> WorkerQueueThread mapping
        self._workers: dict[str, WorkerQueueThread] = {}
        self._lock = threading.Lock()

    def register_worker(
        self,
        worker_id: str,
        worker_endpoint: str,
        model_id: str,
    ) -> None:
        """Register a new worker and start its queue thread.

        Args:
            worker_id: Unique identifier for the worker.
            worker_endpoint: HTTP endpoint for worker's model API.
            model_id: Model ID running on this worker.

        Raises:
            ValueError: If worker is already registered.
        """
        with self._lock:
            if worker_id in self._workers:
                raise ValueError(f"Worker {worker_id} already registered")

            thread = WorkerQueueThread(
                worker_id=worker_id,
                worker_endpoint=worker_endpoint,
                model_id=model_id,
                callback=self._callback,
                http_timeout=self._http_timeout,
                max_retries=self._max_retries,
                retry_delay=self._retry_delay,
                on_task_started=self._on_task_started,
            )
            thread.start()
            self._workers[worker_id] = thread

            logger.info(
                f"Registered worker {worker_id} at {worker_endpoint} "
                f"(model: {model_id})"
            )

    def deregister_worker(
        self,
        worker_id: str,
        stop_timeout: float = 10.0,
    ) -> list[QueuedTask]:
        """Deregister a worker and stop its queue thread.

        Args:
            worker_id: Worker to deregister.
            stop_timeout: Max time to wait for thread to stop.

        Returns:
            List of pending tasks that were in the worker's queue.
        """
        with self._lock:
            thread = self._workers.pop(worker_id, None)

        if thread is None:
            logger.warning(f"Worker {worker_id} not found for deregistration")
            return []

        pending_tasks = thread.stop(timeout=stop_timeout)

        logger.info(
            f"Deregistered worker {worker_id}, "
            f"returned {len(pending_tasks)} pending tasks"
        )

        return pending_tasks

    def get_worker(self, worker_id: str) -> WorkerQueueThread | None:
        """Get the WorkerQueueThread for a specific worker.

        Args:
            worker_id: Worker to look up.

        Returns:
            WorkerQueueThread or None if not found.
        """
        with self._lock:
            return self._workers.get(worker_id)

    def has_worker(self, worker_id: str) -> bool:
        """Check if a worker is registered.

        Args:
            worker_id: Worker to check.

        Returns:
            True if worker is registered.
        """
        with self._lock:
            return worker_id in self._workers

    def enqueue_task(self, worker_id: str, task: QueuedTask) -> int:
        """Enqueue a task to a specific worker.

        Args:
            worker_id: Worker to receive the task.
            task: Task to enqueue.

        Returns:
            Queue size after enqueue.

        Raises:
            ValueError: If worker is not registered.
        """
        thread = self.get_worker(worker_id)
        if thread is None:
            raise ValueError(f"Worker {worker_id} not registered")

        return thread.enqueue(task)

    def enqueue_priority_task(self, worker_id: str, task: QueuedTask) -> None:
        """Enqueue a lightweight task to the front of a worker's queue.

        Priority tasks skip ahead of normal tasks and do not affect
        queue depth or estimated wait time calculations.

        Args:
            worker_id: Worker to receive the task.
            task: Task to enqueue with priority.

        Raises:
            ValueError: If worker is not registered.
        """
        thread = self.get_worker(worker_id)
        if thread is None:
            raise ValueError(f"Worker {worker_id} not registered")

        thread.enqueue_priority(task)

    def get_queue_depth(self, worker_id: str) -> int:
        """Get the current queue depth for a worker.

        Args:
            worker_id: Worker to query.

        Returns:
            Queue depth (0 if worker not found).
        """
        thread = self.get_worker(worker_id)
        return thread.queue_size() if thread else 0

    def get_all_queue_depths(self) -> dict[str, int]:
        """Get queue depths for all registered workers.

        Returns:
            Dictionary mapping worker_id to queue depth.
        """
        with self._lock:
            workers = list(self._workers.items())

        return {worker_id: thread.queue_size() for worker_id, thread in workers}

    def get_estimated_wait_times(
        self,
        avg_exec_time_ms: float,
    ) -> dict[str, float]:
        """Get estimated wait times for all workers.

        Args:
            avg_exec_time_ms: Average execution time per task.

        Returns:
            Dictionary mapping worker_id to estimated wait time (ms).
        """
        with self._lock:
            workers = list(self._workers.items())

        return {
            worker_id: thread.get_estimated_wait_time(avg_exec_time_ms)
            for worker_id, thread in workers
        }

    def get_worker_count(self) -> int:
        """Get the number of registered workers.

        Returns:
            Number of registered workers.
        """
        with self._lock:
            return len(self._workers)

    def get_worker_ids(self) -> list[str]:
        """Get list of all registered worker IDs.

        Returns:
            List of worker IDs.
        """
        with self._lock:
            return list(self._workers.keys())

    def get_total_queue_depth(self) -> int:
        """Get total queue depth across all workers.

        Returns:
            Sum of all worker queue depths.
        """
        return sum(self.get_all_queue_depths().values())

    def shutdown(self, timeout: float = 30.0) -> list[QueuedTask]:
        """Shutdown all workers and return pending tasks.

        Args:
            timeout: Max time to wait for all threads to stop.

        Returns:
            Combined list of pending tasks from all workers.
        """
        logger.info("Shutting down WorkerQueueManager...")

        with self._lock:
            worker_ids = list(self._workers.keys())

        all_pending = []
        per_worker_timeout = timeout / max(len(worker_ids), 1)

        for worker_id in worker_ids:
            pending = self.deregister_worker(
                worker_id,
                stop_timeout=per_worker_timeout,
            )
            all_pending.extend(pending)

        logger.info(
            f"WorkerQueueManager shutdown complete, "
            f"{len(all_pending)} total pending tasks"
        )

        return all_pending

    def get_stats(self) -> dict:
        """Get statistics about all workers.

        Returns:
            Dictionary with worker statistics.
        """
        with self._lock:
            workers = list(self._workers.items())

        stats = {
            "total_workers": len(workers),
            "total_queue_depth": 0,
            "workers": {},
        }

        for worker_id, thread in workers:
            queue_depth = thread.queue_size()
            has_running = thread.has_running_task()

            stats["workers"][worker_id] = {
                "queue_depth": queue_depth,
                "has_running_task": has_running,
                "current_task_id": thread.get_current_task_id(),
            }
            stats["total_queue_depth"] += queue_depth

        return stats
