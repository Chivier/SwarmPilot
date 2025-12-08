"""
Task queue management with FIFO processing
"""

import asyncio
import heapq
import time
import traceback
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

from loguru import logger


def log_error_with_traceback(
    error: Exception,
    context: str,
    additional_info: str = "",
) -> None:
    """
    Log error with detailed information and traceback.

    Args:
        error: The exception that occurred
        context: Context description of where the error occurred
        additional_info: Additional context information
    """
    tb_str = traceback.format_exc()
    if additional_info:
        logger.error(
            f"[TaskQueue] [{context}] Error occurred:\n"
            f"  Internal error: {type(error).__name__}: {error}\n"
            f"  {additional_info}\n"
            f"  Traceback:\n{tb_str}"
        )
    else:
        logger.error(
            f"[TaskQueue] [{context}] Error occurred:\n"
            f"  Internal error: {type(error).__name__}: {error}\n"
            f"  Traceback:\n{tb_str}"
        )

from .config import config
from .manager_factory import get_docker_manager
from .models import Task, TaskStatus
from .scheduler_client import get_scheduler_client


class TaskQueue:
    """
    Manages task queue with priority-based processing.

    Tasks are processed based on their enqueue_time (earliest first) using a min-heap.
    """

    def __init__(self):
        self.tasks: Dict[str, Task] = {}  # All tasks by task_id
        self.queue: List[Tuple[float, str]] = []  # Priority queue: (enqueue_time, task_id)
        self._insertion_order: deque = deque()  # Track insertion order for O(1) LIFO fetch
        self._queue_lock: asyncio.Lock = asyncio.Lock()  # Ensure thread-safe queue operations
        self.current_task_id: Optional[str] = None
        self._detached_task_id: Optional[str] = None  # Task detached during deregister timeout
        self.is_processing = False
        self._processing_task: Optional[asyncio.Task] = None

    async def submit_task(self, task: Task, enqueue_time: Optional[float] = None) -> int:
        """
        Submit a new task to the queue.

        Args:
            task: Task object to submit
            enqueue_time: Optional Unix timestamp for task priority. If not provided,
                         uses current time for backward compatibility (FIFO behavior)

        Returns:
            Position in queue (1-indexed)

        Raises:
            ValueError: If task_id already exists
        """
        if task.task_id in self.tasks:
            raise ValueError(f"Task with ID {task.task_id} already exists")

        # Set enqueue_time if provided, otherwise use current time
        if enqueue_time is not None:
            task.enqueue_time = enqueue_time
        elif not hasattr(task, 'enqueue_time') or task.enqueue_time is None:
            task.enqueue_time = time.time()

        # Add to storage and priority queue
        self.tasks[task.task_id] = task

        async with self._queue_lock:
            heapq.heappush(self.queue, (task.enqueue_time, task.task_id))
            self._insertion_order.append(task.task_id)  # Track for LIFO fetch
            queue_size = len(self.queue)

        logger.info(
            f"Task {task.task_id} submitted with enqueue_time={task.enqueue_time:.3f}, "
            f"queue position: {queue_size}, callback_url: {task.callback_url}"
        )

        # Start processing if not already running
        if not self.is_processing:
            self._processing_task = asyncio.create_task(self._process_queue())

        # Return position in queue
        return queue_size

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)

    async def detach_current_task(self) -> Optional[str]:
        """
        Detach the currently running task from the queue tracking.

        This allows deregister operations to proceed without waiting for the
        task to complete. The detached task will continue running in the
        background and will still send its callback when finished.

        Returns:
            The task_id that was detached, or None if no task was running.

        Note:
            - The task continues to execute in _process_queue()
            - When the task completes, _detached_task_id is cleared instead of current_task_id
            - The callback is still sent even after the instance is deregistered
        """
        if self.current_task_id is None:
            return None

        detached_id = self.current_task_id
        self._detached_task_id = detached_id
        self.current_task_id = None

        logger.info(f"Task {detached_id} detached - continuing in background")
        return detached_id

    async def fetch_task(self) -> Optional[Task]:
        """
        Fetch the newest queued task (highest enqueue_time) from the queue tail.

        The task is completely removed from this instance (not just marked).
        Its callback will NOT be executed by this instance.

        Returns:
            Task object if a queued task was found, None otherwise.

        Note:
            - Only QUEUED tasks are considered for fetching
            - RUNNING, COMPLETED, FAILED tasks are skipped
            - The fetched task is removed from self.tasks (not kept)
            - Fetching from tail (newest) - LIFO order for work stealing
            - Uses insertion order tracking for O(1) best-case LIFO access
            - Always keeps at least one task (RUNNING or QUEUED) in the instance
        """
        async with self._queue_lock:
            if not self._insertion_order:
                return None

            # Count active tasks (RUNNING + QUEUED) to ensure at least one remains
            running_count = sum(
                1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING
            )
            queued_count = sum(
                1 for t in self.tasks.values() if t.status == TaskStatus.QUEUED
            )
            active_count = running_count + queued_count

            # Must keep at least one active task in the instance
            if active_count <= 1:
                logger.debug(
                    f"Cannot fetch: must keep at least 1 active task "
                    f"(running={running_count}, queued={queued_count})"
                )
                return None

            # Iterate from the tail (newest) to find a QUEUED task - O(1) best case
            found_task: Optional[Task] = None
            skipped_task_ids: list = []

            while self._insertion_order:
                task_id = self._insertion_order.pop()  # Pop from right (newest)
                task = self.tasks.get(task_id)

                if task and task.status == TaskStatus.QUEUED:
                    found_task = task
                    break
                # Track skipped task_ids to restore them if needed
                skipped_task_ids.append(task_id)

            # Restore skipped task_ids back to insertion order (append in original order)
            for tid in reversed(skipped_task_ids):
                self._insertion_order.append(tid)

            if found_task is None:
                return None

            # Remove from the min-heap (need to rebuild)
            self.queue = [
                (t, tid) for t, tid in self.queue if tid != found_task.task_id
            ]
            heapq.heapify(self.queue)

            # Remove task from storage completely (not just mark as FETCHED)
            del self.tasks[found_task.task_id]

            logger.info(
                f"Task {found_task.task_id} fetched and removed from instance "
                f"(enqueue_time={found_task.enqueue_time:.3f}, "
                f"remaining: running={running_count}, queued={queued_count - 1})"
            )

            return found_task

    async def list_tasks(
        self,
        status_filter: Optional[TaskStatus] = None,
        limit: Optional[int] = None
    ) -> List[Task]:
        """
        List all tasks with optional filtering.

        Args:
            status_filter: Filter by task status
            limit: Maximum number of tasks to return

        Returns:
            List of tasks
        """
        tasks = list(self.tasks.values())

        # Apply status filter
        if status_filter:
            tasks = [t for t in tasks if t.status == status_filter]

        # Sort by submission time (most recent first)
        tasks.sort(key=lambda t: t.submitted_at, reverse=True)

        # Apply limit
        if limit:
            tasks = tasks[:limit]

        return tasks

    async def delete_task(self, task_id: str) -> bool:
        """
        Delete a task.

        Can only delete queued, completed, or failed tasks.
        Cannot delete running tasks.

        Args:
            task_id: Task ID to delete

        Returns:
            True if deleted, False if not found

        Raises:
            ValueError: If task is currently running
        """
        task = self.tasks.get(task_id)
        if not task:
            return False

        if task.status == TaskStatus.RUNNING:
            raise ValueError("Cannot delete a running task")

        # Remove from queue if queued
        if task.status == TaskStatus.QUEUED:
            async with self._queue_lock:
                # Remove task from priority queue
                self.queue = [(t, tid) for t, tid in self.queue if tid != task_id]
                heapq.heapify(self.queue)

        # Remove from storage
        del self.tasks[task_id]
        logger.info(f"Task {task_id} deleted")
        return True

    async def get_queue_stats(self) -> Dict[str, int]:
        """Get task queue statistics"""
        stats = {
            "total": len(self.tasks),
            "queued": sum(1 for t in self.tasks.values() if t.status == TaskStatus.QUEUED),
            "running": sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING),
            "completed": sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED),
            "fetched": sum(1 for t in self.tasks.values() if t.status == TaskStatus.FETCHED),
        }
        return stats

    async def cleanup_fetched_tasks(self) -> int:
        """
        Remove all FETCHED tasks from storage to free memory.

        FETCHED tasks are tasks that were stolen by work-stealing mechanism
        and are no longer needed in this instance.

        Returns:
            Number of tasks cleaned up
        """
        fetched_task_ids = [
            task_id for task_id, task in self.tasks.items()
            if task.status == TaskStatus.FETCHED
        ]

        for task_id in fetched_task_ids:
            del self.tasks[task_id]

        if fetched_task_ids:
            logger.info(f"Cleaned up {len(fetched_task_ids)} fetched tasks from storage")

        return len(fetched_task_ids)

    async def _process_queue(self):
        """
        Process tasks from the priority queue sequentially.

        This runs as a background task and processes tasks one by one,
        prioritizing tasks with the earliest enqueue_time.
        """
        self.is_processing = True
        logger.info("Task queue processing started")

        try:
            while True:
                # Get next task from priority queue
                async with self._queue_lock:
                    if not self.queue:
                        break
                    enqueue_time, task_id = heapq.heappop(self.queue)

                task = self.tasks.get(task_id)

                if not task:
                    logger.error(f"Task {task_id} not found in storage")
                    continue

                self.current_task_id = task_id
                logger.info(
                    f"Processing task {task_id} (enqueue_time={enqueue_time:.3f}, "
                    f"wait_time={time.time() - enqueue_time:.3f}s)"
                )

                try:
                    await self._execute_task(task)
                except Exception as e:
                    log_error_with_traceback(
                        error=e,
                        context=f"_process_queue({task_id})",
                        additional_info=f"Model: {task.model_id}",
                    )
                    task.mark_failed(str(e))

                # Handle task completion: check if task was detached during execution
                if self._detached_task_id == task_id:
                    # Task was detached (deregister timeout) - clear detached state
                    self._detached_task_id = None
                    logger.info(f"Detached task {task_id} completed in background")
                else:
                    # Normal case - clear current task
                    self.current_task_id = None

        finally:
            self.is_processing = False
            logger.info("Task queue processing stopped")

    async def _execute_task(self, task: Task):
        """
        Execute a single task.

        Args:
            task: Task to execute
        """
        # Mark task as started
        task.mark_started()
        logger.info(f"Task {task.task_id} started")

        # Track execution time
        start_time = time.time()
        execution_time_ms = None

        try:
            # Get docker manager
            docker_manager = get_docker_manager()

            # Check if model is running
            if not await docker_manager.is_model_running():
                raise RuntimeError("No model is currently running")

            # Check if model matches
            current_model = await docker_manager.get_current_model()
            if current_model and current_model.model_id != task.model_id:
                raise RuntimeError(
                    f"Model mismatch: task requires {task.model_id}, "
                    f"but {current_model.model_id} is running"
                )

            # Invoke inference
            result = await docker_manager.invoke_inference(task.task_input)

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Mark task as completed
            task.mark_completed(result)
            logger.info(f"Task {task.task_id} completed successfully in {execution_time_ms:.2f}ms")

            # Send callback if URL is provided
            await self._send_callback(
                task_id=task.task_id,
                status="completed",
                result=result,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            # Calculate execution time even for failed tasks
            execution_time_ms = (time.time() - start_time) * 1000

            # Mark task as failed
            error_msg = str(e)
            task.mark_failed(error_msg)
            log_error_with_traceback(
                error=e,
                context=f"_execute_task({task.task_id})",
                additional_info=f"Model: {task.model_id}, Execution time: {execution_time_ms:.2f}ms",
            )

            # Send callback if URL is provided
            await self._send_callback(
                task_id=task.task_id,
                status="failed",
                error=error_msg,
                execution_time_ms=execution_time_ms,
            )

    async def _send_callback(
        self,
        task_id: str,
        status: str,
        result: Optional[Dict] = None,
        error: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
    ):
        """
        Send task result callback to scheduler.

        Tries WebSocket first, falls back to HTTP if WebSocket unavailable.

        Args:
            task_id: ID of the completed task
            status: Task status ("completed" or "failed")
            result: Task result (if completed)
            error: Error message (if failed)
            execution_time_ms: Execution time in milliseconds
        """
        # Try WebSocket first
        from .websocket_client_singleton import get_websocket_client

        ws_client = get_websocket_client()
        websocket_success = False

        if ws_client and ws_client.is_connected():
            try:
                websocket_success = await ws_client.send_task_result(
                    task_id=task_id,
                    status=status,
                    result=result,
                    error=error,
                    execution_time_ms=execution_time_ms,
                )
                if websocket_success:
                    logger.info(f"Task result sent via WebSocket for {task_id}")
                    return
            except Exception as e:
                tb_str = traceback.format_exc()
                logger.warning(
                    f"WebSocket callback failed for {task_id}, falling back to HTTP:\n"
                    f"  Error: {type(e).__name__}: {e}\n"
                    f"  Traceback:\n{tb_str}"
                )

        # Fallback to HTTP
        scheduler_client = get_scheduler_client()
        try:
            task = self.tasks.get(task_id)
            success = await scheduler_client.send_task_result(
                task_id=task_id,
                status=status,
                callback_url=getattr(task, "callback_url", None),
                result=result,
                error=error,
                execution_time_ms=execution_time_ms,
            )
            if success:
                logger.info(f"Task result sent via HTTP for {task_id}")
            else:
                logger.warning(f"HTTP callback failed for task {task_id}")
        except Exception as e:
            log_error_with_traceback(
                error=e,
                context=f"_send_callback({task_id})/http",
                additional_info=f"Status: {status}",
            )

    async def clear_all_tasks(self, force: bool = True) -> Dict[str, int]:
        """
        Clear all tasks from the queue and task storage.

        This will remove all tasks regardless of their status (queued, running,
        completed, or failed).

        Args:
            force: If True (default), clear all tasks including running ones.
                   If False, raise RuntimeError when there are running tasks.

        Returns:
            Dictionary with counts of cleared tasks by status

        Raises:
            RuntimeError: If there are currently running tasks and force=False
        """
        # Get stats before clearing
        stats = await self.get_queue_stats()

        # Check if there are running tasks (unless force mode)
        if not force and stats["running"] > 0:
            raise RuntimeError(
                f"Cannot clear tasks while {stats['running']} task(s) are running. "
                "Wait for running tasks to complete or stop processing first."
            )

        # Clear the queue and insertion order tracking
        async with self._queue_lock:
            self.queue.clear()
            self._insertion_order.clear()

        # Clear all tasks
        cleared_count = {
            "queued": stats["queued"],
            "completed": stats["completed"],
            "failed": stats["failed"],
            "fetched": stats["fetched"],
            "total": stats["total"]
        }

        self.tasks.clear()

        logger.info(
            f"Cleared all tasks: {cleared_count['total']} total "
            f"(queued: {cleared_count['queued']}, "
            f"completed: {cleared_count['completed']}, "
            f"failed: {cleared_count['failed']}, "
            f"fetched: {cleared_count['fetched']})"
        )

        return cleared_count

    async def peek_pending_tasks(self) -> List[Dict[str, Any]]:
        """
        Get all pending (QUEUED) tasks info WITHOUT removing them from the queue.

        Use this to get task info for redistribution, then call remove_task()
        for each successfully resubmitted task.

        Returns:
            List of task dictionaries containing task metadata.
        """
        pending_tasks = []

        async with self._queue_lock:
            for task in self.tasks.values():
                if task.status == TaskStatus.QUEUED:
                    task_info = {
                        "task_id": task.task_id,
                        "model_id": task.model_id,
                        "task_input": task.task_input,
                        "enqueue_time": task.enqueue_time,
                        "submitted_at": task.submitted_at,
                        "callback_url": getattr(task, "callback_url", None),
                        "metadata": getattr(task, "metadata", None),
                    }
                    pending_tasks.append(task_info)

        logger.debug(f"Peeked {len(pending_tasks)} pending tasks")
        return pending_tasks

    async def remove_task(self, task_id: str) -> bool:
        """
        Remove a specific task from the queue and task storage.

        Args:
            task_id: ID of the task to remove

        Returns:
            True if task was removed, False if not found
        """
        async with self._queue_lock:
            if task_id not in self.tasks:
                return False

            # Remove from task storage
            del self.tasks[task_id]

            # Remove from priority queue
            self.queue = [(t, tid) for t, tid in self.queue if tid != task_id]
            heapq.heapify(self.queue)

            # Remove from insertion order
            self._insertion_order = deque(
                tid for tid in self._insertion_order if tid != task_id
            )

            logger.debug(f"Removed task {task_id} from local queue")
            return True

    async def extract_pending_tasks(self) -> List[Dict[str, Any]]:
        """
        Extract all pending (QUEUED) tasks from the queue for redistribution.

        This method is used during instance deregistration to return queued tasks
        to the scheduler for reassignment to other instances.

        Returns:
            List of task dictionaries containing task metadata for redistribution.
            Each dictionary includes: task_id, model_id, task_input, enqueue_time,
            submitted_at, callback_url (if available), and metadata (if available).

        Note:
            - QUEUED tasks are extracted for redistribution
            - FETCHED tasks are removed without redistribution (already picked up)
            - RUNNING tasks are preserved in the queue
            - This operation is thread-safe (uses queue lock)
        """
        extracted_tasks = []
        extracted_task_ids = set()

        fetched_task_ids = set()

        async with self._queue_lock:
            # Extract all queued tasks from priority queue
            queued_items = []
            while self.queue:
                enqueue_time, task_id = heapq.heappop(self.queue)
                task = self.tasks.get(task_id)

                if task and task.status == TaskStatus.QUEUED:
                    # Build task info for redistribution
                    task_info = {
                        "task_id": task.task_id,
                        "model_id": task.model_id,
                        "task_input": task.task_input,
                        "enqueue_time": task.enqueue_time,
                        "submitted_at": task.submitted_at,
                        "callback_url": getattr(task, "callback_url", None),
                        "metadata": getattr(task, "metadata", None),
                    }
                    extracted_tasks.append(task_info)
                    extracted_task_ids.add(task_id)

                    # Remove from task storage
                    del self.tasks[task_id]
                elif task and task.status == TaskStatus.FETCHED:
                    # FETCHED tasks: already picked up, remove without redistribution
                    fetched_task_ids.add(task_id)
                    del self.tasks[task_id]
                    logger.debug(f"Removing fetched task {task_id} without redistribution")
                elif task and task.status == TaskStatus.RUNNING:
                    # Preserve running tasks - add back to queue
                    queued_items.append((enqueue_time, task_id))
                    logger.debug(f"Preserving running task {task_id} during extraction")

            # Restore non-extracted tasks back to queue
            for item in queued_items:
                heapq.heappush(self.queue, item)

            # Remove extracted and fetched tasks from insertion order tracking
            removed_task_ids = extracted_task_ids | fetched_task_ids
            self._insertion_order = deque(
                tid for tid in self._insertion_order if tid not in removed_task_ids
            )

        logger.info(
            f"Extracted {len(extracted_tasks)} pending tasks, "
            f"removed {len(fetched_task_ids)} fetched tasks. "
            f"Remaining tasks in queue: {len(self.queue)}"
        )

        return extracted_tasks

    async def get_current_task_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the currently executing task.

        Returns:
            Dictionary containing task_id and estimated_completion_ms,
            or None if no task is currently running.

        Note:
            The estimated_completion_ms is calculated based on the task's
            start time and a simple heuristic (assuming average task duration).
        """
        if not self.current_task_id:
            return None

        task = self.tasks.get(self.current_task_id)
        if not task or task.status != TaskStatus.RUNNING:
            return None

        # Calculate estimated completion time
        estimated_completion_ms = None
        if task.started_at:
            try:
                from datetime import datetime, UTC
                started_time = datetime.fromisoformat(task.started_at.replace("Z", "+00:00"))
                elapsed_ms = (datetime.now(UTC) - started_time).total_seconds() * 1000

                # Simple heuristic: assume average task takes 5 seconds
                # In production, this could use historical data
                avg_task_duration_ms = 5000
                estimated_remaining_ms = max(0, avg_task_duration_ms - elapsed_ms)
                estimated_completion_ms = estimated_remaining_ms
            except Exception as e:
                tb_str = traceback.format_exc()
                logger.warning(
                    f"Failed to calculate estimated completion time:\n"
                    f"  Error: {type(e).__name__}: {e}\n"
                    f"  Traceback:\n{tb_str}"
                )

        return {
            "task_id": self.current_task_id,
            "estimated_completion_ms": estimated_completion_ms,
        }

    async def stop_processing(self):
        """Stop queue processing (graceful shutdown)"""
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass


# Global task queue instance
_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """Get or create the global task queue instance"""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
    return _task_queue
