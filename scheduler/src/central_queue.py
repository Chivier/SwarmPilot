"""
Central task queue with backpressure control for the scheduler service.

This module provides a FIFO task queue that manages task dispatch to instances
with configurable high/low water marks to prevent instance overload.
"""

from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import asyncio
import time

from loguru import logger

from .model import TaskStatus
from .instance_registry import InstanceRegistry
from .task_registry import TaskRegistry


class QueuedTask:
    """Represents a task waiting in the central queue."""

    def __init__(
        self,
        task_id: str,
        model_id: str,
        task_input: Dict[str, Any],
        metadata: Dict[str, Any],
        enqueue_time: float,
    ):
        self.task_id = task_id
        self.model_id = model_id
        self.task_input = task_input
        self.metadata = metadata
        self.enqueue_time = enqueue_time


class CentralTaskQueue:
    """
    Central task queue with backpressure control.

    Features:
    - FIFO ordering by enqueue time
    - Event-driven dispatch: tasks dispatched when enqueued or when capacity becomes available
    - Backpressure: tasks wait in queue when all instances are at/above high water mark
    - Parallel dispatch with configurable concurrency
    """

    def __init__(
        self,
        task_registry: TaskRegistry,
        instance_registry: InstanceRegistry,
        high_water_mark: int = 10,
        low_water_mark: int = 5,
        max_concurrent_dispatch: int = 50,
    ):
        """
        Initialize the central task queue.

        Args:
            task_registry: Registry for task state management
            instance_registry: Registry for instance management
            high_water_mark: Maximum pending tasks per instance before stopping dispatch
            low_water_mark: Resume dispatching when pending tasks drop below this
            max_concurrent_dispatch: Maximum concurrent dispatch operations
        """
        self._task_registry = task_registry
        self._instance_registry = instance_registry
        self._high_water_mark = high_water_mark
        self._low_water_mark = low_water_mark
        self._max_concurrent_dispatch = max_concurrent_dispatch

        # FIFO queue for pending tasks
        self._queue: deque[QueuedTask] = deque()
        self._queue_lock = asyncio.Lock()

        # Event to signal dispatcher to wake up
        self._dispatch_event = asyncio.Event()

        # Dispatcher task handle
        self._dispatcher_task: Optional[asyncio.Task] = None

        # Shutdown flag
        self._shutdown = False

        # Semaphore for concurrent dispatch
        self._semaphore = asyncio.Semaphore(max_concurrent_dispatch)

        # Reference to scheduling strategy (set by api.py)
        self._scheduling_strategy = None

        # Reference to task dispatcher (set by api.py)
        self._task_dispatcher = None

        logger.info(
            f"CentralTaskQueue initialized with high_water_mark={high_water_mark}, "
            f"low_water_mark={low_water_mark}, max_concurrent={max_concurrent_dispatch}"
        )

    def set_scheduling_strategy(self, strategy) -> None:
        """Set the scheduling strategy to use for task assignment."""
        self._scheduling_strategy = strategy

    def set_task_dispatcher(self, dispatcher) -> None:
        """Set the task dispatcher for sending tasks to instances."""
        self._task_dispatcher = dispatcher

    async def start(self) -> None:
        """Start the background dispatcher loop."""
        if self._dispatcher_task is not None:
            logger.warning("Dispatcher already running")
            return

        self._shutdown = False
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())
        logger.info("Central queue dispatcher started")

    async def shutdown(self) -> None:
        """Shutdown the dispatcher gracefully."""
        logger.info("Shutting down central queue dispatcher...")
        self._shutdown = True
        self._dispatch_event.set()  # Wake up dispatcher

        if self._dispatcher_task:
            try:
                await asyncio.wait_for(self._dispatcher_task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Dispatcher shutdown timed out, cancelling")
                self._dispatcher_task.cancel()
                try:
                    await self._dispatcher_task
                except asyncio.CancelledError:
                    pass

        self._dispatcher_task = None
        logger.info("Central queue dispatcher shutdown complete")

    async def enqueue(
        self,
        task_id: str,
        model_id: str,
        task_input: Dict[str, Any],
        metadata: Dict[str, Any],
        enqueue_time: Optional[float] = None,
    ) -> int:
        """
        Add a task to the central queue.

        Args:
            task_id: Unique task identifier
            model_id: Model/tool to use
            task_input: Input data for the task
            metadata: Task metadata
            enqueue_time: Optional enqueue timestamp (defaults to current time)

        Returns:
            Queue position (1-based)
        """
        if enqueue_time is None:
            enqueue_time = time.time()

        queued_task = QueuedTask(
            task_id=task_id,
            model_id=model_id,
            task_input=task_input,
            metadata=metadata,
            enqueue_time=enqueue_time,
        )

        async with self._queue_lock:
            self._queue.append(queued_task)
            position = len(self._queue)

        logger.debug(f"Task {task_id} enqueued at position {position}")

        # Signal dispatcher to try dispatching
        self._dispatch_event.set()

        return position

    async def notify_capacity_available(self) -> None:
        """
        Notify the queue that an instance may have capacity available.

        Called by task_dispatcher when a task completes.
        """
        self._dispatch_event.set()

    async def get_queue_size(self) -> int:
        """Get the current number of tasks in the queue."""
        async with self._queue_lock:
            return len(self._queue)

    async def get_queue_info(self) -> Dict[str, Any]:
        """Get detailed queue information."""
        async with self._queue_lock:
            # Group tasks by model_id
            model_counts: Dict[str, int] = {}
            for task in self._queue:
                model_counts[task.model_id] = model_counts.get(task.model_id, 0) + 1

            return {
                "total_size": len(self._queue),
                "by_model": model_counts,
                "high_water_mark": self._high_water_mark,
                "low_water_mark": self._low_water_mark,
            }

    async def clear(self) -> int:
        """
        Clear all tasks from the central queue.

        This should be called as part of a full task clear operation to ensure
        consistency between the task registry and the queue.

        Returns:
            Count of tasks that were cleared from the queue
        """
        async with self._queue_lock:
            count = len(self._queue)
            self._queue.clear()

        if count > 0:
            logger.warning(f"Cleared {count} tasks from central queue")

        return count

    async def _dispatch_loop(self) -> None:
        """
        Background loop that dispatches tasks when instances are available.

        This loop:
        1. Waits for dispatch event (task enqueued or capacity available)
        2. Tries to dispatch tasks that have available instances
        3. Tasks for models with no available instances remain in queue
        """
        logger.info("Dispatch loop started")

        while not self._shutdown:
            try:
                # Wait for dispatch event with timeout
                try:
                    await asyncio.wait_for(self._dispatch_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Periodic check even without events
                    pass

                # Clear the event
                self._dispatch_event.clear()

                if self._shutdown:
                    break

                # Try to dispatch tasks
                await self._try_dispatch_tasks()

            except Exception as e:
                logger.error(f"[central_queue] Error in dispatch loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)  # Prevent tight loop on error

        logger.info("Dispatch loop stopped")

    async def _try_dispatch_tasks(self) -> None:
        """
        Try to dispatch tasks from the queue to available instances.

        Tasks are processed in FIFO order. If an instance is not available
        for a task's model, the task is skipped (remains in queue) and we
        try the next task.
        """
        if not self._scheduling_strategy or not self._task_dispatcher:
            logger.warning("Scheduling strategy or task dispatcher not set")
            return

        dispatched_count = 0
        skipped_tasks: List[QueuedTask] = []

        while True:
            # Get next task from queue
            async with self._queue_lock:
                if not self._queue:
                    break
                task = self._queue.popleft()

            # Check if any instance is available for this model
            has_capacity = await self._instance_registry.is_any_instance_available(
                task.model_id, self._high_water_mark
            )

            if not has_capacity:
                # No available instance, put task back
                skipped_tasks.append(task)
                continue

            # Try to dispatch this task
            success = await self._dispatch_single_task(task)

            if success:
                dispatched_count += 1
            else:
                # Dispatch failed (e.g., no instances), put task back
                skipped_tasks.append(task)

        # Put skipped tasks back at the front of the queue (maintain FIFO)
        if skipped_tasks:
            async with self._queue_lock:
                for task in reversed(skipped_tasks):
                    self._queue.appendleft(task)

        if dispatched_count > 0:
            logger.debug(f"Dispatched {dispatched_count} tasks, {len(skipped_tasks)} waiting")

    async def _dispatch_single_task(self, task: QueuedTask) -> bool:
        """
        Dispatch a single task to an instance.

        Args:
            task: The queued task to dispatch

        Returns:
            True if dispatch successful, False otherwise
        """
        async with self._semaphore:
            try:
                # Get available instances below high water mark
                available_instances = await self._instance_registry.get_instances_below_water_mark(
                    task.model_id, self._high_water_mark
                )

                if not available_instances:
                    logger.debug(f"No available instances for model {task.model_id}")
                    return False

                # Use scheduling strategy to select best instance
                try:
                    schedule_result = await self._scheduling_strategy.schedule_task(
                        model_id=task.model_id,
                        metadata=task.metadata,
                        available_instances=available_instances,
                    )
                except Exception as e:
                    logger.error(f"[central_queue] Scheduling failed for task {task.task_id}: {e}", exc_info=True)
                    return False

                if not schedule_result.selected_instance_id:
                    logger.warning(f"No instance selected for task {task.task_id}")
                    return False

                # Get the selected instance object
                selected_instance = None
                for inst in available_instances:
                    if inst.instance_id == schedule_result.selected_instance_id:
                        selected_instance = inst
                        break

                if not selected_instance:
                    logger.error(f"Selected instance {schedule_result.selected_instance_id} not found")
                    return False

                # Update task record with assignment info
                task_record = await self._task_registry.get(task.task_id)
                if task_record:
                    task_record.assigned_instance = selected_instance.instance_id
                    selected_pred = schedule_result.selected_prediction
                    if selected_pred:
                        task_record.predicted_time_ms = selected_pred.predicted_time_ms
                        task_record.predicted_error_margin_ms = selected_pred.error_margin_ms
                        task_record.predicted_quantiles = selected_pred.quantiles

                # Increment pending count on selected instance
                await self._instance_registry.increment_pending(selected_instance.instance_id)

                # Dispatch task to instance (fire and forget)
                self._task_dispatcher.dispatch_task_async(task.task_id)

                logger.debug(
                    f"Dispatched task {task.task_id} to instance {selected_instance.instance_id}"
                )
                return True

            except Exception as e:
                logger.error(f"[central_queue] Error dispatching task {task.task_id}: {e}", exc_info=True)
                return False
