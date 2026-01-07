"""Central task queue for the scheduler service.

This module provides a FIFO task queue that manages task dispatch to instances.
"""

import asyncio
import time
from collections import deque
from contextlib import suppress
from typing import Any

from loguru import logger

from src.registry.instance_registry import InstanceRegistry
from src.registry.task_registry import TaskRegistry


class QueuedTask:
    """Represents a task waiting in the central queue."""

    def __init__(
        self,
        task_id: str,
        model_id: str,
        task_input: dict[str, Any],
        metadata: dict[str, Any],
        enqueue_time: float,
        generation: int = 0,
    ):
        self.task_id = task_id
        self.model_id = model_id
        self.task_input = task_input
        self.metadata = metadata
        self.enqueue_time = enqueue_time
        # Generation counter to track which clear cycle this task belongs to
        self.generation = generation


class CentralTaskQueue:
    """Central task queue for task dispatch.

    Features:
    - FIFO ordering by enqueue time
    - Event-driven dispatch: tasks dispatched when enqueued or when capacity becomes available
    - Parallel dispatch with configurable concurrency
    """

    def __init__(
        self,
        task_registry: TaskRegistry,
        instance_registry: InstanceRegistry,
        max_concurrent_dispatch: int = 50,
    ):
        """Initialize the central task queue.

        Args:
            task_registry: Registry for task state management
            instance_registry: Registry for instance management
            max_concurrent_dispatch: Maximum concurrent dispatch operations
        """
        self._task_registry = task_registry
        self._instance_registry = instance_registry
        self._max_concurrent_dispatch = max_concurrent_dispatch

        # FIFO queue for pending tasks
        self._queue: deque[QueuedTask] = deque()
        self._queue_lock = asyncio.Lock()

        # Event to signal dispatcher to wake up
        self._dispatch_event = asyncio.Event()

        # Dispatcher task handle
        self._dispatcher_task: asyncio.Task | None = None

        # Shutdown flag
        self._shutdown = False

        # Semaphore for concurrent dispatch
        self._semaphore = asyncio.Semaphore(max_concurrent_dispatch)

        # Reference to scheduling strategy (set by api.py)
        self._scheduling_strategy = None

        # Reference to task dispatcher (set by api.py)
        self._task_dispatcher = None

        # Generation counter to invalidate stale tasks after clear
        # Increments on each clear() call, used to prevent race conditions
        # where tasks popped before clear get re-added after clear
        self._generation = 0

        logger.info(
            f"CentralTaskQueue initialized with max_concurrent={max_concurrent_dispatch}"
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
            except TimeoutError:
                logger.warning("Dispatcher shutdown timed out, cancelling")
                self._dispatcher_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._dispatcher_task

        self._dispatcher_task = None
        logger.info("Central queue dispatcher shutdown complete")

    async def enqueue(
        self,
        task_id: str,
        model_id: str,
        task_input: dict[str, Any],
        metadata: dict[str, Any],
        enqueue_time: float | None = None,
    ) -> int:
        """Add a task to the central queue.

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

        async with self._queue_lock:
            current_generation = self._generation
            queued_task = QueuedTask(
                task_id=task_id,
                model_id=model_id,
                task_input=task_input,
                metadata=metadata,
                enqueue_time=enqueue_time,
                generation=current_generation,
            )
            self._queue.append(queued_task)
            position = len(self._queue)

        logger.info(
            f"[QUEUE_ENQUEUE] task_id={task_id} model_id={model_id} "
            f"position={position} queue_size={len(self._queue)}"
        )

        # Signal dispatcher to try dispatching
        self._dispatch_event.set()

        return position

    async def notify_capacity_available(self) -> None:
        """Notify the queue that an instance may have capacity available.

        Called by task_dispatcher when a task completes.
        """
        self._dispatch_event.set()

    async def get_queue_size(self) -> int:
        """Get the current number of tasks in the queue."""
        async with self._queue_lock:
            return len(self._queue)

    async def get_queue_info(self) -> dict[str, Any]:
        """Get detailed queue information."""
        async with self._queue_lock:
            # Group tasks by model_id
            model_counts: dict[str, int] = {}
            for task in self._queue:
                model_counts[task.model_id] = (
                    model_counts.get(task.model_id, 0) + 1
                )

            return {
                "total_size": len(self._queue),
                "by_model": model_counts,
            }

    async def clear(self) -> int:
        """Clear all tasks from the central queue.

        This should be called as part of a full task clear operation to ensure
        consistency between the task registry and the queue.

        This method also increments the generation counter to invalidate any
        tasks that were popped from the queue before the clear and might be
        re-added by _try_dispatch_tasks() after the clear completes.

        Returns:
            Count of tasks that were cleared from the queue
        """
        async with self._queue_lock:
            count = len(self._queue)
            self._queue.clear()
            # Increment generation to invalidate any tasks that were popped
            # before clear and might be re-added after clear completes
            self._generation += 1
            logger.info(f"Queue generation incremented to {self._generation}")

        if count > 0:
            logger.warning(f"Cleared {count} tasks from central queue")

        return count

    async def _dispatch_loop(self) -> None:
        """Background loop that dispatches tasks when instances are available.

        This loop:
        1. Waits for dispatch event (task enqueued or capacity available)
        2. Tries to dispatch tasks that have available instances
        3. Tasks for models with no available instances remain in queue
        """
        logger.info("Dispatch loop started")

        while not self._shutdown:
            try:
                # Wait for dispatch event with timeout (periodic check)
                with suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._dispatch_event.wait(), timeout=1.0
                    )

                # Clear the event
                self._dispatch_event.clear()

                if self._shutdown:
                    break

                # Try to dispatch tasks
                await self._try_dispatch_tasks()

            except Exception as e:
                logger.error(
                    f"[central_queue] Error in dispatch loop: {e}",
                    exc_info=True,
                )
                await asyncio.sleep(1.0)  # Prevent tight loop on error

        logger.info("Dispatch loop stopped")

    async def _try_dispatch_tasks(self) -> None:
        """Try to dispatch tasks from the queue to available instances.

        Tasks are processed in FIFO order. If an instance is not available
        for a task's model, the task is skipped (remains in queue) and we
        try the next task.

        Uses generation counter to prevent race conditions with clear():
        - Tasks are tagged with current generation when enqueued
        - If clear() is called while tasks are being processed, generation increments
        - Skipped tasks are only re-queued if their generation matches current generation
        - This prevents stale tasks from being re-added after a clear operation
        """
        if not self._scheduling_strategy or not self._task_dispatcher:
            logger.warning("Scheduling strategy or task dispatcher not set")
            return

        dispatched_count = 0
        skipped_tasks: list[QueuedTask] = []

        while True:
            # Get next task from queue
            async with self._queue_lock:
                if not self._queue:
                    break
                task = self._queue.popleft()

            # Check if any ACTIVE instance exists for this model
            has_instance = await self._instance_registry.has_active_instance(
                task.model_id
            )

            if not has_instance:
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
        # Only re-queue tasks that still have valid generation (not invalidated by clear())
        if skipped_tasks:
            async with self._queue_lock:
                current_generation = self._generation
                valid_tasks = []
                discarded_count = 0

                for task in skipped_tasks:
                    if task.generation == current_generation:
                        valid_tasks.append(task)
                    else:
                        # Task was invalidated by a clear() call
                        discarded_count += 1

                if discarded_count > 0:
                    logger.warning(
                        f"Discarded {discarded_count} stale tasks from previous generation "
                        f"(current generation: {current_generation})"
                    )

                # Re-queue only valid tasks (maintain FIFO order)
                for task in reversed(valid_tasks):
                    self._queue.appendleft(task)

        if dispatched_count > 0:
            logger.debug(
                f"Dispatched {dispatched_count} tasks, {len(skipped_tasks)} waiting"
            )

    async def _dispatch_single_task(self, task: QueuedTask) -> bool:
        """Dispatch a single task to an instance.

        Args:
            task: The queued task to dispatch

        Returns:
            True if dispatch successful, False otherwise
        """
        async with self._semaphore:
            try:
                # Check if task was invalidated by a clear() call before processing
                async with self._queue_lock:
                    if task.generation != self._generation:
                        logger.debug(
                            f"Task {task.task_id} has stale generation {task.generation}, "
                            f"current is {self._generation}, skipping dispatch"
                        )
                        return (
                            True  # Return True to not re-queue this stale task
                        )

                # Get all ACTIVE instances for this model
                available_instances = (
                    await self._instance_registry.get_active_instances(
                        task.model_id
                    )
                )

                if not available_instances:
                    logger.debug(
                        f"No available instances for model {task.model_id}"
                    )
                    return False

                # Use scheduling strategy to select best instance
                try:
                    schedule_result = (
                        await self._scheduling_strategy.schedule_task(
                            model_id=task.model_id,
                            metadata=task.metadata,
                            available_instances=available_instances,
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"[central_queue] Scheduling failed for task {task.task_id}: {e}",
                        exc_info=True,
                    )
                    return False

                if not schedule_result.selected_instance_id:
                    logger.warning(
                        f"No instance selected for task {task.task_id}"
                    )
                    return False

                # Get the selected instance object
                selected_instance = None
                for inst in available_instances:
                    if inst.instance_id == schedule_result.selected_instance_id:
                        selected_instance = inst
                        break

                if not selected_instance:
                    logger.error(
                        f"Selected instance {schedule_result.selected_instance_id} not found"
                    )
                    return False

                # Update task record with assignment info
                task_record = await self._task_registry.get(task.task_id)
                if task_record:
                    task_record.assigned_instance = (
                        selected_instance.instance_id
                    )
                    selected_pred = schedule_result.selected_prediction
                    if selected_pred:
                        task_record.predicted_time_ms = (
                            selected_pred.predicted_time_ms
                        )
                        task_record.predicted_error_margin_ms = (
                            selected_pred.error_margin_ms
                        )
                        task_record.predicted_quantiles = (
                            selected_pred.quantiles
                        )

                # Increment pending count on selected instance
                await self._instance_registry.increment_pending(
                    selected_instance.instance_id
                )

                # Dispatch task to instance (fire and forget)
                self._task_dispatcher.dispatch_task_async(task.task_id)

                # Calculate wait time in queue
                wait_time_ms = (time.time() - task.enqueue_time) * 1000

                logger.info(
                    f"[QUEUE_DISPATCH] task_id={task.task_id} model_id={task.model_id} "
                    f"selected_instance={selected_instance.instance_id} "
                    f"wait_time_ms={wait_time_ms:.2f}"
                )
                return True

            except Exception as e:
                logger.error(
                    f"[central_queue] Error dispatching task {task.task_id}: {e}",
                    exc_info=True,
                )
                return False
