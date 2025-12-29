"""Background task scheduler for non-blocking task submission.

This module handles CPU-intensive scheduling operations in the background,
allowing API endpoints to return immediately.
"""

import asyncio
from typing import Any

from loguru import logger

from .model import TaskStatus

# Backpressure water marks for task dispatching
# HIGH_WATER_MARK: Maximum pending tasks before instance is considered "full"
# LOW_WATER_MARK: Target pending tasks when selecting instances (prefer less loaded)
HIGH_WATER_MARK = 5
LOW_WATER_MARK = 3


class BackgroundScheduler:
    """Handles task scheduling in the background to prevent blocking API responses.

    When a task is submitted:
    1. API creates task record immediately and returns
    2. BackgroundScheduler processes scheduling asynchronously:
       - Get available instances
       - Call predictor for time estimates
       - Select optimal instance
       - Update queue info (Monte Carlo sampling)
       - Assign instance to task
       - Dispatch task

    This prevents 500 workflow (3500 tasks) scenario from blocking API
    for 35-175 seconds due to serial prediction requests.
    """

    def __init__(
        self,
        scheduling_strategy,
        task_registry,
        instance_registry,
        task_dispatcher,
        max_concurrent_scheduling: int = 50,
    ):
        """Initialize background scheduler.

        Args:
            scheduling_strategy: Strategy for selecting instances
            task_registry: Task registry for updating task state
            instance_registry: Instance registry for stats
            task_dispatcher: Dispatcher for sending tasks to instances
            max_concurrent_scheduling: Maximum concurrent scheduling operations
        """
        self.scheduling_strategy = scheduling_strategy
        self.task_registry = task_registry
        self.instance_registry = instance_registry
        self.task_dispatcher = task_dispatcher

        # Semaphore to limit concurrent scheduling operations
        self._semaphore = asyncio.Semaphore(max_concurrent_scheduling)

        # Track active scheduling tasks
        self._active_tasks: dict[str, asyncio.Task] = {}

        logger.info(
            f"BackgroundScheduler initialized with max_concurrent={max_concurrent_scheduling}"
        )

    def schedule_task_background(
        self,
        task_id: str,
        model_id: str,
        task_input: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Schedule a task in the background (non-blocking).

        Creates an asyncio task that handles the full scheduling workflow.
        Returns immediately without waiting for scheduling to complete.

        Args:
            task_id: Unique task identifier
            model_id: Model/tool to use
            task_input: Input data for the task
            metadata: Metadata for prediction
        """
        # Create background task
        task = asyncio.create_task(
            self._schedule_task_async(task_id, model_id, task_input, metadata)
        )

        # Track the task
        self._active_tasks[task_id] = task

        # Add callback to clean up when done
        task.add_done_callback(lambda t: self._active_tasks.pop(task_id, None))

        logger.debug(f"Created background scheduling task for {task_id}")

    async def _schedule_task_async(
        self,
        task_id: str,
        model_id: str,
        task_input: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Internal async method to handle full task scheduling workflow.

        This runs in the background and performs:
        1. Get available instances
        2. Call scheduling strategy (predictions + selection)
        3. Update task with assigned instance
        4. Update instance stats
        5. Dispatch task to instance

        Args:
            task_id: Unique task identifier
            model_id: Model/tool to use
            task_input: Input data for the task
            metadata: Metadata for prediction
        """
        async with self._semaphore:
            try:
                logger.debug(
                    f"Starting background scheduling for task {task_id}"
                )

                # 1. Get available instances with backpressure
                # Only consider instances below HIGH_WATER_MARK
                available_instances = (
                    await self.instance_registry.get_instances_below_water_mark(
                        model_id=model_id, water_mark=HIGH_WATER_MARK
                    )
                )

                if not available_instances:
                    # All instances at or above HIGH_WATER_MARK, or no instances exist
                    # Check if any active instances exist at all
                    all_active = await self.instance_registry.list_active(
                        model_id=model_id
                    )
                    if not all_active:
                        # No instances at all - mark task as failed
                        await self.task_registry.update_status(
                            task_id, TaskStatus.FAILED
                        )
                        await self.task_registry.set_error(
                            task_id,
                            f"No available instance for model_id: {model_id}",
                        )
                        logger.error(
                            f"Task {task_id}: No available instances for {model_id}"
                        )
                        return
                    else:
                        # Backpressure: all instances full, use least loaded one
                        logger.warning(
                            f"Task {task_id}: All instances above HIGH_WATER_MARK ({HIGH_WATER_MARK}), "
                            f"using all active instances"
                        )
                        available_instances = all_active

                # 2. Schedule task (predictions + selection + queue update)
                try:
                    schedule_result = (
                        await self.scheduling_strategy.schedule_task(
                            model_id=model_id,
                            metadata=metadata,
                            available_instances=available_instances,
                        )
                    )
                except Exception as e:
                    # Scheduling failed - mark task as failed
                    error_msg = str(e)
                    await self.task_registry.update_status(
                        task_id, TaskStatus.FAILED
                    )
                    await self.task_registry.set_error(
                        task_id, f"Scheduling failed: {error_msg}"
                    )
                    logger.error(
                        f"[background_scheduler] Task {task_id}: Scheduling failed - {error_msg}",
                        exc_info=True,
                    )
                    return

                # 3. Update task with scheduling results
                task = await self.task_registry.get(task_id)
                if not task:
                    logger.error(
                        f"Task {task_id}: Task not found after scheduling"
                    )
                    return

                # Update task with prediction info and assigned instance
                selected_pred = schedule_result.selected_prediction
                task.assigned_instance = schedule_result.selected_instance_id
                if selected_pred:
                    task.predicted_time_ms = selected_pred.predicted_time_ms
                    task.predicted_error_margin_ms = (
                        selected_pred.error_margin_ms
                    )
                    task.predicted_quantiles = selected_pred.quantiles

                # 4. Update instance stats
                await self.instance_registry.increment_pending(
                    schedule_result.selected_instance_id
                )

                # 5. Dispatch task to instance
                self.task_dispatcher.dispatch_task_async(task_id)

                logger.info(
                    f"Task {task_id}: Scheduled to {schedule_result.selected_instance_id}"
                )

            except Exception as e:
                # Unexpected error - mark task as failed
                logger.error(
                    f"[background_scheduler] Task {task_id}: Unexpected error in background scheduling - {e}",
                    exc_info=True,
                )
                try:
                    await self.task_registry.update_status(
                        task_id, TaskStatus.FAILED
                    )
                    await self.task_registry.set_error(
                        task_id, f"Internal scheduling error: {e!s}"
                    )
                except Exception:
                    logger.error(
                        f"[background_scheduler] Task {task_id}: Failed to update task status after error",
                        exc_info=True,
                    )

    async def reassign_task(
        self,
        task_id: str,
        model_id: str,
        task_input: dict[str, Any],
        enqueue_time: float | None = None,
        metadata: dict[str, Any] | None = None,
        exclude_instance_id: str | None = None,
    ) -> bool:
        """Reassign a task to a new instance (used during redeployment).

        This is similar to schedule_task_background, but:
        1. Preserves the original enqueue_time for priority ordering
        2. Can exclude specific instances (e.g., the one being redeployed)
        3. Returns success/failure status synchronously

        Args:
            task_id: Unique task identifier
            model_id: Model/tool to use
            task_input: Input data for the task
            enqueue_time: Optional original enqueue time for priority preservation
            metadata: Optional metadata for prediction
            exclude_instance_id: Optional instance ID to exclude from selection

        Returns:
            True if successfully reassigned, False otherwise
        """
        try:
            logger.debug(
                f"Reassigning task {task_id} (excluding {exclude_instance_id})"
            )

            # 1. Get available instances with backpressure (excluding the specified one)
            available_instances = (
                await self.instance_registry.get_instances_below_water_mark(
                    model_id=model_id, water_mark=HIGH_WATER_MARK
                )
            )

            # Filter out excluded instance
            if exclude_instance_id:
                available_instances = [
                    inst
                    for inst in available_instances
                    if inst.instance_id != exclude_instance_id
                ]

            if not available_instances:
                # Backpressure: try all active instances if all are above water mark
                all_active = await self.instance_registry.list_active(
                    model_id=model_id
                )
                if exclude_instance_id:
                    all_active = [
                        inst
                        for inst in all_active
                        if inst.instance_id != exclude_instance_id
                    ]
                if all_active:
                    logger.warning(
                        f"Task {task_id}: All instances above HIGH_WATER_MARK ({HIGH_WATER_MARK}), "
                        f"using all active instances for reassignment"
                    )
                    available_instances = all_active
                else:
                    logger.error(
                        f"Task {task_id}: No available instances for reassignment "
                        f"(model_id: {model_id}, excluding: {exclude_instance_id})"
                    )
                    return False

            # 2. Create task in registry if it doesn't exist
            existing_task = await self.task_registry.get(task_id)
            if not existing_task:
                # Task doesn't exist in registry, create it
                await self.task_registry.create_task(
                    task_id=task_id,
                    model_id=model_id,
                    task_input=task_input,
                    metadata=metadata or {},
                    assigned_instance="",  # Will be assigned by scheduling below
                )
            else:
                # Task exists, update status to PENDING for rescheduling
                await self.task_registry.update_status(
                    task_id, TaskStatus.PENDING
                )

            # 3. Schedule task using current strategy
            schedule_result = await self.scheduling_strategy.schedule_task(
                model_id=model_id,
                metadata=metadata or {},
                available_instances=available_instances,
            )

            # 4. Update task with scheduling results
            task = await self.task_registry.get(task_id)
            if not task:
                logger.error(
                    f"Task {task_id}: Task not found after reassignment"
                )
                return False

            # Update task with prediction info and assigned instance
            selected_pred = schedule_result.selected_prediction
            task.assigned_instance = schedule_result.selected_instance_id
            if selected_pred:
                task.predicted_time_ms = selected_pred.predicted_time_ms
                task.predicted_error_margin_ms = selected_pred.error_margin_ms
                task.predicted_quantiles = selected_pred.quantiles

            # 5. Update instance stats
            await self.instance_registry.increment_pending(
                schedule_result.selected_instance_id
            )

            # 6. Dispatch task to instance with preserved enqueue_time
            await self.task_dispatcher.dispatch_task(
                task_id=task_id,
                enqueue_time=enqueue_time,  # Preserve original priority
            )

            logger.info(
                f"Task {task_id}: Successfully reassigned to {schedule_result.selected_instance_id}"
            )
            return True

        except Exception as e:
            logger.error(
                f"[background_scheduler] Task {task_id}: Failed to reassign task - {e}",
                exc_info=True,
            )
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about background scheduling.

        Returns:
            Dictionary with scheduling statistics
        """
        return {
            "active_scheduling_tasks": len(self._active_tasks),
            "max_concurrent_scheduling": self._semaphore._value
            + len(self._active_tasks),
            "available_slots": self._semaphore._value,
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown background scheduler.

        Waits for all active scheduling tasks to complete.
        """
        if self._active_tasks:
            logger.info(
                f"Shutting down BackgroundScheduler, "
                f"waiting for {len(self._active_tasks)} tasks..."
            )
            await asyncio.gather(
                *self._active_tasks.values(), return_exceptions=True
            )

        logger.info("BackgroundScheduler shutdown complete")
