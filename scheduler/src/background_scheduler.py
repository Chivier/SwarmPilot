"""
Background task scheduler for non-blocking task submission.

This module handles CPU-intensive scheduling operations in the background,
allowing API endpoints to return immediately.
"""

import asyncio
from typing import Dict, Any, Optional
from loguru import logger
from .model import TaskStatus


class BackgroundScheduler:
    """
    Handles task scheduling in the background to prevent blocking API responses.

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
        """
        Initialize background scheduler.

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
        self._active_tasks: Dict[str, asyncio.Task] = {}

        logger.info(
            f"BackgroundScheduler initialized with max_concurrent={max_concurrent_scheduling}"
        )

    def schedule_task_background(
        self,
        task_id: str,
        model_id: str,
        task_input: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Schedule a task in the background (non-blocking).

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
        task_input: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Internal async method to handle full task scheduling workflow.

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
                logger.debug(f"Starting background scheduling for task {task_id}")

                # 1. Get available instances
                available_instances = await self.instance_registry.list_active(
                    model_id=model_id
                )

                if not available_instances:
                    # No instances available - mark task as failed
                    await self.task_registry.update_status(task_id, TaskStatus.FAILED)
                    await self.task_registry.set_error(
                        task_id,
                        f"No available instance for model_id: {model_id}"
                    )
                    logger.error(f"Task {task_id}: No available instances for {model_id}")
                    return

                # 2. Schedule task (predictions + selection + queue update)
                try:
                    schedule_result = await self.scheduling_strategy.schedule_task(
                        model_id=model_id,
                        metadata=metadata,
                        available_instances=available_instances,
                    )
                except Exception as e:
                    # Scheduling failed - mark task as failed
                    error_msg = str(e)
                    await self.task_registry.update_status(task_id, TaskStatus.FAILED)
                    await self.task_registry.set_error(task_id, f"Scheduling failed: {error_msg}")
                    logger.error(f"Task {task_id}: Scheduling failed - {error_msg}")
                    return

                # 3. Update task with scheduling results
                task = await self.task_registry.get(task_id)
                if not task:
                    logger.error(f"Task {task_id}: Task not found after scheduling")
                    return

                # Update task with prediction info and assigned instance
                selected_pred = schedule_result.selected_prediction
                task.assigned_instance = schedule_result.selected_instance_id
                if selected_pred:
                    task.predicted_time_ms = selected_pred.predicted_time_ms
                    task.predicted_error_margin_ms = selected_pred.error_margin_ms
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
                    f"Task {task_id}: Unexpected error in background scheduling - {e}",
                    exc_info=True
                )
                try:
                    await self.task_registry.update_status(task_id, TaskStatus.FAILED)
                    await self.task_registry.set_error(
                        task_id,
                        f"Internal scheduling error: {str(e)}"
                    )
                except Exception:
                    logger.error(f"Task {task_id}: Failed to update task status after error")

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about background scheduling.

        Returns:
            Dictionary with scheduling statistics
        """
        return {
            "active_scheduling_tasks": len(self._active_tasks),
            "max_concurrent_scheduling": self._semaphore._value + len(self._active_tasks),
            "available_slots": self._semaphore._value,
        }

    async def shutdown(self) -> None:
        """
        Gracefully shutdown background scheduler.

        Waits for all active scheduling tasks to complete.
        """
        if self._active_tasks:
            logger.info(
                f"Shutting down BackgroundScheduler, "
                f"waiting for {len(self._active_tasks)} tasks..."
            )
            await asyncio.gather(*self._active_tasks.values(), return_exceptions=True)

        logger.info("BackgroundScheduler shutdown complete")
