"""
Task dispatcher for sending tasks to instances and handling results.

This module manages the actual execution of tasks on instances,
including sending task requests and processing results.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
import asyncio
import httpx

from .model import TaskStatus, InstanceQueueExpectError, InstanceQueueProbabilistic
from .task_registry import TaskRegistry
from .instance_registry import InstanceRegistry
from .websocket_manager import ConnectionManager

if TYPE_CHECKING:
    from .training_client import TrainingClient


class TaskDispatcher:
    """Dispatcher for executing tasks on instances."""

    def __init__(
        self,
        task_registry: TaskRegistry,
        instance_registry: InstanceRegistry,
        websocket_manager: ConnectionManager,
        training_client: Optional["TrainingClient"] = None,
        timeout: float = 60.0,
        callback_base_url: str = "http://localhost:8000",
    ):
        """
        Initialize task dispatcher.

        Args:
            task_registry: Task registry for tracking task state
            instance_registry: Instance registry for accessing instances
            websocket_manager: WebSocket manager for result notifications
            training_client: Optional training client for collecting runtime data
            timeout: Task execution timeout in seconds
            callback_base_url: Base URL for task result callbacks
        """
        self.task_registry = task_registry
        self.instance_registry = instance_registry
        self.websocket_manager = websocket_manager
        self.training_client = training_client
        self.timeout = timeout
        self.callback_base_url = callback_base_url
        # Reusable HTTP client with SSL verification disabled for internal network
        self._http_client = httpx.AsyncClient(
            timeout=timeout,
            verify=False,  # Disable SSL verification for internal network usage
        )

    async def dispatch_task(self, task_id: str, enqueue_time: Optional[float] = None) -> None:
        """
        Dispatch a task to its assigned instance for execution.

        This is an async operation that sends the task to the instance,
        waits for the result, and updates task status accordingly.

        Args:
            task_id: ID of task to dispatch
            enqueue_time: Optional Unix timestamp for task priority ordering
        """
        # Get task record
        task = await self.task_registry.get(task_id)
        if not task:
            return

        # Get instance
        instance = await self.instance_registry.get(task.assigned_instance)
        if not instance:
            # Instance not found - mark task as failed
            await self.task_registry.update_status(task_id, TaskStatus.FAILED)
            await self.task_registry.set_error(
                task_id, f"Instance {task.assigned_instance} not found"
            )
            await self._notify_task_completion(task_id)
            return

        try:
            # Update task status to running (dispatched to instance)
            await self.task_registry.update_status(task_id, TaskStatus.RUNNING)
            await self.instance_registry.decrement_pending(instance.instance_id)

            # Submit task via HTTP
            await self._submit_via_http(
                instance=instance,
                task_id=task.task_id,
                model_id=task.model_id,
                task_input=task.task_input,
                enqueue_time=enqueue_time,
            )

            # Task is now dispatched and running on instance
            # Result will arrive via callback (HTTP)

        except httpx.TimeoutException:
            # Task dispatch timed out (not execution timeout - that's handled by instance)
            await self.task_registry.update_status(task_id, TaskStatus.FAILED)
            await self.task_registry.set_error(
                task_id, f"Task dispatch timed out after {self.timeout}s"
            )
            await self.instance_registry.increment_failed(instance.instance_id)
            await self._notify_task_completion(task_id)

        except httpx.HTTPError as e:
            # HTTP error from instance during dispatch
            await self.task_registry.update_status(task_id, TaskStatus.FAILED)
            await self.task_registry.set_error(task_id, f"Task dispatch failed: {str(e)}")
            await self.instance_registry.increment_failed(instance.instance_id)
            await self._notify_task_completion(task_id)

        except Exception as e:
            # Unexpected error during dispatch
            await self.task_registry.update_status(task_id, TaskStatus.FAILED)
            await self.task_registry.set_error(task_id, f"Task dispatch error: {str(e)}")
            await self.instance_registry.increment_failed(instance.instance_id)
            await self._notify_task_completion(task_id)

    async def handle_task_result(
        self,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
    ) -> None:
        """
        Handle task result callback from instance.

        This is called when an instance sends back task results via the callback endpoint.

        Args:
            task_id: ID of the completed task
            status: Task status ("completed" or "failed")
            result: Task result data (if completed)
            error: Error message (if failed)
            execution_time_ms: Execution time in milliseconds
        """
        # Get task record
        task = await self.task_registry.get(task_id)
        if not task:
            return

        # Get instance for stats update
        instance = await self.instance_registry.get(task.assigned_instance)

        # Update task based on status
        if status == "completed":
            await self.task_registry.update_status(task_id, TaskStatus.COMPLETED)
            if result:
                await self.task_registry.set_result(task_id, result)
            if execution_time_ms is not None:
                task.set_execution_time(execution_time_ms)

            # Update instance stats
            if instance:
                await self.instance_registry.increment_completed(instance.instance_id)

            # Update queue information based on actual execution time
            if instance and execution_time_ms and task.predicted_time_ms is not None:
                await self._update_queue_on_completion(
                    instance_id=instance.instance_id,
                    predicted_time_ms=task.predicted_time_ms,
                    actual_time_ms=execution_time_ms,
                    predicted_error_margin_ms=task.predicted_error_margin_ms,
                    predicted_quantiles=task.predicted_quantiles,
                )

            # Collect training data if enabled
            if self.training_client and instance and execution_time_ms:
                self.training_client.add_sample(
                    model_id=task.model_id,
                    platform_info=instance.platform_info,
                    features=task.metadata,
                    actual_runtime_ms=execution_time_ms,
                )
                # Try to flush if buffer is full
                await self.training_client.flush_if_ready()

        elif status == "failed":
            await self.task_registry.update_status(task_id, TaskStatus.FAILED)
            if error:
                await self.task_registry.set_error(task_id, error)

            # Update instance stats
            if instance:
                await self.instance_registry.increment_failed(instance.instance_id)

        # Notify WebSocket subscribers
        await self._notify_task_completion(task_id)

    async def _notify_task_completion(self, task_id: str) -> None:
        """
        Notify WebSocket subscribers about task completion.

        Args:
            task_id: ID of completed task
        """
        from loguru import logger

        task = await self.task_registry.get(task_id)
        if not task:
            logger.warning(f"Cannot notify completion for {task_id}: task not found in registry")
            return

        # Only notify if task is in terminal state
        if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            logger.debug(f"Skipping notification for {task_id}: status={task.status} (not terminal)")
            return

        logger.info(f"Notifying completion for {task_id}: status={task.status}, execution_time={task.execution_time_ms}ms")

        await self.websocket_manager.broadcast_task_result(
            task_id=task.task_id,
            status=task.status,
            result=task.result,
            error=task.error,
            timestamps=task.get_timestamps(),
            execution_time_ms=task.execution_time_ms,
        )

        logger.debug(f"Notification complete for {task_id}")

    def dispatch_task_async(self, task_id: str) -> None:
        """
        Dispatch a task asynchronously (fire and forget).

        Args:
            task_id: ID of task to dispatch
        """
        asyncio.create_task(self.dispatch_task(task_id))

    async def close(self) -> None:
        """
        Close the HTTP client and cleanup resources.

        Should be called when shutting down the dispatcher.
        """
        await self._http_client.aclose()

    async def _update_queue_on_completion(
        self,
        instance_id: str,
        predicted_time_ms: float,
        actual_time_ms: float,
        predicted_error_margin_ms: Optional[float] = None,
        predicted_quantiles: Optional[Dict[float, float]] = None,
    ) -> None:
        """
        Update queue information when a task completes.

        According to the scheduling strategy documentation:
        - Shortest Queue: Update expect with actual time, keep error unchanged
        - Probabilistic: Update quantiles using Monte Carlo method

        Args:
            instance_id: ID of the instance
            predicted_time_ms: Predicted execution time
            actual_time_ms: Actual execution time
            predicted_error_margin_ms: Predicted error margin (for Shortest Queue)
            predicted_quantiles: Predicted quantiles (for Probabilistic)
        """
        current_queue = await self.instance_registry.get_queue_info(instance_id)
        if not current_queue:
            return

        if isinstance(current_queue, InstanceQueueExpectError):
            # Shortest Queue strategy: Update expect with actual time, keep error unchanged
            # Formula: new_expect = old_expect - predicted_time + actual_time
            new_expected = current_queue.expected_time_ms - predicted_time_ms + actual_time_ms
            # Ensure non-negative
            new_expected = max(0.0, new_expected)

            updated_queue = InstanceQueueExpectError(
                instance_id=instance_id,
                expected_time_ms=new_expected,
                error_margin_ms=current_queue.error_margin_ms,  # Keep error unchanged
            )
            await self.instance_registry.update_queue_info(instance_id, updated_queue)

        elif isinstance(current_queue, InstanceQueueProbabilistic):
            # Probabilistic strategy: Update quantiles using Monte Carlo method
            if predicted_quantiles:
                import numpy as np

                num_samples = 1000

                # Generate random percentiles for sampling
                random_percentiles = np.random.random(num_samples)

                # Sample from current queue distribution
                queue_samples = np.interp(
                    random_percentiles,
                    current_queue.quantiles,
                    current_queue.values
                )

                # Sample from predicted task distribution
                task_quantiles = sorted(predicted_quantiles.keys())
                task_values = [predicted_quantiles[q] for q in task_quantiles]
                task_samples = np.interp(
                    random_percentiles,
                    task_quantiles,
                    task_values
                )

                # Subtract predicted task time and add actual time
                # new_queue = old_queue - predicted_task + actual_task
                updated_samples = queue_samples - task_samples
                # Ensure non-negative
                updated_samples = np.maximum(updated_samples, 0.0)

                # Compute new quantiles from updated samples
                updated_values = [
                    float(np.percentile(updated_samples, q * 100))
                    for q in current_queue.quantiles
                ]

                updated_queue = InstanceQueueProbabilistic(
                    instance_id=instance_id,
                    quantiles=current_queue.quantiles,
                    values=updated_values,
                )
                await self.instance_registry.update_queue_info(instance_id, updated_queue)
            else:
                # Fallback: Simple subtraction and addition for all quantiles
                updated_values = [
                    max(0.0, current_queue.values[i] - predicted_time_ms + actual_time_ms)
                    for i in range(len(current_queue.quantiles))
                ]
                updated_queue = InstanceQueueProbabilistic(
                    instance_id=instance_id,
                    quantiles=current_queue.quantiles,
                    values=updated_values,
                )
                await self.instance_registry.update_queue_info(instance_id, updated_queue)

    async def _submit_via_http(
        self,
        instance: Any,
        task_id: str,
        model_id: str,
        task_input: Dict[str, Any],
        enqueue_time: Optional[float] = None,
    ) -> None:
        """
        Submit task to instance via HTTP (fallback method).

        Args:
            instance: Instance object
            task_id: Task identifier
            model_id: Model identifier
            task_input: Task input data
            enqueue_time: Optional Unix timestamp for task priority ordering

        Raises:
            httpx.HTTPError: If HTTP request fails
        """
        from loguru import logger

        logger.info(f"Submitting task {task_id} to instance {instance.instance_id} via HTTP")

        # Prepare callback URL for result notification
        callback_url = f"{self.callback_base_url}/callback/task_result"

        # Prepare payload
        payload = {
            "task_id": task_id,
            "model_id": model_id,
            "task_input": task_input,
            "callback_url": callback_url,
        }

        # Include enqueue_time if provided (for priority preservation)
        if enqueue_time is not None:
            payload["enqueue_time"] = enqueue_time

        # Send task to instance for execution
        response = await self._http_client.post(
            f"{instance.endpoint}/task/submit",
            json=payload,
        )
        response.raise_for_status()
        submit_result = response.json()

        # Verify task was accepted by instance
        if not submit_result.get("success", False):
            raise ValueError(
                f"Instance rejected task: {submit_result.get('message', 'Unknown error')}"
            )

        logger.info(f"Task {task_id} accepted by instance {instance.instance_id} via HTTP")
