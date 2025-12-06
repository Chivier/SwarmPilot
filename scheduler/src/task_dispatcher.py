"""
Task dispatcher for sending tasks to instances and handling results.

This module manages the actual execution of tasks on instances,
including sending task requests and processing results.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
import asyncio
import httpx
from loguru import logger

from .model import TaskStatus, InstanceQueueExpectError, InstanceQueueProbabilistic
from .task_registry import TaskRegistry
from .instance_registry import InstanceRegistry
from .websocket_manager import ConnectionManager
from .http_error_logger import log_http_error

if TYPE_CHECKING:
    from .training_client import TrainingClient
    from .central_queue import CentralTaskQueue

# Default retry configuration for transient connection errors
DEFAULT_DISPATCH_RETRIES = 3
DEFAULT_DISPATCH_RETRY_DELAY = 0.1  # 100ms initial delay


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
        dispatch_retries: int = DEFAULT_DISPATCH_RETRIES,
        dispatch_retry_delay: float = DEFAULT_DISPATCH_RETRY_DELAY,
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
            dispatch_retries: Number of retry attempts for transient connection errors
            dispatch_retry_delay: Initial delay between retries (uses exponential backoff)
        """
        self.task_registry = task_registry
        self.instance_registry = instance_registry
        self.websocket_manager = websocket_manager
        self.training_client = training_client
        self.timeout = timeout
        self.callback_base_url = callback_base_url
        self.dispatch_retries = dispatch_retries
        self.dispatch_retry_delay = dispatch_retry_delay

        # Configure connection pool with shorter keepalive to avoid stale connections
        # This helps prevent ReadError when server closes idle connections
        # The keepalive_expiry should be shorter than uvicorn's default (5s)
        limits = httpx.Limits(
            max_keepalive_connections=100,
            max_connections=200,
            keepalive_expiry=3.0,  # Close idle connections after 3 seconds
        )

        # Reusable HTTP client with SSL verification disabled for internal network
        self._http_client = httpx.AsyncClient(
            timeout=timeout,
            verify=False,  # Disable SSL verification for internal network usage
            limits=limits,
        )
        # Reference to central queue (set by api.py)
        self._central_queue: Optional["CentralTaskQueue"] = None

    def set_central_queue(self, queue: "CentralTaskQueue") -> None:
        """Set the central queue reference for capacity notifications."""
        self._central_queue = queue

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

        except httpx.TimeoutException as e:
            # Task dispatch timed out (not execution timeout - that's handled by instance)
            log_http_error(
                e,
                request_url=f"{instance.endpoint}/task/submit",
                request_method="POST",
                request_body={
                    "task_id": task.task_id,
                    "model_id": task.model_id,
                    "task_input": task.task_input,
                    "callback_url": f"{self.callback_base_url}/callback/task_result",
                },
                context="task dispatch timeout",
                extra={"task_id": task_id, "instance_id": instance.instance_id},
            )
            await self.task_registry.update_status(task_id, TaskStatus.FAILED)
            await self.task_registry.set_error(
                task_id, f"Task dispatch timed out after {self.timeout}s"
            )
            await self.instance_registry.increment_failed(instance.instance_id)
            await self._notify_task_completion(task_id)

        except httpx.HTTPError as e:
            # HTTP error from instance during dispatch
            log_http_error(
                e,
                request_url=f"{instance.endpoint}/task/submit",
                request_method="POST",
                request_body={
                    "task_id": task.task_id,
                    "model_id": task.model_id,
                    "task_input": task.task_input,
                    "callback_url": f"{self.callback_base_url}/callback/task_result",
                },
                context="task dispatch HTTP error",
                extra={"task_id": task_id, "instance_id": instance.instance_id},
            )
            await self.task_registry.update_status(task_id, TaskStatus.FAILED)
            await self.task_registry.set_error(task_id, f"Task dispatch failed: {str(e)}")
            await self.instance_registry.increment_failed(instance.instance_id)
            await self._notify_task_completion(task_id)

        except Exception as e:
            # Unexpected error during dispatch
            log_http_error(
                e,
                request_url=f"{instance.endpoint}/task/submit",
                request_method="POST",
                request_body={
                    "task_id": task.task_id,
                    "model_id": task.model_id,
                    "task_input": task.task_input,
                    "callback_url": f"{self.callback_base_url}/callback/task_result",
                },
                context="task dispatch unexpected error",
                extra={"task_id": task_id, "instance_id": instance.instance_id},
            )
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

        # Notify central queue that capacity may be available
        if self._central_queue:
            await self._central_queue.notify_capacity_available()

    async def _notify_task_completion(self, task_id: str) -> None:
        """
        Notify WebSocket subscribers about task completion.

        Args:
            task_id: ID of completed task
        """
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
            # Formula: new_expect = old_expect - actual_time
            new_expected = current_queue.expected_time_ms - actual_time_ms
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

    def _is_transient_connection_error(self, error: Exception) -> bool:
        """
        Check if an error is a transient connection error that should be retried.

        Transient errors include:
        - ReadError: Connection was reset by peer (stale keep-alive connection)
        - ConnectError: Temporary network issues
        - RemoteProtocolError: Protocol-level issues

        Args:
            error: The exception to check

        Returns:
            True if the error is transient and should be retried
        """
        # Check for specific httpx error types that indicate transient issues
        error_type = type(error).__name__

        # ReadError occurs when server closes connection unexpectedly
        # This commonly happens with stale keep-alive connections
        if error_type == "ReadError":
            return True

        # ConnectError can be transient (e.g., temporary network issues)
        if error_type == "ConnectError":
            return True

        # RemoteProtocolError indicates protocol-level issues
        if error_type == "RemoteProtocolError":
            return True

        return False

    async def _submit_via_http(
        self,
        instance: Any,
        task_id: str,
        model_id: str,
        task_input: Dict[str, Any],
        enqueue_time: Optional[float] = None,
    ) -> None:
        """
        Submit task to instance via HTTP with retry logic for transient errors.

        This method includes automatic retry for transient connection errors
        (e.g., ReadError from stale keep-alive connections). The retry uses
        exponential backoff to avoid overwhelming the target instance.

        Args:
            instance: Instance object
            task_id: Task identifier
            model_id: Model identifier
            task_input: Task input data
            enqueue_time: Optional Unix timestamp for task priority ordering

        Raises:
            httpx.HTTPError: If HTTP request fails after all retries
        """
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

        url = f"{instance.endpoint}/task/submit"
        last_error: Optional[Exception] = None

        # Retry loop for transient connection errors
        for attempt in range(self.dispatch_retries):
            try:
                # Send task to instance for execution
                response = await self._http_client.post(url, json=payload)
                response.raise_for_status()
                submit_result = response.json()

                # Verify task was accepted by instance
                if not submit_result.get("success", False):
                    raise ValueError(
                        f"Instance rejected task: {submit_result.get('message', 'Unknown error')}"
                    )

                logger.info(f"Task {task_id} accepted by instance {instance.instance_id} via HTTP")
                return  # Success - exit the retry loop

            except httpx.HTTPError as e:
                last_error = e

                # Check if this is a transient error that should be retried
                if self._is_transient_connection_error(e) and attempt < self.dispatch_retries - 1:
                    delay = self.dispatch_retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Task dispatch transient error ({type(e).__name__}) for {task_id}, "
                        f"retrying in {delay:.2f}s (attempt {attempt + 1}/{self.dispatch_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue  # Retry

                # Not a transient error or last attempt - re-raise
                raise

        # Should not reach here, but just in case
        if last_error:
            raise last_error
