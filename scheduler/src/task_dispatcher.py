"""
Task dispatcher for sending tasks to instances and handling results.

This module manages the actual execution of tasks on instances,
including sending task requests and processing results.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
import asyncio
import httpx

from .model import TaskStatus
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

    async def dispatch_task(self, task_id: str) -> None:
        """
        Dispatch a task to its assigned instance for execution.

        This is an async operation that sends the task to the instance,
        waits for the result, and updates task status accordingly.

        Args:
            task_id: ID of task to dispatch
        """
        # Get task record
        task = self.task_registry.get(task_id)
        if not task:
            return

        # Get instance
        instance = self.instance_registry.get(task.assigned_instance)
        if not instance:
            # Instance not found - mark task as failed
            self.task_registry.update_status(task_id, TaskStatus.FAILED)
            self.task_registry.set_error(
                task_id, f"Instance {task.assigned_instance} not found"
            )
            await self._notify_task_completion(task_id)
            return

        try:
            # Update task status to running (dispatched to instance)
            self.task_registry.update_status(task_id, TaskStatus.RUNNING)
            self.instance_registry.decrement_pending(instance.instance_id)

            # Prepare callback URL for result notification
            callback_url = f"{self.callback_base_url}/callback/task_result"

            # Send task to instance for execution
            # Instance will process asynchronously and notify via callback
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{instance.endpoint}/task/submit",
                    json={
                        "task_id": task.task_id,
                        "model_id": task.model_id,
                        "task_input": task.task_input,
                        "callback_url": callback_url,
                    },
                )
                response.raise_for_status()
                submit_result = response.json()

            # Verify task was accepted by instance
            if not submit_result.get("success", False):
                raise ValueError(
                    f"Instance rejected task: {submit_result.get('message', 'Unknown error')}"
                )

            # Task is now dispatched and running on instance
            # Result will arrive via callback endpoint
            # NOTE: Task completion, result, and training data collection
            # are handled in the callback endpoint handler

        except httpx.TimeoutException:
            # Task dispatch timed out (not execution timeout - that's handled by instance)
            self.task_registry.update_status(task_id, TaskStatus.FAILED)
            self.task_registry.set_error(
                task_id, f"Task dispatch timed out after {self.timeout}s"
            )
            self.instance_registry.increment_failed(instance.instance_id)
            await self._notify_task_completion(task_id)

        except httpx.HTTPError as e:
            # HTTP error from instance during dispatch
            self.task_registry.update_status(task_id, TaskStatus.FAILED)
            self.task_registry.set_error(task_id, f"Task dispatch failed: {str(e)}")
            self.instance_registry.increment_failed(instance.instance_id)
            await self._notify_task_completion(task_id)

        except Exception as e:
            # Unexpected error during dispatch
            self.task_registry.update_status(task_id, TaskStatus.FAILED)
            self.task_registry.set_error(task_id, f"Task dispatch error: {str(e)}")
            self.instance_registry.increment_failed(instance.instance_id)
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
        task = self.task_registry.get(task_id)
        if not task:
            return

        # Get instance for stats update
        instance = self.instance_registry.get(task.assigned_instance)

        # Update task based on status
        if status == "completed":
            self.task_registry.update_status(task_id, TaskStatus.COMPLETED)
            if result:
                self.task_registry.set_result(task_id, result)
            if execution_time_ms:
                task.execution_time_ms = execution_time_ms

            # Update instance stats
            if instance:
                self.instance_registry.increment_completed(instance.instance_id)

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
            self.task_registry.update_status(task_id, TaskStatus.FAILED)
            if error:
                self.task_registry.set_error(task_id, error)

            # Update instance stats
            if instance:
                self.instance_registry.increment_failed(instance.instance_id)

        # Notify WebSocket subscribers
        await self._notify_task_completion(task_id)

    async def _notify_task_completion(self, task_id: str) -> None:
        """
        Notify WebSocket subscribers about task completion.

        Args:
            task_id: ID of completed task
        """
        task = self.task_registry.get(task_id)
        if not task:
            return

        # Only notify if task is in terminal state
        if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            return

        await self.websocket_manager.broadcast_task_result(
            task_id=task.task_id,
            status=task.status,
            result=task.result,
            error=task.error,
            timestamps=task.get_timestamps(),
            execution_time_ms=task.execution_time_ms,
        )

    def dispatch_task_async(self, task_id: str) -> None:
        """
        Dispatch a task asynchronously (fire and forget).

        Args:
            task_id: ID of task to dispatch
        """
        asyncio.create_task(self.dispatch_task(task_id))
