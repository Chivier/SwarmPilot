"""
Task dispatcher for sending tasks to instances and handling results.

This module manages the actual execution of tasks on instances,
including sending task requests and processing results.
"""

from typing import Dict, Any, Optional
import asyncio
import httpx

from .model import TaskStatus
from .task_registry import TaskRegistry
from .instance_registry import InstanceRegistry
from .websocket_manager import ConnectionManager


class TaskDispatcher:
    """Dispatcher for executing tasks on instances."""

    def __init__(
        self,
        task_registry: TaskRegistry,
        instance_registry: InstanceRegistry,
        websocket_manager: ConnectionManager,
        timeout: float = 60.0,
    ):
        """
        Initialize task dispatcher.

        Args:
            task_registry: Task registry for tracking task state
            instance_registry: Instance registry for accessing instances
            websocket_manager: WebSocket manager for result notifications
            timeout: Task execution timeout in seconds
        """
        self.task_registry = task_registry
        self.instance_registry = instance_registry
        self.websocket_manager = websocket_manager
        self.timeout = timeout

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
            # Update task status to running
            self.task_registry.update_status(task_id, TaskStatus.RUNNING)
            self.instance_registry.decrement_pending(instance.instance_id)

            # TODO: Send task to instance endpoint
            # TODO: The actual API contract with instances needs to be defined
            # TODO: This is a placeholder for the instance communication protocol

            # Example implementation:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{instance.endpoint}/execute",
                    json={
                        "task_id": task.task_id,
                        "model_id": task.model_id,
                        "input": task.task_input,
                        "metadata": task.metadata,
                    },
                )
                response.raise_for_status()
                result = response.json()

            # Update task with result
            self.task_registry.update_status(task_id, TaskStatus.COMPLETED)
            self.task_registry.set_result(task_id, result)
            self.instance_registry.increment_completed(instance.instance_id)

        except httpx.TimeoutException:
            # Task timed out
            self.task_registry.update_status(task_id, TaskStatus.FAILED)
            self.task_registry.set_error(
                task_id, f"Task execution timed out after {self.timeout}s"
            )
            self.instance_registry.increment_failed(instance.instance_id)

        except httpx.HTTPError as e:
            # HTTP error from instance
            self.task_registry.update_status(task_id, TaskStatus.FAILED)
            self.task_registry.set_error(task_id, f"Instance request failed: {str(e)}")
            self.instance_registry.increment_failed(instance.instance_id)

        except Exception as e:
            # Unexpected error
            self.task_registry.update_status(task_id, TaskStatus.FAILED)
            self.task_registry.set_error(task_id, f"Unexpected error: {str(e)}")
            self.instance_registry.increment_failed(instance.instance_id)

        finally:
            # Notify WebSocket subscribers of completion
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
