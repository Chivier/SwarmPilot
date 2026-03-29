"""Task result callback handler for worker threads.

This module implements PYLET-016: Library-Based Callback Mechanism, providing
a callback handler that bridges worker threads with the main asyncio event loop.

Key features:
- Thread-safe callback from worker threads to event loop
- Updates TaskRegistry with status/result/error
- Updates InstanceRegistry with statistics
- Records throughput for planner reporting
- Broadcasts results to WebSocket subscribers
"""

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger

from swarmpilot.scheduler.models import TaskStatus
from swarmpilot.scheduler.registry.instance_registry import InstanceRegistry
from swarmpilot.scheduler.registry.task_registry import TaskRegistry
from swarmpilot.scheduler.services.websocket_manager import ConnectionManager

if TYPE_CHECKING:
    from swarmpilot.scheduler.clients.training_library_client import (
        TrainingClient,
    )
    from swarmpilot.scheduler.services.worker_queue_thread import TaskResult
    from swarmpilot.scheduler.utils.throughput_tracker import ThroughputTracker


class TaskResultCallback:
    """Callback handler for task results from worker threads.

    This class bridges the worker threads (which execute synchronously)
    with the main asyncio event loop (which manages registries and
    WebSocket connections).

    Key responsibilities:
    1. Receive results from worker threads
    2. Update TaskRegistry with status/result/error
    3. Update InstanceRegistry with statistics
    4. Record throughput for planner reporting
    5. Broadcast results to WebSocket subscribers

    Example:
        ```python
        callback_handler = TaskResultCallback(
            task_registry=task_registry,
            instance_registry=instance_registry,
            websocket_manager=websocket_manager,
            throughput_tracker=throughput_tracker,
        )

        # Get thread-safe callback for worker threads
        loop = asyncio.get_event_loop()
        thread_callback = callback_handler.create_thread_callback(loop)

        # Pass to WorkerQueueThread
        worker_thread = WorkerQueueThread(
            worker_id="worker-1",
            callback=thread_callback,
            ...
        )
        ```
    """

    def __init__(
        self,
        task_registry: TaskRegistry,
        instance_registry: InstanceRegistry,
        websocket_manager: ConnectionManager,
        throughput_tracker: "ThroughputTracker | None" = None,
        training_client: "TrainingClient | None" = None,
    ):
        """Initialize callback handler.

        Args:
            task_registry: Registry for task state management.
            instance_registry: Registry for instance statistics.
            websocket_manager: Manager for WebSocket notifications.
            throughput_tracker: Optional tracker for planner reporting.
            training_client: Optional training client for auto-training.
        """
        self.task_registry = task_registry
        self.instance_registry = instance_registry
        self.websocket_manager = websocket_manager
        self.throughput_tracker = throughput_tracker
        self.training_client = training_client

        # Event loop reference (set when creating thread callback)
        self._loop: asyncio.AbstractEventLoop | None = None

        # Per-task Future pool: task_id -> asyncio.Future[TaskResult]
        self._futures: dict[str, asyncio.Future] = {}

    def register_future(self, task_id: str) -> asyncio.Future:
        """Register a Future for a task, to be resolved when the result arrives.

        Args:
            task_id: Task identifier to register.

        Returns:
            An asyncio.Future that will be resolved with the TaskResult.

        Raises:
            ValueError: If a Future is already registered for this task_id.
        """
        if task_id in self._futures:
            raise ValueError(f"Future already registered for task {task_id}")

        loop = self._loop or asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._futures[task_id] = future
        logger.debug(f"Registered future for task {task_id}")
        return future

    def has_future(self, task_id: str) -> bool:
        """Check if a Future is registered for a task.

        Args:
            task_id: Task identifier.

        Returns:
            True if a Future exists for this task_id.
        """
        return task_id in self._futures

    def cleanup_future(self, task_id: str) -> None:
        """Remove and cancel an unresolved Future.

        Args:
            task_id: Task identifier to clean up.
        """
        future = self._futures.pop(task_id, None)
        if future is not None and not future.done():
            future.cancel()
            logger.debug(f"Cleaned up future for task {task_id}")

    def create_thread_callback(
        self,
        loop: asyncio.AbstractEventLoop,
    ) -> Callable[["TaskResult"], None]:
        """Create a callback function that can be called from worker threads.

        This returns a closure that safely schedules the async handle_result
        coroutine on the main event loop using run_coroutine_threadsafe.

        Args:
            loop: The main asyncio event loop.

        Returns:
            A synchronous callback function for use in worker threads.
        """
        self._loop = loop

        def thread_safe_callback(result: "TaskResult") -> None:
            """Callback function safe to call from any thread."""
            coro = self.handle_result(result)
            try:
                try:
                    asyncio.run_coroutine_threadsafe(
                        coro,
                        loop,
                    )
                    # Don't wait for result - fire and forget
                    # Errors are logged in handle_result
                except BaseException as e:
                    coro.close()
                    raise RuntimeError(
                        f"Failed to schedule callback for task {result.task_id}: {e}"
                    ) from e
            except RuntimeError as e:
                logger.error(
                    f"Failed to schedule callback for task {result.task_id}: {e}"
                )

        return thread_safe_callback

    def create_thread_start_callback(
        self,
        loop: asyncio.AbstractEventLoop,
    ) -> Callable[[str], None]:
        """Create a callback for when a task starts executing.

        Returns a closure that safely schedules the async
        _handle_task_started coroutine on the main event loop.

        Args:
            loop: The main asyncio event loop.

        Returns:
            A synchronous callback accepting a task_id string.
        """

        def thread_safe_start_callback(task_id: str) -> None:
            """Callback for task start, safe to call from any thread."""
            coro = self._handle_task_started(task_id)
            try:
                try:
                    asyncio.run_coroutine_threadsafe(
                        coro,
                        loop,
                    )
                except BaseException as e:
                    coro.close()
                    raise RuntimeError(
                        f"Failed to schedule start callback for task {task_id}: {e}"
                    ) from e
            except RuntimeError as e:
                logger.error(
                    f"Failed to schedule start callback for task {task_id}: {e}"
                )

        return thread_safe_start_callback

    async def _handle_task_started(self, task_id: str) -> None:
        """Handle task execution start by setting RUNNING status.

        Args:
            task_id: Task that started executing.
        """
        try:
            await self.task_registry.update_status(task_id, TaskStatus.RUNNING)
            logger.debug(f"Task {task_id} status set to RUNNING")
        except KeyError:
            logger.warning(f"Task {task_id} not found when setting RUNNING")
        except ValueError as e:
            logger.error(f"Error setting task {task_id} to RUNNING: {e}")

    async def handle_result(self, result: "TaskResult") -> None:
        """Handle task completion result.

        This coroutine runs in the main event loop and updates all
        relevant registries and notifies subscribers.

        Args:
            result: Task result from worker thread.
        """
        task_id = result.task_id
        logger.debug(f"Handling result for task {task_id}: {result.status}")

        try:
            # Get task record
            task = await self.task_registry.get(task_id)
            if not task:
                logger.warning(f"Task {task_id} not found in registry")
                self._resolve_future(task_id, result)
                return

            # Get instance for stats update
            instance = await self.instance_registry.get(result.worker_id)

            if result.status == "completed":
                await self._handle_success(task_id, result, task, instance)
            else:
                await self._handle_failure(task_id, result, instance)

            # Notify WebSocket subscribers
            await self._notify_subscribers(task_id, result)

            # Resolve Future if registered (for synchronous proxy)
            self._resolve_future(task_id, result)

        except Exception as e:
            logger.opt(exception=True).error(
                f"Error handling result for task {task_id}: {e}"
            )
            # Still try to resolve the future on error so proxy doesn't hang
            self._resolve_future(task_id, result)

    def _resolve_future(
        self,
        task_id: str,
        result: "TaskResult",
    ) -> None:
        """Resolve the Future for a task if one is registered.

        Args:
            task_id: Task identifier.
            result: Task result to set on the Future.
        """
        future = self._futures.pop(task_id, None)
        if future is not None and not future.done():
            future.set_result(result)
            logger.debug(f"Resolved future for task {task_id}")

    async def _handle_success(
        self,
        task_id: str,
        result: "TaskResult",
        task,
        instance,
    ) -> None:
        """Handle successful task completion.

        Args:
            task_id: Task identifier.
            result: Task result.
            task: Task record from registry.
            instance: Instance that executed the task.
        """
        # Update task status
        await self.task_registry.update_status(task_id, TaskStatus.COMPLETED)

        # Set result data
        if result.result:
            await self.task_registry.set_result(task_id, result.result)

        # Set execution time
        if task:
            task.set_execution_time(result.execution_time_ms)

        # Update instance statistics
        if instance:
            await self.instance_registry.increment_completed(
                instance.instance_id
            )

        # Record throughput for planner
        if self.throughput_tracker and instance:
            await self.throughput_tracker.record_execution_time(
                instance_endpoint=instance.endpoint,
                execution_time_ms=result.execution_time_ms,
            )

        # Record training sample for auto-training
        if self.training_client and not instance:
            logger.warning(
                f"Skipping training sample for task {task_id}: "
                f"instance {result.worker_id} not found in registry"
            )
        if self.training_client and instance and task:
            try:
                self.training_client.add_sample(
                    model_id=task.model_id,
                    platform_info=instance.platform_info,
                    features=task.metadata,
                    actual_runtime_ms=result.execution_time_ms,
                )
                await self.training_client.flush_if_ready()
            except (ValueError, RuntimeError, OSError) as e:
                logger.warning(
                    f"Training sample recording failed for task {task_id}: {e}"
                )

        logger.info(
            f"Task {task_id} completed on {result.worker_id} "
            f"in {result.execution_time_ms:.2f}ms"
        )

    async def _handle_failure(
        self,
        task_id: str,
        result: "TaskResult",
        instance,
    ) -> None:
        """Handle failed task completion.

        Args:
            task_id: Task identifier.
            result: Task result with error.
            instance: Instance that executed the task.
        """
        # Update task status
        await self.task_registry.update_status(task_id, TaskStatus.FAILED)

        # Set error message
        if result.error:
            await self.task_registry.set_error(task_id, result.error)

        # Update instance statistics
        if instance:
            await self.instance_registry.increment_failed(instance.instance_id)

        logger.warning(
            f"Task {task_id} failed on {result.worker_id} "
            f"after {result.execution_time_ms:.2f}ms: {result.error}"
        )

    async def _notify_subscribers(
        self,
        task_id: str,
        result: "TaskResult",
    ) -> None:
        """Notify WebSocket subscribers of task completion.

        Args:
            task_id: Task identifier.
            result: Task result.
        """
        task = await self.task_registry.get(task_id)
        if not task:
            return

        await self.websocket_manager.broadcast_task_result(
            task_id=task_id,
            status=task.status,
            result=result.result,
            error=result.error,
            timestamps=task.get_timestamps(),
            execution_time_ms=int(result.execution_time_ms),
        )

        logger.debug(f"Broadcasted result for task {task_id}")
