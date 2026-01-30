"""Instance synchronization module for PYLET-019: API Integration.

This module implements the declarative instance sync API where the Planner
submits the complete target instance list and the scheduler computes the diff.

Key features:
- Declarative sync: Planner submits target list, scheduler handles additions/removals
- Task rescheduling: When instances are removed, pending tasks are rescheduled
- Model validation: Ensures all instances match the configured model_id

Example:
    ```python
    request = InstanceSyncRequest(
        instances=[
            InstanceInfo(instance_id="worker-1", endpoint="http://...", model_id="gpt-4"),
            InstanceInfo(instance_id="worker-2", endpoint="http://...", model_id="gpt-4"),
        ]
    )
    result = await handle_instance_sync(
        request=request,
        config_model_id="gpt-4",
        instance_registry=registry,
        worker_queue_manager=manager,
        scheduling_strategy=strategy,
    )
    # result.added = ["worker-1", "worker-2"] if these were new
    # result.removed = [] if no workers were removed
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from swarmpilot.scheduler.algorithms.base import SchedulingStrategy
    from swarmpilot.scheduler.registry.instance_registry import InstanceRegistry
    from swarmpilot.scheduler.services.worker_queue_manager import WorkerQueueManager
    from swarmpilot.scheduler.services.worker_queue_thread import QueuedTask


@dataclass
class InstanceInfo:
    """Information about a single instance for sync.

    Attributes:
        instance_id: Unique identifier for the instance.
        endpoint: HTTP endpoint for the instance's API.
        model_id: Model ID running on this instance.
    """

    instance_id: str
    endpoint: str
    model_id: str


@dataclass
class InstanceSyncRequest:
    """Request for syncing instance list.

    The Planner submits the complete target instance list. The scheduler
    computes the diff and handles additions/removals internally.

    Attributes:
        instances: Complete list of target instances.
    """

    instances: list[InstanceInfo] = field(default_factory=list)


@dataclass
class InstanceSyncResponse:
    """Response from instance sync operation.

    Attributes:
        success: Whether the sync completed successfully.
        added: List of instance IDs that were added.
        removed: List of instance IDs that were removed.
        rescheduled: Number of tasks rescheduled from removed instances.
        message: Optional message about the sync operation.
    """

    success: bool
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    rescheduled: int = 0
    message: str = ""


async def handle_instance_sync(
    request: InstanceSyncRequest,
    config_model_id: str,
    instance_registry: InstanceRegistry,
    worker_queue_manager: WorkerQueueManager,
    scheduling_strategy: SchedulingStrategy,
) -> InstanceSyncResponse:
    """Handle instance sync request.

    Computes the diff between current and target instance lists,
    then handles additions and removals.

    Args:
        request: Sync request with target instance list.
        config_model_id: Configured model ID for this scheduler.
        instance_registry: Registry for instance management.
        worker_queue_manager: Manager for worker queue threads.
        scheduling_strategy: Strategy for rescheduling tasks.

    Returns:
        InstanceSyncResponse with sync results.

    Raises:
        ValueError: If any instance has a mismatched model_id.
    """
    # 1. Validate all instances have correct model_id
    for inst in request.instances:
        if inst.model_id != config_model_id:
            raise ValueError(
                f"Model mismatch: {inst.model_id} != {config_model_id} "
                f"(instance: {inst.instance_id})"
            )

    # 2. Compute diff
    current_instances = await instance_registry.list_all()
    current_ids = {inst.instance_id for inst in current_instances}
    target_ids = {inst.instance_id for inst in request.instances}
    target_map = {inst.instance_id: inst for inst in request.instances}

    to_add = target_ids - current_ids
    to_remove = current_ids - target_ids

    logger.info(
        f"Instance sync: current={len(current_ids)}, target={len(target_ids)}, "
        f"to_add={len(to_add)}, to_remove={len(to_remove)}"
    )

    response = InstanceSyncResponse(success=True)

    # 3. Remove instances first (allows task rescheduling to remaining instances)
    for instance_id in to_remove:
        result = await handle_instance_removal(
            instance_id=instance_id,
            instance_registry=instance_registry,
            worker_queue_manager=worker_queue_manager,
            scheduling_strategy=scheduling_strategy,
        )
        response.removed.append(instance_id)
        response.rescheduled += result["rescheduled"]

    # 4. Add new instances
    for instance_id in to_add:
        inst = target_map[instance_id]
        await handle_instance_addition(
            instance_info=inst,
            instance_registry=instance_registry,
            worker_queue_manager=worker_queue_manager,
        )
        response.added.append(instance_id)

    response.message = (
        f"Sync complete: {len(response.added)} added, "
        f"{len(response.removed)} removed, {response.rescheduled} tasks rescheduled"
    )

    logger.info(response.message)

    return response


async def handle_instance_addition(
    instance_info: InstanceInfo,
    instance_registry: InstanceRegistry,
    worker_queue_manager: WorkerQueueManager,
) -> None:
    """Add a new instance: register and create queue thread.

    Args:
        instance_info: Information about the instance to add.
        instance_registry: Registry for instance management.
        worker_queue_manager: Manager for worker queue threads.
    """
    # Import here to avoid circular imports
    from swarmpilot.scheduler.models import InstanceStatus
    from swarmpilot.scheduler.models.core import Instance

    # Create Instance object
    instance = Instance(
        instance_id=instance_info.instance_id,
        model_id=instance_info.model_id,
        endpoint=instance_info.endpoint,
        platform_info={
            "software_name": "pylet",
            "software_version": "1.0",
            "hardware_name": "unknown",
        },
        status=InstanceStatus.ACTIVE,
    )

    # Register in instance registry
    await instance_registry.register(instance)

    # Create worker queue thread
    worker_queue_manager.register_worker(
        worker_id=instance_info.instance_id,
        worker_endpoint=instance_info.endpoint,
        model_id=instance_info.model_id,
    )

    logger.info(
        f"Added instance {instance_info.instance_id} at {instance_info.endpoint}"
    )


async def handle_instance_removal(
    instance_id: str,
    instance_registry: InstanceRegistry,
    worker_queue_manager: WorkerQueueManager,
    scheduling_strategy: SchedulingStrategy,
) -> dict:
    """Remove an instance: stop queue and reschedule pending tasks.

    Args:
        instance_id: ID of the instance to remove.
        instance_registry: Registry for instance management.
        worker_queue_manager: Manager for worker queue threads.
        scheduling_strategy: Strategy for rescheduling tasks.

    Returns:
        Dictionary with removal results:
        - pending_tasks: Number of pending tasks from removed instance
        - rescheduled: Number of tasks successfully rescheduled
    """
    # 1. Stop worker queue thread and get pending tasks
    pending_tasks = worker_queue_manager.deregister_worker(
        instance_id, stop_timeout=5.0
    )

    result = {
        "pending_tasks": len(pending_tasks),
        "rescheduled": 0,
    }

    # 2. Reschedule each pending task
    for task in pending_tasks:
        success = await reschedule_task(
            task=task,
            exclude_instance=instance_id,
            instance_registry=instance_registry,
            worker_queue_manager=worker_queue_manager,
            scheduling_strategy=scheduling_strategy,
        )
        if success:
            result["rescheduled"] += 1
        else:
            logger.warning(
                f"Cannot reschedule task {task.task_id}: no workers available"
            )

    # 3. Remove from instance registry
    await instance_registry.remove(instance_id)

    logger.info(
        f"Removed instance {instance_id}, "
        f"rescheduled {result['rescheduled']}/{result['pending_tasks']} tasks"
    )

    return result


async def reschedule_task(
    task: QueuedTask,
    exclude_instance: str,
    instance_registry: InstanceRegistry,
    worker_queue_manager: WorkerQueueManager,
    scheduling_strategy: SchedulingStrategy,
) -> bool:
    """Reschedule a task to another worker.

    Uses the existing scheduling algorithm to select a new worker.
    Preserves the original enqueue_time for priority ordering.

    Args:
        task: Task to reschedule.
        exclude_instance: Instance ID to exclude from selection.
        instance_registry: Registry for instance management.
        worker_queue_manager: Manager for worker queue threads.
        scheduling_strategy: Strategy for selecting new worker.

    Returns:
        True if successfully rescheduled, False if no workers available.
    """
    # Get available instances (excluding the removed one)
    available = await instance_registry.get_active_instances(task.model_id)
    available = [i for i in available if i.instance_id != exclude_instance]

    if not available:
        return False

    # Use existing scheduling algorithm
    schedule_result = await scheduling_strategy.schedule_task(
        model_id=task.model_id,
        metadata=task.metadata,
        available_instances=available,
    )

    if not schedule_result.selected_instance_id:
        return False

    # Enqueue to new worker (task keeps original enqueue_time for priority)
    worker_queue_manager.enqueue_task(
        schedule_result.selected_instance_id,
        task,
    )

    logger.info(
        f"Rescheduled task {task.task_id} to {schedule_result.selected_instance_id}"
    )

    return True
