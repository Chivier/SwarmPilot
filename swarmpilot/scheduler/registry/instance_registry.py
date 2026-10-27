"""Instance registry for managing compute instances.

This module provides thread-safe storage and management of instances
that can execute tasks.
"""

import asyncio
import time
from datetime import UTC, datetime

from swarmpilot.scheduler.models import (
    Instance,
    InstanceQueueBase,
    InstanceQueueExpectError,
    InstanceQueueProbabilistic,
    InstanceStats,
    InstanceStatus,
)
from swarmpilot.scheduler.registry.queue_sketch import (
    DEFAULT_QUANTILES,
    Charge,
    apply,
    empty_like,
)


class InstanceRegistry:
    """Thread-safe registry for managing instances."""

    def __init__(self, queue_info_type: str = "probabilistic"):
        """Initialize instance registry.

        Args:
            queue_info_type: Type of queue information to maintain
                            ("probabilistic" or "expect_error")
        """
        self._instances: dict[str, Instance] = {}
        self._queue_info: dict[str, InstanceQueueBase] = {}
        self._stats: dict[str, InstanceStats] = {}
        self._lock = asyncio.Lock()
        self._queue_info_type = queue_info_type
        # Store quantiles configuration for probabilistic queues
        self._quantiles = list(DEFAULT_QUANTILES)
        # What each routed, unfinished task added to its instance's sketch,
        # and how many charges each sketch holds.
        self._charges: dict[str, Charge] = {}
        self._charge_counts: dict[str, int] = {}
        # Requests each instance serves at once (1 when not set).
        self._servers: dict[str, int] = {}
        # When each UNHEALTHY instance was found unreachable (monotonic).
        self._unhealthy_since: dict[str, float] = {}

    async def register(self, instance: Instance) -> None:
        """Register a new instance.

        Args:
            instance: Instance to register

        Raises:
            ValueError: If instance with this ID already exists
        """
        async with self._lock:
            if instance.instance_id in self._instances:
                raise ValueError(
                    f"Instance {instance.instance_id} already exists"
                )

            self._instances[instance.instance_id] = instance

            # Initialize queue info based on scheduling strategy type
            if self._queue_info_type == "expect_error":
                self._queue_info[instance.instance_id] = (
                    InstanceQueueExpectError(
                        instance_id=instance.instance_id,
                        expected_time_ms=0.0,
                        error_margin_ms=0.0,
                    )
                )
            else:  # Default to probabilistic
                # Use stored quantiles configuration
                values = [0.0] * len(self._quantiles)
                self._queue_info[instance.instance_id] = (
                    InstanceQueueProbabilistic(
                        instance_id=instance.instance_id,
                        quantiles=self._quantiles.copy(),
                        values=values,
                    )
                )

            # Initialize statistics
            self._stats[instance.instance_id] = InstanceStats(
                pending_tasks=0,
                completed_tasks=0,
                failed_tasks=0,
            )

    async def remove(self, instance_id: str) -> Instance:
        """Remove an instance from the registry.

        Args:
            instance_id: ID of instance to remove

        Returns:
            The removed instance

        Raises:
            KeyError: If instance not found
        """
        async with self._lock:
            if instance_id not in self._instances:
                raise KeyError(f"Instance {instance_id} not found")

            instance = self._instances.pop(instance_id)
            self._queue_info.pop(instance_id, None)
            self._stats.pop(instance_id, None)
            self._drop_charges(instance_id)
            self._servers.pop(instance_id, None)
            self._unhealthy_since.pop(instance_id, None)

            return instance

    async def get(self, instance_id: str) -> Instance | None:
        """Get an instance by ID.

        Args:
            instance_id: ID of instance to retrieve

        Returns:
            Instance if found, None otherwise
        """
        async with self._lock:
            return self._instances.get(instance_id)

    async def update_status(
        self, instance_id: str, status: InstanceStatus
    ) -> None:
        """Update the status of an instance.

        Args:
            instance_id: ID of instance to update
            status: New instance status

        Raises:
            KeyError: If instance not found
        """
        async with self._lock:
            if instance_id not in self._instances:
                raise KeyError(f"Instance {instance_id} not found")

            instance = self._instances[instance_id]
            instance.status = status

            # Update drain_initiated_at timestamp if transitioning to DRAINING
            if (
                status == InstanceStatus.DRAINING
                and not instance.drain_initiated_at
            ):
                instance.drain_initiated_at = (
                    datetime.now(UTC).isoformat().replace("+00:00", "Z")
                )

    async def mark_unhealthy(self, instance_id: str) -> bool:
        """Stop routing to an ACTIVE instance whose engine is unreachable.

        Args:
            instance_id: The instance.

        Returns:
            True if it was ACTIVE and is now UNHEALTHY; False if it is
            unknown or in another state (draining, already unhealthy).
        """
        async with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None or instance.status != InstanceStatus.ACTIVE:
                return False
            instance.status = InstanceStatus.UNHEALTHY
            self._unhealthy_since[instance_id] = time.monotonic()
            return True

    async def mark_healthy(self, instance_id: str) -> bool:
        """Route to an UNHEALTHY instance again.

        Returns:
            True if it was UNHEALTHY and is now ACTIVE.
        """
        async with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None or instance.status != InstanceStatus.UNHEALTHY:
                return False
            instance.status = InstanceStatus.ACTIVE
            self._unhealthy_since.pop(instance_id, None)
            return True

    async def list_unhealthy(self) -> list[tuple[Instance, float]]:
        """Return UNHEALTHY instances with seconds since they became so."""
        now = time.monotonic()
        async with self._lock:
            return [
                (i, now - self._unhealthy_since.get(i.instance_id, now))
                for i in self._instances.values()
                if i.status == InstanceStatus.UNHEALTHY
            ]

    async def list_all(self, model_id: str | None = None) -> list[Instance]:
        """List all instances, optionally filtered by model_id.

        Args:
            model_id: Optional model ID filter

        Returns:
            List of instances
        """
        async with self._lock:
            instances = list(self._instances.values())

            if model_id:
                instances = [i for i in instances if i.model_id == model_id]

            return instances

    async def get_queue_info(
        self, instance_id: str
    ) -> InstanceQueueBase | None:
        """Get queue information for an instance.

        Args:
            instance_id: ID of instance

        Returns:
            Queue information if found, None otherwise
        """
        async with self._lock:
            return self._queue_info.get(instance_id)

    async def get_all_queue_info(
        self, instance_ids: list[str] | None = None
    ) -> dict[str, InstanceQueueBase]:
        """Get queue information for all instances in a single lock acquisition.

        This method is optimized for collecting queue info for multiple instances
        at once, reducing lock contention from O(N) to O(1) per call.

        Args:
            instance_ids: Optional list of instance IDs to filter.
                         If None, returns queue info for all registered instances.

        Returns:
            Dictionary mapping instance_id to queue information.
        """
        async with self._lock:
            if instance_ids is None:
                # Return all queue info (shallow copy to prevent external mutation)
                return dict(self._queue_info)
            else:
                # Return filtered queue info
                return {
                    iid: self._queue_info[iid]
                    for iid in instance_ids
                    if iid in self._queue_info
                }

    async def update_queue_info(
        self, instance_id: str, queue_info: InstanceQueueBase
    ) -> None:
        """Update queue information for an instance.

        Args:
            instance_id: ID of instance
            queue_info: New queue information
        """
        async with self._lock:
            if instance_id in self._instances:
                self._queue_info[instance_id] = queue_info

    # ------------------------------------------------------------------
    # Queue sketch charges
    # ------------------------------------------------------------------

    def _apply_locked(self, charge: Charge, sign: int) -> None:
        queue = self._queue_info.get(charge.instance_id)
        if queue is None:
            return
        count = self._charge_counts.get(charge.instance_id, 0) + sign
        self._charge_counts[charge.instance_id] = max(count, 0)
        # An empty sketch is exactly empty, not a rounding residue.
        self._queue_info[charge.instance_id] = (
            empty_like(queue)
            if count <= 0
            else apply(
                queue,
                charge,
                sign,
                self._servers.get(charge.instance_id, 1),
            )
        )

    def _drop_charges(self, instance_id: str) -> None:
        self._charge_counts.pop(instance_id, None)
        stale = [
            t for t, c in self._charges.items() if c.instance_id == instance_id
        ]
        for task_id in stale:
            del self._charges[task_id]

    async def charge(self, task_id: str | None, charge: Charge) -> None:
        """Add a task's predicted runtime to its instance's sketch.

        A task charged before (re-routed or re-predicted) has its old charge
        removed first, in the same step.

        Args:
            task_id: The task, or None for a charge that is never released.
            charge: The task's contribution.
        """
        async with self._lock:
            if task_id is not None:
                old = self._charges.pop(task_id, None)
                if old is not None:
                    self._apply_locked(old, -1)
                self._charges[task_id] = charge
            self._apply_locked(charge, 1)

    async def release_charge(self, task_id: str) -> Charge | None:
        """Remove a finished or moved task from its instance's sketch.

        Args:
            task_id: The task.

        Returns:
            The removed charge, or None if the task held none.
        """
        async with self._lock:
            old = self._charges.pop(task_id, None)
            if old is not None:
                self._apply_locked(old, -1)
            return old

    async def move_charge(self, task_id: str, instance_id: str) -> bool:
        """Move a task's charge, unchanged, to another instance.

        Returns:
            True if the task held a charge.
        """
        async with self._lock:
            old = self._charges.pop(task_id, None)
            if old is None:
                return False
            self._apply_locked(old, -1)
            new = Charge(
                instance_id=instance_id,
                time_ms=old.time_ms,
                error_ms=old.error_ms,
                quantiles=old.quantiles,
            )
            self._charges[task_id] = new
            self._apply_locked(new, 1)
            return True

    async def set_capacity(self, instance_id: str, servers: int) -> None:
        """Set how many requests an instance serves at once.

        The instance's sketch is rebuilt from its open charges, so tasks
        charged before the change are counted the same way as later ones.

        Args:
            instance_id: The instance.
            servers: Requests in flight at once; values below 1 count as 1.
        """
        servers = max(1, int(servers))
        async with self._lock:
            queue = self._queue_info.get(instance_id)
            if queue is None or self.get_capacity(instance_id) == servers:
                return
            self._servers[instance_id] = servers
            self._queue_info[instance_id] = empty_like(queue)
            self._charge_counts.pop(instance_id, None)
            for charge in self._charges.values():
                if charge.instance_id == instance_id:
                    self._apply_locked(charge, 1)

    def get_capacity(self, instance_id: str) -> int:
        """Return how many requests the instance serves at once."""
        return self._servers.get(instance_id, 1)

    def get_charge(self, task_id: str) -> Charge | None:
        """Return the task's current charge, if any."""
        return self._charges.get(task_id)

    async def reset_queues(
        self, queue_info_type: str, quantiles: list[float] | None = None
    ) -> None:
        """Switch every sketch to a new kind and rebuild it from charges.

        Args:
            queue_info_type: "probabilistic" or "expect_error".
            quantiles: Quantile grid for probabilistic sketches.
        """
        async with self._lock:
            self._queue_info_type = queue_info_type
            if quantiles:
                self._quantiles = sorted(set(quantiles))
            for iid in self._instances:
                if queue_info_type == "expect_error":
                    queue: InstanceQueueBase = InstanceQueueExpectError(
                        instance_id=iid,
                        expected_time_ms=0.0,
                        error_margin_ms=0.0,
                    )
                else:
                    queue = InstanceQueueProbabilistic(
                        instance_id=iid,
                        quantiles=list(self._quantiles),
                        values=[0.0] * len(self._quantiles),
                    )
                self._queue_info[iid] = queue
            self._charge_counts.clear()
            for charge in self._charges.values():
                self._apply_locked(charge, 1)

    async def get_stats(self, instance_id: str) -> InstanceStats | None:
        """Get statistics for an instance.

        Args:
            instance_id: ID of instance

        Returns:
            Statistics if found, None otherwise
        """
        async with self._lock:
            return self._stats.get(instance_id)

    async def increment_pending(self, instance_id: str) -> None:
        """Increment pending task count for an instance."""
        async with self._lock:
            if instance_id in self._stats:
                self._stats[instance_id].pending_tasks += 1

    async def decrement_pending(self, instance_id: str) -> None:
        """Decrement pending task count for an instance."""
        async with self._lock:
            if instance_id in self._stats:
                self._stats[instance_id].pending_tasks = max(
                    0, self._stats[instance_id].pending_tasks - 1
                )

    async def increment_completed(self, instance_id: str) -> None:
        """Increment completed task count for an instance."""
        async with self._lock:
            if instance_id in self._stats:
                self._stats[instance_id].completed_tasks += 1

    async def increment_failed(self, instance_id: str) -> None:
        """Increment failed task count for an instance."""
        async with self._lock:
            if instance_id in self._stats:
                self._stats[instance_id].failed_tasks += 1

    async def reset_all_pending_tasks(self) -> int:
        """Reset pending_tasks counter to 0 for all instances.

        This should be called when clearing all tasks from the task registry
        to ensure instance stats remain consistent.

        Returns:
            Count of instances whose pending_tasks were reset
        """
        async with self._lock:
            count = 0
            for stats in self._stats.values():
                if stats.pending_tasks > 0:
                    stats.pending_tasks = 0
                    count += 1
            return count

    async def start_draining(self, instance_id: str) -> Instance:
        """Mark instance as draining - stops accepting new tasks.

        Args:
            instance_id: ID of instance to start draining

        Returns:
            The instance being drained

        Raises:
            KeyError: If instance not found
            ValueError: If instance is not in ACTIVE state
        """
        async with self._lock:
            if instance_id not in self._instances:
                raise KeyError(f"Instance {instance_id} not found")

            instance = self._instances[instance_id]
            if instance.status != InstanceStatus.ACTIVE:
                raise ValueError(
                    f"Instance {instance_id} is already in {instance.status} state"
                )

            instance.status = InstanceStatus.DRAINING
            instance.drain_initiated_at = (
                datetime.now(UTC).isoformat().replace("+00:00", "Z")
            )
            return instance

    async def get_drain_status(self, instance_id: str) -> dict:
        """Get draining status for an instance.

        Args:
            instance_id: ID of instance

        Returns:
            Dictionary with drain status information

        Raises:
            KeyError: If instance not found
        """
        async with self._lock:
            instance = self._instances.get(instance_id)
            if not instance:
                raise KeyError(f"Instance {instance_id} not found")

            stats = self._stats.get(instance_id)
            if not stats:
                return {
                    "instance_id": instance_id,
                    "status": instance.status,
                    "pending_tasks": 0,
                    "running_tasks": 0,
                    "can_remove": True,
                    "drain_initiated_at": instance.drain_initiated_at,
                }

            can_remove = (
                instance.status == InstanceStatus.DRAINING
                and stats.pending_tasks == 0
            )

            return {
                "instance_id": instance_id,
                "status": instance.status,
                "pending_tasks": stats.pending_tasks,
                "running_tasks": 0,  # pending_tasks includes running
                "can_remove": can_remove,
                "drain_initiated_at": instance.drain_initiated_at,
            }

    async def list_active(self, model_id: str | None = None) -> list[Instance]:
        """List only ACTIVE instances (excludes draining/removing).

        Args:
            model_id: Optional model ID filter

        Returns:
            List of active instances
        """
        async with self._lock:
            instances = [
                i
                for i in self._instances.values()
                if i.status == InstanceStatus.ACTIVE
            ]

            if model_id:
                instances = [i for i in instances if i.model_id == model_id]

            return instances

    async def safe_remove(self, instance_id: str) -> Instance:
        """Safely remove an instance - only if draining and no pending tasks.

        Args:
            instance_id: ID of instance to remove

        Returns:
            The removed instance

        Raises:
            KeyError: If instance not found
            ValueError: If instance cannot be safely removed
        """
        async with self._lock:
            if instance_id not in self._instances:
                raise KeyError(f"Instance {instance_id} not found")

            instance = self._instances[instance_id]
            stats = self._stats.get(instance_id)

            # Check if safe to remove
            if instance.status != InstanceStatus.DRAINING:
                raise ValueError(
                    f"Instance must be in DRAINING state. "
                    f"Current state: {instance.status}. "
                    f"Use /instance/drain endpoint first."
                )

            if stats and stats.pending_tasks > 0:
                raise ValueError(
                    f"Instance has {stats.pending_tasks} pending tasks. "
                    f"Wait for completion before removing."
                )

            # Safe to remove
            instance = self._instances.pop(instance_id)
            self._queue_info.pop(instance_id, None)
            self._stats.pop(instance_id, None)
            self._drop_charges(instance_id)

            return instance

    async def get_total_count(self) -> int:
        """Get total number of registered instances."""
        async with self._lock:
            return len(self._instances)

    async def get_active_count(self) -> int:
        """Get count of active instances (instances with no failed status)."""
        # TODO: Implement health checking mechanism
        # For now, assume all registered instances are active
        return await self.get_total_count()

    async def get_instances_below_water_mark(
        self, model_id: str, water_mark: int
    ) -> list[Instance]:
        """Get active instances with pending_tasks below the specified water mark.

        Args:
            model_id: Model ID filter
            water_mark: Maximum pending tasks threshold

        Returns:
            List of instances that can accept more tasks
        """
        async with self._lock:
            result = []
            for instance in self._instances.values():
                if instance.model_id != model_id:
                    continue
                if instance.status != InstanceStatus.ACTIVE:
                    continue

                stats = self._stats.get(instance.instance_id)
                if stats and stats.pending_tasks < water_mark:
                    result.append(instance)

            return result

    async def is_any_instance_available(
        self, model_id: str, high_water_mark: int
    ) -> bool:
        """Check if any active instance for the model can accept new tasks.

        Args:
            model_id: Model ID filter
            high_water_mark: Maximum pending tasks threshold

        Returns:
            True if at least one instance is below the high water mark
        """
        async with self._lock:
            for instance in self._instances.values():
                if instance.model_id != model_id:
                    continue
                if instance.status != InstanceStatus.ACTIVE:
                    continue

                stats = self._stats.get(instance.instance_id)
                if stats and stats.pending_tasks < high_water_mark:
                    return True

            return False

    async def get_pending_tasks_count(self, instance_id: str) -> int:
        """Get the current pending tasks count for an instance.

        Args:
            instance_id: ID of instance

        Returns:
            Current pending tasks count, or 0 if instance not found
        """
        async with self._lock:
            stats = self._stats.get(instance_id)
            return stats.pending_tasks if stats else 0

    async def has_active_instance(self, model_id: str) -> bool:
        """Check if any ACTIVE instance exists for the model.

        Args:
            model_id: Model ID filter

        Returns:
            True if at least one ACTIVE instance exists
        """
        async with self._lock:
            for instance in self._instances.values():
                if (
                    instance.model_id == model_id
                    and instance.status == InstanceStatus.ACTIVE
                ):
                    return True
            return False

    async def get_active_instances(self, model_id: str) -> list[Instance]:
        """Get all ACTIVE instances for the model.

        Args:
            model_id: Model ID filter

        Returns:
            List of all ACTIVE instances for the model
        """
        async with self._lock:
            return [
                inst
                for inst in self._instances.values()
                if inst.model_id == model_id
                and inst.status == InstanceStatus.ACTIVE
            ]

    async def clear_all(self) -> int:
        """Clear all instances from the registry.

        Returns:
            Count of instances that were cleared
        """
        async with self._lock:
            count = len(self._instances)
            self._instances.clear()
            self._queue_info.clear()
            self._stats.clear()
            self._charges.clear()
            self._charge_counts.clear()
            return count
