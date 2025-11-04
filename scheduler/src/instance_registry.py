"""
Instance registry for managing compute instances.

This module provides thread-safe storage and management of instances
that can execute tasks.
"""

from typing import Dict, List, Optional
import asyncio
from datetime import datetime

from .model import (
    Instance,
    InstanceStatus,
    InstanceStats,
    InstanceQueueBase,
    InstanceQueueProbabilistic,
    InstanceQueueExpectError,
)


class InstanceRegistry:
    """Thread-safe registry for managing instances."""

    def __init__(self, queue_info_type: str = "probabilistic"):
        """
        Initialize instance registry.

        Args:
            queue_info_type: Type of queue information to maintain
                            ("probabilistic" or "expect_error")
        """
        self._instances: Dict[str, Instance] = {}
        self._queue_info: Dict[str, InstanceQueueBase] = {}
        self._stats: Dict[str, InstanceStats] = {}
        self._lock = asyncio.Lock()
        self._queue_info_type = queue_info_type

    async def register(self, instance: Instance) -> None:
        """
        Register a new instance.

        Args:
            instance: Instance to register

        Raises:
            ValueError: If instance with this ID already exists
        """
        async with self._lock:
            if instance.instance_id in self._instances:
                raise ValueError(f"Instance {instance.instance_id} already exists")

            self._instances[instance.instance_id] = instance

            # Initialize queue info based on scheduling strategy type
            if self._queue_info_type == "expect_error":
                self._queue_info[instance.instance_id] = InstanceQueueExpectError(
                    instance_id=instance.instance_id,
                    expected_time_ms=0.0,
                    error_margin_ms=0.0,
                )
            else:  # Default to probabilistic
                self._queue_info[instance.instance_id] = InstanceQueueProbabilistic(
                    instance_id=instance.instance_id,
                    quantiles=[0.5, 0.9, 0.95, 0.99],
                    values=[0.0, 0.0, 0.0, 0.0],
                )

            # Initialize statistics
            self._stats[instance.instance_id] = InstanceStats(
                pending_tasks=0,
                completed_tasks=0,
                failed_tasks=0,
            )

    async def remove(self, instance_id: str) -> Instance:
        """
        Remove an instance from the registry.

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

            return instance

    async def get(self, instance_id: str) -> Optional[Instance]:
        """
        Get an instance by ID.

        Args:
            instance_id: ID of instance to retrieve

        Returns:
            Instance if found, None otherwise
        """
        async with self._lock:
            return self._instances.get(instance_id)

    async def list_all(self, model_id: Optional[str] = None) -> List[Instance]:
        """
        List all instances, optionally filtered by model_id.

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

    async def get_queue_info(self, instance_id: str) -> Optional[InstanceQueueBase]:
        """
        Get queue information for an instance.

        Args:
            instance_id: ID of instance

        Returns:
            Queue information if found, None otherwise
        """
        async with self._lock:
            return self._queue_info.get(instance_id)

    async def update_queue_info(self, instance_id: str, queue_info: InstanceQueueBase) -> None:
        """
        Update queue information for an instance.

        Args:
            instance_id: ID of instance
            queue_info: New queue information
        """
        async with self._lock:
            if instance_id in self._instances:
                self._queue_info[instance_id] = queue_info

    async def get_stats(self, instance_id: str) -> Optional[InstanceStats]:
        """
        Get statistics for an instance.

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
        """
        Reset pending_tasks counter to 0 for all instances.

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
        """
        Mark instance as draining - stops accepting new tasks.

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
            instance.drain_initiated_at = datetime.utcnow().isoformat() + "Z"
            return instance

    async def get_drain_status(self, instance_id: str) -> Dict:
        """
        Get draining status for an instance.

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
                    "drain_initiated_at": instance.drain_initiated_at
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
                "drain_initiated_at": instance.drain_initiated_at
            }

    async def list_active(self, model_id: Optional[str] = None) -> List[Instance]:
        """
        List only ACTIVE instances (excludes draining/removing).

        Args:
            model_id: Optional model ID filter

        Returns:
            List of active instances
        """
        async with self._lock:
            instances = [
                i for i in self._instances.values()
                if i.status == InstanceStatus.ACTIVE
            ]

            if model_id:
                instances = [i for i in instances if i.model_id == model_id]

            return instances

    async def safe_remove(self, instance_id: str) -> Instance:
        """
        Safely remove an instance - only if draining and no pending tasks.

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
