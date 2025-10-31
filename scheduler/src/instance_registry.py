"""
Instance registry for managing compute instances.

This module provides thread-safe storage and management of instances
that can execute tasks.
"""

from typing import Dict, List, Optional
from threading import Lock
from datetime import datetime

from .model import Instance, InstanceStats, InstanceQueueBase, InstanceQueueProbabilistic


class InstanceRegistry:
    """Thread-safe registry for managing instances."""

    def __init__(self):
        self._instances: Dict[str, Instance] = {}
        self._queue_info: Dict[str, InstanceQueueBase] = {}
        self._stats: Dict[str, InstanceStats] = {}
        self._lock = Lock()

    def register(self, instance: Instance) -> None:
        """
        Register a new instance.

        Args:
            instance: Instance to register

        Raises:
            ValueError: If instance with this ID already exists
        """
        with self._lock:
            if instance.instance_id in self._instances:
                raise ValueError(f"Instance {instance.instance_id} already exists")

            self._instances[instance.instance_id] = instance

            # TODO: Initialize queue info based on scheduling strategy
            # For now, initialize with empty probabilistic queue
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

    def remove(self, instance_id: str) -> Instance:
        """
        Remove an instance from the registry.

        Args:
            instance_id: ID of instance to remove

        Returns:
            The removed instance

        Raises:
            KeyError: If instance not found
        """
        with self._lock:
            if instance_id not in self._instances:
                raise KeyError(f"Instance {instance_id} not found")

            instance = self._instances.pop(instance_id)
            self._queue_info.pop(instance_id, None)
            self._stats.pop(instance_id, None)

            return instance

    def get(self, instance_id: str) -> Optional[Instance]:
        """
        Get an instance by ID.

        Args:
            instance_id: ID of instance to retrieve

        Returns:
            Instance if found, None otherwise
        """
        with self._lock:
            return self._instances.get(instance_id)

    def list_all(self, model_id: Optional[str] = None) -> List[Instance]:
        """
        List all instances, optionally filtered by model_id.

        Args:
            model_id: Optional model ID filter

        Returns:
            List of instances
        """
        with self._lock:
            instances = list(self._instances.values())

            if model_id:
                instances = [i for i in instances if i.model_id == model_id]

            return instances

    def get_queue_info(self, instance_id: str) -> Optional[InstanceQueueBase]:
        """
        Get queue information for an instance.

        Args:
            instance_id: ID of instance

        Returns:
            Queue information if found, None otherwise
        """
        with self._lock:
            return self._queue_info.get(instance_id)

    def update_queue_info(self, instance_id: str, queue_info: InstanceQueueBase) -> None:
        """
        Update queue information for an instance.

        Args:
            instance_id: ID of instance
            queue_info: New queue information
        """
        with self._lock:
            if instance_id in self._instances:
                self._queue_info[instance_id] = queue_info

    def get_stats(self, instance_id: str) -> Optional[InstanceStats]:
        """
        Get statistics for an instance.

        Args:
            instance_id: ID of instance

        Returns:
            Statistics if found, None otherwise
        """
        with self._lock:
            return self._stats.get(instance_id)

    def increment_pending(self, instance_id: str) -> None:
        """Increment pending task count for an instance."""
        with self._lock:
            if instance_id in self._stats:
                self._stats[instance_id].pending_tasks += 1

    def decrement_pending(self, instance_id: str) -> None:
        """Decrement pending task count for an instance."""
        with self._lock:
            if instance_id in self._stats:
                self._stats[instance_id].pending_tasks = max(
                    0, self._stats[instance_id].pending_tasks - 1
                )

    def increment_completed(self, instance_id: str) -> None:
        """Increment completed task count for an instance."""
        with self._lock:
            if instance_id in self._stats:
                self._stats[instance_id].completed_tasks += 1

    def increment_failed(self, instance_id: str) -> None:
        """Increment failed task count for an instance."""
        with self._lock:
            if instance_id in self._stats:
                self._stats[instance_id].failed_tasks += 1

    def get_total_count(self) -> int:
        """Get total number of registered instances."""
        with self._lock:
            return len(self._instances)

    def get_active_count(self) -> int:
        """Get count of active instances (instances with no failed status)."""
        # TODO: Implement health checking mechanism
        # For now, assume all registered instances are active
        return self.get_total_count()
