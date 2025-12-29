"""Store for available instances during migration-based redeployment."""

import asyncio

from loguru import logger
from pydantic import BaseModel


class AvailableInstance(BaseModel):
    """Represents an available instance for migration.

    Attributes:
        model_id: The model ID this instance is running.
        endpoint: The HTTP endpoint URL of the instance.
    """

    model_id: str
    endpoint: str


class AvailableInstanceStore:
    """Store for available instances.

    This class is designed for migration-based re-deployment.
    Uses List instead of Queue for better iteration, indexing, and JSON
    serialization support.

    Attributes:
        available_instances: Dictionary mapping model_id to list of instances.
        lock: Asyncio lock for thread-safe access.
    """

    def __init__(self):
        """Initialize an empty instance store."""
        self.available_instances: dict[str, list[AvailableInstance]] = {}
        self.lock = asyncio.Lock()

    async def get_available_instances_by_model_id(
        self, model_id: str
    ) -> list[AvailableInstance]:
        """Get all available instances for a specific model.

        Args:
            model_id: The model ID to query.

        Returns:
            List of available instances for the model.
        """
        async with self.lock:
            return list(self.available_instances.get(model_id, []))

    async def get_available_instances(self) -> list[AvailableInstance]:
        """Get all available instances across all models.

        Returns:
            Flat list of all available instances.
        """
        async with self.lock:
            return [
                instance
                for instances in self.available_instances.values()
                for instance in instances
            ]

    async def add_available_instance(self, instance: AvailableInstance):
        """Add an instance to the available pool.

        Args:
            instance: The instance to add.
        """
        async with self.lock:
            if instance.model_id not in self.available_instances:
                self.available_instances[instance.model_id] = []
            self.available_instances[instance.model_id].append(instance)

    async def remove_available_instance(self, instance: AvailableInstance):
        """Remove an instance from the available pool.

        Args:
            instance: The instance to remove.
        """
        async with self.lock:
            if instance.model_id in self.available_instances:
                try:
                    self.available_instances[instance.model_id].remove(instance)
                except ValueError:
                    logger.warning(
                        f"Instance {instance.endpoint} not found "
                        f"in model {instance.model_id} list"
                    )

    async def fetch_one_available_instance(
        self, model_id: str
    ) -> AvailableInstance | None:
        """Fetch and remove one available instance for a model.

        Args:
            model_id: The model ID to fetch an instance for.

        Returns:
            An available instance, or None if none available.
        """
        async with self.lock:
            if model_id not in self.available_instances:
                logger.error(
                    f"Model {model_id} not found in available instances"
                )
                logger.error(
                    f"Available instances: {list(self.available_instances.keys())}"
                )
                return None
            if len(self.available_instances[model_id]) == 0:
                logger.error(
                    f"Model {model_id} has no available instances, list is empty"
                )
                return None
            return self.available_instances[model_id].pop(0)


_available_model_store: AvailableInstanceStore | None = None


def get_available_instance_store() -> AvailableInstanceStore:
    """Get the global available instance store singleton.

    Returns:
        The global AvailableInstanceStore instance.
    """
    global _available_model_store
    if _available_model_store is None:
        _available_model_store = AvailableInstanceStore()
    return _available_model_store
