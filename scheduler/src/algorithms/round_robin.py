"""Round-robin scheduling strategy.

Simple strategy that cycles through instances in order.
Useful for load balancing when predictions are not available.
"""

from typing import TYPE_CHECKING

from src.algorithms.base import SchedulingStrategy
from src.clients.models import Prediction

if TYPE_CHECKING:
    from src.clients.predictor_library_client import PredictorClient
    from src.models import InstanceQueueBase
    from src.registry.instance_registry import InstanceRegistry


class RoundRobinStrategy(SchedulingStrategy):
    """Round-robin scheduling strategy.

    Simple strategy that cycles through instances in order.
    Useful for load balancing when predictions are not available.
    """

    def __init__(
        self,
        predictor_client: "PredictorClient",
        instance_registry: "InstanceRegistry",
    ):
        """Initialize RoundRobinStrategy."""
        super().__init__(predictor_client, instance_registry)
        self._counter = 0

    def select_instance(
        self,
        predictions: list[Prediction],
        queue_info: dict[str, "InstanceQueueBase"],
    ) -> str | None:
        """Select next instance in round-robin order."""
        if not predictions:
            return None

        selected = predictions[self._counter % len(predictions)]
        self._counter += 1
        return selected.instance_id

    async def update_queue(
        self,
        instance_id: str,
        prediction: Prediction,
    ) -> None:
        """No-op for RoundRobinStrategy.

        RoundRobin doesn't use queue predictions for scheduling decisions,
        so no queue update is necessary.

        Args:
            instance_id: Selected instance
            prediction: Prediction for the task
        """
        # No-op: RoundRobin doesn't maintain queue state
        pass
