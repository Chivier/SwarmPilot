"""Random scheduling strategy.

Simple strategy that randomly selects an instance.
Represents the worst case of probabilistic scheduling.
"""

import random
from typing import TYPE_CHECKING

from src.algorithms.base import SchedulingStrategy
from src.clients.models import Prediction

if TYPE_CHECKING:
    from src.clients.predictor_library_client import PredictorClient
    from src.models import InstanceQueueBase
    from src.registry.instance_registry import InstanceRegistry


class RandomStrategy(SchedulingStrategy):
    """Random scheduling strategy.

    Worst case of probabilistic scheduling - randomly selects an instance.
    """

    def __init__(
        self,
        predictor_client: "PredictorClient",
        instance_registry: "InstanceRegistry",
    ):
        """Initialize RandomStrategy."""
        super().__init__(predictor_client, instance_registry)
        self._counter = 0

    def select_instance(
        self,
        predictions: list[Prediction],
        queue_info: dict[str, "InstanceQueueBase"],
    ) -> str | None:
        """Randomly select an instance."""
        if not predictions:
            return None

        return random.choice(predictions).instance_id

    async def update_queue(
        self,
        instance_id: str,
        prediction: Prediction,
    ) -> None:
        """No-op for RandomStrategy.

        Random selection doesn't use queue predictions for scheduling decisions,
        so no queue update is necessary.

        Args:
            instance_id: Selected instance
            prediction: Prediction for the task
        """
        # No-op: Random doesn't maintain queue state
        pass
