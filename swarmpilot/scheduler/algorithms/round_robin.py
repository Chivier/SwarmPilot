"""Round-robin scheduling strategy.

Simple strategy that cycles through instances in order.
Useful for load balancing when predictions are not available.
"""

from typing import TYPE_CHECKING, Any

from loguru import logger

from swarmpilot.scheduler.algorithms.base import SchedulingStrategy
from swarmpilot.scheduler.clients.models import Prediction

if TYPE_CHECKING:
    from swarmpilot.scheduler.clients.predictor_library_client import (
        PredictorClient,
    )
    from swarmpilot.scheduler.models import Instance, InstanceQueueBase
    from swarmpilot.scheduler.registry.instance_registry import InstanceRegistry


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

    async def get_predictions(
        self,
        model_id: str,
        metadata: dict[str, Any],
        available_instances: list["Instance"],
    ) -> list[Prediction]:
        """Get predictions, falling back to dummy values.

        RoundRobin doesn't need real predictions, so if the predictor
        fails (e.g., no trained model), return dummy predictions.

        Args:
            model_id: Model identifier.
            metadata: Task metadata.
            available_instances: Instances to predict for.

        Returns:
            List of predictions (real or dummy).
        """
        try:
            return await super().get_predictions(
                model_id, metadata, available_instances
            )
        except (ValueError, Exception) as e:
            logger.debug(
                f"RoundRobin: predictor unavailable ({e}), "
                f"using dummy predictions for {len(available_instances)} instances"
            )
            return [
                Prediction(
                    instance_id=inst.instance_id,
                    predicted_time_ms=0.0,
                )
                for inst in available_instances
            ]

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
