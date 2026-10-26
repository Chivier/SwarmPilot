"""Power of Two Choices scheduling strategy.

This strategy randomly selects two instances and picks the one with
lower expected queue completion time.
"""

from typing import TYPE_CHECKING

from swarmpilot.scheduler.algorithms.base import SchedulingStrategy
from swarmpilot.scheduler.clients.models import Prediction

if TYPE_CHECKING:
    from swarmpilot.scheduler.clients.predictor_library_client import (
        PredictorClient,
    )
    from swarmpilot.scheduler.models import InstanceQueueBase
    from swarmpilot.scheduler.registry.instance_registry import InstanceRegistry


class PowerOfTwoStrategy(SchedulingStrategy):
    """Power of Two Choices scheduling strategy.

    Randomly selects two instances and picks the one with lower queue time.
    """

    def __init__(
        self,
        predictor_client: "PredictorClient",
        instance_registry: "InstanceRegistry",
        seed: int | None = None,
    ):
        """Initialize PowerOfTwoStrategy."""
        super().__init__(predictor_client, instance_registry, seed=seed)
        self._counter = 0

    def select_instance(
        self,
        predictions: list[Prediction],
        queue_info: dict[str, "InstanceQueueBase"],
    ) -> str | None:
        """Select the better of two randomly chosen instances."""
        if not predictions:
            return None

        # Every task counts as one unit, so the queues compare by length.
        for idx in range(len(predictions)):
            predictions[idx].predicted_time_ms = 1.0

        # Two distinct replicas, uniformly at random
        if len(predictions) == 1:
            return predictions[0].instance_id
        i, j = self._rng.choice(len(predictions), size=2, replace=False)
        pred_1, pred_2 = predictions[i], predictions[j]
        instance_1, instance_2 = pred_1.instance_id, pred_2.instance_id

        from swarmpilot.scheduler.models import InstanceQueueExpectError

        def total(queue: "InstanceQueueBase | None", pred: Prediction) -> float:
            # Queue expected time plus the new task; prediction alone when
            # the instance has no expect-error queue.
            if isinstance(queue, InstanceQueueExpectError):
                return queue.expected_time_ms + pred.predicted_time_ms
            return pred.predicted_time_ms

        total_time_1 = total(queue_info.get(instance_1), pred_1)
        total_time_2 = total(queue_info.get(instance_2), pred_2)
        return instance_1 if total_time_1 <= total_time_2 else instance_2
