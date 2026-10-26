"""Minimum Expected Time scheduling strategy.

This strategy selects the instance with minimum expected queue completion time,
considering both the current queue state and the predicted time for the new task.
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


class MinimumExpectedTimeStrategy(SchedulingStrategy):
    """Strategy that selects the instance with minimum expected queue completion time.

    This strategy considers both the current queue state (expected time and error margin)
    and the predicted time for the new task. It selects the instance that minimizes
    the total expected completion time including uncertainty.
    """

    def __init__(
        self,
        predictor_client: "PredictorClient",
        instance_registry: "InstanceRegistry",
    ):
        """Initialize MinimumExpectedTimeStrategy."""
        super().__init__(predictor_client, instance_registry)

    def select_instance(
        self,
        predictions: list[Prediction],
        queue_info: dict[str, "InstanceQueueBase"],
    ) -> str | None:
        """Select instance with minimum total expected time (queue + new task).

        For each instance, calculates: queue_expected + queue_error + task_expected
        and selects the instance with the minimum value.
        """
        from swarmpilot.scheduler.models import InstanceQueueExpectError

        if not predictions:
            return None

        best_instance_id = None
        best_total_time = float("inf")

        for pred in predictions:
            # Get queue information for this instance
            queue = queue_info.get(pred.instance_id)

            if queue and isinstance(queue, InstanceQueueExpectError):
                # Calculate total time: queue expected + queue error + new task expected
                total_time = (
                    queue.expected_time_ms
                    + queue.error_margin_ms
                    + pred.predicted_time_ms
                )
            else:
                # Fallback: no queue info, just use prediction
                total_time = pred.predicted_time_ms

            if total_time < best_total_time:
                best_total_time = total_time
                best_instance_id = pred.instance_id

        return best_instance_id

