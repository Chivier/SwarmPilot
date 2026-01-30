"""Serverless-optimized Minimum Expected Time scheduling strategy.

Similar to MinimumExpectedTimeStrategy but optimized for serverless environments.
"""

import math
from typing import TYPE_CHECKING

from loguru import logger

from swarmpilot.scheduler.algorithms.base import SchedulingStrategy
from swarmpilot.scheduler.clients.models import Prediction

if TYPE_CHECKING:
    from swarmpilot.scheduler.clients.predictor_library_client import PredictorClient
    from swarmpilot.scheduler.models import InstanceQueueBase
    from swarmpilot.scheduler.registry.instance_registry import InstanceRegistry


class MinimumExpectedTimeServerlessStrategy(SchedulingStrategy):
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
        """Initialize MinimumExpectedTimeServerlessStrategy."""
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

    async def update_queue(
        self,
        instance_id: str,
        prediction: Prediction,
    ) -> None:
        """Update queue using error accumulation formula.

        Formula:
        - new_expected = current_expected + task_expected
        - new_error = sqrt(current_error^2 + task_error^2)

        Args:
            instance_id: Selected instance
            prediction: Prediction for the task
        """
        from swarmpilot.scheduler.models import InstanceQueueExpectError

        current_queue = await self.instance_registry.get_queue_info(instance_id)

        if not current_queue:
            # If no queue exists, initialize with correct type
            current_queue = InstanceQueueExpectError(
                instance_id=instance_id,
                expected_time_ms=0.0,
                error_margin_ms=0.0,
            )
        elif not isinstance(current_queue, InstanceQueueExpectError):
            # Type mismatch - this shouldn't happen if strategy switch was done properly
            logger.warning(
                f"Queue info type mismatch for {instance_id}: "
                f"expected InstanceQueueExpectError, got {type(current_queue).__name__}. "
                f"This indicates the strategy switch didn't properly reinitialize queues. Skipping update."
            )
            return

        task_expected = prediction.predicted_time_ms
        task_error = prediction.error_margin_ms or 0.0

        # Calculate new queue expected time (simple addition)
        new_expected = current_queue.expected_time_ms + task_expected

        # Calculate new queue error margin (error accumulation)
        new_error = math.sqrt(current_queue.error_margin_ms**2 + task_error**2)

        updated_queue = InstanceQueueExpectError(
            instance_id=instance_id,
            expected_time_ms=new_expected,
            error_margin_ms=new_error,
        )

        await self.instance_registry.update_queue_info(instance_id, updated_queue)
        logger.debug(
            f"Updated queue (expect_error) for {instance_id}: "
            f"expected_time_ms={new_expected:.2f}, error_margin_ms={new_error:.2f}"
        )
