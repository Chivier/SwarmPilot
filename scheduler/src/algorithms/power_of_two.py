"""Power of Two Choices scheduling strategy.

This strategy randomly selects two instances and picks the one with
lower expected queue completion time.
"""

import math
import random
from typing import TYPE_CHECKING

from loguru import logger

from src.algorithms.base import SchedulingStrategy
from src.clients.predictor_client import Prediction

if TYPE_CHECKING:
    from src.registry.instance_registry import InstanceRegistry
    from src.model import InstanceQueueBase
    from src.clients.predictor_client import PredictorClient


class PowerOfTwoStrategy(SchedulingStrategy):
    """Power of Two Choices scheduling strategy.

    Randomly selects two instances and picks the one with lower queue time.
    """

    def __init__(
        self,
        predictor_client: "PredictorClient",
        instance_registry: "InstanceRegistry",
    ):
        """Initialize PowerOfTwoStrategy."""
        super().__init__(predictor_client, instance_registry)
        self._counter = 0

    def select_instance(
        self,
        predictions: list[Prediction],
        queue_info: dict[str, "InstanceQueueBase"],
    ) -> str | None:
        """Select the better of two randomly chosen instances."""
        if not predictions:
            return None

        for idx in range(len(predictions)):
            predictions[idx].predicted_time_ms = 1.0

        # Select Queue
        pred_1 = random.choice(predictions)
        pred_2 = random.choice(predictions)
        instance_1 = pred_1.instance_id
        instance_2 = pred_2.instance_id

        from src.model import InstanceQueueExpectError

        if not predictions:
            return None

        # Retrive queue info
        queue_1 = queue_info.get(instance_1)
        queue_2 = queue_info.get(instance_2)

        # Compute metrics for queue 1
        if queue_1 and isinstance(queue_1, InstanceQueueExpectError):
            # Calculate total time: queue expected + queue error + new task expected
            total_time_1 = queue_1.expected_time_ms + pred_1.predicted_time_ms
        else:
            # Fallback: no queue info, just use prediction
            total_time_1 = pred_1.predicted_time_ms

        # Compute metrics for queue 2
        if queue_2 and isinstance(queue_1, InstanceQueueExpectError):
            # Calculate total time: queue expected + queue error + new task expected
            total_time_2 = queue_2.expected_time_ms + pred_2.predicted_time_ms
        else:
            # Fallback: no queue info, just use prediction
            total_time_2 = pred_2.predicted_time_ms

        if total_time_1 < total_time_2:
            return instance_1
        else:
            return instance_2

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
        from src.model import InstanceQueueExpectError

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

        await self.instance_registry.update_queue_info(
            instance_id, updated_queue
        )
        logger.debug(
            f"Updated queue (expect_error) for {instance_id}: "
            f"expected_time_ms={new_expected:.2f}, error_margin_ms={new_error:.2f}"
        )
