"""Probabilistic scheduling strategy based on quantile sampling.

This strategy considers both the queue distribution and the new task distribution.
For each instance, it samples from the queue's quantile distribution to estimate
completion time, then selects the instance with the minimum sampled value.
"""

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from src.algorithms.base import SchedulingStrategy
from src.predictor_client import Prediction

if TYPE_CHECKING:
    from src.instance_registry import InstanceRegistry
    from src.model import InstanceQueueBase
    from src.predictor_client import PredictorClient


class ProbabilisticSchedulingStrategy(SchedulingStrategy):
    """Probabilistic scheduling strategy based on quantile sampling.

    This strategy considers both the queue distribution and the new task distribution.
    For each instance, it samples from the queue's quantile distribution to estimate
    completion time, then selects the instance with the minimum sampled value.
    """

    def __init__(
        self,
        predictor_client: "PredictorClient",
        instance_registry: "InstanceRegistry",
        target_quantile: float = 0.9,
    ):
        """Initialize probabilistic strategy.

        Args:
            predictor_client: Client for getting predictions
            instance_registry: Registry for instance and queue management
            target_quantile: Target quantile for probabilistic selection (default: 0.9)
        """
        super().__init__(predictor_client, instance_registry)
        self.quantiles = instance_registry._quantiles
        self.target_quantile = target_quantile

    def get_prediction_type(self) -> str:
        """Probabilistic strategy requires quantile predictions."""
        return "quantile"

    def select_instance(
        self,
        predictions: list[Prediction],
        queue_info: dict[str, "InstanceQueueBase"],
    ) -> str | None:
        """Select instance using Monte Carlo simulation on combined prediction + queue time.

        Performs vectorized Monte Carlo sampling (default: 10 samples) by:
        1. For each instance, interpolate both prediction quantiles and queue quantiles
        2. Sample from the same random percentiles to get prediction_time + queue_time
        3. Find the instance with minimum total time for each sample
        4. Return the instance with the most wins across all samples

        Args:
            predictions: List of predictions with quantile information
            queue_info: Dictionary mapping instance_id to queue information

        Returns:
            Selected instance ID (the one with most wins across Monte Carlo samples)
        """
        from src.model import InstanceQueueProbabilistic

        if not predictions:
            return None

        num_samples = 10

        # Build prediction dictionary for quick lookup
        pred_dict = {p.instance_id: p for p in predictions}

        # Use all prediction instance IDs (queue_info is optional)
        instance_ids = list(pred_dict.keys())
        num_instances = len(instance_ids)

        # Generate all random percentiles at once (shared across all instances)
        random_percentiles = np.random.random(num_samples)

        # Build combined time matrix: shape (num_instances, num_samples)
        # Each entry = prediction_time + queue_time
        total_times_matrix = np.zeros((num_instances, num_samples))

        # Vectorized sampling for all instances
        for i, instance_id in enumerate(instance_ids):
            pred = pred_dict[instance_id]

            # Sample prediction times
            if pred.quantiles and len(pred.quantiles) > 0:
                # Convert dict to sorted arrays for interpolation
                pred_quantiles = np.array(sorted(pred.quantiles.keys()))
                pred_values = np.array(
                    [pred.quantiles[q] for q in pred_quantiles]
                )

                # Vectorized prediction time sampling
                prediction_times = np.interp(
                    random_percentiles, pred_quantiles, pred_values
                )
            else:
                # Fallback: use predicted_time_ms as constant
                prediction_times = np.full(num_samples, pred.predicted_time_ms)

            # Sample queue times (optional)
            if instance_id in queue_info:
                queue = queue_info[instance_id]
                if isinstance(queue, InstanceQueueProbabilistic):
                    # Vectorized queue time sampling
                    queue_times = np.interp(
                        random_percentiles, queue.quantiles, queue.values
                    )
                else:
                    # Fallback: no probabilistic queue info, assume zero queue
                    queue_times = np.zeros(num_samples)
            else:
                # No queue info available for this instance
                queue_times = np.zeros(num_samples)

            # Combine prediction + queue time
            total_times_matrix[i, :] = prediction_times + queue_times

        # Find winner for each sample (argmin along axis 0)
        winners = np.argmin(total_times_matrix, axis=0)

        # Count wins for each instance
        win_counts = np.bincount(winners, minlength=num_instances)

        # Select instance with most wins
        best_instance_idx = np.argmax(win_counts)

        return instance_ids[best_instance_idx]

    async def update_queue(
        self,
        instance_id: str,
        prediction: Prediction,
    ) -> None:
        """Update queue using Monte Carlo sampling method.

        Uses 1000 samples to compute new quantile distribution by combining
        queue distribution and task distribution.

        Args:
            instance_id: Selected instance
            prediction: Prediction for the task
        """
        from src.model import InstanceQueueProbabilistic

        current_queue = await self.instance_registry.get_queue_info(instance_id)

        if not current_queue:
            # If no queue exists, initialize with correct type
            current_queue = InstanceQueueProbabilistic(
                instance_id=instance_id,
                quantiles=[0.5, 0.9, 0.95, 0.99],
                values=[0.0, 0.0, 0.0, 0.0],
            )
        elif not isinstance(current_queue, InstanceQueueProbabilistic):
            # Type mismatch - this shouldn't happen if strategy switch was done properly
            logger.warning(
                f"Queue info type mismatch for {instance_id}: "
                f"expected InstanceQueueProbabilistic, got {type(current_queue).__name__}. "
                f"This indicates the strategy switch didn't properly reinitialize queues. Skipping update."
            )
            return

        if not prediction.quantiles:
            # Fallback: use predicted_time_ms for all quantiles
            updated_values = [
                current_queue.values[i] + prediction.predicted_time_ms
                for i in range(len(current_queue.quantiles))
            ]
            updated_queue = InstanceQueueProbabilistic(
                instance_id=instance_id,
                quantiles=current_queue.quantiles,
                values=updated_values,
            )
            await self.instance_registry.update_queue_info(
                instance_id, updated_queue
            )
            logger.debug(
                f"Updated queue (probabilistic, fallback) for {instance_id}: "
                f"quantiles={current_queue.quantiles}, "
                f"values={[f'{v:.2f}' for v in updated_values]}"
            )
            return

        # Monte Carlo method
        num_samples = 1000
        random_percentiles = np.random.random(num_samples)

        # Sample from queue distribution using vectorized numpy interpolation
        queue_samples = np.interp(
            random_percentiles, current_queue.quantiles, current_queue.values
        )

        # Sample from task distribution
        task_quantiles = sorted(prediction.quantiles.keys())
        task_values = [prediction.quantiles[q] for q in task_quantiles]
        task_samples = np.interp(
            random_percentiles, task_quantiles, task_values
        )

        # Compute total time samples
        total_samples = queue_samples + task_samples

        # Compute new quantiles from samples using numpy.percentile
        updated_values = [
            float(np.percentile(total_samples, q * 100))
            for q in current_queue.quantiles
        ]

        updated_queue = InstanceQueueProbabilistic(
            instance_id=instance_id,
            quantiles=current_queue.quantiles,
            values=updated_values,
        )

        await self.instance_registry.update_queue_info(
            instance_id, updated_queue
        )
        logger.debug(
            f"Updated queue (probabilistic, Monte Carlo) for {instance_id}: "
            f"quantiles={current_queue.quantiles}, "
            f"values={[f'{v:.2f}' for v in updated_values]}"
        )
