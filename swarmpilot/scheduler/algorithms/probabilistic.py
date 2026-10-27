"""Probabilistic scheduling strategy based on quantile sampling.

This strategy considers both the queue distribution and the new task distribution.
For each instance, it samples from the queue's quantile distribution to estimate
completion time, then selects the instance with the minimum sampled value.
"""

from typing import TYPE_CHECKING

import numpy as np

from swarmpilot.scheduler import core
from swarmpilot.scheduler.algorithms.base import SchedulingStrategy
from swarmpilot.scheduler.clients.models import Prediction

if TYPE_CHECKING:
    from swarmpilot.scheduler.clients.predictor_library_client import (
        PredictorClient,
    )
    from swarmpilot.scheduler.models import InstanceQueueBase
    from swarmpilot.scheduler.registry.instance_registry import InstanceRegistry


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
        seed: int | None = None,
    ):
        """Initialize probabilistic strategy.

        Args:
            predictor_client: Client for getting predictions
            instance_registry: Registry for instance and queue management
            target_quantile: Target quantile for probabilistic selection (default: 0.9)
            seed: Seed for the random generator (experiments only)
        """
        super().__init__(predictor_client, instance_registry, seed=seed)
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
        from swarmpilot.scheduler.models import InstanceQueueProbabilistic

        if not predictions:
            return None

        num_samples = 10

        # Build prediction dictionary for quick lookup
        pred_dict = {p.instance_id: p for p in predictions}

        # Use all prediction instance IDs (queue_info is optional)
        instance_ids = list(pred_dict.keys())
        num_instances = len(instance_ids)

        # Generate all random percentiles at once (shared across all instances)
        random_percentiles = self._rng.random(num_samples)

        rc = core.rust()
        if rc is not None:
            k = rc.select_probabilistic(
                [p.instance_id for p in predictions],
                [p.predicted_time_ms for p in predictions],
                [
                    sorted(p.quantiles.items()) if p.quantiles else None
                    for p in predictions
                ],
                [
                    (q.quantiles, q.values)
                    if isinstance(q, InstanceQueueProbabilistic)
                    else None
                    for q in (
                        queue_info.get(p.instance_id) for p in predictions
                    )
                ],
                random_percentiles.tolist(),
            )
            return predictions[k].instance_id

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
