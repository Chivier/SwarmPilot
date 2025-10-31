"""
Scheduling strategies for task assignment to instances.

This module implements various scheduling strategies to select the best
instance for executing a task based on predictions.
"""

from typing import List, Optional, Dict, TYPE_CHECKING
from abc import ABC, abstractmethod

from .predictor_client import Prediction

if TYPE_CHECKING:
    from .model import InstanceQueueBase


class SchedulingStrategy(ABC):
    """Abstract base class for scheduling strategies."""

    @abstractmethod
    def select_instance(
        self, predictions: List[Prediction], queue_info: Dict[str, "InstanceQueueBase"]
    ) -> Optional[str]:
        """
        Select the best instance from predictions.

        Args:
            predictions: List of predictions for different instances
            queue_info: Dictionary mapping instance_id to queue information

        Returns:
            Selected instance ID, or None if no suitable instance found
        """
        pass

    def get_prediction_type(self) -> str:
        """
        Get the prediction type required by this strategy.

        Returns:
            Prediction type: "expect_error" or "quantile"
        """
        # Default to expect_error for simplicity
        return "expect_error"


class MinimumExpectedTimeStrategy(SchedulingStrategy):
    """
    Strategy that selects the instance with minimum expected queue completion time.

    This strategy considers both the current queue state (expected time and error margin)
    and the predicted time for the new task. It selects the instance that minimizes
    the total expected completion time including uncertainty.
    """

    def select_instance(
        self, predictions: List[Prediction], queue_info: Dict[str, "InstanceQueueBase"]
    ) -> Optional[str]:
        """
        Select instance with minimum total expected time (queue + new task).

        For each instance, calculates: queue_expected + queue_error + task_expected
        and selects the instance with the minimum value.
        """
        from .model import InstanceQueueExpectError

        if not predictions:
            return None

        best_instance_id = None
        best_total_time = float("inf")

        for pred in predictions:
            # Get queue information for this instance
            queue = queue_info.get(pred.instance_id)

            if queue and isinstance(queue, InstanceQueueExpectError):
                # Calculate total time: queue expected + queue error + new task expected
                total_time = queue.expected_time_ms + queue.error_margin_ms + pred.predicted_time_ms
            else:
                # Fallback: no queue info, just use prediction
                total_time = pred.predicted_time_ms

            if total_time < best_total_time:
                best_total_time = total_time
                best_instance_id = pred.instance_id

        return best_instance_id


class ProbabilisticSchedulingStrategy(SchedulingStrategy):
    """
    Probabilistic scheduling strategy based on quantile sampling.

    This strategy considers both the queue distribution and the new task distribution.
    For each instance, it samples from the queue's quantile distribution to estimate
    completion time, then selects the instance with the minimum sampled value.
    """

    def __init__(self, target_quantile: float = 0.9):
        """
        Initialize probabilistic strategy.

        Args:
            target_quantile: Quantile to optimize for (default 0.9 = 90th percentile)
        """
        self.target_quantile = target_quantile

    def get_prediction_type(self) -> str:
        """Probabilistic strategy requires quantile predictions."""
        return "quantile"

    def select_instance(
        self, predictions: List[Prediction], queue_info: Dict[str, "InstanceQueueBase"]
    ) -> Optional[str]:
        """
        Select instance based on sampling from queue quantile distribution.

        For each instance, samples once from its queue distribution by interpolating
        between quantile values, then selects the instance with minimum sampled time.

        Args:
            predictions: List of predictions with quantile information
            queue_info: Dictionary mapping instance_id to queue information

        Returns:
            Selected instance ID
        """
        import numpy as np
        from .model import InstanceQueueProbabilistic

        if not predictions:
            return None

        best_instance_id = None
        best_sampled_time = float("inf")

        for pred in predictions:
            # Get queue information for this instance
            queue = queue_info.get(pred.instance_id)

            if queue and isinstance(queue, InstanceQueueProbabilistic):
                # Sample from queue distribution
                # Use a random percentile between 0 and 1
                random_percentile = np.random.random()

                # Interpolate queue time from quantiles using numpy
                queue_time = np.interp(
                    random_percentile,
                    queue.quantiles,
                    queue.values
                )

                # Sample from task distribution
                if pred.quantiles:
                    task_quantiles = sorted(pred.quantiles.keys())
                    task_values = [pred.quantiles[q] for q in task_quantiles]
                    task_time = np.interp(random_percentile, task_quantiles, task_values)
                else:
                    task_time = pred.predicted_time_ms

                total_time = queue_time + task_time
            else:
                # Fallback: no queue info or wrong type
                total_time = pred.predicted_time_ms

            if total_time < best_sampled_time:
                best_sampled_time = total_time
                best_instance_id = pred.instance_id

        return best_instance_id


class RoundRobinStrategy(SchedulingStrategy):
    """
    Round-robin scheduling strategy.

    Simple strategy that cycles through instances in order.
    Useful for load balancing when predictions are not available.
    """

    def __init__(self):
        self._counter = 0

    def select_instance(
        self, predictions: List[Prediction], queue_info: Dict[str, "InstanceQueueBase"]
    ) -> Optional[str]:
        """Select next instance in round-robin order."""
        if not predictions:
            return None

        selected = predictions[self._counter % len(predictions)]
        self._counter += 1
        return selected.instance_id




# Factory function to get strategy by name
def get_strategy(
    strategy_name: str = "probabilistic", **kwargs
) -> SchedulingStrategy:
    """
    Get scheduling strategy by name.

    Args:
        strategy_name: Name of strategy
                      ("min_time", "probabilistic", "round_robin")
        **kwargs: Additional keyword arguments passed to strategy constructor
                 For probabilistic: target_quantile (default 0.9)

    Returns:
        Scheduling strategy instance
    """
    strategies = {
        "min_time": MinimumExpectedTimeStrategy,
        "probabilistic": ProbabilisticSchedulingStrategy,
        "round_robin": RoundRobinStrategy,
    }

    strategy_class = strategies.get(strategy_name, ProbabilisticSchedulingStrategy)

    # Pass kwargs to strategies that support them
    if strategy_name == "probabilistic" and "target_quantile" in kwargs:
        return strategy_class(target_quantile=kwargs["target_quantile"])
    elif strategy_class == MinimumExpectedTimeStrategy or strategy_class == RoundRobinStrategy:
        return strategy_class()
    else:
        return strategy_class()
