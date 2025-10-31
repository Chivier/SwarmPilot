"""
Scheduling strategies for task assignment to instances.

This module implements various scheduling strategies to select the best
instance for executing a task based on predictions.
"""

from typing import List, Optional
from abc import ABC, abstractmethod

from .predictor_client import Prediction


class SchedulingStrategy(ABC):
    """Abstract base class for scheduling strategies."""

    @abstractmethod
    def select_instance(self, predictions: List[Prediction]) -> Optional[str]:
        """
        Select the best instance from predictions.

        Args:
            predictions: List of predictions for different instances

        Returns:
            Selected instance ID, or None if no suitable instance found
        """
        pass


class MinimumExpectedTimeStrategy(SchedulingStrategy):
    """
    Strategy that selects the instance with minimum expected execution time.

    This is a simple greedy strategy that chooses the instance predicted
    to complete the task fastest.
    """

    def select_instance(self, predictions: List[Prediction]) -> Optional[str]:
        """Select instance with minimum predicted time."""
        if not predictions:
            return None

        # Find prediction with minimum predicted time
        best_prediction = min(predictions, key=lambda p: p.predicted_time_ms)
        return best_prediction.instance_id


class ProbabilisticSchedulingStrategy(SchedulingStrategy):
    """
    Probabilistic scheduling strategy based on quantile analysis.

    This strategy considers not just the expected time but also the
    uncertainty/variance in predictions. It may prefer instances with
    more consistent performance over those with lower average but higher variance.
    """

    def __init__(self, target_quantile: float = 0.9):
        """
        Initialize probabilistic strategy.

        Args:
            target_quantile: Quantile to optimize for (default 0.9 = 90th percentile)
        """
        self.target_quantile = target_quantile

    def select_instance(self, predictions: List[Prediction]) -> Optional[str]:
        """
        Select instance based on target quantile.

        Args:
            predictions: List of predictions with quantile information

        Returns:
            Selected instance ID
        """
        if not predictions:
            return None

        # TODO: Implement more sophisticated selection logic based on quantiles
        # TODO: Consider current queue length on each instance
        # TODO: Consider historical performance variance

        # Simple implementation: select based on target quantile value
        best_prediction = None
        best_quantile_value = float("inf")

        for pred in predictions:
            if pred.quantiles and self.target_quantile in pred.quantiles:
                quantile_value = pred.quantiles[self.target_quantile]
                if quantile_value < best_quantile_value:
                    best_quantile_value = quantile_value
                    best_prediction = pred

        # Fallback to minimum expected time if quantile not available
        if best_prediction is None:
            return MinimumExpectedTimeStrategy().select_instance(predictions)

        return best_prediction.instance_id


class RoundRobinStrategy(SchedulingStrategy):
    """
    Round-robin scheduling strategy.

    Simple strategy that cycles through instances in order.
    Useful for load balancing when predictions are not available.
    """

    def __init__(self):
        self._counter = 0

    def select_instance(self, predictions: List[Prediction]) -> Optional[str]:
        """Select next instance in round-robin order."""
        if not predictions:
            return None

        selected = predictions[self._counter % len(predictions)]
        self._counter += 1
        return selected.instance_id




# Factory function to get strategy by name
def get_strategy(strategy_name: str = "probabilistic") -> SchedulingStrategy:
    """
    Get scheduling strategy by name.

    Args:
        strategy_name: Name of strategy
                      ("min_time", "probabilistic", "round_robin", "weighted")

    Returns:
        Scheduling strategy instance
    """
    strategies = {
        "min_time": MinimumExpectedTimeStrategy,
        "probabilistic": ProbabilisticSchedulingStrategy,
        "round_robin": RoundRobinStrategy,
    }

    strategy_class = strategies.get(strategy_name, ProbabilisticSchedulingStrategy)
    return strategy_class()
