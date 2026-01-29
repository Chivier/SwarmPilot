"""Factory function for creating scheduling strategies.

Provides a centralized way to instantiate scheduling strategies by name.
"""

from typing import TYPE_CHECKING

from src.algorithms.min_expected_time import MinimumExpectedTimeStrategy
from src.algorithms.min_expected_time_dt import MinimumExpectedTimeDTStrategy
from src.algorithms.min_expected_time_lr import MinimumExpectedTimeLRStrategy
from src.algorithms.power_of_two import PowerOfTwoStrategy
from src.algorithms.probabilistic import ProbabilisticSchedulingStrategy
from src.algorithms.random import RandomStrategy
from src.algorithms.round_robin import RoundRobinStrategy
from src.algorithms.serverless import MinimumExpectedTimeServerlessStrategy

if TYPE_CHECKING:
    from src.algorithms.base import SchedulingStrategy
    from src.clients.predictor_library_client import PredictorClient
    from src.registry.instance_registry import InstanceRegistry


def get_strategy(
    strategy_name: str,
    predictor_client: "PredictorClient",
    instance_registry: "InstanceRegistry",
    target_quantile: float = 0.9,
    **kwargs,
) -> "SchedulingStrategy":
    """Get scheduling strategy by name.

    Args:
        strategy_name: Name of strategy
                      ("min_time", "probabilistic", "round_robin", "min_time_lr", "min_time_dt")
        predictor_client: Predictor client instance
        instance_registry: Instance registry instance
        target_quantile: Target quantile for probabilistic strategy (default: 0.9)
        **kwargs: Additional keyword arguments (reserved for future use)

    Returns:
        Configured scheduling strategy instance
    """
    if strategy_name == "min_time":
        return MinimumExpectedTimeStrategy(predictor_client, instance_registry)
    elif strategy_name == "min_time_lr":
        return MinimumExpectedTimeLRStrategy(predictor_client, instance_registry)
    elif strategy_name == "min_time_dt":
        return MinimumExpectedTimeDTStrategy(predictor_client, instance_registry)
    elif strategy_name == "probabilistic":
        return ProbabilisticSchedulingStrategy(
            predictor_client, instance_registry, target_quantile=target_quantile
        )
    elif strategy_name == "round_robin":
        return RoundRobinStrategy(predictor_client, instance_registry)
    elif strategy_name == "random":
        return RandomStrategy(predictor_client, instance_registry)
    elif strategy_name == "po2":
        return PowerOfTwoStrategy(predictor_client, instance_registry)
    elif strategy_name == "severless":
        return MinimumExpectedTimeServerlessStrategy(
            predictor_client, instance_registry
        )
    else:
        # Default to probabilistic
        return ProbabilisticSchedulingStrategy(
            predictor_client, instance_registry, target_quantile=target_quantile
        )
