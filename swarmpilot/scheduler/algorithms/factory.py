"""Factory function for creating scheduling strategies.

Provides a centralized way to instantiate scheduling strategies by name.
"""

from typing import TYPE_CHECKING

from swarmpilot.scheduler.algorithms.adaptive_bootstrap import AdaptiveBootstrapStrategy
from swarmpilot.scheduler.algorithms.min_expected_time import MinimumExpectedTimeStrategy
from swarmpilot.scheduler.algorithms.power_of_two import PowerOfTwoStrategy
from swarmpilot.scheduler.algorithms.probabilistic import ProbabilisticSchedulingStrategy
from swarmpilot.scheduler.algorithms.random import RandomStrategy
from swarmpilot.scheduler.algorithms.round_robin import RoundRobinStrategy
from swarmpilot.scheduler.algorithms.serverless import MinimumExpectedTimeServerlessStrategy

if TYPE_CHECKING:
    from swarmpilot.scheduler.algorithms.base import SchedulingStrategy
    from swarmpilot.scheduler.clients.predictor_library_client import PredictorClient
    from swarmpilot.scheduler.registry.instance_registry import InstanceRegistry


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
                      ("min_time", "probabilistic", "round_robin", etc.)
        predictor_client: Predictor client instance
        instance_registry: Instance registry instance
        target_quantile: Target quantile for probabilistic strategy (default: 0.9)
        **kwargs: Additional keyword arguments (reserved for future use)

    Returns:
        Configured scheduling strategy instance
    """
    if strategy_name == "min_time":
        return MinimumExpectedTimeStrategy(predictor_client, instance_registry)
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
    elif strategy_name == "adaptive_bootstrap":
        return AdaptiveBootstrapStrategy(
            predictor_client, instance_registry, target_quantile=target_quantile
        )
    else:
        # Default to adaptive bootstrap
        return AdaptiveBootstrapStrategy(
            predictor_client, instance_registry, target_quantile=target_quantile
        )
