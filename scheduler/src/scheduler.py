"""Scheduling strategies for task assignment to instances.

This module provides backward compatibility by re-exporting all
scheduling strategies from the src.algorithms package.

For new code, prefer importing directly from src.algorithms:
    from src.algorithms import MinimumExpectedTimeStrategy, get_strategy
"""

# Re-export all public classes and functions from algorithms package
from src.algorithms import (
    MinimumExpectedTimeDTStrategy,
    MinimumExpectedTimeLRStrategy,
    MinimumExpectedTimeServerlessStrategy,
    MinimumExpectedTimeStrategy,
    PowerOfTwoStrategy,
    ProbabilisticSchedulingStrategy,
    RandomStrategy,
    RoundRobinStrategy,
    ScheduleResult,
    SchedulingStrategy,
    get_strategy,
)

__all__ = [
    "SchedulingStrategy",
    "ScheduleResult",
    "MinimumExpectedTimeStrategy",
    "ProbabilisticSchedulingStrategy",
    "RoundRobinStrategy",
    "RandomStrategy",
    "PowerOfTwoStrategy",
    "MinimumExpectedTimeServerlessStrategy",
    "MinimumExpectedTimeLRStrategy",
    "MinimumExpectedTimeDTStrategy",
    "get_strategy",
]
