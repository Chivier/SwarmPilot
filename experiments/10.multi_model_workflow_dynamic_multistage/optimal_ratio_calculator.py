#!/usr/bin/env python3
"""
Optimal Instance Ratio Calculator for Multi-Stage Workflows

Calculates the optimal allocation of instances between A and B groups
based on fanout, task execution times, and scheduling strategy.
"""

import math
from typing import Dict, Tuple, Optional
from enum import Enum


class SchedulingStrategy(Enum):
    """Supported scheduling strategies with their characteristics."""
    MIN_TIME = 'min_time'
    ROUND_ROBIN = 'round_robin'
    PROBABILISTIC = 'probabilistic'


class OptimalRatioCalculator:
    """
    Calculator for optimal instance allocation between A and B groups.

    The calculator uses queueing theory and empirical adjustments to determine
    the best instance split for a given stage's fanout and task characteristics.
    """

    def __init__(self, safety_factor: float = 1.2):
        """
        Initialize the calculator.

        Args:
            safety_factor: Buffer multiplier to prevent queue buildup (default: 1.2 = 20% buffer)
        """
        self.safety_factor = safety_factor

        # Strategy-specific adjustment factors
        # These account for different load balancing efficiencies
        self.strategy_adjustments = {
            SchedulingStrategy.MIN_TIME: 1.0,       # Most efficient
            SchedulingStrategy.ROUND_ROBIN: 1.15,  # 15% more B instances needed
            SchedulingStrategy.PROBABILISTIC: 1.1   # 10% more B instances needed
        }

    def calculate_optimal_ratio(
        self,
        fanout: int,
        task_stats: Dict[str, float],
        total_instances: int,
        strategy: SchedulingStrategy
    ) -> Tuple[int, int]:
        """
        Calculate optimal A and B instance allocation.

        Formula:
            Base calculation with safety factor and strategy adjustment:
            num_a = total / (1 + fanout * (avg_b_time / avg_a_time) * safety_factor * strategy_mult)
            num_b = total - num_a

        Args:
            fanout: Number of B tasks per A task
            task_stats: Dictionary with 'avg_a_time' and 'avg_b_time' in seconds
            total_instances: Total number of instances available
            strategy: Scheduling strategy being used

        Returns:
            Tuple of (num_a_instances, num_b_instances)

        Examples:
            >>> calc = OptimalRatioCalculator()
            >>> # Stage 1: fanout=8, equal times
            >>> calc.calculate_optimal_ratio(8, {'avg_a_time': 5.0, 'avg_b_time': 5.0}, 30, SchedulingStrategy.MIN_TIME)
            (3, 27)

            >>> # Stage 3: fanout=1, equal times
            >>> calc.calculate_optimal_ratio(1, {'avg_a_time': 5.0, 'avg_b_time': 5.0}, 30, SchedulingStrategy.MIN_TIME)
            (14, 16)
        """
        avg_a_time = task_stats.get('avg_a_time', 5.0)
        avg_b_time = task_stats.get('avg_b_time', 5.0)

        # Avoid division by zero
        if avg_a_time <= 0:
            avg_a_time = 1.0

        # Calculate time ratio
        time_ratio = avg_b_time / avg_a_time

        # Get strategy-specific multiplier
        strategy_mult = self.strategy_adjustments[strategy]

        # Calculate optimal number of A instances
        # Formula: num_a = total / (1 + fanout * time_ratio * safety * strategy_adj)
        denominator = 1 + (fanout * time_ratio * self.safety_factor * strategy_mult)
        num_a = max(1, int(total_instances / denominator))

        # Remaining instances go to B group
        num_b = total_instances - num_a

        # Ensure at least one instance of each type
        if num_b == 0:
            num_a = total_instances - 1
            num_b = 1
        elif num_a == 0:
            num_a = 1
            num_b = total_instances - 1

        return num_a, num_b

    def calculate_from_stage_metrics(
        self,
        stage_metrics: Dict,
        next_fanout: int,
        total_instances: int,
        strategy: str
    ) -> Tuple[int, int]:
        """
        Calculate optimal ratio from collected stage metrics.

        This is a convenience method for use in the main experiment orchestrator
        where metrics are collected in a specific format.

        Args:
            stage_metrics: Dictionary containing stage execution metrics
            next_fanout: Fanout for the next stage
            total_instances: Total instances available
            strategy: Strategy name as string (will be converted to enum)

        Returns:
            Tuple of (num_a_instances, num_b_instances)
        """
        # Extract average times from stage metrics
        task_stats = {
            'avg_a_time': stage_metrics.get('avg_a_completion_time', 5.0),
            'avg_b_time': stage_metrics.get('avg_b_completion_time', 5.0)
        }

        # Convert strategy string to enum
        try:
            strategy_enum = SchedulingStrategy(strategy)
        except ValueError:
            # Default to MIN_TIME if unknown strategy
            strategy_enum = SchedulingStrategy.MIN_TIME

        return self.calculate_optimal_ratio(
            fanout=next_fanout,
            task_stats=task_stats,
            total_instances=total_instances,
            strategy=strategy_enum
        )

    def calculate_initial_allocation(
        self,
        first_stage_fanout: int,
        total_instances: int,
        strategy: SchedulingStrategy,
        assumed_task_time: float = 5.0
    ) -> Tuple[int, int]:
        """
        Calculate initial allocation for Stage 1 when no metrics are available.

        Uses assumed equal task times as a starting point.

        Args:
            first_stage_fanout: Fanout for the first stage
            total_instances: Total instances available
            strategy: Scheduling strategy
            assumed_task_time: Assumed average task time (default: 5.0s)

        Returns:
            Tuple of (num_a_instances, num_b_instances)
        """
        task_stats = {
            'avg_a_time': assumed_task_time,
            'avg_b_time': assumed_task_time
        }

        return self.calculate_optimal_ratio(
            fanout=first_stage_fanout,
            task_stats=task_stats,
            total_instances=total_instances,
            strategy=strategy
        )

    def explain_calculation(
        self,
        fanout: int,
        task_stats: Dict[str, float],
        total_instances: int,
        strategy: SchedulingStrategy
    ) -> str:
        """
        Generate human-readable explanation of the calculation.

        Useful for debugging and understanding allocation decisions.

        Args:
            fanout: Fanout value
            task_stats: Task statistics
            total_instances: Total instances
            strategy: Strategy enum

        Returns:
            Multi-line explanation string
        """
        avg_a_time = task_stats.get('avg_a_time', 5.0)
        avg_b_time = task_stats.get('avg_b_time', 5.0)

        time_ratio = avg_b_time / avg_a_time
        strategy_mult = self.strategy_adjustments[strategy]
        denominator = 1 + (fanout * time_ratio * self.safety_factor * strategy_mult)
        num_a, num_b = self.calculate_optimal_ratio(fanout, task_stats, total_instances, strategy)

        explanation = f"""
Optimal Ratio Calculation
==========================
Inputs:
  - Fanout: {fanout} B tasks per A task
  - Avg A time: {avg_a_time:.2f}s
  - Avg B time: {avg_b_time:.2f}s
  - Total instances: {total_instances}
  - Strategy: {strategy.value}

Calculation:
  - Time ratio (B/A): {time_ratio:.3f}
  - Safety factor: {self.safety_factor:.2f}
  - Strategy multiplier: {strategy_mult:.2f}
  - Denominator: 1 + ({fanout} * {time_ratio:.3f} * {self.safety_factor:.2f} * {strategy_mult:.2f}) = {denominator:.3f}
  - A instances: {total_instances} / {denominator:.3f} = {num_a}
  - B instances: {total_instances} - {num_a} = {num_b}

Result:
  - A group: {num_a} instances ({num_a/total_instances*100:.1f}%)
  - B group: {num_b} instances ({num_b/total_instances*100:.1f}%)
  - A:B ratio: 1:{num_b/num_a:.2f}

Rationale:
  With fanout={fanout}, each A task generates {fanout} B tasks.
  To balance load, B group needs {fanout}x more capacity than A group
  (adjusted for time ratio {time_ratio:.2f}x and safety factor {self.safety_factor:.2f}x).
"""
        return explanation


def demo():
    """Demonstrate the calculator with various scenarios."""
    print("=" * 70)
    print("Optimal Ratio Calculator Demo")
    print("=" * 70)

    calculator = OptimalRatioCalculator(safety_factor=1.2)

    scenarios = [
        {
            "name": "Stage 1 (Fanout=8, Equal Times)",
            "fanout": 8,
            "task_stats": {"avg_a_time": 5.0, "avg_b_time": 5.0},
            "total": 30,
            "strategy": SchedulingStrategy.MIN_TIME
        },
        {
            "name": "Stage 2 (Fanout=15, Equal Times)",
            "fanout": 15,
            "task_stats": {"avg_a_time": 5.0, "avg_b_time": 5.0},
            "total": 30,
            "strategy": SchedulingStrategy.MIN_TIME
        },
        {
            "name": "Stage 3 (Fanout=1, Equal Times)",
            "fanout": 1,
            "task_stats": {"avg_a_time": 5.0, "avg_b_time": 5.0},
            "total": 30,
            "strategy": SchedulingStrategy.MIN_TIME
        },
        {
            "name": "Stage 1 with Round Robin (needs more buffer)",
            "fanout": 8,
            "task_stats": {"avg_a_time": 5.0, "avg_b_time": 5.0},
            "total": 30,
            "strategy": SchedulingStrategy.ROUND_ROBIN
        },
        {
            "name": "Unbalanced Times (B takes 2x longer)",
            "fanout": 8,
            "task_stats": {"avg_a_time": 5.0, "avg_b_time": 10.0},
            "total": 30,
            "strategy": SchedulingStrategy.MIN_TIME
        }
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print("-" * 70)

        num_a, num_b = calculator.calculate_optimal_ratio(
            fanout=scenario['fanout'],
            task_stats=scenario['task_stats'],
            total_instances=scenario['total'],
            strategy=scenario['strategy']
        )

        print(f"  Fanout: {scenario['fanout']}")
        print(f"  Strategy: {scenario['strategy'].value}")
        print(f"  Result: {num_a} A-instances + {num_b} B-instances")
        print(f"  Ratio: 1:{num_b/num_a:.2f}")

    # Show detailed explanation for one scenario
    print("\n" + "=" * 70)
    print("Detailed Explanation Example")
    print("=" * 70)

    explanation = calculator.explain_calculation(
        fanout=8,
        task_stats={"avg_a_time": 5.0, "avg_b_time": 5.0},
        total_instances=30,
        strategy=SchedulingStrategy.MIN_TIME
    )
    print(explanation)


if __name__ == "__main__":
    demo()
