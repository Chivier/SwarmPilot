#!/usr/bin/env python3
"""
Workload generation module for experiment 02.

Provides two types of workload distributions:
1. Bimodal distribution (reused from exp01)
2. Pareto long-tail distribution (new for exp02)
"""

import numpy as np
from typing import List
from dataclasses import dataclass


# Bimodal distribution parameters (from exp01)
LEFT_PEAK_MIN = 1.0
LEFT_PEAK_MAX = 3.0
LEFT_PEAK_MEAN = 2.0
LEFT_PEAK_STD = 0.4

RIGHT_PEAK_MIN = 7.0
RIGHT_PEAK_MAX = 10.0
RIGHT_PEAK_MEAN = 8.5
RIGHT_PEAK_STD = 0.6

PEAK_RATIO = 0.5  # 50% left peak, 50% right peak

# Pareto distribution parameters
PARETO_MIN = 30.0
PARETO_MAX = 60.0
PARETO_ALPHA = 1.5  # Shape parameter (smaller = more skewed/long-tail)


@dataclass
class WorkloadConfig:
    """Configuration for a workload distribution."""
    name: str
    min_time: float
    max_time: float
    mean_time: float
    std_time: float
    description: str


def generate_bimodal_distribution(num_tasks: int, seed: int = 42) -> tuple[List[float], WorkloadConfig]:
    """
    Generate task execution times from a bimodal distribution.

    The distribution has two peaks:
    - Left peak: 1-3 seconds (mean=2.0s, std=0.4s)
    - Right peak: 7-10 seconds (mean=8.5s, std=0.6s)
    - Ratio: 50% left, 50% right

    Args:
        num_tasks: Number of tasks to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (task_times, config)
    """
    np.random.seed(seed)

    # Calculate number of tasks for each peak
    num_left_peak = int(num_tasks * PEAK_RATIO)
    num_right_peak = num_tasks - num_left_peak

    # Generate left peak
    left_times = np.random.normal(LEFT_PEAK_MEAN, LEFT_PEAK_STD, num_left_peak)
    left_times = np.clip(left_times, LEFT_PEAK_MIN, LEFT_PEAK_MAX)

    # Generate right peak
    right_times = np.random.normal(RIGHT_PEAK_MEAN, RIGHT_PEAK_STD, num_right_peak)
    right_times = np.clip(right_times, RIGHT_PEAK_MIN, RIGHT_PEAK_MAX)

    # Combine and shuffle
    times = np.concatenate([left_times, right_times])
    np.random.shuffle(times)

    # Calculate statistics
    mean_time = float(np.mean(times))
    std_time = float(np.std(times))

    config = WorkloadConfig(
        name="bimodal",
        min_time=LEFT_PEAK_MIN,
        max_time=RIGHT_PEAK_MAX,
        mean_time=mean_time,
        std_time=std_time,
        description=f"Bimodal: {PEAK_RATIO:.0%} at {LEFT_PEAK_MEAN}s, {1-PEAK_RATIO:.0%} at {RIGHT_PEAK_MEAN}s"
    )

    return times.tolist(), config


def generate_pareto_distribution(num_tasks: int,
                                min_time: float = PARETO_MIN,
                                max_time: float = PARETO_MAX,
                                alpha: float = PARETO_ALPHA,
                                seed: int = 42) -> tuple[List[float], WorkloadConfig]:
    """
    Generate task execution times from a Pareto (long-tail) distribution.

    The Pareto distribution creates a long-tail effect where:
    - Most tasks (80%) complete in short time
    - Few tasks (20%) take much longer

    This is scaled and truncated to [min_time, max_time] range.

    Args:
        num_tasks: Number of tasks to generate
        min_time: Minimum task execution time in seconds
        max_time: Maximum task execution time in seconds
        alpha: Pareto shape parameter (smaller = more long-tail, typical: 1.5)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (task_times, config)
    """
    np.random.seed(seed)

    # Generate Pareto samples (returns values >= 1.0)
    # Using numpy's pareto distribution: f(x) = alpha / x^(alpha+1)
    raw_samples = np.random.pareto(alpha, num_tasks) + 1.0

    # Scale to [0, 1] range using log transform for better distribution
    # Log transform helps spread the long tail more evenly
    log_samples = np.log(raw_samples)
    log_min = log_samples.min()
    log_max = log_samples.max()
    normalized = (log_samples - log_min) / (log_max - log_min)

    # Scale to desired range [min_time, max_time]
    times = min_time + normalized * (max_time - min_time)

    # Shuffle to randomize order
    np.random.shuffle(times)

    # Calculate statistics
    mean_time = float(np.mean(times))
    std_time = float(np.std(times))
    p80 = float(np.percentile(times, 80))
    p95 = float(np.percentile(times, 95))

    config = WorkloadConfig(
        name="pareto",
        min_time=min_time,
        max_time=max_time,
        mean_time=mean_time,
        std_time=std_time,
        description=f"Pareto (alpha={alpha}): 80%<{p80:.1f}s, 95%<{p95:.1f}s, long-tail to {max_time:.1f}s"
    )

    return times.tolist(), config


def print_distribution_stats(times: List[float], config: WorkloadConfig):
    """
    Print detailed statistics about a workload distribution.

    Args:
        times: List of task execution times
        config: Workload configuration
    """
    times_array = np.array(times)

    print(f"\nWorkload: {config.name}")
    print(f"Description: {config.description}")
    print(f"Statistics:")
    print(f"  Count:  {len(times)}")
    print(f"  Min:    {times_array.min():.3f}s")
    print(f"  Max:    {times_array.max():.3f}s")
    print(f"  Mean:   {config.mean_time:.3f}s")
    print(f"  Median: {np.median(times_array):.3f}s")
    print(f"  Std:    {config.std_time:.3f}s")
    print(f"  P50:    {np.percentile(times_array, 50):.3f}s")
    print(f"  P80:    {np.percentile(times_array, 80):.3f}s")
    print(f"  P90:    {np.percentile(times_array, 90):.3f}s")
    print(f"  P95:    {np.percentile(times_array, 95):.3f}s")
    print(f"  P99:    {np.percentile(times_array, 99):.3f}s")


if __name__ == "__main__":
    """Demo and testing of workload generators."""
    import argparse

    parser = argparse.ArgumentParser(description="Test workload generation")
    parser.add_argument("--num-tasks", type=int, default=100, help="Number of tasks to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("Workload Generator Demo")
    print("=" * 60)

    # Generate and display bimodal distribution
    bimodal_times, bimodal_config = generate_bimodal_distribution(args.num_tasks, args.seed)
    print_distribution_stats(bimodal_times, bimodal_config)

    # Generate and display Pareto distribution
    pareto_times, pareto_config = generate_pareto_distribution(args.num_tasks, seed=args.seed)
    print_distribution_stats(pareto_times, pareto_config)

    print("\n" + "=" * 60)
