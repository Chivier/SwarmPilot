#!/usr/bin/env python3
"""
Workload generation module for unified multi-model workflow experiments.

Provides workload distributions for experiments 04-07:
1. Bimodal distribution for A and B tasks
2. Fanout distribution (number of B tasks per A task)
3. B1/B2 split distributions for experiment 07
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


# Bimodal distribution parameters for A tasks (Experiment 04 standard)
A_LEFT_PEAK_MIN = 1.0
A_LEFT_PEAK_MAX = 3.0
A_LEFT_PEAK_MEAN = 2.0
A_LEFT_PEAK_STD = 0.4

A_RIGHT_PEAK_MIN = 7.0
A_RIGHT_PEAK_MAX = 10.0
A_RIGHT_PEAK_MEAN = 8.5
A_RIGHT_PEAK_STD = 0.6

A_PEAK_RATIO = 0.5  # 50% left peak (fast), 50% right peak (slow)

# Bimodal distribution parameters for B tasks (same as A tasks in experiments 04-06)
B_LEFT_PEAK_MIN = 1.0
B_LEFT_PEAK_MAX = 3.0
B_LEFT_PEAK_MEAN = 2.0
B_LEFT_PEAK_STD = 0.4

B_RIGHT_PEAK_MIN = 7.0
B_RIGHT_PEAK_MAX = 10.0
B_RIGHT_PEAK_MEAN = 8.5
B_RIGHT_PEAK_STD = 0.6

B_PEAK_RATIO = 0.5  # 50% left peak (fast), 50% right peak (slow)

# B1/B2 distribution parameters for experiment 07
# B1: All slow peak only (7-10s)
B1_PEAK_MIN = 7.0
B1_PEAK_MAX = 10.0
B1_PEAK_MEAN = 8.5
B1_PEAK_STD = 0.6

# B2: All fast peak only (1-3s)
B2_PEAK_MIN = 1.0
B2_PEAK_MAX = 3.0
B2_PEAK_MEAN = 2.0
B2_PEAK_STD = 0.4

# Fanout distribution parameters (Experiment 04 standard)
FANOUT_MIN = 3  # Minimum number of B tasks per A task
FANOUT_MAX = 8  # Maximum number of B tasks per A task


@dataclass
class WorkloadConfig:
    """Configuration for a workload distribution."""
    name: str
    min_time: float
    max_time: float
    mean_time: float
    std_time: float
    description: str


@dataclass
class FanoutConfig:
    """Configuration for fanout distribution."""
    name: str
    min_fanout: int
    max_fanout: int
    mean_fanout: float
    std_fanout: float
    description: str


def generate_bimodal_distribution(num_tasks: int, seed: int = 42) -> Tuple[List[float], WorkloadConfig]:
    """
    Generate task execution times from a bimodal distribution.

    The distribution has two peaks (Experiment 04 standard):
    - Left peak (fast): 1-3 seconds (mean=2.0s, std=0.4s)
    - Right peak (slow): 7-10 seconds (mean=8.5s, std=0.6s)
    - Ratio: 50% left, 50% right

    Args:
        num_tasks: Number of tasks to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (task_times, config)
    """
    np.random.seed(seed)

    # Calculate number of tasks for each peak
    num_left_peak = int(num_tasks * A_PEAK_RATIO)
    num_right_peak = num_tasks - num_left_peak

    # Generate left peak (fast tasks)
    left_times = np.random.normal(A_LEFT_PEAK_MEAN, A_LEFT_PEAK_STD, num_left_peak)
    left_times = np.clip(left_times, A_LEFT_PEAK_MIN, A_LEFT_PEAK_MAX)

    # Generate right peak (slow tasks)
    right_times = np.random.normal(A_RIGHT_PEAK_MEAN, A_RIGHT_PEAK_STD, num_right_peak)
    right_times = np.clip(right_times, A_RIGHT_PEAK_MIN, A_RIGHT_PEAK_MAX)

    # Combine and shuffle
    times = np.concatenate([left_times, right_times])
    np.random.shuffle(times)

    # Calculate statistics
    mean_time = float(np.mean(times))
    std_time = float(np.std(times))

    config = WorkloadConfig(
        name="bimodal",
        min_time=A_LEFT_PEAK_MIN,
        max_time=A_RIGHT_PEAK_MAX,
        mean_time=mean_time,
        std_time=std_time,
        description=f"Bimodal: {A_PEAK_RATIO:.0%} fast ({A_LEFT_PEAK_MEAN}s), {1-A_PEAK_RATIO:.0%} slow ({A_RIGHT_PEAK_MEAN}s)"
    )

    return times.tolist(), config


def generate_b_task_bimodal_distribution(num_tasks: int, seed: int = 42) -> Tuple[List[float], WorkloadConfig]:
    """
    Generate B task execution times from a bimodal distribution.

    The distribution has two peaks (same as A tasks for experiments 04-06):
    - Left peak (fast): 1-3 seconds (mean=2.0s, std=0.4s)
    - Right peak (slow): 7-10 seconds (mean=8.5s, std=0.6s)
    - Ratio: 50% left, 50% right

    Args:
        num_tasks: Number of B tasks to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (task_times, config)
    """
    np.random.seed(seed)

    # Calculate number of tasks for each peak
    num_left_peak = int(num_tasks * B_PEAK_RATIO)
    num_right_peak = num_tasks - num_left_peak

    # Generate left peak (fast tasks)
    left_times = np.random.normal(B_LEFT_PEAK_MEAN, B_LEFT_PEAK_STD, num_left_peak)
    left_times = np.clip(left_times, B_LEFT_PEAK_MIN, B_LEFT_PEAK_MAX)

    # Generate right peak (slow tasks)
    right_times = np.random.normal(B_RIGHT_PEAK_MEAN, B_RIGHT_PEAK_STD, num_right_peak)
    right_times = np.clip(right_times, B_RIGHT_PEAK_MIN, B_RIGHT_PEAK_MAX)

    # Combine and shuffle
    times = np.concatenate([left_times, right_times])
    np.random.shuffle(times)

    # Calculate statistics
    mean_time = float(np.mean(times))
    std_time = float(np.std(times))

    config = WorkloadConfig(
        name="b_task_bimodal",
        min_time=B_LEFT_PEAK_MIN,
        max_time=B_RIGHT_PEAK_MAX,
        mean_time=mean_time,
        std_time=std_time,
        description=f"B-task Bimodal: {B_PEAK_RATIO:.0%} fast ({B_LEFT_PEAK_MEAN}s), {1-B_PEAK_RATIO:.0%} slow ({B_RIGHT_PEAK_MEAN}s)"
    )

    return times.tolist(), config


def generate_b1_task_distribution(num_tasks: int, seed: int = 42) -> Tuple[List[float], WorkloadConfig]:
    """
    Generate B1 task execution times (Experiment 07 only).

    B1 tasks use only the slow peak:
    - All slow: 7-10 seconds (mean=8.5s, std=0.6s)

    Args:
        num_tasks: Number of B1 tasks to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (task_times, config)
    """
    np.random.seed(seed)

    # Generate all slow tasks
    times = np.random.normal(B1_PEAK_MEAN, B1_PEAK_STD, num_tasks)
    times = np.clip(times, B1_PEAK_MIN, B1_PEAK_MAX)

    # Calculate statistics
    mean_time = float(np.mean(times))
    std_time = float(np.std(times))

    config = WorkloadConfig(
        name="b1_slow_only",
        min_time=B1_PEAK_MIN,
        max_time=B1_PEAK_MAX,
        mean_time=mean_time,
        std_time=std_time,
        description=f"B1-task (slow only): {B1_PEAK_MIN}-{B1_PEAK_MAX}s (mean={B1_PEAK_MEAN}s)"
    )

    return times.tolist(), config


def generate_b2_task_distribution(num_tasks: int, seed: int = 42) -> Tuple[List[float], WorkloadConfig]:
    """
    Generate B2 task execution times (Experiment 07 only).

    B2 tasks use only the fast peak:
    - All fast: 1-3 seconds (mean=2.0s, std=0.4s)

    Args:
        num_tasks: Number of B2 tasks to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (task_times, config)
    """
    np.random.seed(seed)

    # Generate all fast tasks
    times = np.random.normal(B2_PEAK_MEAN, B2_PEAK_STD, num_tasks)
    times = np.clip(times, B2_PEAK_MIN, B2_PEAK_MAX)

    # Calculate statistics
    mean_time = float(np.mean(times))
    std_time = float(np.std(times))

    config = WorkloadConfig(
        name="b2_fast_only",
        min_time=B2_PEAK_MIN,
        max_time=B2_PEAK_MAX,
        mean_time=mean_time,
        std_time=std_time,
        description=f"B2-task (fast only): {B2_PEAK_MIN}-{B2_PEAK_MAX}s (mean={B2_PEAK_MEAN}s)"
    )

    return times.tolist(), config


def generate_fanout_distribution(num_workflows: int,
                                 min_fanout: int = FANOUT_MIN,
                                 max_fanout: int = FANOUT_MAX,
                                 seed: int = 42) -> Tuple[List[int], FanoutConfig]:
    """
    Generate fanout values (number of B tasks per A task) from a uniform distribution.

    Each A task will generate a random number of B tasks uniformly distributed
    between min_fanout and max_fanout (inclusive).

    Standard values from Experiment 04: 3-8 B tasks per A task.

    Args:
        num_workflows: Number of workflows (A tasks)
        min_fanout: Minimum number of B tasks per A task (default: 3)
        max_fanout: Maximum number of B tasks per A task (default: 8)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (fanout_values, config)
    """
    np.random.seed(seed)

    # Generate uniform integer distribution [min_fanout, max_fanout] inclusive
    fanout_values = np.random.randint(min_fanout, max_fanout + 1, num_workflows)

    # Calculate statistics
    mean_fanout = float(np.mean(fanout_values))
    std_fanout = float(np.std(fanout_values))

    config = FanoutConfig(
        name="uniform_fanout",
        min_fanout=min_fanout,
        max_fanout=max_fanout,
        mean_fanout=mean_fanout,
        std_fanout=std_fanout,
        description=f"Uniform fanout: {min_fanout}-{max_fanout} B tasks per A task (mean={mean_fanout:.1f})"
    )

    return fanout_values.tolist(), config


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


def print_fanout_stats(fanout_values: List[int], config: FanoutConfig):
    """
    Print detailed statistics about fanout distribution.

    Args:
        fanout_values: List of fanout values (number of B tasks per A task)
        config: Fanout configuration
    """
    fanout_array = np.array(fanout_values)

    print(f"\nFanout: {config.name}")
    print(f"Description: {config.description}")
    print(f"Statistics:")
    print(f"  Count:  {len(fanout_values)}")
    print(f"  Min:    {fanout_array.min()}")
    print(f"  Max:    {fanout_array.max()}")
    print(f"  Mean:   {config.mean_fanout:.2f}")
    print(f"  Median: {np.median(fanout_array):.2f}")
    print(f"  Std:    {config.std_fanout:.2f}")
    print(f"  Total B tasks: {fanout_array.sum()}")
    print(f"\nDistribution:")
    unique, counts = np.unique(fanout_array, return_counts=True)
    for value, count in zip(unique, counts):
        percentage = (count / len(fanout_values)) * 100
        print(f"  {value} B tasks: {count} workflows ({percentage:.1f}%)")


if __name__ == "__main__":
    """Demo and testing of workload generators."""
    import argparse

    parser = argparse.ArgumentParser(description="Test workload generation")
    parser.add_argument("--num-tasks", type=int, default=100, help="Number of tasks to generate")
    parser.add_argument("--num-workflows", type=int, default=100, help="Number of workflows for fanout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("Unified Workload Generator Demo")
    print("=" * 60)

    # Generate and display A task bimodal distribution
    print("\n--- A TASKS (Experiments 04-07) ---")
    a_times, a_config = generate_bimodal_distribution(args.num_tasks, args.seed)
    print_distribution_stats(a_times, a_config)

    # Generate and display B task bimodal distribution
    print("\n--- B TASKS (Experiments 04-06) ---")
    b_times, b_config = generate_b_task_bimodal_distribution(args.num_tasks, args.seed + 1)
    print_distribution_stats(b_times, b_config)

    # Generate and display B1 task distribution
    print("\n--- B1 TASKS (Experiment 07 only) ---")
    b1_times, b1_config = generate_b1_task_distribution(args.num_tasks, args.seed + 1)
    print_distribution_stats(b1_times, b1_config)

    # Generate and display B2 task distribution
    print("\n--- B2 TASKS (Experiment 07 only) ---")
    b2_times, b2_config = generate_b2_task_distribution(args.num_tasks, args.seed + 2)
    print_distribution_stats(b2_times, b2_config)

    # Generate and display fanout distribution
    print("\n--- FANOUT (All Experiments) ---")
    fanout_values, fanout_config = generate_fanout_distribution(args.num_workflows, seed=args.seed)
    print_fanout_stats(fanout_values, fanout_config)

    print("\n" + "=" * 60)
