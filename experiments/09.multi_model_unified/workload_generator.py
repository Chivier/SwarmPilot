#!/usr/bin/env python3
"""
Unified workload generation module for experiment 09.

Provides workload distributions for all workflow modes (ocr, t2img, merge, dr):
1. Bimodal distribution - A tasks (50% fast 1-3s, 50% slow 7-10s)
2. B task bimodal distribution - B tasks for ocr/t2img/merge modes
3. Fast peak distribution - B2 tasks for dr mode (1-3s)
4. Slow peak distribution - B1 tasks for dr mode (7-10s)
5. Pareto long-tail distribution - Optional alternative distribution
6. Fanout distribution - Number of B tasks per A task (3-8)
7. Merge task distribution - Merge tasks (0.5x original A time)
"""

import numpy as np
from typing import List
from dataclasses import dataclass


# A task bimodal distribution parameters
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
PARETO_MIN = 1.0
PARETO_MAX = 10.0
PARETO_ALPHA = 1.5  # Shape parameter (smaller = more skewed/long-tail)

# B task bimodal distribution parameters (for ocr/t2img/merge modes)
B_LEFT_PEAK_MIN = 1.0
B_LEFT_PEAK_MAX = 3.0
B_LEFT_PEAK_MEAN = 2.0
B_LEFT_PEAK_STD = 0.4

B_RIGHT_PEAK_MIN = 8.0
B_RIGHT_PEAK_MAX = 12.0
B_RIGHT_PEAK_MEAN = 10.0
B_RIGHT_PEAK_STD = 0.6

B_PEAK_RATIO = 0.5  # 50% left peak (2.0s), 50% right peak (10.0s)

# Fanout distribution parameters
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


def generate_bimodal_distribution(num_tasks: int, seed: int = 42) -> tuple[List[float], WorkloadConfig]:
    """
    Generate A task execution times from a bimodal distribution.

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


def generate_b_task_bimodal_distribution(num_tasks: int, seed: int = 42) -> tuple[List[float], WorkloadConfig]:
    """
    Generate B task execution times from a bimodal distribution.
    Used for ocr, t2img, and merge modes.

    The distribution has two peaks:
    - Left peak: 1-3 seconds (mean=2.0s, std=0.4s)
    - Right peak: 8-12 seconds (mean=10.0s, std=0.6s)
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

    # Generate left peak
    left_times = np.random.normal(B_LEFT_PEAK_MEAN, B_LEFT_PEAK_STD, num_left_peak)
    left_times = np.clip(left_times, B_LEFT_PEAK_MIN, B_LEFT_PEAK_MAX)

    # Generate right peak
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
        description=f"B-task Bimodal: {B_PEAK_RATIO:.0%} at {B_LEFT_PEAK_MEAN}s, {1-B_PEAK_RATIO:.0%} at {B_RIGHT_PEAK_MEAN}s"
    )

    return times.tolist(), config


def generate_fast_peak_distribution(num_tasks: int, seed: int = 42) -> tuple[List[float], WorkloadConfig]:
    """
    Generate task execution times from the fast peak (left peak) only.
    Used for B2 tasks in dr mode.

    This generates only the fast tasks:
    - Range: 1-3 seconds (mean=2.0s, std=0.4s)

    Args:
        num_tasks: Number of tasks to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (task_times, config)
    """
    np.random.seed(seed)

    # Generate left peak only
    times = np.random.normal(LEFT_PEAK_MEAN, LEFT_PEAK_STD, num_tasks)
    times = np.clip(times, LEFT_PEAK_MIN, LEFT_PEAK_MAX)

    # Calculate statistics
    mean_time = float(np.mean(times))
    std_time = float(np.std(times))

    config = WorkloadConfig(
        name="fast_peak",
        min_time=LEFT_PEAK_MIN,
        max_time=LEFT_PEAK_MAX,
        mean_time=mean_time,
        std_time=std_time,
        description=f"Fast peak only: {LEFT_PEAK_MEAN}s (range: {LEFT_PEAK_MIN}-{LEFT_PEAK_MAX}s)"
    )

    return times.tolist(), config


def generate_slow_peak_distribution(num_tasks: int, seed: int = 42) -> tuple[List[float], WorkloadConfig]:
    """
    Generate task execution times from the slow peak (right peak) only.
    Used for B1 tasks in dr mode.

    This generates only the slow tasks:
    - Range: 7-10 seconds (mean=8.5s, std=0.6s)

    Args:
        num_tasks: Number of tasks to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (task_times, config)
    """
    np.random.seed(seed)

    # Generate right peak only
    times = np.random.normal(RIGHT_PEAK_MEAN, RIGHT_PEAK_STD, num_tasks)
    times = np.clip(times, RIGHT_PEAK_MIN, RIGHT_PEAK_MAX)

    # Calculate statistics
    mean_time = float(np.mean(times))
    std_time = float(np.std(times))

    config = WorkloadConfig(
        name="slow_peak",
        min_time=RIGHT_PEAK_MIN,
        max_time=RIGHT_PEAK_MAX,
        mean_time=mean_time,
        std_time=std_time,
        description=f"Slow peak only: {RIGHT_PEAK_MEAN}s (range: {RIGHT_PEAK_MIN}-{RIGHT_PEAK_MAX}s)"
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


def generate_fanout_distribution(num_workflows: int,
                                 min_fanout: int = FANOUT_MIN,
                                 max_fanout: int = FANOUT_MAX,
                                 seed: int = 42) -> tuple[List[int], FanoutConfig]:
    """
    Generate fanout values (number of B tasks per A task) from a uniform distribution.

    Each A task will generate a random number of B tasks uniformly distributed
    between min_fanout and max_fanout (inclusive).

    Args:
        num_workflows: Number of workflows (A tasks)
        min_fanout: Minimum number of B tasks per A task
        max_fanout: Maximum number of B tasks per A task
        seed: Random seed for reproducibility

    Returns:
        Tuple of (fanout_values, config)
    """
    np.random.seed(seed)

    # Generate uniform integer distribution
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


def generate_merge_task_distribution(original_a_times: List[float], seed: int = 42) -> tuple[List[float], WorkloadConfig]:
    """
    Generate merge task execution times based on original A task times.
    Used for merge and dr modes.

    Merge tasks take 0.5x the original A task execution time.

    Args:
        original_a_times: List of original A task execution times
        seed: Random seed for reproducibility

    Returns:
        Tuple of (merge_times, config)
    """
    np.random.seed(seed)

    # Merge task = 0.5x original A task time
    merge_times = [t * 0.5 for t in original_a_times]

    times_array = np.array(merge_times)
    mean_time = float(np.mean(times_array))
    std_time = float(np.std(times_array))

    config = WorkloadConfig(
        name="merge_task",
        min_time=float(times_array.min()),
        max_time=float(times_array.max()),
        mean_time=mean_time,
        std_time=std_time,
        description=f"Merge task: 0.5x original A task time (mean={mean_time:.2f}s)"
    )

    return merge_times, config


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

    parser = argparse.ArgumentParser(description="Test unified workload generation")
    parser.add_argument("--num-tasks", type=int, default=100, help="Number of tasks to generate")
    parser.add_argument("--num-workflows", type=int, default=100, help="Number of workflows for fanout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("Unified Workload Generator Demo (Experiment 09)")
    print("=" * 60)

    # Generate and display A task bimodal distribution
    bimodal_times, bimodal_config = generate_bimodal_distribution(args.num_tasks, args.seed)
    print_distribution_stats(bimodal_times, bimodal_config)

    # Generate and display B task bimodal distribution (for ocr/t2img/merge)
    b_task_times, b_task_config = generate_b_task_bimodal_distribution(args.num_tasks, seed=args.seed)
    print_distribution_stats(b_task_times, b_task_config)

    # Generate and display slow peak distribution (for dr mode B1)
    slow_peak_times, slow_peak_config = generate_slow_peak_distribution(args.num_tasks, seed=args.seed)
    print_distribution_stats(slow_peak_times, slow_peak_config)

    # Generate and display fast peak distribution (for dr mode B2)
    fast_peak_times, fast_peak_config = generate_fast_peak_distribution(args.num_tasks, seed=args.seed)
    print_distribution_stats(fast_peak_times, fast_peak_config)

    # Generate and display Pareto distribution
    pareto_times, pareto_config = generate_pareto_distribution(args.num_tasks, seed=args.seed)
    print_distribution_stats(pareto_times, pareto_config)

    # Generate and display fanout distribution
    fanout_values, fanout_config = generate_fanout_distribution(args.num_workflows, seed=args.seed)
    print_fanout_stats(fanout_values, fanout_config)

    # Generate and display merge task distribution
    merge_times, merge_config = generate_merge_task_distribution(bimodal_times[:args.num_workflows], seed=args.seed)
    print_distribution_stats(merge_times, merge_config)

    print("\n" + "=" * 60)
