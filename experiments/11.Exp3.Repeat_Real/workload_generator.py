#!/usr/bin/env python3
"""
Workload generation module for experiment 11.

Provides four types of workload distributions:
1. Bimodal distribution - A task execution times (reused from exp01/03)
2. 4-Peak distribution - B task execution times (from exp04)
3. Fanout distribution - number of B tasks per A task (from exp04)
4. Repeat distribution - number of times each workflow executes (NEW for exp11)
"""

import numpy as np
from typing import List
from dataclasses import dataclass


# Bimodal distribution parameters (from exp01)
LEFT_PEAK_MIN = 0.5
LEFT_PEAK_MAX = 0.7
LEFT_PEAK_MEAN = 1.0
LEFT_PEAK_STD = 0.4

RIGHT_PEAK_MIN = 10.0
RIGHT_PEAK_MAX = 15.0
RIGHT_PEAK_MEAN = 20
RIGHT_PEAK_STD = 0.6

PEAK_RATIO = 0.5  # 50% left peak, 50% right peak

# Pareto distribution parameters
PARETO_MIN = 1.0
PARETO_MAX = 10.0
PARETO_ALPHA = 1.5  # Shape parameter (smaller = more skewed/long-tail)

# B task bimodal distribution parameters
B_1ST_PEAK_MIN = 0.5
B_1ST_PEAK_MAX = 1.5
B_1ST_PEAK_MEAN = 1.0
B_1ST_PEAK_STD = 0.2

B_2ND_PEAK_MIN = 4.0
B_2ND_PEAK_MAX = 6.0
B_2ND_PEAK_MEAN = 4.0
B_2ND_PEAK_STD = 0.4

B_3RD_PEAK_MIN = 10
B_3RD_PEAK_MAX = 30
B_3RD_PEAK_MEAN = 20
B_3RD_PEAK_STD = 1

B_4TH_PEAK_MIN = 60
B_4TH_PEAK_MAX = 120
B_4TH_PEAK_MEAN = 100
B_4TH_PEAK_STD = 0.4

B_PEAK_RATIO = 0.25  # 50% left peak (2.0s), 50% right peak (10.0s)

# Fanout distribution parameters
FANOUT_MIN = 5  # Minimum number of B tasks per A task
FANOUT_MAX = 15  # Maximum number of B tasks per A task

# Repeat distribution parameters
REPEAT_MIN = 1  # Minimum number of times to repeat a workflow
REPEAT_MAX = 3  # Maximum number of times to repeat a workflow


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


@dataclass
class RepeatConfig:
    """Configuration for repeat distribution."""
    name: str
    min_repeat: int
    max_repeat: int
    mean_repeat: float
    std_repeat: float
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


def generate_b_task_bimodal_distribution(num_tasks: int, seed: int = 42) -> tuple[List[float], WorkloadConfig]:
    """
    Generate B task execution times from a bimodal distribution.

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
    num_each_peak = int(num_tasks * (B_PEAK_RATIO + 0.5))


    peak_1st_times = np.random.normal(B_1ST_PEAK_MEAN, B_1ST_PEAK_STD, num_each_peak)
    peak_1st_times = np.clip(peak_1st_times, B_1ST_PEAK_MIN, B_1ST_PEAK_MAX)

    peak_2nd_times = np.random.normal(B_2ND_PEAK_MEAN, B_2ND_PEAK_STD, num_each_peak)
    peak_2nd_times = np.clip(peak_2nd_times, B_2ND_PEAK_MIN, B_2ND_PEAK_MAX)
    
    peak_3rd_times = np.random.normal(B_3RD_PEAK_MEAN, B_3RD_PEAK_STD, num_each_peak)
    peak_3rd_times = np.clip(peak_3rd_times, B_3RD_PEAK_MIN, B_3RD_PEAK_MAX)
    
    peak_4th_times = np.random.normal(B_4TH_PEAK_MEAN, B_4TH_PEAK_STD, num_each_peak)
    peak_4th_times = np.clip(peak_4th_times, B_4TH_PEAK_MIN, B_4TH_PEAK_MAX)

    # Combine and shuffle
    times = np.concatenate([peak_1st_times, peak_2nd_times, peak_3rd_times, peak_4th_times])
    np.random.shuffle(times)

    # Calculate statistics
    mean_time = float(np.mean(times))
    std_time = float(np.std(times))

    config = WorkloadConfig(
        name="b_task_bimodal",
        min_time=B_1ST_PEAK_MIN,
        max_time=B_4TH_PEAK_MAX,
        mean_time=mean_time,
        std_time=std_time,
        description=f"B-task 4 peaks"
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


def generate_repeat_distribution(num_workflows: int,
                                  min_repeat: int = REPEAT_MIN,
                                  max_repeat: int = REPEAT_MAX,
                                  seed: int = 42) -> tuple[List[int], RepeatConfig]:
    """
    Generate repeat values (number of times each workflow executes) from a uniform distribution.

    Each workflow will be executed a random number of times uniformly distributed
    between min_repeat and max_repeat (inclusive).

    Args:
        num_workflows: Number of workflows
        min_repeat: Minimum number of times to repeat a workflow (default: 1)
        max_repeat: Maximum number of times to repeat a workflow (default: 3)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (repeat_values, config)
    """
    np.random.seed(seed)

    # Generate uniform integer distribution
    repeat_values = np.random.randint(min_repeat, max_repeat + 1, num_workflows)

    # Calculate statistics
    mean_repeat = float(np.mean(repeat_values))
    std_repeat = float(np.std(repeat_values))

    config = RepeatConfig(
        name="uniform_repeat",
        min_repeat=min_repeat,
        max_repeat=max_repeat,
        mean_repeat=mean_repeat,
        std_repeat=std_repeat,
        description=f"Uniform repeat: {min_repeat}-{max_repeat} executions per workflow (mean={mean_repeat:.1f})"
    )

    return repeat_values.tolist(), config


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


def print_repeat_stats(repeat_values: List[int], config: RepeatConfig):
    """
    Print detailed statistics about repeat distribution.

    Args:
        repeat_values: List of repeat values (number of times each workflow executes)
        config: Repeat configuration
    """
    repeat_array = np.array(repeat_values)

    print(f"\nRepeat: {config.name}")
    print(f"Description: {config.description}")
    print(f"Statistics:")
    print(f"  Count:  {len(repeat_values)}")
    print(f"  Min:    {repeat_array.min()}")
    print(f"  Max:    {repeat_array.max()}")
    print(f"  Mean:   {config.mean_repeat:.2f}")
    print(f"  Median: {np.median(repeat_array):.2f}")
    print(f"  Std:    {config.std_repeat:.2f}")
    print(f"  Total iterations: {repeat_array.sum()}")
    print(f"\nDistribution:")
    unique, counts = np.unique(repeat_array, return_counts=True)
    for value, count in zip(unique, counts):
        percentage = (count / len(repeat_values)) * 100
        print(f"  {value} iterations: {count} workflows ({percentage:.1f}%)")


if __name__ == "__main__":
    """Demo and testing of workload generators."""
    import argparse

    parser = argparse.ArgumentParser(description="Test workload generation")
    parser.add_argument("--num-tasks", type=int, default=100, help="Number of tasks to generate")
    parser.add_argument("--num-workflows", type=int, default=100, help="Number of workflows for fanout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("Workload Generator Demo")
    print("=" * 60)

    # Generate and display A task bimodal distribution
    bimodal_times, bimodal_config = generate_bimodal_distribution(args.num_tasks, args.seed)
    print_distribution_stats(bimodal_times, bimodal_config)

    # Generate and display B task bimodal distribution
    b_task_times, b_task_config = generate_b_task_bimodal_distribution(args.num_tasks, seed=args.seed)
    print_distribution_stats(b_task_times, b_task_config)

    # Generate and display Pareto distribution
    pareto_times, pareto_config = generate_pareto_distribution(args.num_tasks, seed=args.seed)
    print_distribution_stats(pareto_times, pareto_config)

    # Generate and display fanout distribution
    fanout_values, fanout_config = generate_fanout_distribution(args.num_workflows, seed=args.seed)
    print_fanout_stats(fanout_values, fanout_config)

    print("\n" + "=" * 60)
