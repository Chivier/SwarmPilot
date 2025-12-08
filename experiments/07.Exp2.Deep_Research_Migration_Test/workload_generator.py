#!/usr/bin/env python3
"""
Workload generation module for experiment 07.

Provides workload distributions for B1/B2 split workflow:
1. Bimodal distribution (reused from exp01/03/06)
2. Fast peak distribution (for B2 tasks) - left peak only (1-3s)
3. Slow peak distribution (for B1 tasks) - right peak only (7-10s)
4. Pareto long-tail distribution (from exp02/03)
5. Fanout distribution (from exp06) - number of B tasks per A task

Synthetic Data Generation using Four Normal Distributions:
This module generates synthetic workload data using four normal distributions
to model the four task phases:
- dr_boot: Task A left peak (boot times) ~ N(30, 2²)
- dr_summary_dict: Task A right peak (summary times) ~ N(25, 2²), keyed by fanout
- dr_query: Task B left peak (query times) ~ N(3, 0.3²)
- dr_criteria: Task B right peak (criteria evaluation times) ~ N(2, 0.2²)

Usage:
    # Generate synthetic workloads (default mode)
    python workload_generator.py --num-workflows 100

    # Or explicitly specify synthetic mode
    python workload_generator.py --mode synthetic --num-workflows 100

    # Generate trace-based workloads (using four normal distributions)
    python workload_generator.py --mode trace --num-workflows 100

    # Generate both for comparison
    python workload_generator.py --mode both --num-workflows 100

    # Use in code:
    from workload_generator import generate_workflow_from_traces

    workflow, config = generate_workflow_from_traces(num_workflows=100, seed=42)
    # Access workflow data:
    # - workflow.a1_times: List of A1 (boot) times ~ N(30, 2²)
    # - workflow.a2_times: List of A2 (summary) times ~ N(25, 2²)
    # - workflow.b1_times: List of lists of B1 (query) times ~ N(3, 0.3²) per workflow
    # - workflow.b2_times: List of lists of B2 (criteria) times ~ N(2, 0.2²) per workflow
    # - workflow.fanout_values: List of fanout values per workflow
"""

import numpy as np
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
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
PARETO_MIN = 1.0
PARETO_MAX = 10.0
PARETO_ALPHA = 1.5  # Shape parameter (smaller = more skewed/long-tail)

# Fanout distribution parameters
FANOUT_MIN = 3  # Minimum number of B tasks per A task
FANOUT_MAX = 8  # Maximum number of B tasks per A task

# Data directory path
DATA_DIR = Path(__file__).parent / "data"


def load_trace_data() -> Tuple[List[float], Dict[str, List[float]], List[float], List[float]]:
    """
    Generate synthetic trace data using four normal distributions for four task phases.

    This function uses normal distributions to generate task execution times
    for the four phases of the workflow:
    1. Task A left peak (boot): N(30, 2^2) - Fast boot phase
    2. Task A right peak (summary): N(25, 2^2) - Summary generation phase
    3. Task B left peak (query): N(3, 0.3^2) - Fast query phase
    4. Task B right peak (criteria): N(2, 0.2^2) - Criteria evaluation phase

    Returns:
        Tuple containing:
        - dr_boot: List of boot times (Task A left peak)
        - dr_summary_dict: Dict mapping fanout to summary times (Task A right peak)
        - dr_query: List of query times (Task B left peak)
        - dr_criteria: List of criteria evaluation times (Task B right peak)
    """
    # Load template data to get size information
    with open(DATA_DIR / "dr_boot.json", "r") as f:
        dr_boot = json.load(f)

    # Generate Task A left peak (boot) using normal distribution: N(30, 2^2)
    dr_boot = np.random.normal(30, 2, size=len(dr_boot))
    dr_boot = dr_boot.tolist()

    # Load template data for summary
    with open(DATA_DIR / "dr_summary_dict.json", "r") as f:
        dr_summary_dict = json.load(f)

    # Generate Task A right peak (summary) using normal distribution: N(25, 2^2)
    for i in range(5, 16):
        dr_summary_dict[str(i)] = np.random.normal(25, 2, size=len(dr_summary_dict[str(i)]))
        dr_summary_dict[str(i)] = dr_summary_dict[str(i)].tolist()

    # Load template data for query
    with open(DATA_DIR / "dr_query.json", "r") as f:
        dr_query = json.load(f)

    # Generate Task B left peak (query) using normal distribution: N(3, 0.3^2)
    dr_query = np.random.normal(3, 0.3, size=len(dr_query))
    dr_query = dr_query.tolist()

    # Load template data for criteria
    with open(DATA_DIR / "dr_criteria.json", "r") as f:
        dr_criteria = json.load(f)

    # Generate Task B right peak (criteria) using normal distribution: N(2, 0.2^2)
    dr_criteria = np.random.normal(2, 0.2, size=len(dr_criteria))
    dr_criteria = dr_criteria.tolist()

    return dr_boot, dr_summary_dict, dr_query, dr_criteria


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
    random.seed(seed)

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


def generate_fast_peak_distribution(num_tasks: int, seed: int = 42) -> tuple[List[float], WorkloadConfig]:
    """
    Generate task execution times from the fast peak (left peak) of bimodal distribution.

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
    Generate task execution times from the slow peak (right peak) of bimodal distribution.

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


def generate_pattern_based_fanout(
    num_workflows: int,
    seed: int = 42,
    mean_low: int = 6,
    mean_high: int = 14,
    std: float = 0.6,
    min_fanout: int = 5,
    max_fanout: int = 15
) -> tuple[List[int], FanoutConfig]:
    """
    Generate fanout values using alternating normal distributions in 25% segments.

    Pattern (for non-warmup workflows):
    - First 25%: Normal distribution with mean=mean_low, std=std
    - Next 25%: Normal distribution with mean=mean_high, std=std
    - Next 25%: Normal distribution with mean=mean_low, std=std (repeat)
    - Last 25%: Normal distribution with mean=mean_high, std=std (repeat)

    This creates a wave pattern with clear separation between low and high fanout phases.

    Args:
        num_workflows: Number of workflows to generate
        seed: Random seed for reproducibility
        mean_low: Mean for low fanout phase (default: 6)
        mean_high: Mean for high fanout phase (default: 14)
        std: Standard deviation for both distributions (default: 0.6)
        min_fanout: Minimum fanout value (hard limit, default: 5)
        max_fanout: Maximum fanout value (hard limit, default: 15)

    Returns:
        Tuple of (fanout_values, config)
    """
    np.random.seed(seed)
    random.seed(seed)

    # Calculate segment sizes (handle non-divisible by 4)
    segment_size = num_workflows // 4
    remainder = num_workflows % 4

    # Distribute remainder across segments (add 1 to first 'remainder' segments)
    segment_sizes = [segment_size + (1 if i < remainder else 0) for i in range(4)]

    # Pattern: low, high, low, high
    means = [mean_low, mean_high, mean_low, mean_high]

    fanout_values = []

    for segment_idx, (size, mean) in enumerate(zip(segment_sizes, means)):
        # Generate values from normal distribution
        segment_values = np.random.normal(mean, std, size)

        # Round to integers
        segment_values = np.round(segment_values).astype(int)

        # Clip to valid range [min_fanout, max_fanout]
        segment_values = np.clip(segment_values, min_fanout, max_fanout)

        fanout_values.extend(segment_values.tolist())

    # Calculate statistics
    fanout_array = np.array(fanout_values)
    mean_fanout = float(np.mean(fanout_array))
    std_fanout = float(np.std(fanout_array))

    config = FanoutConfig(
        name="pattern_based_fanout",
        min_fanout=min_fanout,
        max_fanout=max_fanout,
        mean_fanout=mean_fanout,
        std_fanout=std_fanout,
        description=f"Pattern-based fanout: 4 segments alternating between N({mean_low},{std}²) and N({mean_high},{std}²)"
    )

    return fanout_values, config


def validate_fanout_pattern(fanout_values: List[int], num_workflows: int, expected_means: List[int] = [6, 14, 6, 14]):
    """
    Validate and log fanout pattern distribution.

    Prints statistics for each 25% segment and overall distribution.

    Args:
        fanout_values: List of fanout values to validate
        num_workflows: Expected number of workflows
        expected_means: Expected means for each segment (default: [6, 14, 6, 14])
    """
    segment_size = num_workflows // 4
    remainder = num_workflows % 4
    segment_sizes = [segment_size + (1 if i < remainder else 0) for i in range(4)]

    print("\n" + "=" * 70)
    print("FANOUT PATTERN VALIDATION")
    print("=" * 70)

    start_idx = 0

    for seg_num, (size, expected_mean) in enumerate(zip(segment_sizes, expected_means), 1):
        segment = fanout_values[start_idx:start_idx + size]
        seg_array = np.array(segment)

        print(f"\nSegment {seg_num} (workflows {start_idx}-{start_idx+size-1}):")
        print(f"  Expected: N({expected_mean}, 0.6²)")
        print(f"  Actual mean: {seg_array.mean():.2f}")
        print(f"  Actual std: {seg_array.std():.2f}")
        print(f"  Range: [{seg_array.min()}, {seg_array.max()}]")

        # Distribution
        unique, counts = np.unique(seg_array, return_counts=True)
        print(f"  Distribution:")
        for val, cnt in zip(unique, counts):
            pct = (cnt / len(seg_array)) * 100
            bar = '█' * max(1, int(pct / 5))
            print(f"    {val:2d}: {bar} {cnt:3d} ({pct:5.1f}%)")

        start_idx += size

    print("\n" + "=" * 70)


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


@dataclass
class WorkflowWorkload:
    """Configuration for a complete workflow workload."""
    name: str
    a1_times: List[float]  # Task A1 (boot) times
    a2_times: List[float]  # Task A2 (summary) times
    b1_times: List[List[float]]  # Task B1 (query) times for each workflow
    b2_times: List[List[float]]  # Task B2 (criteria) times for each workflow
    fanout_values: List[int]  # Number of B tasks for each workflow
    description: str


def generate_workflow_from_traces(
    num_workflows: int,
    seed: int = 42,
    use_pattern_fanout: bool = False
) -> Tuple[WorkflowWorkload, WorkloadConfig]:
    """
    Generate workflow workload data from synthetic trace data.

    Process:
    1. Generate synthetic trace data using four normal distributions
    2. Select a fanout from dr_summary_dict keys (or use pattern-based generation)
    3. For each workflow:
       - Select one dr_boot value as A1 time
       - Select fanout number of dr_query values as B1 times
       - Select fanout number of dr_criteria values as B2 times
       - Select one value from dr_summary_dict[fanout] as A2 time

    Args:
        num_workflows: Number of workflows to generate
        seed: Random seed for reproducibility
        use_pattern_fanout: If True, use pattern-based fanout generation (25% segments)
                           with alternating N(6, 0.6²) and N(14, 0.6²). If False,
                           use uniform random selection from available fanout values.

    Returns:
        Tuple of (WorkflowWorkload, WorkloadConfig)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Generate synthetic trace data using four normal distributions
    dr_boot, dr_summary_dict, dr_query, dr_criteria = load_trace_data()

    # Get available fanout values (keys of dr_summary_dict)
    available_fanouts = [int(k) for k in dr_summary_dict.keys()]

    # Generate data for each workflow
    a1_times = []
    a2_times = []
    b1_times = []
    b2_times = []

    # Generate fanout values based on mode
    if use_pattern_fanout:
        # Use pattern-based generation (25% segments alternating between N(6, 0.6²) and N(14, 0.6²))
        fanout_values, _ = generate_pattern_based_fanout(
            num_workflows=num_workflows,
            seed=seed
        )
    else:
        # Use original uniform random selection
        fanout_values = []
        for _ in range(num_workflows):
            fanout = random.choice(available_fanouts)
            fanout_values.append(fanout)

    # Generate workflow data using the fanout values
    for workflow_idx in range(num_workflows):
        fanout = fanout_values[workflow_idx]

        # Select A1 time from dr_boot
        a1_time = random.choice(dr_boot)
        a1_times.append(a1_time)

        # Select fanout number of B1 times from dr_query
        b1_workflow = random.sample(dr_query, fanout)
        b1_times.append(b1_workflow)

        # Select fanout number of B2 times from dr_criteria
        b2_workflow = random.sample(dr_criteria, fanout)
        b2_times.append(b2_workflow)

        # Select A2 time from dr_summary_dict[fanout]
        a2_time = random.choice(dr_summary_dict[str(fanout)])
        a2_times.append(a2_time)

    # Calculate statistics
    all_a1 = np.array(a1_times)
    all_a2 = np.array(a2_times)
    all_b1 = np.array([t for workflow in b1_times for t in workflow])
    all_b2 = np.array([t for workflow in b2_times for t in workflow])
    all_fanouts = np.array(fanout_values)

    workflow_workload = WorkflowWorkload(
        name="synthetic_workflow",
        a1_times=a1_times,
        a2_times=a2_times,
        b1_times=b1_times,
        b2_times=b2_times,
        fanout_values=fanout_values,
        description=f"Workflow from synthetic normal distributions: {num_workflows} workflows, "
                    f"A1~N(30,2²), A2~N(25,2²), B1~N(3,0.3²), B2~N(2,0.2²)"
    )

    # Overall config for statistics
    config = WorkloadConfig(
        name="synthetic_workflow",
        min_time=min(all_a1.min(), all_a2.min(), all_b1.min(), all_b2.min()),
        max_time=max(all_a1.max(), all_a2.max(), all_b1.max(), all_b2.max()),
        mean_time=(all_a1.mean() + all_a2.mean() + all_b1.mean() + all_b2.mean()) / 4,
        std_time=(all_a1.std() + all_a2.std() + all_b1.std() + all_b2.std()) / 4,
        description=f"Synthetic workflow: {num_workflows} workflows, "
                    f"4 normal distributions, avg fanout={all_fanouts.mean():.1f}"
    )

    return workflow_workload, config


def print_workflow_stats(workload: WorkflowWorkload):
    """
    Print detailed statistics about a workflow workload.

    Args:
        workload: WorkflowWorkload to analyze
    """
    a1_times = np.array(workload.a1_times)
    a2_times = np.array(workload.a2_times)
    all_b1 = np.array([t for workflow in workload.b1_times for t in workflow])
    all_b2 = np.array([t for workflow in workload.b2_times for t in workflow])
    fanouts = np.array(workload.fanout_values)

    print(f"\nWorkflow: {workload.name}")
    print(f"Description: {workload.description}")
    print(f"\nA1 (Boot) Statistics:")
    print(f"  Count:  {len(a1_times)}")
    print(f"  Min:    {a1_times.min():.3f}s")
    print(f"  Max:    {a1_times.max():.3f}s")
    print(f"  Mean:   {a1_times.mean():.3f}s")
    print(f"  Median: {np.median(a1_times):.3f}s")
    print(f"  Std:    {a1_times.std():.3f}s")

    print(f"\nA2 (Summary) Statistics:")
    print(f"  Count:  {len(a2_times)}")
    print(f"  Min:    {a2_times.min():.3f}s")
    print(f"  Max:    {a2_times.max():.3f}s")
    print(f"  Mean:   {a2_times.mean():.3f}s")
    print(f"  Median: {np.median(a2_times):.3f}s")
    print(f"  Std:    {a2_times.std():.3f}s")

    print(f"\nB1 (Query) Statistics:")
    print(f"  Count:  {len(all_b1)}")
    print(f"  Min:    {all_b1.min():.3f}s")
    print(f"  Max:    {all_b1.max():.3f}s")
    print(f"  Mean:   {all_b1.mean():.3f}s")
    print(f"  Median: {np.median(all_b1):.3f}s")
    print(f"  Std:    {all_b1.std():.3f}s")

    print(f"\nB2 (Criteria) Statistics:")
    print(f"  Count:  {len(all_b2)}")
    print(f"  Min:    {all_b2.min():.3f}s")
    print(f"  Max:    {all_b2.max():.3f}s")
    print(f"  Mean:   {all_b2.mean():.3f}s")
    print(f"  Median: {np.median(all_b2):.3f}s")
    print(f"  Std:    {all_b2.std():.3f}s")

    print(f"\nFanout Distribution:")
    print(f"  Min:    {fanouts.min()}")
    print(f"  Max:    {fanouts.max()}")
    print(f"  Mean:   {fanouts.mean():.2f}")
    print(f"  Median: {np.median(fanouts):.2f}")
    print(f"  Std:    {fanouts.std():.2f}")
    print(f"  Total B tasks: {fanouts.sum()}")
    print(f"\nDistribution:")
    unique, counts = np.unique(fanouts, return_counts=True)
    for value, count in zip(unique, counts):
        percentage = (count / len(fanouts)) * 100
        print(f"  Fanout {value}: {count} workflows ({percentage:.1f}%)")


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
    parser.add_argument("--mode", type=str, default="trace",
                        choices=["synthetic", "trace", "both"],
                        help="Generation mode: trace (using 4 normal distributions, default), synthetic (original), or both")
    args = parser.parse_args()

    print("=" * 60)
    print("Workload Generator Demo")
    print("=" * 60)

    if args.mode in ["synthetic", "both"]:
        print("\n" + "=" * 60)
        print("SYNTHETIC WORKLOAD GENERATION")
        print("=" * 60)

        # Generate and display bimodal distribution
        bimodal_times, bimodal_config = generate_bimodal_distribution(args.num_tasks, args.seed)
        print_distribution_stats(bimodal_times, bimodal_config)

        # Generate and display Pareto distribution
        pareto_times, pareto_config = generate_pareto_distribution(args.num_tasks, seed=args.seed)
        print_distribution_stats(pareto_times, pareto_config)

        # Generate and display fanout distribution
        fanout_values, fanout_config = generate_fanout_distribution(args.num_workflows, seed=args.seed)
        print_fanout_stats(fanout_values, fanout_config)

    if args.mode in ["trace", "both"]:
        print("\n" + "=" * 60)
        print("SYNTHETIC WORKLOAD GENERATION (4 Normal Distributions)")
        print("=" * 60)
        print("Using four normal distributions:")
        print("  A1 (boot): N(30, 2²)")
        print("  A2 (summary): N(25, 2²)")
        print("  B1 (query): N(3, 0.3²)")
        print("  B2 (criteria): N(2, 0.2²)")

        # Generate workflow from synthetic normal distributions
        workflow_workload, workflow_config = generate_workflow_from_traces(
            args.num_workflows, seed=args.seed
        )
        print_workflow_stats(workflow_workload)

        print(f"\nOverall Config:")
        print(f"  Name: {workflow_config.name}")
        print(f"  Min time: {workflow_config.min_time:.3f}s")
        print(f"  Max time: {workflow_config.max_time:.3f}s")
        print(f"  Mean time: {workflow_config.mean_time:.3f}s")
        print(f"  Std time: {workflow_config.std_time:.3f}s")

    print("\n" + "=" * 60)
