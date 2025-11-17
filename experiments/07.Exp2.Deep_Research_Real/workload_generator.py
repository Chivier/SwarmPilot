#!/usr/bin/env python3
"""
Workload generation module for experiment 07.

Provides workload distributions for B1/B2 split workflow:
1. Bimodal distribution (reused from exp01/03/06)
2. Fast peak distribution (for B2 tasks) - left peak only (1-3s)
3. Slow peak distribution (for B1 tasks) - right peak only (7-10s)
4. Pareto long-tail distribution (from exp02/03)
5. Fanout distribution (from exp06) - number of B tasks per A task

Data Loading from Real Traces:
- dr_boot: Task A left peak (boot times)
- dr_summary_dict: Task A right peak (summary times), keyed by fanout
- dr_query: Task B left peak (query times)
- dr_criteria: Task B right peak (criteria evaluation times)

Usage:
    # Generate trace-based workloads from real data (default mode)
    python workload_generator.py --num-workflows 100

    # Or explicitly specify trace mode
    python workload_generator.py --mode trace --num-workflows 100

    # Generate synthetic workloads (original functionality)
    python workload_generator.py --mode synthetic --num-workflows 100

    # Generate both for comparison
    python workload_generator.py --mode both --num-workflows 100

    # Use in code:
    from workload_generator import generate_workflow_from_traces

    workflow, config = generate_workflow_from_traces(num_workflows=100, seed=42)
    # Access workflow data:
    # - workflow.a1_times: List of A1 (boot) times
    # - workflow.a2_times: List of A2 (summary) times
    # - workflow.b1_times: List of lists of B1 (query) times per workflow
    # - workflow.b2_times: List of lists of B2 (criteria) times per workflow
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
    Load real trace data from JSON files.

    Returns:
        Tuple containing:
        - dr_boot: List of boot times (Task A left peak)
        - dr_summary_dict: Dict mapping fanout to summary times (Task A right peak)
        - dr_query: List of query times (Task B left peak)
        - dr_criteria: List of criteria evaluation times (Task B right peak)
    """
    with open(DATA_DIR / "dr_boot.json", "r") as f:
        dr_boot = json.load(f)

    with open(DATA_DIR / "dr_summary_dict.json", "r") as f:
        dr_summary_dict = json.load(f)

    with open(DATA_DIR / "dr_query.json", "r") as f:
        dr_query = json.load(f)

    with open(DATA_DIR / "dr_criteria.json", "r") as f:
        dr_criteria = json.load(f)

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
    seed: int = 42
) -> Tuple[WorkflowWorkload, WorkloadConfig]:
    """
    Generate workflow workload data from real traces.

    Process:
    1. Load real trace data
    2. Select a fanout from dr_summary_dict keys
    3. For each workflow:
       - Select one dr_boot value as A1 time
       - Select fanout number of dr_query values as B1 times
       - Select fanout number of dr_criteria values as B2 times
       - Select one value from dr_summary_dict[fanout] as A2 time

    Args:
        num_workflows: Number of workflows to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (WorkflowWorkload, WorkloadConfig)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Load trace data
    dr_boot, dr_summary_dict, dr_query, dr_criteria = load_trace_data()

    # Get available fanout values (keys of dr_summary_dict)
    available_fanouts = [int(k) for k in dr_summary_dict.keys()]

    # Generate data for each workflow
    a1_times = []
    a2_times = []
    b1_times = []
    b2_times = []
    fanout_values = []

    for _ in range(num_workflows):
        # Select a fanout value
        fanout = random.choice(available_fanouts)
        fanout_values.append(fanout)

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
        name="trace_based_workflow",
        a1_times=a1_times,
        a2_times=a2_times,
        b1_times=b1_times,
        b2_times=b2_times,
        fanout_values=fanout_values,
        description=f"Workflow from real traces: {num_workflows} workflows, "
                    f"A1 from dr_boot, A2 from dr_summary_dict, "
                    f"B1 from dr_query, B2 from dr_criteria"
    )

    # Overall config for statistics
    config = WorkloadConfig(
        name="trace_based_workflow",
        min_time=min(all_a1.min(), all_a2.min(), all_b1.min(), all_b2.min()),
        max_time=max(all_a1.max(), all_a2.max(), all_b1.max(), all_b2.max()),
        mean_time=(all_a1.mean() + all_a2.mean() + all_b1.mean() + all_b2.mean()) / 4,
        std_time=(all_a1.std() + all_a2.std() + all_b1.std() + all_b2.std()) / 4,
        description=f"Trace-based workflow: {num_workflows} workflows, "
                    f"avg fanout={all_fanouts.mean():.1f}"
    )

    return workflow_workload, config


@dataclass
class DatasetEntry:
    """Single entry from dataset.jsonl."""
    boot: str  # A1 task input (boot prompt)
    queries: List[str]  # B1/B2 task inputs (query array)
    summary: str  # A2 task input (summary prompt)
    fanout: int  # Number of queries (computed)


def load_dataset(dataset_path: Path) -> List[DatasetEntry]:
    """
    Load dataset from JSONL file.

    Args:
        dataset_path: Path to dataset.jsonl file

    Returns:
        List of DatasetEntry objects
    """
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry_dict = json.loads(line)
                # Extract queries - handle both dict format and string format
                queries_raw = entry_dict.get("queries", [])
                if isinstance(queries_raw, list) and len(queries_raw) > 0:
                    # If queries is a list of dicts with 'input' key
                    if isinstance(queries_raw[0], dict):
                        queries = [q.get("input", str(q)) for q in queries_raw]
                    else:
                        # If queries is already a list of strings
                        queries = queries_raw
                else:
                    queries = []

                entry = DatasetEntry(
                    boot=entry_dict.get("boot", ""),
                    queries=queries,
                    summary=entry_dict.get("summary", ""),
                    fanout=len(queries)
                )
                dataset.append(entry)

    return dataset


def generate_workflow_from_dataset(
    dataset_path: Path,
    num_workflows: int,
    seed: int = 42
) -> Tuple[WorkflowWorkload, WorkloadConfig]:
    """
    Generate workflow workload data by sampling from dataset.jsonl WITH REPLACEMENT.

    Process:
    1. Load dataset.jsonl
    2. Sample num_workflows entries WITH REPLACEMENT (using random.choices)
    3. For each sampled entry:
       - boot field → A1 task input
       - summary field → A2 task input
       - queries array → B1 and B2 task inputs
       - fanout = len(queries)

    Note: This function does NOT generate execution times. The actual execution
    time will be determined by the LLM service based on the input content and
    max_tokens parameters.

    Args:
        dataset_path: Path to dataset.jsonl file
        num_workflows: Number of workflows to generate (can be > dataset size)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (WorkflowWorkload, WorkloadConfig)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Load dataset
    dataset = load_dataset(dataset_path)

    if len(dataset) == 0:
        raise ValueError(f"Dataset at {dataset_path} is empty!")

    # Sample WITH REPLACEMENT using random.choices
    sampled_entries = random.choices(dataset, k=num_workflows)

    # Build workflow data
    # Note: Setting times to 0.0 since we're using real LLM execution
    a1_times = [0.0] * num_workflows  # Placeholder (actual time from LLM)
    a2_times = [0.0] * num_workflows  # Placeholder (actual time from LLM)
    b1_times = []  # List of lists (one list per workflow)
    b2_times = []  # List of lists (one list per workflow)
    fanout_values = []

    for entry in sampled_entries:
        fanout = entry.fanout
        fanout_values.append(fanout)

        # Each B task gets 0.0 as placeholder time
        b1_workflow = [0.0] * fanout
        b2_workflow = [0.0] * fanout

        b1_times.append(b1_workflow)
        b2_times.append(b2_workflow)

    # Calculate statistics (fanout only, since times are all 0.0)
    all_fanouts = np.array(fanout_values)

    workflow_workload = WorkflowWorkload(
        name="dataset_based_workflow",
        a1_times=a1_times,
        a2_times=a2_times,
        b1_times=b1_times,
        b2_times=b2_times,
        fanout_values=fanout_values,
        description=f"Workflow from dataset.jsonl: {num_workflows} workflows sampled "
                    f"WITH REPLACEMENT from {len(dataset)} entries, "
                    f"A1 from boot field, A2 from summary field, "
                    f"B1/B2 from queries array"
    )

    # Overall config (times are 0.0 since we use real LLM execution)
    config = WorkloadConfig(
        name="dataset_based_workflow",
        min_time=0.0,
        max_time=0.0,
        mean_time=0.0,
        std_time=0.0,
        description=f"Dataset-based workflow: {num_workflows} workflows, "
                    f"avg fanout={all_fanouts.mean():.1f}, "
                    f"fanout range=[{all_fanouts.min()}, {all_fanouts.max()}]"
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
                        help="Generation mode: trace (from real data, default), synthetic (original), or both")
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
        print("TRACE-BASED WORKLOAD GENERATION")
        print("=" * 60)

        # Generate workflow from real traces
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
