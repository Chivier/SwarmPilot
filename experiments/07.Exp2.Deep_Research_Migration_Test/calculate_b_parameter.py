#!/usr/bin/env python3
"""
Calculate B parameter for redeploy based on actual execution times from trace data.

This script analyzes the test dataset to compute realistic throughput values (QPS)
for Model A and Model B based on their actual execution times.

Deep Research Workflow Pattern:
- Model A: A1 (boot) → n×(B1→B2) → A2 (summary)
- Model B: B1 (query) + B2 (criteria)

B Parameter:
- B[i,j] = throughput (QPS) of instance i for model j
- Used by optimizer to calculate service capacity and instance allocation
"""

import json
import statistics
from pathlib import Path


def load_trace_data(data_dir: Path):
    """Load all trace data files."""
    with open(data_dir / "dr_boot.json") as f:
        boot_times = json.load(f)  # A1 execution times (ms)

    with open(data_dir / "dr_query.json") as f:
        query_times = json.load(f)  # B1 execution times (ms)

    with open(data_dir / "dr_criteria.json") as f:
        criteria_times = json.load(f)  # B2 execution times (ms)

    with open(data_dir / "dr_summary_dict.json") as f:
        summary_dict = json.load(f)  # A2 execution times by fanout (ms)

    return boot_times, query_times, criteria_times, summary_dict


def calculate_statistics(times_ms, name):
    """Calculate and print statistics for execution times."""
    times_s = [t / 1000 for t in times_ms]  # Convert to seconds

    mean_s = statistics.mean(times_s)
    median_s = statistics.median(times_s)
    stdev_s = statistics.stdev(times_s) if len(times_s) > 1 else 0

    print(f"\n{name}:")
    print(f"  Count: {len(times_s)}")
    print(f"  Mean: {mean_s:.3f}s")
    print(f"  Median: {median_s:.3f}s")
    print(f"  Std Dev: {stdev_s:.3f}s")

    return median_s


def main():
    """Calculate B parameter from trace data."""
    print("=" * 80)
    print("B Parameter Calculation from Trace Data")
    print("=" * 80)

    # Load data
    data_dir = Path(__file__).parent / "data"
    boot_times, query_times, criteria_times, summary_dict = load_trace_data(data_dir)

    print("\n1. Individual Task Statistics")
    print("-" * 80)

    # A1: boot task
    median_boot = calculate_statistics(boot_times, "A1 (boot)")

    # B1: query task
    median_query = calculate_statistics(query_times, "B1 (query)")

    # B2: criteria task
    median_criteria = calculate_statistics(criteria_times, "B2 (criteria)")

    # A2: summary task (fanout=10)
    fanout = 10
    summary_times_fanout10 = summary_dict.get(str(fanout), [])
    median_summary = calculate_statistics(
        summary_times_fanout10,
        f"A2 (summary, fanout={fanout})"
    )

    # Calculate combined execution times
    print("\n2. Combined Task Execution Times")
    print("-" * 80)

    # Model A: A1 + A2 (boot → summary)
    combined_model_a = median_boot + median_summary
    print(f"\nModel A (A1 + A2):")
    print(f"  A1 (boot): {median_boot:.3f}s")
    print(f"  A2 (summary, fanout={fanout}): {median_summary:.3f}s")
    print(f"  Combined: {combined_model_a:.3f}s")

    # Model B: B1 + B2 (query → criteria)
    combined_model_b = median_query + median_criteria
    print(f"\nModel B (B1 + B2):")
    print(f"  B1 (query): {median_query:.3f}s")
    print(f"  B2 (criteria): {median_criteria:.3f}s")
    print(f"  Combined: {combined_model_b:.3f}s")

    # Calculate throughput (QPS)
    print("\n3. Throughput Calculation (QPS = 1 / execution_time)")
    print("-" * 80)

    qps_model_a = 1.0 / combined_model_a
    qps_model_b = 1.0 / combined_model_b

    print(f"\nModel A QPS: 1 / {combined_model_a:.3f}s = {qps_model_a:.6f} req/s")
    print(f"Model B QPS: 1 / {combined_model_b:.3f}s = {qps_model_b:.6f} req/s")
    print(f"\nThroughput Ratio (B/A): {qps_model_b / qps_model_a:.2f}:1")

    # Generate B parameter
    print("\n4. B Parameter for redeply.py")
    print("-" * 80)

    print(f"\nCalculated B matrix:")
    print(f"B = [[{qps_model_a:.6f}, {qps_model_b:.6f}]] * num_instances")

    print(f"\nRounded to 6 decimal places:")
    print(f"B = [[{qps_model_a:.6f}, {qps_model_b:.6f}]] * (instance_a_num + instance_b_num)")

    # Comparison with current setting
    print("\n5. Comparison with Current Setting")
    print("-" * 80)

    current_a = 1
    current_b = 15
    current_ratio = current_b / current_a
    new_ratio = qps_model_b / qps_model_a

    print(f"\nCurrent B: [[{current_a}, {current_b}]]")
    print(f"  Current ratio (B/A): {current_ratio:.2f}:1")
    print(f"\nNew B: [[{qps_model_a:.6f}, {qps_model_b:.6f}]]")
    print(f"  New ratio (B/A): {new_ratio:.2f}:1")
    print(f"\nDifference: {abs(new_ratio - current_ratio):.2f}x")

    # Recommendations
    print("\n6. Recommendations")
    print("-" * 80)

    print("""
To update redeply.py:

1. Open experiments/07.Exp2.Deep_Research_Migration_Test/redeply.py
2. Find the line: B = [[1, 15]] * (instance_a_num + instance_b_num)
3. Replace with:
   # B[i,j] = throughput (QPS) of instance i for model j
   # Calculated from median execution times in trace data:
   #   Model A: A1 (boot) + A2 (summary, fanout=10) = {combined_model_a:.3f}s
   #   Model B: B1 (query) + B2 (criteria) = {combined_model_b:.3f}s
   B = [[{qps_model_a:.6f}, {qps_model_b:.6f}]] * (instance_a_num + instance_b_num)

4. Save and test with a small experiment
""".format(
        combined_model_a=combined_model_a,
        combined_model_b=combined_model_b,
        qps_model_a=qps_model_a,
        qps_model_b=qps_model_b
    ))

    print("=" * 80)
    print("Calculation Complete!")
    print("=" * 80)

    return {
        'qps_model_a': qps_model_a,
        'qps_model_b': qps_model_b,
        'combined_model_a': combined_model_a,
        'combined_model_b': combined_model_b,
    }


if __name__ == "__main__":
    result = main()
