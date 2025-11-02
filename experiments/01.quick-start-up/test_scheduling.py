#!/usr/bin/env python3
"""
Test script for 01.quick-start-up experiment.

Tests all three scheduling strategies with tasks that have normally distributed
execution times (1.5s-5s range).
"""

import time
import json
import requests
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
import sys
import os


# Configuration
SCHEDULER_URL = "http://localhost:8100"
NUM_TASKS = 100  # Number of tasks to submit per strategy
TASK_MEAN = 3.25  # Mean task execution time (seconds)
TASK_STD = 0.583  # Standard deviation (ensures 99.7% in 1.5s-5s range)
TASK_MIN = 1.5    # Minimum task time
TASK_MAX = 5.0    # Maximum task time
MODEL_ID = "sleep_model"

# Strategies to test
STRATEGIES = ["round_robin", "min_time", "probabilistic"]


def generate_task_times(num_tasks: int) -> List[float]:
    """
    Generate task execution times from a normal distribution.

    Args:
        num_tasks: Number of tasks to generate

    Returns:
        List of task execution times in seconds
    """
    # Generate normally distributed values
    times = np.random.normal(TASK_MEAN, TASK_STD, num_tasks)

    # Clip to ensure all values are in valid range
    times = np.clip(times, TASK_MIN, TASK_MAX)

    return times.tolist()


def clear_tasks() -> bool:
    """
    Clear all tasks from the scheduler.

    Returns:
        True if successful, False otherwise
    """
    url = f"{SCHEDULER_URL}/task/clear"

    try:
        response = requests.post(url, timeout=10)
        response.raise_for_status()
        result = response.json()
        cleared_count = result.get("cleared_count", 0)
        print(f"✓ Cleared {cleared_count} tasks from scheduler")
        return True
    except Exception as e:
        print(f"✗ Failed to clear tasks: {e}")
        return False


def set_strategy(strategy_name: str, target_quantile: float = 0.9) -> bool:
    """
    Set the scheduling strategy.

    Args:
        strategy_name: Name of the strategy (round_robin, min_time, probabilistic)
        target_quantile: Target quantile for probabilistic strategy

    Returns:
        True if successful, False otherwise
    """
    url = f"{SCHEDULER_URL}/strategy/set"
    payload = {"strategy_name": strategy_name}

    if strategy_name == "probabilistic":
        payload["target_quantile"] = target_quantile

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        print(f"✓ Strategy set to: {strategy_name}")
        if "cleared_tasks" in result:
            print(f"  Cleared {result['cleared_tasks']} tasks during strategy switch")
        if "reinitialized_instances" in result:
            print(f"  Reinitialized {result['reinitialized_instances']} instances")
        return True
    except Exception as e:
        print(f"✗ Failed to set strategy: {e}")
        return False


def submit_task(task_id: str, sleep_time: float, exp_runtime: float) -> Tuple[bool, float]:
    """
    Submit a task to the scheduler.

    Args:
        task_id: Unique task identifier
        sleep_time: Actual sleep time for the task (seconds)
        exp_runtime: Expected runtime for prediction (milliseconds)

    Returns:
        Tuple of (success, submit_timestamp)
    """
    url = f"{SCHEDULER_URL}/task/submit"
    payload = {
        "task_id": task_id,
        "model_id": MODEL_ID,
        "task_input": {
            "sleep_time": sleep_time
        },
        "metadata": {
            "exp_runtime": exp_runtime
        }
    }

    try:
        submit_time = time.time()
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True, submit_time
    except Exception as e:
        print(f"✗ Failed to submit task {task_id}: {e}")
        return False, 0.0


def get_task_info(task_id: str) -> Dict:
    """
    Get task information from the scheduler.

    Args:
        task_id: Task identifier

    Returns:
        Task information dictionary
    """
    url = f"{SCHEDULER_URL}/task/info"
    params = {"task_id": task_id}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"✗ Failed to get task info for {task_id}: {e}")
        return {}


def wait_for_task_completion(task_id: str, timeout: int = 60) -> Tuple[bool, float]:
    """
    Wait for a task to complete.

    Args:
        task_id: Task identifier
        timeout: Maximum time to wait (seconds)

    Returns:
        Tuple of (success, completion_timestamp)
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        info = get_task_info(task_id)

        if not info:
            time.sleep(0.5)
            continue

        status = info.get("status", "")

        if status == "COMPLETED":
            return True, time.time()
        elif status == "FAILED":
            print(f"✗ Task {task_id} failed")
            return False, 0.0

        time.sleep(0.5)

    print(f"✗ Task {task_id} timed out")
    return False, 0.0


def test_strategy(strategy_name: str, task_times: List[float]) -> Dict:
    """
    Test a scheduling strategy with the given task times.

    Args:
        strategy_name: Name of the strategy
        task_times: List of task execution times (seconds)

    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*60}")
    print(f"Testing strategy: {strategy_name.upper()}")
    print(f"{'='*60}")

    # Step 1: Clear all tasks from scheduler
    print("\nStep 1: Clearing existing tasks...")
    if not clear_tasks():
        return {"error": "Failed to clear tasks"}

    # Give scheduler time to clean up
    time.sleep(1)

    # Step 2: Set the strategy
    print("\nStep 2: Setting scheduling strategy...")
    if not set_strategy(strategy_name):
        return {"error": "Failed to set strategy"}

    # Give scheduler time to initialize
    time.sleep(2)

    # Calculate expected runtime for each task
    # For round_robin and probabilistic: use actual task time
    # For min_time: use mean task time
    if strategy_name == "min_time":
        exp_runtimes = [TASK_MEAN * 1000] * len(task_times)  # Use mean for all tasks
    else:
        exp_runtimes = [t * 1000 for t in task_times]  # Use actual times

    # Submit all tasks
    print(f"\nSubmitting {len(task_times)} tasks...")
    task_records = []
    first_submit_time = None

    for i, (sleep_time, exp_runtime) in enumerate(zip(task_times, exp_runtimes)):
        task_id = f"task-{strategy_name}-{i:04d}"
        success, submit_time = submit_task(task_id, sleep_time, exp_runtime)

        if success:
            if first_submit_time is None:
                first_submit_time = submit_time

            task_records.append({
                "task_id": task_id,
                "sleep_time": sleep_time,
                "exp_runtime": exp_runtime,
                "submit_time": submit_time
            })

            if (i + 1) % 10 == 0:
                print(f"  Submitted {i + 1}/{len(task_times)} tasks")
        else:
            print(f"  Failed to submit task {i}")

    print(f"✓ Submitted {len(task_records)}/{len(task_times)} tasks")

    # Wait for all tasks to complete
    print(f"\nWaiting for tasks to complete...")
    completion_times = []
    last_completion_time = None

    for i, record in enumerate(task_records):
        task_id = record["task_id"]
        success, complete_time = wait_for_task_completion(task_id, timeout=120)

        if success:
            record["complete_time"] = complete_time
            completion_times.append(complete_time - record["submit_time"])
            last_completion_time = complete_time

            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{len(task_records)} tasks")
        else:
            print(f"  Failed to complete task {task_id}")

    print(f"✓ Completed {len(completion_times)}/{len(task_records)} tasks")

    # Calculate metrics
    if not completion_times:
        return {"error": "No tasks completed"}

    avg_completion_time = np.mean(completion_times)
    median_completion_time = np.median(completion_times)
    p95_completion_time = np.percentile(completion_times, 95)
    p99_completion_time = np.percentile(completion_times, 99)
    total_time = last_completion_time - first_submit_time if last_completion_time else 0

    # Get task distribution across instances
    instance_distribution = {}
    for record in task_records:
        if "complete_time" in record:
            info = get_task_info(record["task_id"])
            instance_id = info.get("instance_id", "unknown")
            instance_distribution[instance_id] = instance_distribution.get(instance_id, 0) + 1

    results = {
        "strategy": strategy_name,
        "num_tasks": len(task_records),
        "num_completed": len(completion_times),
        "avg_completion_time": avg_completion_time,
        "median_completion_time": median_completion_time,
        "p95_completion_time": p95_completion_time,
        "p99_completion_time": p99_completion_time,
        "total_time": total_time,
        "instance_distribution": instance_distribution
    }

    # Print results
    print(f"\n{'='*60}")
    print(f"Results for {strategy_name.upper()}")
    print(f"{'='*60}")
    print(f"Total tasks:              {results['num_tasks']}")
    print(f"Completed tasks:          {results['num_completed']}")
    print(f"Avg completion time:      {results['avg_completion_time']:.3f}s")
    print(f"Median completion time:   {results['median_completion_time']:.3f}s")
    print(f"P95 completion time:      {results['p95_completion_time']:.3f}s")
    print(f"P99 completion time:      {results['p99_completion_time']:.3f}s")
    print(f"Total execution time:     {results['total_time']:.3f}s")
    print(f"\nTask distribution:")
    for instance_id in sorted(instance_distribution.keys()):
        count = instance_distribution[instance_id]
        print(f"  {instance_id}: {count} tasks")

    return results


def main():
    """Main test function."""
    # Create results directory
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("01.quick-start-up Experiment")
    print("=" * 60)
    print(f"Testing {len(STRATEGIES)} scheduling strategies")
    print(f"Tasks per strategy: {NUM_TASKS}")
    print(f"Task time distribution: N({TASK_MEAN}, {TASK_STD}²) seconds")
    print(f"Task time range: [{TASK_MIN}, {TASK_MAX}] seconds")
    print("=" * 60)

    # Check scheduler health
    try:
        response = requests.get(f"{SCHEDULER_URL}/health", timeout=5)
        response.raise_for_status()
        print("✓ Scheduler is healthy")
    except Exception as e:
        print(f"✗ Scheduler is not responding: {e}")
        print("Please ensure all services are running (./start_all_services.sh)")
        sys.exit(1)

    # Generate task times (same for all strategies for fair comparison)
    print(f"\nGenerating {NUM_TASKS} task times...")
    task_times = generate_task_times(NUM_TASKS)
    print(f"✓ Generated task times: min={min(task_times):.3f}s, max={max(task_times):.3f}s, mean={np.mean(task_times):.3f}s")

    # Test all strategies
    all_results = []

    for strategy in STRATEGIES:
        try:
            results = test_strategy(strategy, task_times)
            all_results.append(results)
        except Exception as e:
            print(f"✗ Error testing {strategy}: {e}")
            import traceback
            traceback.print_exc()

    # Save results to file
    output_file = f"results/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({
            "experiment": "01.quick-start-up",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_tasks": NUM_TASKS,
                "task_mean": TASK_MEAN,
                "task_std": TASK_STD,
                "task_min": TASK_MIN,
                "task_max": TASK_MAX
            },
            "results": all_results
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

    # Print comparison
    print(f"\n{'='*60}")
    print("Strategy Comparison")
    print(f"{'='*60}")
    print(f"{'Strategy':<20} {'Avg Time':<12} {'P95 Time':<12} {'Total Time':<12}")
    print("-" * 60)
    for result in all_results:
        if "error" not in result:
            print(f"{result['strategy']:<20} {result['avg_completion_time']:>10.3f}s {result['p95_completion_time']:>10.3f}s {result['total_time']:>10.3f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
