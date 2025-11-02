#!/usr/bin/env python3
"""
Test script for 01.quick-start-up experiment with Poisson process task submission.

Features:
- WebSocket-based event-driven task status updates
- Two-threaded architecture:
  - Thread 1: Task submission following Poisson process
  - Thread 2: WebSocket result receiver (starts first)
- Configurable QPS (queries per second) and task count
- Pre-generated task data shared across all tests
"""

import time
import json
import requests
import numpy as np
import asyncio
import websockets
import threading
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from queue import Queue
import sys
import os


# Configuration
SCHEDULER_URL = "http://localhost:8100"
SCHEDULER_WS_URL = "ws://localhost:8100"
MODEL_ID = "sleep_model"

# Task generation parameters (bimodal distribution)
# Left peak: 1-3 seconds, Right peak: 7-10 seconds, Ratio: 1:1
LEFT_PEAK_MIN = 1.0
LEFT_PEAK_MAX = 3.0
LEFT_PEAK_MEAN = 2.0
LEFT_PEAK_STD = 0.4

RIGHT_PEAK_MIN = 7.0
RIGHT_PEAK_MAX = 10.0
RIGHT_PEAK_MEAN = 8.5
RIGHT_PEAK_STD = 0.6

PEAK_RATIO = 0.5  # 50% left peak, 50% right peak

# Calculated parameters (for compatibility)
TASK_MIN = LEFT_PEAK_MIN
TASK_MAX = RIGHT_PEAK_MAX
TASK_MEAN = LEFT_PEAK_MEAN * PEAK_RATIO + RIGHT_PEAK_MEAN * (1 - PEAK_RATIO)  # Weighted mean
TASK_STD = ((LEFT_PEAK_STD**2 + (LEFT_PEAK_MEAN - TASK_MEAN)**2) * PEAK_RATIO +
            (RIGHT_PEAK_STD**2 + (RIGHT_PEAK_MEAN - TASK_MEAN)**2) * (1 - PEAK_RATIO))**0.5

# Strategies to test
STRATEGIES = ["min_time", "round_robin", "probabilistic"]

np.random.seed(42)

@dataclass
class TaskData:
    """Pre-generated task data."""
    task_id: str
    sleep_time: float
    exp_runtime: float


@dataclass
class TaskRecord:
    """Task execution record."""
    task_id: str
    sleep_time: float
    exp_runtime: float
    submit_time: Optional[float] = None
    complete_time: Optional[float] = None
    status: Optional[str] = None
    assigned_instance: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None


class WebSocketResultReceiver:
    """WebSocket-based result receiver thread."""

    def __init__(self, ws_url: str, task_ids: List[str], result_queue: Queue):
        """
        Initialize WebSocket result receiver.

        Args:
            ws_url: WebSocket URL (ws://host:port/task/get_result)
            task_ids: List of task IDs to subscribe to
            result_queue: Queue to put results into
        """
        self.ws_url = ws_url
        self.task_ids = task_ids
        self.result_queue = result_queue
        self.running = False
        self.thread = None
        self.loop = None
        self.websocket = None

    def start(self):
        """Start the receiver thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()
        # Wait for connection to be established
        time.sleep(1)

    def stop(self):
        """Stop the receiver thread."""
        self.running = False
        if self.loop and self.websocket:
            asyncio.run_coroutine_threadsafe(self._close_websocket(), self.loop)
        if self.thread:
            self.thread.join(timeout=5)

    def _run_async_loop(self):
        """Run the asyncio event loop in this thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._receive_results())
        finally:
            self.loop.close()

    async def _close_websocket(self):
        """Close the WebSocket connection."""
        if self.websocket:
            await self.websocket.close()

    async def _receive_results(self):
        """Connect to WebSocket and receive results."""
        try:
            async with websockets.connect(self.ws_url) as websocket:
                self.websocket = websocket
                print(f"✓ WebSocket connected to {self.ws_url}")

                # Subscribe to task IDs
                subscribe_msg = {
                    "type": "subscribe",
                    "task_ids": self.task_ids
                }
                await websocket.send(json.dumps(subscribe_msg))
                print(f"✓ Subscribed to {len(self.task_ids)} tasks")

                # Receive results
                while self.running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)

                        if data.get("type") == "result":
                            result = {
                                "task_id": data.get("task_id"),
                                "status": data.get("status"),
                                "result": data.get("result"),
                                "error": data.get("error"),
                                "execution_time_ms": data.get("execution_time_ms"),
                                "timestamp": time.time()
                            }
                            self.result_queue.put(result)

                        elif data.get("type") == "error":
                            result = {
                                "task_id": data.get("task_id"),
                                "status": "failed",
                                "result": None,
                                "error": data.get("error"),
                                "execution_time_ms": None,
                                "timestamp": time.time()
                            }
                            self.result_queue.put(result)

                        elif data.get("type") == "ack":
                            print(f"  WebSocket: {data.get('message')}")

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        if self.running:
                            print(f"✗ Error receiving message: {e}")
                        break

        except Exception as e:
            print(f"✗ WebSocket connection error: {e}")
        finally:
            print("✓ WebSocket connection closed")


class PoissonTaskSubmitter:
    """Task submitter following Poisson process."""

    def __init__(self, scheduler_url: str, tasks: List[TaskData], qps: float):
        """
        Initialize Poisson task submitter.

        Args:
            scheduler_url: Scheduler HTTP URL
            tasks: List of tasks to submit
            qps: Target queries per second (QPS)
        """
        self.scheduler_url = scheduler_url
        self.tasks = tasks
        self.qps = qps
        self.running = False
        self.thread = None
        self.submitted_tasks = []
        self.first_submit_time = None
        self.last_submit_time = None

    def start(self):
        """Start the submitter thread."""
        self.running = True
        self.thread = threading.Thread(target=self._submit_tasks, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the submitter thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def wait_completion(self):
        """Wait for all tasks to be submitted."""
        if self.thread:
            self.thread.join()

    def _submit_tasks(self):
        """Submit tasks following Poisson process."""
        print(f"\n{'='*60}")
        print(f"Starting task submission (QPS={self.qps})")
        print(f"{'='*60}")

        # Generate inter-arrival times (exponential distribution)
        lambda_rate = self.qps
        inter_arrival_times = np.random.exponential(1.0 / lambda_rate, len(self.tasks))

        self.first_submit_time = time.time()

        for i, (task, wait_time) in enumerate(zip(self.tasks, inter_arrival_times)):
            if not self.running:
                break

            # Wait for the inter-arrival time
            if i > 0:  # Don't wait before the first task
                time.sleep(wait_time)

            # Submit task
            success, submit_time, assigned_instance = self._submit_task(task)

            if success:
                self.submitted_tasks.append({
                    "task_id": task.task_id,
                    "sleep_time": task.sleep_time,
                    "exp_runtime": task.exp_runtime,
                    "submit_time": submit_time,
                    "assigned_instance": assigned_instance
                })
                self.last_submit_time = submit_time

                if (i + 1) % 10 == 0:
                    print(f"  Submitted {i + 1}/{len(self.tasks)} tasks")
            else:
                print(f"  Failed to submit task {i}")

        print(f"✓ Submitted {len(self.submitted_tasks)}/{len(self.tasks)} tasks")
        print(f"  Total submission time: {self.last_submit_time - self.first_submit_time:.3f}s")
        print(f"  Actual QPS: {len(self.submitted_tasks) / (self.last_submit_time - self.first_submit_time):.2f}")

    def _submit_task(self, task: TaskData) -> Tuple[bool, float, Optional[str]]:
        """
        Submit a single task.

        Args:
            task: Task data to submit

        Returns:
            Tuple of (success, submit_timestamp, assigned_instance)
        """
        url = f"{self.scheduler_url}/task/submit"
        payload = {
            "task_id": task.task_id,
            "model_id": MODEL_ID,
            "task_input": {
                "sleep_time": task.sleep_time
            },
            "metadata": {
                "exp_runtime": task.exp_runtime
            }
        }

        try:
            submit_time = time.time()
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()

            # Parse response to get assigned instance
            result = response.json()
            assigned_instance = None
            if result.get("success") and "task" in result:
                assigned_instance = result["task"].get("assigned_instance")

            return True, submit_time, assigned_instance
        except Exception as e:
            print(f"✗ Failed to submit task {task.task_id}: {e}")
            return False, 0.0, None


def generate_task_times(num_tasks: int) -> List[float]:
    """
    Generate task execution times from a bimodal distribution.

    The distribution has two peaks:
    - Left peak: configurable range (default: 1-3 seconds)
    - Right peak: configurable range (default: 7-10 seconds)
    Tasks are generated according to PEAK_RATIO and shuffled randomly.

    Args:
        num_tasks: Number of tasks to generate

    Returns:
        List of task execution times in seconds (randomly shuffled)
    """
    # Calculate number of tasks for each peak based on PEAK_RATIO
    num_left_peak = int(num_tasks * PEAK_RATIO)
    num_right_peak = num_tasks - num_left_peak

    # Left peak
    left_times = np.random.normal(LEFT_PEAK_MEAN, LEFT_PEAK_STD, num_left_peak)
    left_times = np.clip(left_times, LEFT_PEAK_MIN, LEFT_PEAK_MAX)

    # Right peak
    right_times = np.random.normal(RIGHT_PEAK_MEAN, RIGHT_PEAK_STD, num_right_peak)
    right_times = np.clip(right_times, RIGHT_PEAK_MIN, RIGHT_PEAK_MAX)

    # Combine and shuffle
    times = np.concatenate([left_times, right_times])
    np.random.shuffle(times)

    return times.tolist()


def generate_task_data(num_tasks: int, strategy_name: str, task_times: List[float]) -> List[TaskData]:
    """
    Generate pre-generated task data.

    Args:
        num_tasks: Number of tasks to generate
        strategy_name: Name of the strategy (for task ID prefix)
        task_times: Pre-generated task execution times

    Returns:
        List of TaskData objects
    """
    tasks = []

    # Calculate expected runtime for each task
    # For round_robin and probabilistic: use actual task time
    # For min_time: use mean task time
    if strategy_name == "min_time":
        exp_runtimes = [TASK_MEAN * 1000] * len(task_times)  # Use mean for all tasks
    else:
        exp_runtimes = [t * 1000 for t in task_times]  # Use actual times

    for i, (sleep_time, exp_runtime) in enumerate(zip(task_times, exp_runtimes)):
        task_id = f"task-{strategy_name}-{i:04d}"
        tasks.append(TaskData(
            task_id=task_id,
            sleep_time=sleep_time,
            exp_runtime=exp_runtime
        ))

    return tasks


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


def test_strategy_with_poisson(strategy_name: str, tasks: List[TaskData], qps: float) -> Dict:
    """
    Test a scheduling strategy with Poisson process task submission.

    Args:
        strategy_name: Name of the strategy
        tasks: List of pre-generated tasks
        qps: Target queries per second

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

    # Create result queue
    result_queue = Queue()

    # Start WebSocket receiver (Thread 2 - starts first)
    task_ids = [task.task_id for task in tasks]
    ws_url = f"{SCHEDULER_WS_URL}/task/get_result"
    receiver = WebSocketResultReceiver(ws_url, task_ids, result_queue)
    receiver.start()

    # Wait for WebSocket connection to be established
    time.sleep(2)

    # Start Poisson task submitter (Thread 1 - starts second)
    submitter = PoissonTaskSubmitter(SCHEDULER_URL, tasks, qps)
    submitter.start()

    # Wait for all tasks to be submitted
    submitter.wait_completion()

    # Collect results from WebSocket
    print(f"\n{'='*60}")
    print(f"Waiting for task results...")
    print(f"{'='*60}")

    task_records = {task.task_id: TaskRecord(
        task_id=task.task_id,
        sleep_time=task.sleep_time,
        exp_runtime=task.exp_runtime
    ) for task in tasks}

    # Update submit times and assigned instances
    for submitted in submitter.submitted_tasks:
        task_id = submitted["task_id"]
        if task_id in task_records:
            task_records[task_id].submit_time = submitted["submit_time"]
            task_records[task_id].assigned_instance = submitted["assigned_instance"]

    completed_count = 0
    target_count = len(submitter.submitted_tasks)
    last_print_count = 0

    # Wait for all results (with timeout)
    timeout = 300  # 5 minutes
    start_time = time.time()

    while completed_count < target_count:
        if time.time() - start_time > timeout:
            print(f"✗ Timeout waiting for results ({completed_count}/{target_count} completed)")
            break

        try:
            # Non-blocking get with timeout
            result = result_queue.get(timeout=1.0)

            task_id = result["task_id"]
            if task_id in task_records:
                task_records[task_id].complete_time = result["timestamp"]
                task_records[task_id].status = result["status"]
                task_records[task_id].result = result["result"]
                task_records[task_id].error = result["error"]
                task_records[task_id].execution_time_ms = result["execution_time_ms"]

                completed_count += 1

                if completed_count - last_print_count >= 10:
                    print(f"  Completed {completed_count}/{target_count} tasks")
                    last_print_count = completed_count

        except:
            # Timeout on queue.get, continue waiting
            continue

    print(f"✓ Completed {completed_count}/{target_count} tasks")

    # Stop receiver
    receiver.stop()

    # Calculate metrics
    completion_times = []
    successful_tasks = []

    for task_id, record in task_records.items():
        if record.submit_time and record.complete_time and record.status == "completed":
            completion_time = record.complete_time - record.submit_time
            completion_times.append(completion_time)
            successful_tasks.append(record)

    if not completion_times:
        return {"error": "No tasks completed"}

    avg_completion_time = np.mean(completion_times)
    median_completion_time = np.median(completion_times)
    p95_completion_time = np.percentile(completion_times, 95)
    p99_completion_time = np.percentile(completion_times, 99)

    first_submit_time = submitter.first_submit_time
    last_completion_time = max(record.complete_time for record in successful_tasks)
    total_time = last_completion_time - first_submit_time

    # Get task distribution across instances (from real-time submission data)
    instance_distribution_submitted = {}
    instance_distribution_completed = {}

    for task_id, record in task_records.items():
        # Count all submitted tasks by assigned instance
        if record.assigned_instance:
            instance_id = record.assigned_instance
            instance_distribution_submitted[instance_id] = instance_distribution_submitted.get(instance_id, 0) + 1

            # Also track completed tasks separately
            if record.status == "completed":
                instance_distribution_completed[instance_id] = instance_distribution_completed.get(instance_id, 0) + 1

    results = {
        "strategy": strategy_name,
        "qps": qps,
        "num_tasks": len(tasks),
        "num_submitted": len(submitter.submitted_tasks),
        "num_completed": len(successful_tasks),
        "avg_completion_time": avg_completion_time,
        "median_completion_time": median_completion_time,
        "p95_completion_time": p95_completion_time,
        "p99_completion_time": p99_completion_time,
        "total_time": total_time,
        "submission_time": submitter.last_submit_time - submitter.first_submit_time,
        "actual_qps": len(submitter.submitted_tasks) / (submitter.last_submit_time - submitter.first_submit_time),
        "instance_distribution_submitted": instance_distribution_submitted,
        "instance_distribution_completed": instance_distribution_completed
    }

    # Print results
    print(f"\n{'='*60}")
    print(f"Results for {strategy_name.upper()}")
    print(f"{'='*60}")
    print(f"Total tasks:              {results['num_tasks']}")
    print(f"Submitted tasks:          {results['num_submitted']}")
    print(f"Completed tasks:          {results['num_completed']}")
    print(f"Target QPS:               {results['qps']:.2f}")
    print(f"Actual QPS:               {results['actual_qps']:.2f}")
    print(f"Submission time:          {results['submission_time']:.3f}s")
    print(f"Avg completion time:      {results['avg_completion_time']:.3f}s")
    print(f"Median completion time:   {results['median_completion_time']:.3f}s")
    print(f"P95 completion time:      {results['p95_completion_time']:.3f}s")
    print(f"P99 completion time:      {results['p99_completion_time']:.3f}s")
    print(f"Total execution time:     {results['total_time']:.3f}s")
    print(f"\nTask distribution (submitted):")
    for instance_id in sorted(instance_distribution_submitted.keys()):
        count = instance_distribution_submitted[instance_id]
        percentage = (count / results['num_submitted']) * 100
        print(f"  {instance_id}: {count} tasks ({percentage:.1f}%)")
    print(f"\nTask distribution (completed):")
    for instance_id in sorted(instance_distribution_completed.keys()):
        count = instance_distribution_completed[instance_id]
        percentage = (count / results['num_completed']) * 100
        print(f"  {instance_id}: {count} tasks ({percentage:.1f}%)")

    return results


def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test scheduling strategies with Poisson process")
    parser.add_argument("--qps", type=float, default=10.0, help="Target queries per second (default: 10.0)")
    parser.add_argument("--num-tasks", type=int, default=100, help="Number of tasks per strategy (default: 100)")
    parser.add_argument("--strategies", type=str, nargs="+", default=STRATEGIES,
                        help="Strategies to test (default: all)")
    args = parser.parse_args()

    # Create results directory
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("01.quick-start-up Experiment (Poisson Process)")
    print("=" * 60)
    print(f"Testing {len(args.strategies)} scheduling strategies")
    print(f"Tasks per strategy: {args.num_tasks}")
    print(f"Target QPS: {args.qps}")
    print(f"Task time distribution: Bimodal")
    print(f"  Left peak:  [{LEFT_PEAK_MIN}, {LEFT_PEAK_MAX}]s, mean={LEFT_PEAK_MEAN}s, ratio={PEAK_RATIO:.0%}")
    print(f"  Right peak: [{RIGHT_PEAK_MIN}, {RIGHT_PEAK_MAX}]s, mean={RIGHT_PEAK_MEAN}s, ratio={1-PEAK_RATIO:.0%}")
    print(f"  Overall: mean={TASK_MEAN:.3f}s, std={TASK_STD:.3f}s, range=[{TASK_MIN}, {TASK_MAX}]s")
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
    print(f"\nGenerating {args.num_tasks} task times...")
    task_times = generate_task_times(args.num_tasks)
    print(f"✓ Generated task times: min={min(task_times):.3f}s, max={max(task_times):.3f}s, mean={np.mean(task_times):.3f}s")

    # Test all strategies
    all_results = []

    for strategy in args.strategies:
        try:
            # Generate task data for this strategy
            tasks = generate_task_data(args.num_tasks, strategy, task_times)

            # Test strategy
            results = test_strategy_with_poisson(strategy, tasks, args.qps)
            all_results.append(results)
        except Exception as e:
            print(f"✗ Error testing {strategy}: {e}")
            import traceback
            traceback.print_exc()

    # Save results to file
    output_file = f"results/results_poisson_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({
            "experiment": "01.quick-start-up-poisson",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_tasks": args.num_tasks,
                "qps": args.qps,
                "distribution": "bimodal",
                "left_peak": {
                    "min": LEFT_PEAK_MIN,
                    "max": LEFT_PEAK_MAX,
                    "mean": LEFT_PEAK_MEAN,
                    "std": LEFT_PEAK_STD,
                    "ratio": PEAK_RATIO
                },
                "right_peak": {
                    "min": RIGHT_PEAK_MIN,
                    "max": RIGHT_PEAK_MAX,
                    "mean": RIGHT_PEAK_MEAN,
                    "std": RIGHT_PEAK_STD,
                    "ratio": 1 - PEAK_RATIO
                },
                "overall": {
                    "mean": TASK_MEAN,
                    "std": TASK_STD,
                    "min": TASK_MIN,
                    "max": TASK_MAX
                }
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
    print(f"{'Strategy':<20} {'QPS':<8} {'Avg Time':<12} {'P95 Time':<12} {'Total Time':<12}")
    print("-" * 60)
    for result in all_results:
        if "error" not in result:
            print(f"{result['strategy']:<20} {result['actual_qps']:>6.2f} {result['avg_completion_time']:>10.3f}s {result['p95_completion_time']:>10.3f}s {result['total_time']:>10.3f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
