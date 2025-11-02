#!/usr/bin/env python3
"""
Test script for 02.multi_model_no_dep experiment with dual-scheduler setup.

Features:
- Two independent schedulers handling different workloads
- WebSocket-based event-driven task status updates (one per scheduler)
- Four-threaded architecture:
  - Thread 1: WebSocket receiver for Scheduler A
  - Thread 2: WebSocket receiver for Scheduler B
  - Thread 3: Poisson submitter for Scheduler A (bimodal workload)
  - Thread 4: Poisson submitter for Scheduler B (Pareto workload)
- Independent Poisson processes for each scheduler
- Configurable instance counts, QPS, and task counts
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

# Import workload generator
from workload_generator import (
    generate_bimodal_distribution,
    generate_pareto_distribution,
    print_distribution_stats
)

# Configuration
SCHEDULER_A_URL = "http://localhost:8100"
SCHEDULER_A_WS_URL = "ws://localhost:8100"
SCHEDULER_B_URL = "http://localhost:8200"
SCHEDULER_B_WS_URL = "ws://localhost:8200"
MODEL_ID = "sleep_model"

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

    def __init__(self, name: str, ws_url: str, task_ids: List[str], result_queue: Queue):
        """
        Initialize WebSocket result receiver.

        Args:
            name: Receiver name (for logging)
            ws_url: WebSocket URL (ws://host:port/task/get_result)
            task_ids: List of task IDs to subscribe to
            result_queue: Queue to put results into
        """
        self.name = name
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
                print(f"✓ [{self.name}] WebSocket connected to {self.ws_url}")

                # Subscribe to task IDs
                subscribe_msg = {
                    "type": "subscribe",
                    "task_ids": self.task_ids
                }
                await websocket.send(json.dumps(subscribe_msg))
                print(f"✓ [{self.name}] Subscribed to {len(self.task_ids)} tasks")

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
                            print(f"  [{self.name}] WebSocket: {data.get('message')}")

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        if self.running:
                            print(f"✗ [{self.name}] Error receiving message: {e}")
                        break

        except Exception as e:
            print(f"✗ [{self.name}] WebSocket connection error: {e}")
        finally:
            print(f"✓ [{self.name}] WebSocket connection closed")


class PoissonTaskSubmitter:
    """Task submitter following Poisson process."""

    def __init__(self, name: str, scheduler_url: str, tasks: List[TaskData], qps: float):
        """
        Initialize Poisson task submitter.

        Args:
            name: Submitter name (for logging)
            scheduler_url: Scheduler HTTP URL
            tasks: List of tasks to submit
            qps: Target queries per second (QPS)
        """
        self.name = name
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
        print(f"[{self.name}] Starting task submission (QPS={self.qps})")
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
                    print(f"  [{self.name}] Submitted {i + 1}/{len(self.tasks)} tasks")
            else:
                print(f"  [{self.name}] Failed to submit task {i}")

        print(f"✓ [{self.name}] Submitted {len(self.submitted_tasks)}/{len(self.tasks)} tasks")
        print(f"  [{self.name}] Total submission time: {self.last_submit_time - self.first_submit_time:.3f}s")
        print(f"  [{self.name}] Actual QPS: {len(self.submitted_tasks) / (self.last_submit_time - self.first_submit_time):.2f}")

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
            print(f"✗ [{self.name}] Failed to submit task {task.task_id}: {e}")
            return False, 0.0, None


def generate_task_data(num_tasks: int, strategy_name: str, scheduler_name: str,
                      task_times: List[float], mean_time: float) -> List[TaskData]:
    """
    Generate pre-generated task data.

    Args:
        num_tasks: Number of tasks to generate
        strategy_name: Name of the strategy (for task ID prefix)
        scheduler_name: Name of the scheduler ("A" or "B")
        task_times: Pre-generated task execution times
        mean_time: Mean task execution time (for min_time strategy)

    Returns:
        List of TaskData objects
    """
    tasks = []

    # Calculate expected runtime for each task
    # For round_robin and probabilistic: use actual task time
    # For min_time: use mean task time
    if strategy_name == "min_time":
        exp_runtimes = [mean_time * 1000] * len(task_times)  # Use mean for all tasks
    else:
        exp_runtimes = [t * 1000 for t in task_times]  # Use actual times

    for i, (sleep_time, exp_runtime) in enumerate(zip(task_times, exp_runtimes)):
        task_id = f"task-{scheduler_name}-{strategy_name}-{i:04d}"
        tasks.append(TaskData(
            task_id=task_id,
            sleep_time=sleep_time,
            exp_runtime=exp_runtime
        ))

    return tasks


def clear_tasks(scheduler_url: str, scheduler_name: str) -> bool:
    """
    Clear all tasks from a scheduler.

    Args:
        scheduler_url: Scheduler HTTP URL
        scheduler_name: Scheduler name (for logging)

    Returns:
        True if successful, False otherwise
    """
    url = f"{scheduler_url}/task/clear"

    try:
        response = requests.post(url, timeout=10)
        response.raise_for_status()
        result = response.json()
        cleared_count = result.get("cleared_count", 0)
        print(f"✓ [{scheduler_name}] Cleared {cleared_count} tasks")
        return True
    except Exception as e:
        print(f"✗ [{scheduler_name}] Failed to clear tasks: {e}")
        return False


def set_strategy(scheduler_url: str, scheduler_name: str, strategy_name: str,
                target_quantile: float = 0.9) -> bool:
    """
    Set the scheduling strategy for a scheduler.

    Args:
        scheduler_url: Scheduler HTTP URL
        scheduler_name: Scheduler name (for logging)
        strategy_name: Name of the strategy (round_robin, min_time, probabilistic)
        target_quantile: Target quantile for probabilistic strategy

    Returns:
        True if successful, False otherwise
    """
    url = f"{scheduler_url}/strategy/set"
    payload = {"strategy_name": strategy_name}

    if strategy_name == "probabilistic":
        payload["target_quantile"] = target_quantile

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        print(f"✓ [{scheduler_name}] Strategy set to: {strategy_name}")
        if "cleared_tasks" in result:
            print(f"  [{scheduler_name}] Cleared {result['cleared_tasks']} tasks during strategy switch")
        if "reinitialized_instances" in result:
            print(f"  [{scheduler_name}] Reinitialized {result['reinitialized_instances']} instances")
        return True
    except Exception as e:
        print(f"✗ [{scheduler_name}] Failed to set strategy: {e}")
        return False


def collect_results(receiver: WebSocketResultReceiver, submitter: PoissonTaskSubmitter,
                   tasks: List[TaskData], scheduler_name: str) -> Dict[str, TaskRecord]:
    """
    Collect results from WebSocket receiver.

    Args:
        receiver: WebSocket receiver instance
        submitter: Task submitter instance
        tasks: List of tasks
        scheduler_name: Scheduler name (for logging)

    Returns:
        Dictionary of task_id -> TaskRecord
    """
    print(f"\n{'='*60}")
    print(f"[{scheduler_name}] Waiting for task results...")
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

    received_count = 0  # Total results received (success + failed)
    completed_count = 0  # Successfully completed tasks
    failed_count = 0     # Failed tasks
    target_count = len(submitter.submitted_tasks)
    last_print_count = 0

    # Wait for all results (with timeout)
    timeout = 300  # 5 minutes
    start_time = time.time()

    while received_count < target_count:
        if time.time() - start_time > timeout:
            print(f"✗ [{scheduler_name}] Timeout waiting for results ({received_count}/{target_count} received, {completed_count} completed, {failed_count} failed)")
            break

        try:
            # Non-blocking get with timeout
            result = receiver.result_queue.get(timeout=1.0)

            task_id = result["task_id"]
            if task_id in task_records:
                task_records[task_id].complete_time = result["timestamp"]
                task_records[task_id].status = result["status"]
                task_records[task_id].result = result["result"]
                task_records[task_id].error = result["error"]
                task_records[task_id].execution_time_ms = result["execution_time_ms"]

                received_count += 1

                if result["status"] == "completed":
                    completed_count += 1
                else:
                    failed_count += 1

                if received_count - last_print_count >= 10:
                    print(f"  [{scheduler_name}] Received {received_count}/{target_count} results ({completed_count} completed, {failed_count} failed)")
                    last_print_count = received_count

        except:
            # Timeout on queue.get, continue waiting
            continue

    print(f"✓ [{scheduler_name}] Received {received_count}/{target_count} results ({completed_count} completed, {failed_count} failed)")

    # Process any remaining results in the queue
    print(f"  [{scheduler_name}] Processing remaining results in queue...")
    remaining_count = 0
    while not receiver.result_queue.empty():
        try:
            result = receiver.result_queue.get_nowait()
            task_id = result["task_id"]
            if task_id in task_records:
                # Only update if not already processed
                if task_records[task_id].status is None:
                    task_records[task_id].complete_time = result["timestamp"]
                    task_records[task_id].status = result["status"]
                    task_records[task_id].result = result["result"]
                    task_records[task_id].error = result["error"]
                    task_records[task_id].execution_time_ms = result["execution_time_ms"]

                    received_count += 1
                    remaining_count += 1

                    if result["status"] == "completed":
                        completed_count += 1
                    else:
                        failed_count += 1
        except:
            break

    if remaining_count > 0:
        print(f"  [{scheduler_name}] Processed {remaining_count} remaining results")
        print(f"✓ [{scheduler_name}] Final: {received_count} total ({completed_count} completed, {failed_count} failed)")

    return task_records


def calculate_metrics(task_records: Dict[str, TaskRecord], submitter: PoissonTaskSubmitter,
                     strategy_name: str, qps: float, scheduler_name: str) -> Dict:
    """
    Calculate performance metrics from task records.

    Args:
        task_records: Dictionary of task records
        submitter: Task submitter instance
        strategy_name: Strategy name
        qps: Target QPS
        scheduler_name: Scheduler name

    Returns:
        Dictionary with performance metrics
    """
    completion_times = []
    successful_tasks = []
    failed_tasks = []

    for task_id, record in task_records.items():
        if record.submit_time and record.complete_time:
            if record.status == "completed":
                completion_time = record.complete_time - record.submit_time
                completion_times.append(completion_time)
                successful_tasks.append(record)
            else:
                failed_tasks.append(record)

    if not completion_times:
        # Debug: print why no tasks completed
        completed_count = sum(1 for r in task_records.values() if r.status == "completed")
        with_submit_time = sum(1 for r in task_records.values() if r.submit_time)
        with_complete_time = sum(1 for r in task_records.values() if r.complete_time)
        status_values = set(r.status for r in task_records.values() if r.status)
        print(f"[{scheduler_name}] Debug: {completed_count} completed, {with_submit_time} with submit_time, {with_complete_time} with complete_time")
        print(f"[{scheduler_name}] Debug: status values seen: {status_values}")
        return {"error": "No tasks completed"}

    avg_completion_time = float(np.mean(completion_times))
    median_completion_time = float(np.median(completion_times))
    p95_completion_time = float(np.percentile(completion_times, 95))
    p99_completion_time = float(np.percentile(completion_times, 99))

    first_submit_time = submitter.first_submit_time
    last_completion_time = max(record.complete_time for record in successful_tasks)
    total_time = last_completion_time - first_submit_time

    # Get task distribution across instances
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
        "scheduler": scheduler_name,
        "strategy": strategy_name,
        "qps": qps,
        "num_tasks": len([t for t in task_records.values()]),
        "num_submitted": len(submitter.submitted_tasks),
        "num_completed": len(successful_tasks),
        "num_failed": len(failed_tasks),
        "num_received": len(successful_tasks) + len(failed_tasks),
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

    return results


def print_metrics(results: Dict):
    """Print performance metrics."""
    scheduler_name = results['scheduler']
    print(f"\n{'='*60}")
    print(f"Results for [{scheduler_name}] {results['strategy'].upper()}")
    print(f"{'='*60}")
    print(f"Total tasks:              {results['num_tasks']}")
    print(f"Submitted tasks:          {results['num_submitted']}")
    print(f"Received results:         {results['num_received']}")
    print(f"  ├─ Completed:           {results['num_completed']}")
    print(f"  └─ Failed:              {results['num_failed']}")
    print(f"Target QPS:               {results['qps']:.2f}")
    print(f"Actual QPS:               {results['actual_qps']:.2f}")
    print(f"Submission time:          {results['submission_time']:.3f}s")
    print(f"Avg completion time:      {results['avg_completion_time']:.3f}s")
    print(f"Median completion time:   {results['median_completion_time']:.3f}s")
    print(f"P95 completion time:      {results['p95_completion_time']:.3f}s")
    print(f"P99 completion time:      {results['p99_completion_time']:.3f}s")
    print(f"Total execution time:     {results['total_time']:.3f}s")
    print(f"\nTask distribution (submitted):")
    for instance_id in sorted(results['instance_distribution_submitted'].keys()):
        count = results['instance_distribution_submitted'][instance_id]
        percentage = (count / results['num_submitted']) * 100
        print(f"  {instance_id}: {count} tasks ({percentage:.1f}%)")
    print(f"\nTask distribution (completed):")
    for instance_id in sorted(results['instance_distribution_completed'].keys()):
        count = results['instance_distribution_completed'][instance_id]
        percentage = (count / results['num_completed']) * 100
        print(f"  {instance_id}: {count} tasks ({percentage:.1f}%)")


def test_strategy_dual_scheduler(strategy_name: str,
                                 tasks_a: List[TaskData], tasks_b: List[TaskData],
                                 qps_a: float, qps_b: float,
                                 mean_time_a: float, mean_time_b: float) -> Dict:
    """
    Test a scheduling strategy with dual schedulers.

    Args:
        strategy_name: Name of the strategy
        tasks_a: Tasks for Scheduler A
        tasks_b: Tasks for Scheduler B
        qps_a: Target QPS for Scheduler A
        qps_b: Target QPS for Scheduler B
        mean_time_a: Mean time for Scheduler A tasks
        mean_time_b: Mean time for Scheduler B tasks

    Returns:
        Dictionary with test results for both schedulers
    """
    print(f"\n{'='*60}")
    print(f"Testing strategy: {strategy_name.upper()}")
    print(f"{'='*60}")

    # Step 1: Clear all tasks from both schedulers
    print("\nStep 1: Clearing existing tasks...")
    if not clear_tasks(SCHEDULER_A_URL, "Scheduler A"):
        return {"error": "Failed to clear tasks from Scheduler A"}
    if not clear_tasks(SCHEDULER_B_URL, "Scheduler B"):
        return {"error": "Failed to clear tasks from Scheduler B"}

    # Give schedulers time to clean up
    time.sleep(1)

    # Step 2: Set the strategy on both schedulers
    print("\nStep 2: Setting scheduling strategy...")
    if not set_strategy(SCHEDULER_A_URL, "Scheduler A", strategy_name):
        return {"error": "Failed to set strategy on Scheduler A"}
    if not set_strategy(SCHEDULER_B_URL, "Scheduler B", strategy_name):
        return {"error": "Failed to set strategy on Scheduler B"}

    # Give schedulers time to initialize
    time.sleep(2)

    # Create result queues
    result_queue_a = Queue()
    result_queue_b = Queue()

    # Start WebSocket receivers (Threads 1 & 2)
    task_ids_a = [task.task_id for task in tasks_a]
    task_ids_b = [task.task_id for task in tasks_b]

    ws_url_a = f"{SCHEDULER_A_WS_URL}/task/get_result"
    ws_url_b = f"{SCHEDULER_B_WS_URL}/task/get_result"

    receiver_a = WebSocketResultReceiver("Scheduler A", ws_url_a, task_ids_a, result_queue_a)
    receiver_b = WebSocketResultReceiver("Scheduler B", ws_url_b, task_ids_b, result_queue_b)

    receiver_a.start()
    receiver_b.start()

    # Wait for WebSocket connections to be established
    time.sleep(2)

    # Start Poisson task submitters (Threads 3 & 4)
    submitter_a = PoissonTaskSubmitter("Scheduler A", SCHEDULER_A_URL, tasks_a, qps_a)
    submitter_b = PoissonTaskSubmitter("Scheduler B", SCHEDULER_B_URL, tasks_b, qps_b)

    submitter_a.start()
    submitter_b.start()

    # Wait for all tasks to be submitted
    submitter_a.wait_completion()
    submitter_b.wait_completion()

    # Collect results from both schedulers
    task_records_a = collect_results(receiver_a, submitter_a, tasks_a, "Scheduler A")
    task_records_b = collect_results(receiver_b, submitter_b, tasks_b, "Scheduler B")

    # Stop receivers
    receiver_a.stop()
    receiver_b.stop()

    # Calculate metrics
    results_a = calculate_metrics(task_records_a, submitter_a, strategy_name, qps_a, "Scheduler A")
    results_b = calculate_metrics(task_records_b, submitter_b, strategy_name, qps_b, "Scheduler B")

    # Print results
    if "error" not in results_a:
        print_metrics(results_a)
    else:
        print(f"✗ [Scheduler A] {results_a['error']}")

    if "error" not in results_b:
        print_metrics(results_b)
    else:
        print(f"✗ [Scheduler B] {results_b['error']}")

    return {
        "strategy": strategy_name,
        "scheduler_a": results_a,
        "scheduler_b": results_b
    }


def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test dual-scheduler setup with different workloads")
    parser.add_argument("--n1", type=int, default=10, help="Number of instances in Group A (default: 10)")
    parser.add_argument("--n2", type=int, default=6, help="Number of instances in Group B (default: 6)")
    parser.add_argument("--qps1", type=float, default=8.0, help="QPS for Scheduler A (default: 8.0)")
    parser.add_argument("--qps2", type=float, default=5.0, help="QPS for Scheduler B (default: 5.0)")
    parser.add_argument("--num-tasks", type=int, default=100, help="Number of tasks per scheduler per strategy (default: 100)")
    parser.add_argument("--strategies", type=str, nargs="+", default=STRATEGIES,
                        help="Strategies to test (default: all)")
    args = parser.parse_args()

    # Create results directory
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("02.multi_model_no_dep Experiment")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Group A: {args.n1} instances")
    print(f"  Group B: {args.n2} instances")
    print(f"  Scheduler A QPS: {args.qps1}")
    print(f"  Scheduler B QPS: {args.qps2}")
    print(f"  Tasks per scheduler: {args.num_tasks}")
    print(f"  Strategies: {', '.join(args.strategies)}")
    print("=" * 60)

    # Check scheduler health
    print("\nChecking scheduler health...")
    try:
        response = requests.get(f"{SCHEDULER_A_URL}/health", timeout=5)
        response.raise_for_status()
        print("✓ Scheduler A is healthy")
    except Exception as e:
        print(f"✗ Scheduler A is not responding: {e}")
        print("Please ensure all services are running (./start_all_services.sh)")
        sys.exit(1)

    try:
        response = requests.get(f"{SCHEDULER_B_URL}/health", timeout=5)
        response.raise_for_status()
        print("✓ Scheduler B is healthy")
    except Exception as e:
        print(f"✗ Scheduler B is not responding: {e}")
        print("Please ensure all services are running (./start_all_services.sh)")
        sys.exit(1)

    # Generate workloads (same for all strategies for fair comparison)
    print(f"\nGenerating workloads...")
    task_times_a, config_a = generate_bimodal_distribution(args.num_tasks, seed=42)
    task_times_b, config_b = generate_pareto_distribution(args.num_tasks, seed=43)

    print_distribution_stats(task_times_a, config_a)
    print_distribution_stats(task_times_b, config_b)

    # Test all strategies
    all_results = []

    for strategy in args.strategies:
        try:
            # Generate task data for this strategy
            tasks_a = generate_task_data(args.num_tasks, strategy, "A", task_times_a, config_a.mean_time)
            tasks_b = generate_task_data(args.num_tasks, strategy, "B", task_times_b, config_b.mean_time)

            # Test strategy
            results = test_strategy_dual_scheduler(
                strategy, tasks_a, tasks_b,
                args.qps1, args.qps2,
                config_a.mean_time, config_b.mean_time
            )
            all_results.append(results)
        except Exception as e:
            print(f"✗ Error testing {strategy}: {e}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()

    # Save results to file
    output_file = f"results/results_dual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({
            "experiment": "02.multi_model_no_dep",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "n1": args.n1,
                "n2": args.n2,
                "qps1": args.qps1,
                "qps2": args.qps2,
                "num_tasks": args.num_tasks,
                "workload_a": {
                    "type": config_a.name,
                    "description": config_a.description,
                    "mean": config_a.mean_time,
                    "std": config_a.std_time,
                    "min": config_a.min_time,
                    "max": config_a.max_time
                },
                "workload_b": {
                    "type": config_b.name,
                    "description": config_b.description,
                    "mean": config_b.mean_time,
                    "std": config_b.std_time,
                    "min": config_b.min_time,
                    "max": config_b.max_time
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
    print(f"{'Strategy':<20} {'Scheduler':<12} {'QPS':<8} {'Avg Time':<12} {'P95 Time':<12} {'Total Time':<12}")
    print("-" * 80)
    for result in all_results:
        if "error" not in result:
            strategy = result['strategy']
            for sched_key in ['scheduler_a', 'scheduler_b']:
                sched_result = result[sched_key]
                if "error" not in sched_result:
                    print(f"{strategy:<20} {sched_result['scheduler']:<12} {sched_result['actual_qps']:>6.2f} {sched_result['avg_completion_time']:>10.3f}s {sched_result['p95_completion_time']:>10.3f}s {sched_result['total_time']:>10.3f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
