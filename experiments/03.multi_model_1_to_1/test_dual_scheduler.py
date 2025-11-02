#!/usr/bin/env python3
"""
Test script for 03.multi_model_1_to_1 experiment with dual-scheduler setup and 1-to-1 task dependencies.

Features:
- Two independent schedulers handling different workloads
- 1-to-1 workflow dependency: each A task completion triggers a B task
- WebSocket-based event-driven task status updates (one per scheduler)
- Four-threaded architecture:
  - Thread 1: WebSocket receiver for Scheduler A (also submits B tasks upon A completion)
  - Thread 2: WebSocket receiver for Scheduler B
  - Thread 3: Poisson submitter for Scheduler A (bimodal workload)
  - Thread 4: (not used - B tasks submitted by A receiver)
- QPS control only applies to A-type tasks
- Workflow-level metrics: A submit → B complete time tracking
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
class WorkflowTaskData:
    """Pre-generated task data with workflow tracking."""
    task_id: str
    workflow_id: str  # Unique workflow identifier
    task_type: str    # "A" or "B"
    sleep_time: float
    exp_runtime: float


@dataclass
class TaskRecord:
    """Task execution record with workflow tracking."""
    task_id: str
    workflow_id: str
    task_type: str
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
    """WebSocket-based result receiver thread with B task submission capability."""

    def __init__(self, name: str, ws_url: str, task_ids: List[str], result_queue: Queue,
                 scheduler_b_url: Optional[str] = None, b_task_generator=None):
        """
        Initialize WebSocket result receiver.

        Args:
            name: Receiver name (for logging)
            ws_url: WebSocket URL (ws://host:port/task/get_result)
            task_ids: List of task IDs to subscribe to
            result_queue: Queue to put results into
            scheduler_b_url: Scheduler B URL (for submitting B tasks when A completes)
            b_task_generator: Function to generate B task data from A task_id
        """
        self.name = name
        self.ws_url = ws_url
        self.task_ids = task_ids
        self.result_queue = result_queue
        self.scheduler_b_url = scheduler_b_url
        self.b_task_generator = b_task_generator
        self.running = False
        self.thread = None
        self.loop = None
        self.websocket = None
        self.b_tasks_submitted = []  # Track submitted B tasks

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

    def _submit_b_task(self, a_task_id: str):
        """
        Submit B task when A task completes.

        Args:
            a_task_id: The completed A task ID
        """
        if not self.scheduler_b_url or not self.b_task_generator:
            return

        try:
            # Generate B task data
            b_task = self.b_task_generator(a_task_id)

            url = f"{self.scheduler_b_url}/task/submit"
            payload = {
                "task_id": b_task.task_id,
                "model_id": MODEL_ID,
                "task_input": {
                    "sleep_time": b_task.sleep_time
                },
                "metadata": {
                    "exp_runtime": b_task.exp_runtime,
                    "workflow_id": b_task.workflow_id,
                    "parent_task_id": a_task_id
                }
            }

            submit_time = time.time()
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()

            # Parse response to get assigned instance
            result = response.json()
            assigned_instance = None
            if result.get("success") and "task" in result:
                assigned_instance = result["task"].get("assigned_instance")

            self.b_tasks_submitted.append({
                "task_id": b_task.task_id,
                "workflow_id": b_task.workflow_id,
                "sleep_time": b_task.sleep_time,
                "exp_runtime": b_task.exp_runtime,
                "submit_time": submit_time,
                "assigned_instance": assigned_instance,
                "parent_task_id": a_task_id
            })

            print(f"  [{self.name}] → Submitted B task {b_task.task_id} for workflow {b_task.workflow_id}")

        except Exception as e:
            print(f"✗ [{self.name}] Failed to submit B task for {a_task_id}: {e}")

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

                            # If this is an A task completion, submit corresponding B task
                            task_id = data.get("task_id")
                            if task_id and "-A-" in task_id and data.get("status") == "completed":
                                self._submit_b_task(task_id)

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

    def __init__(self, name: str, scheduler_url: str, tasks: List[WorkflowTaskData], qps: float):
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
                    "workflow_id": task.workflow_id,
                    "task_type": task.task_type,
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

    def _submit_task(self, task: WorkflowTaskData) -> Tuple[bool, float, Optional[str]]:
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
                "exp_runtime": task.exp_runtime,
                "workflow_id": task.workflow_id
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


def generate_workflow_task_data(num_workflows: int, strategy_name: str,
                                task_times_a: List[float], task_times_b: List[float],
                                mean_time_a: float, mean_time_b: float) -> Tuple[List[WorkflowTaskData], List[WorkflowTaskData]]:
    """
    Generate pre-generated workflow task data.

    Args:
        num_workflows: Number of workflows (A-B pairs) to generate
        strategy_name: Name of the strategy (for task ID prefix)
        task_times_a: Pre-generated A task execution times
        task_times_b: Pre-generated B task execution times
        mean_time_a: Mean A task execution time (for min_time strategy)
        mean_time_b: Mean B task execution time (for min_time strategy)

    Returns:
        Tuple of (A_tasks, B_tasks) lists
    """
    tasks_a = []
    tasks_b = []

    # Calculate expected runtime for each task
    if strategy_name == "min_time":
        exp_runtimes_a = [mean_time_a * 1000] * len(task_times_a)
        exp_runtimes_b = [mean_time_b * 1000] * len(task_times_b)
    else:
        exp_runtimes_a = [t * 1000 for t in task_times_a]
        exp_runtimes_b = [t * 1000 for t in task_times_b]

    for i in range(num_workflows):
        workflow_id = f"wf-{strategy_name}-{i:04d}"

        # A task
        task_id_a = f"task-A-{strategy_name}-workflow-{i:04d}-A"
        tasks_a.append(WorkflowTaskData(
            task_id=task_id_a,
            workflow_id=workflow_id,
            task_type="A",
            sleep_time=task_times_a[i],
            exp_runtime=exp_runtimes_a[i]
        ))

        # B task (will be submitted after A completes)
        task_id_b = f"task-B-{strategy_name}-workflow-{i:04d}-B"
        tasks_b.append(WorkflowTaskData(
            task_id=task_id_b,
            workflow_id=workflow_id,
            task_type="B",
            sleep_time=task_times_b[i],
            exp_runtime=exp_runtimes_b[i]
        ))

    return tasks_a, tasks_b


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


def collect_results(receiver_a: WebSocketResultReceiver, receiver_b: WebSocketResultReceiver,
                   submitter_a: PoissonTaskSubmitter, tasks_a: List[WorkflowTaskData],
                   tasks_b: List[WorkflowTaskData]) -> Tuple[Dict[str, TaskRecord], Dict[str, TaskRecord]]:
    """
    Collect results from both WebSocket receivers.

    Args:
        receiver_a: WebSocket receiver for Scheduler A
        receiver_b: WebSocket receiver for Scheduler B
        submitter_a: Task submitter for A tasks
        tasks_a: List of A tasks
        tasks_b: List of B tasks

    Returns:
        Tuple of (task_records_a, task_records_b)
    """
    print(f"\n{'='*60}")
    print(f"Waiting for task results...")
    print(f"{'='*60}")

    # Initialize task records
    task_records_a = {task.task_id: TaskRecord(
        task_id=task.task_id,
        workflow_id=task.workflow_id,
        task_type=task.task_type,
        sleep_time=task.sleep_time,
        exp_runtime=task.exp_runtime
    ) for task in tasks_a}

    task_records_b = {task.task_id: TaskRecord(
        task_id=task.task_id,
        workflow_id=task.workflow_id,
        task_type=task.task_type,
        sleep_time=task.sleep_time,
        exp_runtime=task.exp_runtime
    ) for task in tasks_b}

    # Update A task submit times and assigned instances
    for submitted in submitter_a.submitted_tasks:
        task_id = submitted["task_id"]
        if task_id in task_records_a:
            task_records_a[task_id].submit_time = submitted["submit_time"]
            task_records_a[task_id].assigned_instance = submitted["assigned_instance"]

    # Update B task submit times and assigned instances (from receiver_a's submitted B tasks)
    # Wait a bit for B tasks to be generated
    time.sleep(2)

    for submitted in receiver_a.b_tasks_submitted:
        task_id = submitted["task_id"]
        if task_id in task_records_b:
            task_records_b[task_id].submit_time = submitted["submit_time"]
            task_records_b[task_id].assigned_instance = submitted["assigned_instance"]

    received_a = 0
    completed_a = 0
    failed_a = 0
    target_a = len(submitter_a.submitted_tasks)

    received_b = 0
    completed_b = 0
    failed_b = 0
    target_b = len(receiver_a.b_tasks_submitted)

    last_print_a = 0
    last_print_b = 0

    # Wait for all results (with timeout)
    timeout = 300  # 5 minutes
    start_time = time.time()

    while (received_a < target_a or received_b < target_b):
        if time.time() - start_time > timeout:
            print(f"✗ Timeout waiting for results")
            print(f"  A: {received_a}/{target_a} received ({completed_a} completed, {failed_a} failed)")
            print(f"  B: {received_b}/{target_b} received ({completed_b} completed, {failed_b} failed)")
            break

        try:
            # Check A results
            if received_a < target_a:
                try:
                    result = receiver_a.result_queue.get(timeout=0.1)
                    task_id = result["task_id"]
                    if task_id in task_records_a:
                        task_records_a[task_id].complete_time = result["timestamp"]
                        task_records_a[task_id].status = result["status"]
                        task_records_a[task_id].result = result["result"]
                        task_records_a[task_id].error = result["error"]
                        task_records_a[task_id].execution_time_ms = result["execution_time_ms"]

                        received_a += 1
                        if result["status"] == "completed":
                            completed_a += 1
                        else:
                            failed_a += 1

                        if received_a - last_print_a >= 10:
                            print(f"  [A] Received {received_a}/{target_a} results ({completed_a} completed, {failed_a} failed)")
                            last_print_a = received_a
                except:
                    pass

            # Check B results
            if received_b < target_b:
                try:
                    result = receiver_b.result_queue.get(timeout=0.1)
                    task_id = result["task_id"]
                    if task_id in task_records_b:
                        task_records_b[task_id].complete_time = result["timestamp"]
                        task_records_b[task_id].status = result["status"]
                        task_records_b[task_id].result = result["result"]
                        task_records_b[task_id].error = result["error"]
                        task_records_b[task_id].execution_time_ms = result["execution_time_ms"]

                        received_b += 1
                        if result["status"] == "completed":
                            completed_b += 1
                        else:
                            failed_b += 1

                        if received_b - last_print_b >= 10:
                            print(f"  [B] Received {received_b}/{target_b} results ({completed_b} completed, {failed_b} failed)")
                            last_print_b = received_b
                except:
                    pass

        except:
            continue

    print(f"✓ [A] Received {received_a}/{target_a} results ({completed_a} completed, {failed_a} failed)")
    print(f"✓ [B] Received {received_b}/{target_b} results ({completed_b} completed, {failed_b} failed)")

    return task_records_a, task_records_b


def calculate_workflow_metrics(task_records_a: Dict[str, TaskRecord],
                               task_records_b: Dict[str, TaskRecord],
                               submitter_a: PoissonTaskSubmitter,
                               strategy_name: str, qps: float) -> Dict:
    """
    Calculate performance metrics including workflow-level metrics.

    Args:
        task_records_a: Dictionary of A task records
        task_records_b: Dictionary of B task records
        submitter_a: Task submitter instance for A tasks
        strategy_name: Strategy name
        qps: Target QPS

    Returns:
        Dictionary with performance metrics for A, B, and workflows
    """
    # Calculate A task metrics
    completion_times_a = []
    successful_tasks_a = []
    failed_tasks_a = []

    for task_id, record in task_records_a.items():
        if record.submit_time and record.complete_time:
            if record.status == "completed":
                completion_time = record.complete_time - record.submit_time
                completion_times_a.append(completion_time)
                successful_tasks_a.append(record)
            else:
                failed_tasks_a.append(record)

    # Calculate B task metrics
    completion_times_b = []
    successful_tasks_b = []
    failed_tasks_b = []

    for task_id, record in task_records_b.items():
        if record.submit_time and record.complete_time:
            if record.status == "completed":
                completion_time = record.complete_time - record.submit_time
                completion_times_b.append(completion_time)
                successful_tasks_b.append(record)
            else:
                failed_tasks_b.append(record)

    # Calculate workflow metrics (A submit → B complete)
    workflow_completion_times = []
    successful_workflows = []

    for record_a in successful_tasks_a:
        workflow_id = record_a.workflow_id
        # Find corresponding B task
        record_b = None
        for task_id_b, rec_b in task_records_b.items():
            if rec_b.workflow_id == workflow_id and rec_b.status == "completed":
                record_b = rec_b
                break

        if record_b and record_a.submit_time and record_b.complete_time:
            workflow_time = record_b.complete_time - record_a.submit_time
            workflow_completion_times.append(workflow_time)
            successful_workflows.append({
                "workflow_id": workflow_id,
                "a_task_id": record_a.task_id,
                "b_task_id": record_b.task_id,
                "workflow_time": workflow_time
            })

    # Build results
    results = {
        "strategy": strategy_name,
        "qps": qps,
        "num_workflows": len(task_records_a),

        # A task metrics
        "a_tasks": {
            "num_tasks": len(task_records_a),
            "num_submitted": len(submitter_a.submitted_tasks),
            "num_completed": len(successful_tasks_a),
            "num_failed": len(failed_tasks_a),
            "avg_completion_time": float(np.mean(completion_times_a)) if completion_times_a else 0,
            "median_completion_time": float(np.median(completion_times_a)) if completion_times_a else 0,
            "p95_completion_time": float(np.percentile(completion_times_a, 95)) if completion_times_a else 0,
            "p99_completion_time": float(np.percentile(completion_times_a, 99)) if completion_times_a else 0,
        },

        # B task metrics
        "b_tasks": {
            "num_tasks": len(task_records_b),
            "num_submitted": len([r for r in task_records_b.values() if r.submit_time]),
            "num_completed": len(successful_tasks_b),
            "num_failed": len(failed_tasks_b),
            "avg_completion_time": float(np.mean(completion_times_b)) if completion_times_b else 0,
            "median_completion_time": float(np.median(completion_times_b)) if completion_times_b else 0,
            "p95_completion_time": float(np.percentile(completion_times_b, 95)) if completion_times_b else 0,
            "p99_completion_time": float(np.percentile(completion_times_b, 99)) if completion_times_b else 0,
        },

        # Workflow metrics
        "workflows": {
            "num_completed": len(successful_workflows),
            "avg_completion_time": float(np.mean(workflow_completion_times)) if workflow_completion_times else 0,
            "median_completion_time": float(np.median(workflow_completion_times)) if workflow_completion_times else 0,
            "p50_completion_time": float(np.percentile(workflow_completion_times, 50)) if workflow_completion_times else 0,
            "p95_completion_time": float(np.percentile(workflow_completion_times, 95)) if workflow_completion_times else 0,
            "p99_completion_time": float(np.percentile(workflow_completion_times, 99)) if workflow_completion_times else 0,
        },

        # Timing
        "submission_time": submitter_a.last_submit_time - submitter_a.first_submit_time if submitter_a.first_submit_time else 0,
        "actual_qps": len(submitter_a.submitted_tasks) / (submitter_a.last_submit_time - submitter_a.first_submit_time) if submitter_a.first_submit_time and submitter_a.last_submit_time else 0,
    }

    return results


def print_workflow_metrics(results: Dict):
    """Print workflow performance metrics."""
    print(f"\n{'='*60}")
    print(f"Results for {results['strategy'].upper()}")
    print(f"{'='*60}")

    print(f"\n[A Tasks - Scheduler A]")
    print(f"  Total tasks:              {results['a_tasks']['num_tasks']}")
    print(f"  Submitted:                {results['a_tasks']['num_submitted']}")
    print(f"  Completed:                {results['a_tasks']['num_completed']}")
    print(f"  Failed:                   {results['a_tasks']['num_failed']}")
    print(f"  Avg completion time:      {results['a_tasks']['avg_completion_time']:.3f}s")
    print(f"  Median completion time:   {results['a_tasks']['median_completion_time']:.3f}s")
    print(f"  P95 completion time:      {results['a_tasks']['p95_completion_time']:.3f}s")
    print(f"  P99 completion time:      {results['a_tasks']['p99_completion_time']:.3f}s")

    print(f"\n[B Tasks - Scheduler B]")
    print(f"  Total tasks:              {results['b_tasks']['num_tasks']}")
    print(f"  Submitted:                {results['b_tasks']['num_submitted']}")
    print(f"  Completed:                {results['b_tasks']['num_completed']}")
    print(f"  Failed:                   {results['b_tasks']['num_failed']}")
    print(f"  Avg completion time:      {results['b_tasks']['avg_completion_time']:.3f}s")
    print(f"  Median completion time:   {results['b_tasks']['median_completion_time']:.3f}s")
    print(f"  P95 completion time:      {results['b_tasks']['p95_completion_time']:.3f}s")
    print(f"  P99 completion time:      {results['b_tasks']['p99_completion_time']:.3f}s")

    print(f"\n[Workflows (A submit → B complete)]")
    print(f"  Total workflows:          {results['num_workflows']}")
    print(f"  Completed workflows:      {results['workflows']['num_completed']}")
    print(f"  Avg workflow time:        {results['workflows']['avg_completion_time']:.3f}s")
    print(f"  Median workflow time:     {results['workflows']['median_completion_time']:.3f}s")
    print(f"  P50 workflow time:        {results['workflows']['p50_completion_time']:.3f}s")
    print(f"  P95 workflow time:        {results['workflows']['p95_completion_time']:.3f}s")
    print(f"  P99 workflow time:        {results['workflows']['p99_completion_time']:.3f}s")

    print(f"\n[Submission Stats]")
    print(f"  Target QPS:               {results['qps']:.2f}")
    print(f"  Actual QPS:               {results['actual_qps']:.2f}")
    print(f"  Submission time:          {results['submission_time']:.3f}s")


def test_strategy_workflow(strategy_name: str,
                           task_times_a: List[float], task_times_b: List[float],
                           qps_a: float, mean_time_a: float, mean_time_b: float,
                           num_workflows: int) -> Dict:
    """
    Test a scheduling strategy with 1-to-1 workflow dependencies.

    Args:
        strategy_name: Name of the strategy
        task_times_a: Pre-generated A task times
        task_times_b: Pre-generated B task times
        qps_a: Target QPS for A tasks
        mean_time_a: Mean time for A tasks
        mean_time_b: Mean time for B tasks
        num_workflows: Number of workflows to test

    Returns:
        Dictionary with test results
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

    time.sleep(1)

    # Step 2: Set the strategy on both schedulers
    print("\nStep 2: Setting scheduling strategy...")
    if not set_strategy(SCHEDULER_A_URL, "Scheduler A", strategy_name):
        return {"error": "Failed to set strategy on Scheduler A"}
    if not set_strategy(SCHEDULER_B_URL, "Scheduler B", strategy_name):
        return {"error": "Failed to set strategy on Scheduler B"}

    time.sleep(2)

    # Step 3: Generate workflow task data
    tasks_a, tasks_b = generate_workflow_task_data(
        num_workflows, strategy_name,
        task_times_a, task_times_b,
        mean_time_a, mean_time_b
    )

    # Create B task lookup for receiver_a
    b_tasks_dict = {task.workflow_id: task for task in tasks_b}

    def b_task_generator(a_task_id: str) -> WorkflowTaskData:
        """Generate B task from A task ID."""
        # Extract workflow_id from A task
        # Format: task-A-{strategy}-workflow-{i:04d}-A
        parts = a_task_id.split("-")
        workflow_idx = parts[4]  # Get the workflow index
        workflow_id = f"wf-{strategy_name}-{workflow_idx}"
        return b_tasks_dict[workflow_id]

    # Create result queues
    result_queue_a = Queue()
    result_queue_b = Queue()

    # Start WebSocket receivers
    task_ids_a = [task.task_id for task in tasks_a]
    task_ids_b = [task.task_id for task in tasks_b]

    ws_url_a = f"{SCHEDULER_A_WS_URL}/task/get_result"
    ws_url_b = f"{SCHEDULER_B_WS_URL}/task/get_result"

    # Receiver A also submits B tasks upon A completion
    receiver_a = WebSocketResultReceiver(
        "Scheduler A", ws_url_a, task_ids_a, result_queue_a,
        scheduler_b_url=SCHEDULER_B_URL,
        b_task_generator=b_task_generator
    )
    receiver_b = WebSocketResultReceiver("Scheduler B", ws_url_b, task_ids_b, result_queue_b)

    receiver_a.start()
    receiver_b.start()

    time.sleep(2)

    # Start Poisson task submitter for A tasks only
    submitter_a = PoissonTaskSubmitter("Scheduler A", SCHEDULER_A_URL, tasks_a, qps_a)
    submitter_a.start()

    # Wait for A tasks to be submitted
    submitter_a.wait_completion()

    # Collect results
    task_records_a, task_records_b = collect_results(
        receiver_a, receiver_b, submitter_a, tasks_a, tasks_b
    )

    # Stop receivers
    receiver_a.stop()
    receiver_b.stop()

    # Calculate metrics
    results = calculate_workflow_metrics(
        task_records_a, task_records_b, submitter_a,
        strategy_name, qps_a
    )

    # Print results
    if "error" not in results:
        print_workflow_metrics(results)
    else:
        print(f"✗ {results['error']}")

    return results


def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test 1-to-1 workflow dependencies with dual schedulers")
    parser.add_argument("--n1", type=int, default=10, help="Number of instances in Group A (default: 10)")
    parser.add_argument("--n2", type=int, default=6, help="Number of instances in Group B (default: 6)")
    parser.add_argument("--qps1", type=float, default=8.0, help="QPS for A tasks (default: 8.0)")
    parser.add_argument("--num-workflows", type=int, default=100, help="Number of workflows per strategy (default: 100)")
    parser.add_argument("--strategies", type=str, nargs="+", default=STRATEGIES,
                        help="Strategies to test (default: all)")
    args = parser.parse_args()

    # Create results directory
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("03.multi_model_1_to_1 Experiment")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Group A: {args.n1} instances (Scheduler A)")
    print(f"  Group B: {args.n2} instances (Scheduler B)")
    print(f"  A Tasks QPS: {args.qps1}")
    print(f"  Workflows per strategy: {args.num_workflows}")
    print(f"  Strategies: {', '.join(args.strategies)}")
    print(f"  Workflow: Each A task → triggers one B task")
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
    task_times_a, config_a = generate_bimodal_distribution(args.num_workflows, seed=42)
    task_times_b, config_b = generate_pareto_distribution(args.num_workflows, seed=43)

    print_distribution_stats(task_times_a, config_a)
    print_distribution_stats(task_times_b, config_b)

    # Test all strategies
    all_results = []

    for strategy in args.strategies:
        try:
            results = test_strategy_workflow(
                strategy, task_times_a, task_times_b,
                args.qps1, config_a.mean_time, config_b.mean_time,
                args.num_workflows
            )
            all_results.append(results)
        except Exception as e:
            print(f"✗ Error testing {strategy}: {e}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()

    # Save results to file
    output_file = f"results/results_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({
            "experiment": "03.multi_model_1_to_1",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "n1": args.n1,
                "n2": args.n2,
                "qps1": args.qps1,
                "num_workflows": args.num_workflows,
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
    print("Strategy Comparison - Workflow Metrics")
    print(f"{'='*60}")
    print(f"{'Strategy':<20} {'A Avg':<12} {'B Avg':<12} {'Workflow Avg':<15} {'Workflow P95':<15}")
    print("-" * 80)
    for result in all_results:
        if "error" not in result:
            strategy = result['strategy']
            a_avg = result['a_tasks']['avg_completion_time']
            b_avg = result['b_tasks']['avg_completion_time']
            wf_avg = result['workflows']['avg_completion_time']
            wf_p95 = result['workflows']['p95_completion_time']
            print(f"{strategy:<20} {a_avg:>10.3f}s {b_avg:>10.3f}s {wf_avg:>13.3f}s {wf_p95:>13.3f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
