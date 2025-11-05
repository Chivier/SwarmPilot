#!/usr/bin/env python3
"""
Common utilities and data structures for unified multi-model workflow experiments.

This module provides shared functionality used across all workflow experiment types (04-07).
"""

import time
import threading
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ============================================================================
# Rate Limiter
# ============================================================================

class RateLimiter:
    """
    Thread-safe rate limiter using token bucket algorithm.

    Used to control global QPS across multiple threads submitting tasks.
    """

    def __init__(self, rate: float):
        """
        Initialize rate limiter.

        Args:
            rate: Target rate in requests per second (e.g., 10.0 for 10 QPS)
        """
        self.rate = rate
        self.tokens = 0.0  # Start with 0 tokens to enforce strict rate limit from beginning
        self.max_tokens = rate
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens from the bucket, blocking if necessary.

        Args:
            tokens: Number of tokens to acquire (default: 1)

        Returns:
            Time spent waiting in seconds
        """
        wait_start = time.time()

        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update

                # Refill tokens based on elapsed time
                self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
                self.last_update = now

                # If enough tokens available, consume and return
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    wait_time = time.time() - wait_start
                    return wait_time

            # Not enough tokens, sleep briefly and retry
            time.sleep(0.01)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class WorkflowTaskData:
    """Pre-generated task data for workflow execution."""
    task_id: str
    workflow_id: str
    task_type: str  # "A", "B", "B1", "B2", or "merge"
    sleep_time: float
    exp_runtime: float  # Expected runtime in milliseconds
    is_warmup: bool = False  # Whether this is a warmup task


@dataclass
class TaskRecord:
    """Complete record of a task's lifecycle."""
    task_id: str
    workflow_id: str
    task_type: str
    sleep_time: float
    exp_runtime: float

    # Tracked at submission
    submit_time: Optional[float] = None
    assigned_instance: Optional[str] = None

    # Tracked at completion
    complete_time: Optional[float] = None
    status: Optional[str] = None
    execution_time_ms: Optional[float] = None

    # Results
    result: Optional[Dict] = None
    error: Optional[str] = None

    # Warmup flag
    is_warmup: bool = False


@dataclass
class WorkflowState:
    """Base state tracking for a single workflow (used by experiments 04 and 05)."""
    workflow_id: str
    strategy: str
    a_task_id: str
    b_task_ids: List[str]
    total_b_tasks: int
    completed_b_tasks: int = 0
    is_warmup: bool = False  # Whether this is a warmup workflow

    # Timestamps
    a_submit_time: Optional[float] = None
    a_complete_time: Optional[float] = None
    b_complete_times: Dict[str, float] = field(default_factory=dict)
    workflow_complete_time: Optional[float] = None

    # For experiment 05 (sequential B task submission)
    next_b_task_index: int = 0

    def is_complete(self) -> bool:
        """Check if workflow is complete (all B tasks done)."""
        return self.completed_b_tasks >= self.total_b_tasks

    def mark_b_task_complete(self, b_task_id: str, complete_time: float):
        """Mark a B task as complete and update workflow state."""
        if b_task_id not in self.b_complete_times:
            self.b_complete_times[b_task_id] = complete_time
            self.completed_b_tasks += 1

            # Update workflow complete time if this is the last B task
            if self.is_complete():
                self.workflow_complete_time = max(self.b_complete_times.values())


@dataclass
class WorkflowStateMerge(WorkflowState):
    """Extended workflow state for experiments with merge task (06 and 07)."""
    merge_task_id: str = ""
    all_b_complete_time: Optional[float] = None  # When all B tasks finished
    merge_submit_time: Optional[float] = None     # When merge task submitted
    merge_complete_time: Optional[float] = None   # When merge task completed (workflow end)

    def mark_merge_task_complete(self, complete_time: float):
        """Mark merge task as complete."""
        self.merge_complete_time = complete_time
        self.workflow_complete_time = complete_time  # Workflow ends when merge completes


@dataclass
class WorkflowStateB1B2(WorkflowStateMerge):
    """Extended workflow state for experiment 07 with B1/B2 split."""
    b1_task_ids: List[str] = field(default_factory=list)
    b2_task_ids: List[str] = field(default_factory=list)
    total_b1_tasks: int = 0
    total_b2_tasks: int = 0
    completed_b1_tasks: int = 0
    completed_b2_tasks: int = 0

    # Timestamps for B1 and B2 separately
    b1_complete_times: Dict[str, float] = field(default_factory=dict)
    b2_complete_times: Dict[str, float] = field(default_factory=dict)
    all_b1_complete_time: Optional[float] = None
    all_b2_complete_time: Optional[float] = None

    def is_all_b1_complete(self) -> bool:
        """Check if all B1 tasks are complete."""
        return self.completed_b1_tasks >= self.total_b1_tasks

    def is_all_b2_complete(self) -> bool:
        """Check if all B2 tasks are complete."""
        return self.completed_b2_tasks >= self.total_b2_tasks

    def mark_b1_task_complete(self, b1_task_id: str, complete_time: float):
        """Mark a B1 task as complete."""
        if b1_task_id not in self.b1_complete_times:
            self.b1_complete_times[b1_task_id] = complete_time
            self.completed_b1_tasks += 1

            if self.is_all_b1_complete():
                self.all_b1_complete_time = max(self.b1_complete_times.values())

    def mark_b2_task_complete(self, b2_task_id: str, complete_time: float):
        """Mark a B2 task as complete."""
        if b2_task_id not in self.b2_complete_times:
            self.b2_complete_times[b2_task_id] = complete_time
            self.completed_b2_tasks += 1

            if self.is_all_b2_complete():
                self.all_b2_complete_time = max(self.b2_complete_times.values())
                self.all_b_complete_time = self.all_b2_complete_time  # For compatibility


@dataclass
class WorkflowCompletionEvent:
    """Event indicating a workflow has completed."""
    workflow_id: str
    workflow_time: float  # A submit → last task complete time (seconds)
    a_task_id: str
    b_task_ids: List[str]
    total_b_tasks: int
    a_submit_time: float
    a_complete_time: float
    workflow_complete_time: float
    is_warmup: bool = False  # Whether this is a warmup workflow


# ============================================================================
# Utility Functions
# ============================================================================

def clear_scheduler_tasks(scheduler_url: str):
    """Clear all tasks from a scheduler."""
    import requests
    logger = logging.getLogger("Utils")
    try:
        response = requests.post(f"{scheduler_url}/task/clear")
        response.raise_for_status()
        logger.info(f"Cleared tasks from {scheduler_url}")
    except Exception as e:
        logger.error(f"Failed to clear tasks from {scheduler_url}: {e}")


def set_scheduling_strategy(scheduler_url: str, strategy: str):
    """Set the scheduling strategy for a scheduler."""
    import requests
    logger = logging.getLogger("Utils")
    try:
        # Prepare request payload
        payload = {"strategy_name": strategy}

        # For probabilistic strategy, use custom quantiles
        if strategy == "probabilistic":
            payload["quantiles"] = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.99]
            logger.info(f"Setting probabilistic strategy with quantiles: {payload['quantiles']}")

        response = requests.post(
            f"{scheduler_url}/strategy/set",
            json=payload
        )
        response.raise_for_status()
        logger.info(f"Set strategy to '{strategy}' on {scheduler_url}")
    except Exception as e:
        logger.error(f"Failed to set strategy on {scheduler_url}: {e}")


def generate_task_ids_simple(num_workflows: int, fanout_values: List[int], strategy: str) -> tuple[
    List[str], List[str], Dict[str, List[str]]]:
    """
    Pre-generate all task IDs for WebSocket subscription (experiments 04, 05).

    Args:
        num_workflows: Number of workflows
        fanout_values: List of fanout values (number of B tasks per workflow)
        strategy: Scheduling strategy name

    Returns:
        Tuple of (a_task_ids, all_b_task_ids, b_task_ids_by_workflow)
    """
    a_task_ids = []
    all_b_task_ids = []
    b_task_ids_by_workflow = {}

    for i in range(num_workflows):
        workflow_id = f"wf-{strategy}-{i:04d}"

        # A task ID
        a_task_id = f"task-A-{strategy}-workflow-{i:04d}-A"
        a_task_ids.append(a_task_id)

        # B task IDs for this workflow
        n = fanout_values[i]
        b_task_ids = []
        for j in range(n):
            b_task_id = f"task-B-{strategy}-workflow-{i:04d}-B-{j:02d}"
            b_task_ids.append(b_task_id)
            all_b_task_ids.append(b_task_id)

        b_task_ids_by_workflow[workflow_id] = b_task_ids

    return a_task_ids, all_b_task_ids, b_task_ids_by_workflow


def generate_task_ids_merge(num_workflows: int, fanout_values: List[int], strategy: str) -> tuple[
    List[str], List[str], List[str], Dict[str, List[str]]]:
    """
    Pre-generate all task IDs including merge tasks (experiment 06).

    Args:
        num_workflows: Number of workflows
        fanout_values: List of fanout values (number of B tasks per workflow)
        strategy: Scheduling strategy name

    Returns:
        Tuple of (a_task_ids, all_b_task_ids, merge_task_ids, b_task_ids_by_workflow)
    """
    a_task_ids = []
    all_b_task_ids = []
    merge_task_ids = []
    b_task_ids_by_workflow = {}

    for i in range(num_workflows):
        workflow_id = f"wf-{strategy}-{i:04d}"

        # A task ID
        a_task_id = f"task-A-{strategy}-workflow-{i:04d}-A"
        a_task_ids.append(a_task_id)

        # Merge task ID
        merge_task_id = f"task-A-{strategy}-workflow-{i:04d}-merge"
        merge_task_ids.append(merge_task_id)

        # B task IDs for this workflow
        n = fanout_values[i]
        b_task_ids = []
        for j in range(n):
            b_task_id = f"task-B-{strategy}-workflow-{i:04d}-B-{j:02d}"
            b_task_ids.append(b_task_id)
            all_b_task_ids.append(b_task_id)

        b_task_ids_by_workflow[workflow_id] = b_task_ids

    return a_task_ids, all_b_task_ids, merge_task_ids, b_task_ids_by_workflow


def generate_task_ids_b1b2(num_workflows: int, fanout_values: List[int], strategy: str) -> tuple[
    List[str], List[str], List[str], List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Pre-generate all task IDs with B1/B2 split (experiment 07).

    Args:
        num_workflows: Number of workflows
        fanout_values: List of fanout values (number of B tasks per workflow)
        strategy: Scheduling strategy name

    Returns:
        Tuple of (a_task_ids, all_b1_task_ids, all_b2_task_ids, merge_task_ids,
                  b1_task_ids_by_workflow, b2_task_ids_by_workflow)
    """
    a_task_ids = []
    all_b1_task_ids = []
    all_b2_task_ids = []
    merge_task_ids = []
    b1_task_ids_by_workflow = {}
    b2_task_ids_by_workflow = {}

    for i in range(num_workflows):
        workflow_id = f"wf-{strategy}-{i:04d}"

        # A task ID
        a_task_id = f"task-A-{strategy}-workflow-{i:04d}-A"
        a_task_ids.append(a_task_id)

        # Merge task ID
        merge_task_id = f"task-A-{strategy}-workflow-{i:04d}-merge"
        merge_task_ids.append(merge_task_id)

        # B1 and B2 task IDs for this workflow
        n = fanout_values[i]
        b1_task_ids = []
        b2_task_ids = []
        for j in range(n):
            b1_task_id = f"task-B1-{strategy}-workflow-{i:04d}-B1-{j:02d}"
            b2_task_id = f"task-B2-{strategy}-workflow-{i:04d}-B2-{j:02d}"
            b1_task_ids.append(b1_task_id)
            b2_task_ids.append(b2_task_id)
            all_b1_task_ids.append(b1_task_id)
            all_b2_task_ids.append(b2_task_id)

        b1_task_ids_by_workflow[workflow_id] = b1_task_ids
        b2_task_ids_by_workflow[workflow_id] = b2_task_ids

    return a_task_ids, all_b1_task_ids, all_b2_task_ids, merge_task_ids, b1_task_ids_by_workflow, b2_task_ids_by_workflow


# ============================================================================
# Metrics Calculation
# ============================================================================

def calculate_task_metrics(records: List[TaskRecord], task_type: str) -> Dict:
    """
    Calculate metrics for a list of task records.

    NOTE: Warmup tasks (is_warmup=True) are excluded from statistics.

    Args:
        records: List of task records
        task_type: "A", "B", "B1", "B2", or "merge"

    Returns:
        Dictionary of metrics
    """
    if not records:
        return {
            "task_type": task_type,
            "num_generated": 0,
            "num_submitted": 0,
            "num_completed": 0,
            "num_failed": 0,
            "num_warmup": 0,
            "completion_times": [],
            "avg_completion_time": 0.0,
            "median_completion_time": 0.0,
            "p95_completion_time": 0.0,
            "p99_completion_time": 0.0
        }

    # Filter records by task type and exclude warmup tasks
    filtered = [r for r in records if r.task_type == task_type and not r.is_warmup]
    num_warmup = sum(1 for r in records if r.task_type == task_type and r.is_warmup)

    # Merge records by task_id to combine submit and complete info
    merged_records: Dict[str, TaskRecord] = {}
    for r in filtered:
        task_id = r.task_id
        if task_id not in merged_records:
            merged_records[task_id] = r
        else:
            # Merge information from both records
            existing = merged_records[task_id]
            if r.submit_time is not None:
                existing.submit_time = r.submit_time
            if r.complete_time is not None:
                existing.complete_time = r.complete_time
            if r.status is not None:
                existing.status = r.status
            if r.execution_time_ms is not None:
                existing.execution_time_ms = r.execution_time_ms
            if r.result is not None:
                existing.result = r.result
            if r.error is not None:
                existing.error = r.error
            if r.assigned_instance is not None:
                existing.assigned_instance = r.assigned_instance

    # Work with merged records
    merged_list = list(merged_records.values())

    # Count statuses
    num_generated = len(merged_list)
    num_submitted = sum(1 for r in merged_list if r.submit_time is not None)
    completed = [r for r in merged_list if r.status == "completed"]
    num_completed = len(completed)
    num_failed = sum(1 for r in merged_list if r.status and r.status != "completed")

    # Calculate completion times (submit → complete)
    completion_times = []
    for r in completed:
        if r.submit_time is not None and r.complete_time is not None:
            completion_times.append(r.complete_time - r.submit_time)

    # Calculate statistics
    if completion_times:
        completion_times_arr = np.array(completion_times)
        avg_completion = float(np.mean(completion_times_arr))
        median_completion = float(np.median(completion_times_arr))
        p95_completion = float(np.percentile(completion_times_arr, 95))
        p99_completion = float(np.percentile(completion_times_arr, 99))
    else:
        avg_completion = 0.0
        median_completion = 0.0
        p95_completion = 0.0
        p99_completion = 0.0

    return {
        "task_type": task_type,
        "num_generated": num_generated,
        "num_submitted": num_submitted,
        "num_completed": num_completed,
        "num_failed": num_failed,
        "num_warmup": num_warmup,
        "completion_times": completion_times,
        "avg_completion_time": avg_completion,
        "median_completion_time": median_completion,
        "p95_completion_time": p95_completion,
        "p99_completion_time": p99_completion
    }


def calculate_workflow_metrics(completed_workflows: List[WorkflowCompletionEvent]) -> Dict:
    """
    Calculate workflow-level metrics.

    NOTE: Warmup workflows (is_warmup=True) are excluded from statistics.

    Args:
        completed_workflows: List of workflow completion events

    Returns:
        Dictionary of workflow metrics
    """
    if not completed_workflows:
        return {
            "num_completed": 0,
            "num_warmup": 0,
            "workflow_times": [],
            "avg_workflow_time": 0.0,
            "median_workflow_time": 0.0,
            "p50_workflow_time": 0.0,
            "p95_workflow_time": 0.0,
            "p99_workflow_time": 0.0,
            "fanout_distribution": {},
            "avg_fanout": 0.0
        }

    # Filter out warmup workflows
    actual_workflows = [e for e in completed_workflows if not e.is_warmup]
    num_warmup = sum(1 for e in completed_workflows if e.is_warmup)

    if not actual_workflows:
        return {
            "num_completed": 0,
            "num_warmup": num_warmup,
            "workflow_times": [],
            "avg_workflow_time": 0.0,
            "median_workflow_time": 0.0,
            "p50_workflow_time": 0.0,
            "p95_workflow_time": 0.0,
            "p99_workflow_time": 0.0,
            "fanout_distribution": {},
            "avg_fanout": 0.0
        }

    # Extract workflow times (excluding warmup)
    workflow_times = [event.workflow_time for event in actual_workflows]
    workflow_times_arr = np.array(workflow_times)

    # Extract fanout values (excluding warmup)
    fanout_values = [event.total_b_tasks for event in actual_workflows]
    fanout_arr = np.array(fanout_values)

    # Calculate fanout distribution
    unique_fanouts, counts = np.unique(fanout_arr, return_counts=True)
    fanout_distribution = {int(f): int(c) for f, c in zip(unique_fanouts, counts)}

    return {
        "num_completed": len(actual_workflows),
        "num_warmup": num_warmup,
        "workflow_times": workflow_times,
        "avg_workflow_time": float(np.mean(workflow_times_arr)),
        "median_workflow_time": float(np.median(workflow_times_arr)),
        "p50_workflow_time": float(np.percentile(workflow_times_arr, 50)),
        "p95_workflow_time": float(np.percentile(workflow_times_arr, 95)),
        "p99_workflow_time": float(np.percentile(workflow_times_arr, 99)),
        "fanout_distribution": fanout_distribution,
        "avg_fanout": float(np.mean(fanout_arr))
    }


def print_metrics_summary(strategy: str, a_metrics: Dict, b_metrics: Dict, wf_metrics: Dict):
    """
    Print a summary of metrics for a test run.

    Args:
        strategy: Strategy name
        a_metrics: A task metrics
        b_metrics: B task metrics
        wf_metrics: Workflow metrics
    """
    print("\n" + "=" * 80)
    print(f"Results Summary: {strategy}")
    print("=" * 80)

    print("\nA Tasks:")
    print(f"  Generated:  {a_metrics['num_generated']} (excl. {a_metrics['num_warmup']} warmup)")
    print(f"  Submitted:  {a_metrics['num_submitted']}")
    print(f"  Completed:  {a_metrics['num_completed']}")
    print(f"  Failed:     {a_metrics['num_failed']}")
    if a_metrics['avg_completion_time'] > 0:
        print(f"  Avg time:   {a_metrics['avg_completion_time']:.2f}s")
        print(f"  Median:     {a_metrics['median_completion_time']:.2f}s")
        print(f"  P95:        {a_metrics['p95_completion_time']:.2f}s")

    print("\nB Tasks:")
    print(f"  Generated:  {b_metrics['num_generated']} (excl. {b_metrics['num_warmup']} warmup)")
    print(f"  Submitted:  {b_metrics['num_submitted']}")
    print(f"  Completed:  {b_metrics['num_completed']}")
    print(f"  Failed:     {b_metrics['num_failed']}")
    if b_metrics['avg_completion_time'] > 0:
        print(f"  Avg time:   {b_metrics['avg_completion_time']:.2f}s")
        print(f"  Median:     {b_metrics['median_completion_time']:.2f}s")
        print(f"  P95:        {b_metrics['p95_completion_time']:.2f}s")

    print("\nWorkflows:")
    print(f"  Completed:  {wf_metrics['num_completed']} (excl. {wf_metrics['num_warmup']} warmup)")
    print(f"  Avg fanout: {wf_metrics['avg_fanout']:.1f} B tasks per A task")
    if wf_metrics['avg_workflow_time'] > 0:
        print(f"  Avg time:   {wf_metrics['avg_workflow_time']:.2f}s")
        print(f"  Median:     {wf_metrics['median_workflow_time']:.2f}s")
        print(f"  P95:        {wf_metrics['p95_workflow_time']:.2f}s")
        print(f"  P99:        {wf_metrics['p99_workflow_time']:.2f}s")

    print("\nFanout Distribution:")
    for fanout, count in sorted(wf_metrics['fanout_distribution'].items()):
        percentage = (count / wf_metrics['num_completed']) * 100 if wf_metrics['num_completed'] > 0 else 0
        print(f"  {fanout} B tasks: {count} workflows ({percentage:.1f}%)")

    print("=" * 80)
