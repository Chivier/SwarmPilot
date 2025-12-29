#!/usr/bin/env python3
"""
Experiment 07: Multi-Model Workflow with B1/B2 Split and Merge (1-to-n-to-n-to-1)

This experiment tests workflow dependencies where each A task generates
a variable number (n) of B1 tasks (slow peak, 7-10s), each B1 task triggers
a corresponding B2 task (fast peak, 1-3s), and upon completion of all B2 tasks,
a merge A task is triggered. A workflow is complete when the merge task has finished.

Architecture:
- Thread 1: Submit initial A tasks (Poisson process, QPS-controlled)
- Thread 2: Receive A task results, submit n B1 tasks per A task
- Thread 3: Receive B1 task results, submit corresponding B2 tasks
- Thread 4: Receive B2 task results, push to merge_ready_queue when all B2 tasks done
- Thread 5: Submit merge A tasks (triggered by all B2 tasks completing)
- Thread 6: Receive merge task results, push final completion events
- Thread 7: Monitor workflow completion (from merge completion), calculate statistics

Key differences from experiment 06 (1-to-n-to-1):
- B task split: Each B task is split into B1 (slow peak) and B2 (fast peak)
- Task pairing: Each B1 task is paired with exactly one B2 task (same b_index)
- Sequential B1→B2: B2 task submitted only after corresponding B1 task completes
- Separate tracking: B1 and B2 tasks tracked separately in workflow state
- Merge trigger: Merge task triggered after all B2 tasks complete (not B1)
"""

import uvloop
import asyncio
import websockets
import requests
import json
import os
import time
import numpy as np
import threading
from queue import Queue, Empty
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Set, Any
import random
from datetime import datetime
import logging
import traceback

# Import workload generators
from workload_generator import (
    generate_bimodal_distribution,
    generate_fast_peak_distribution,
    generate_slow_peak_distribution,
    generate_fanout_distribution,
    generate_workflow_from_traces,
    validate_fanout_pattern,
    WorkloadConfig,
    FanoutConfig,
    WorkflowWorkload,
    print_distribution_stats,
    print_fanout_stats,
    print_workflow_stats
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Scheduler endpoints
SCHEDULER_A_URL = "http://localhost:8100"
SCHEDULER_B_URL = "http://localhost:8200"
SCHEDULER_A_WS = "ws://localhost:8100/task/get_result"
SCHEDULER_B_WS = "ws://localhost:8200/task/get_result"

# Planner endpoint
PLANNER_URL = "http://localhost:8202"

# MAPE (Mean Absolute Percentage Error) for prediction error simulation
MAPE_PERCENTAGE = 50.0  # 50% MAPE for min_time strategy


# ============================================================================
# Helper Functions
# ============================================================================

def apply_mape_error(exp_runtime: float, mape_percentage: float = MAPE_PERCENTAGE) -> float:
    """
    Apply MAPE (Mean Absolute Percentage Error) to exp_runtime.

    For example, with 50% MAPE:
    - Error range: [-50%, +50%]
    - Predicted value = actual value * (1 + uniform(-0.5, 0.5))

    This simulates prediction error in runtime estimation.

    Args:
        exp_runtime: Original expected runtime in milliseconds
        mape_percentage: MAPE percentage (default: 50%)

    Returns:
        Modified exp_runtime with applied error
    """
    # Convert MAPE percentage to multiplier
    # e.g., 50% -> 0.5
    mape_multiplier = mape_percentage / 100.0

    # Generate random error in range [-mape_multiplier, +mape_multiplier]
    # For 50% MAPE: [-0.5, +0.5]
    error = np.random.uniform(-mape_multiplier, mape_multiplier)

    # Apply error: new_value = original * (1 + error)
    modified_runtime = exp_runtime * (1.0 + error)

    # Ensure non-negative result (though with 50% MAPE, this should never be needed)
    return max(0.0, modified_runtime)

random.seed(42)
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
    task_type: str  # "A", "B1", "B2", or "merge"
    sleep_time: float
    exp_runtime: float  # Expected runtime in milliseconds
    b_index: Optional[int] = None  # Index for B1/B2 pairing (0-based)
    is_warmup: bool = False  # Whether this is a warmup task


@dataclass
class TaskRecord:
    """Complete record of a task's lifecycle."""
    task_id: str
    workflow_id: str
    task_type: str  # "A", "B1", "B2", or "merge"
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

    # B1/B2 pairing
    b_index: Optional[int] = None  # Index for B1/B2 pairing (0-based)

    # Warmup flag
    is_warmup: bool = False


@dataclass
class WorkflowState:
    """State tracking for a single workflow.

    Note: Uses atomic dict length operations instead of separate counters
    to avoid race conditions in multi-threaded task completion tracking.
    """
    workflow_id: str
    strategy: str
    a_task_id: str
    b1_task_ids: List[str]  # B1 task IDs
    b2_task_ids: List[str]  # B2 task IDs
    merge_task_id: str  # merge A task ID
    total_b1_tasks: int
    total_b2_tasks: int
    is_warmup: bool = False  # Whether this is a warmup workflow
    is_target_for_stats: bool = True  # Whether to include in statistics (continuous mode)

    # Timestamps
    a_submit_time: Optional[float] = None
    a_complete_time: Optional[float] = None
    b1_complete_times: Dict[str, float] = field(default_factory=dict)
    b2_complete_times: Dict[str, float] = field(default_factory=dict)
    all_b1_complete_time: Optional[float] = None  # When all B1 tasks are done
    all_b2_complete_time: Optional[float] = None  # When all B2 tasks are done
    merge_submit_time: Optional[float] = None  # when merge task is submitted
    merge_complete_time: Optional[float] = None  # when merge task completes
    workflow_complete_time: Optional[float] = None  # same as merge_complete_time

    def are_all_b1_tasks_complete(self) -> bool:
        """Check if all B1 tasks are complete.

        Uses atomic dict length operation instead of counter to avoid race conditions.
        """
        return len(self.b1_complete_times) >= self.total_b1_tasks

    def are_all_b2_tasks_complete(self) -> bool:
        """Check if all B2 tasks are complete.

        Uses atomic dict length operation instead of counter to avoid race conditions.
        """
        return len(self.b2_complete_times) >= self.total_b2_tasks

    def is_workflow_complete(self) -> bool:
        """Check if workflow is complete (merge task done)."""
        return self.merge_complete_time is not None

    def mark_b1_task_complete(self, b1_task_id: str, complete_time: float):
        """Mark a B1 task as complete and update workflow state.

        Thread-safe: Uses dict insertion (more atomic than counter increment).
        Each b1_task_id is unique, so no two threads will insert the same key.
        """
        if b1_task_id not in self.b1_complete_times:
            self.b1_complete_times[b1_task_id] = complete_time

            # Update all_b1_complete_time if this is the last B1 task
            # Uses atomic len() operation instead of counter
            if self.are_all_b1_tasks_complete():
                self.all_b1_complete_time = max(self.b1_complete_times.values())

    def mark_b2_task_complete(self, b2_task_id: str, complete_time: float):
        """Mark a B2 task as complete and update workflow state.

        Thread-safe: Uses dict insertion (more atomic than counter increment).
        Each b2_task_id is unique, so no two threads will insert the same key.
        """
        if b2_task_id not in self.b2_complete_times:
            self.b2_complete_times[b2_task_id] = complete_time

            # Update all_b2_complete_time if this is the last B2 task
            # Uses atomic len() operation instead of counter
            if self.are_all_b2_tasks_complete():
                self.all_b2_complete_time = max(self.b2_complete_times.values())

    def mark_merge_task_complete(self, complete_time: float):
        """Mark the merge task as complete and finalize workflow."""
        self.merge_complete_time = complete_time
        self.workflow_complete_time = complete_time


@dataclass
class WorkflowCompletionEvent:
    """Event indicating a workflow has completed."""
    workflow_id: str
    workflow_time: float  # A submit → merge complete time (seconds)
    a_task_id: str
    b1_task_ids: List[str]  # B1 task IDs
    b2_task_ids: List[str]  # B2 task IDs
    merge_task_id: str  # merge A task ID
    total_b1_tasks: int
    total_b2_tasks: int
    a_submit_time: float
    a_complete_time: float
    all_b1_complete_time: float  # when all B1 tasks finished
    all_b2_complete_time: float  # when all B2 tasks finished
    merge_submit_time: float  # when merge task was submitted
    merge_complete_time: float  # when merge task completed
    workflow_complete_time: float  # Same as merge_complete_time
    is_warmup: bool = False  # Whether this is a warmup workflow
    is_target_for_stats: bool = True  # Whether to include in statistics (continuous mode)


# ============================================================================
# MigrationPlan and OracleDeployer for Predictive Redeployment
# ============================================================================

@dataclass
class MigrationPlan:
    """Pre-computed migration plan for predictive redeployment."""
    trigger_time: float      # Time to trigger migration (relative to experiment start)
    target_a_instances: int  # Target number of A model instances
    target_b_instances: int  # Target number of B model instances
    reason: str              # Reason for migration (for debugging)


class OracleDeployer:
    """
    Thread 8: Oracle-based Predictive Redeployment Controller.

    Uses pre-generated workflow data to predict future load and trigger
    migrations BEFORE load changes occur, rather than reacting to queue backlogs.

    Key insight: Since all workflow data is pre-generated, we have perfect
    foresight of future load patterns and can proactively migrate instances.

    IMPORTANT:
    1. Pass the SAME inter_arrival_times that will be used by PoissonTaskSubmitter
       to ensure migration timeline matches actual task submission.
    2. Call signal_task_submission_started() when actual task submission begins
       to synchronize the migration timeline with task submission.
    """

    def __init__(self,
                 planner_url: str,
                 scheduler_a_url: str,
                 scheduler_b_url: str,
                 total_instances: int,
                 fanout_values: List[int],  # Combined fanout values (warmup + actual)
                 inter_arrival_times: np.ndarray,  # Pre-computed inter-arrival times (MUST match PoissonTaskSubmitter)
                 window_size_s: float = 5.0,
                 lookahead_s: float = 3.0,
                 min_change_threshold: int = 2,
                 two_phase_mode: bool = False,
                 two_phase_split_index: int = 0,
                 num_warmup_workflows: int = 0):
        """
        Initialize Oracle Deployer.

        Args:
            planner_url: Planner API endpoint
            scheduler_a_url: Scheduler A endpoint (for querying instances)
            scheduler_b_url: Scheduler B endpoint (for querying instances)
            total_instances: Total number of instances in the system
            fanout_values: Combined fanout values list (warmup + actual, must match inter_arrival_times length)
            inter_arrival_times: Pre-computed inter-arrival times (MUST be the same array used by PoissonTaskSubmitter)
            window_size_s: Size of load estimation window in seconds
            lookahead_s: How far ahead to trigger migrations (seconds)
            min_change_threshold: Minimum instance change to trigger migration
            two_phase_mode: If True, use simplified two-phase migration (single migration point)
            two_phase_split_index: Index where large-fanout phase starts (within main workflows, not including warmup)
            num_warmup_workflows: Number of warmup workflows (to offset two_phase_split_index)
        """
        self.planner_url = planner_url
        self.scheduler_a_url = scheduler_a_url
        self.scheduler_b_url = scheduler_b_url
        self.total_instances = total_instances
        self.window_size = window_size_s
        self.lookahead = lookahead_s
        self.min_change = min_change_threshold
        self.two_phase_mode = two_phase_mode
        self.two_phase_split_index = two_phase_split_index
        self.num_warmup_workflows = num_warmup_workflows

        # Initialize logger first (needed by _precompute_migration_plan)
        self.logger = logging.getLogger("Thread8.OracleDeployer")

        # Synchronization: wait for task submission to start before triggering migrations
        self._task_submission_started = threading.Event()

        # Task execution time estimates (seconds) - from workload_generator.py
        self.a1_duration = 30.0     # A1 (boot): N(30, 2²) = 30 seconds
        self.a2_duration = 25.0     # A2 (summary): N(25, 2²) = 25 seconds
        self.b1_duration = 3.0      # B1 (query): N(3, 0.3²) = 3 seconds
        self.b2_duration = 2.0      # B2 (criteria): N(2, 0.2²) = 2 seconds

        # Pre-compute migration plan using the SAME inter-arrival times and fanout values
        if two_phase_mode:
            self.migration_plan = self._precompute_two_phase_migration(fanout_values, inter_arrival_times)
        else:
            self.migration_plan = self._precompute_migration_plan(fanout_values, inter_arrival_times)
        self.plan_index = 0

        # Current allocation tracking
        self.current_a_instances = total_instances // 2
        self.current_b_instances = total_instances - self.current_a_instances

        # Thread control
        self.running = False
        self.thread = None
        self.experiment_start_time = None

    def _precompute_migration_plan(self, fanout_values: List[int], inter_arrival_times: np.ndarray) -> List[MigrationPlan]:
        """
        Pre-compute the entire migration timeline from workflow data.

        Uses workload patterns to determine optimal instance allocation
        for each time window, then creates migration plans that trigger
        BEFORE the load changes (lookahead).

        Args:
            fanout_values: Combined fanout values list (warmup + actual)
            inter_arrival_times: Pre-computed inter-arrival times (same as PoissonTaskSubmitter uses)
        """
        plans = []

        # Validate lengths match
        if len(fanout_values) != len(inter_arrival_times):
            self.logger.warning(f"Length mismatch: fanout_values={len(fanout_values)}, inter_arrival_times={len(inter_arrival_times)}")

        # Use the SAME inter-arrival times that PoissonTaskSubmitter will use
        # This ensures migration timeline matches actual task submission
        a_submit_times = np.cumsum(inter_arrival_times)

        # Compute load for each time window
        max_time = a_submit_times[-1] + 60  # Add buffer
        current_allocation = (self.total_instances // 2, self.total_instances // 2)

        for window_start in np.arange(0, max_time, self.window_size):
            window_end = window_start + self.window_size

            # Find workflows that start in this window
            workflows_in_window = []
            for i, submit_time in enumerate(a_submit_times):
                if window_start <= submit_time < window_end:
                    workflows_in_window.append(fanout_values[i])

            if not workflows_in_window:
                continue

            # Compute workload for A and B models
            num_a_tasks = len(workflows_in_window)
            total_fanout = sum(workflows_in_window)

            # A workload: A1 (initial) + A2 (merge) per workflow
            a_work = num_a_tasks * (self.a1_duration + self.a2_duration)
            # B workload: B1 + B2 per fanout item
            b_work = total_fanout * (self.b1_duration + self.b2_duration)

            total_work = a_work + b_work
            if total_work > 0:
                # Compute optimal allocation based on work ratio
                optimal_a = int(round(self.total_instances * a_work / total_work))
                optimal_a = max(1, min(self.total_instances - 1, optimal_a))
                optimal_b = self.total_instances - optimal_a
            else:
                optimal_a, optimal_b = current_allocation

            # Create migration if change is significant
            if abs(optimal_a - current_allocation[0]) >= self.min_change:
                trigger_time = max(0, window_start - self.lookahead)
                plans.append(MigrationPlan(
                    trigger_time=trigger_time,
                    target_a_instances=optimal_a,
                    target_b_instances=optimal_b,
                    reason=f"window={window_start:.1f}s, a_work={a_work:.4f}, b_work={b_work:.4f}"
                ))
                current_allocation = (optimal_a, optimal_b)

        self.logger.info(f"Pre-computed {len(plans)} migration plans")
        return plans

    def _precompute_two_phase_migration(self, fanout_values: List[int], inter_arrival_times: np.ndarray) -> List[MigrationPlan]:
        """
        Pre-compute a SINGLE migration for two-phase mode.

        Two-phase mode:
        - Phase 1: warmup + first 20% of main workflows use small fanout (mean=6)
        - Phase 2: remaining 80% of main workflows use large fanout (mean=14)
        - Migration is triggered after 5 large-fanout tasks have been submitted

        Returns a list with exactly ONE migration plan.
        """
        plans = []

        # Bias factor: shift resources towards model B
        # A positive value (e.g., 0.15) means 15% more instances allocated to B
        B_BIAS_FACTOR = 0.15

        # Compute submit times from inter-arrival times
        a_submit_times = np.cumsum(inter_arrival_times)

        # Phase 1: warmup + small fanout tasks (indices 0 to warmup + small_fanout_count - 1)
        # Phase 2: large fanout tasks (indices warmup + small_fanout_count onwards)

        # Global index where large-fanout phase starts = warmup + two_phase_split_index
        large_phase_start_idx = self.num_warmup_workflows + self.two_phase_split_index

        self.logger.info(f"Two-phase mode: warmup={self.num_warmup_workflows}, "
                        f"small_fanout_count={self.two_phase_split_index}, "
                        f"large_phase_start_idx={large_phase_start_idx}, "
                        f"B_bias={B_BIAS_FACTOR:.0%}")

        # ================================================================
        # HARDCODED INSTANCE ALLOCATIONS
        # Pre-calculated based on actual workload_generator.py data
        # These values match the hardcoded QPS values in main()
        #
        # Task durations (from workload_generator.py):
        #   A1 (boot): N(30, 2²) = 30 seconds
        #   A2 (summary): N(25, 2²) = 25 seconds
        #   B1 (query): N(3, 0.3²) = 3 seconds
        #   B2 (criteria): N(2, 0.2²) = 2 seconds
        #
        # Total A work = 55 seconds per workflow (A is bottleneck)
        # Total B work = 5 seconds per B task
        #
        # With 48 instances, 15% B bias, 115% overload:
        #   Phase 1 (fanout=6):  A=24, B=24 -> QPS=0.50 wf/s
        #   Phase 2 (fanout=14): A=14, B=34 -> QPS=0.29 wf/s
        # ================================================================
        HARDCODED_PHASE1_A = 24
        HARDCODED_PHASE1_B = 24
        HARDCODED_PHASE2_A = 14
        HARDCODED_PHASE2_B = 34

        # Scale to actual instance count if different from 48
        if self.total_instances != 48:
            scale = self.total_instances / 48.0
            optimal_a_phase1 = max(1, min(self.total_instances - 1, int(round(HARDCODED_PHASE1_A * scale))))
            optimal_a_phase2 = max(1, min(self.total_instances - 1, int(round(HARDCODED_PHASE2_A * scale))))
        else:
            optimal_a_phase1 = HARDCODED_PHASE1_A
            optimal_a_phase2 = HARDCODED_PHASE2_A

        optimal_b_phase1 = self.total_instances - optimal_a_phase1
        optimal_b_phase2 = self.total_instances - optimal_a_phase2

        phase1_fanouts = fanout_values[:large_phase_start_idx]
        phase2_fanouts = fanout_values[large_phase_start_idx:]

        self.logger.info(f"Phase 1 (small fanout={np.mean(phase1_fanouts) if phase1_fanouts else 6:.1f}): "
                        f"HARDCODED A={optimal_a_phase1}, B={optimal_b_phase1}")
        self.logger.info(f"Phase 2 (large fanout={np.mean(phase2_fanouts) if phase2_fanouts else 14:.1f}): "
                        f"HARDCODED A={optimal_a_phase2}, B={optimal_b_phase2}")

        # ========== HARDCODED QPS VALUES ==========
        # These match the values in main() for consistency
        # Calculated: (num_A_instances / 55s) * 1.15 overload
        HARDCODED_PHASE1_QPS = 0.50  # workflows/s (24 A instances / 55s * 1.15)
        HARDCODED_PHASE2_QPS = 0.29  # workflows/s (14 A instances / 55s * 1.15)

        optimal_qps_phase1 = HARDCODED_PHASE1_QPS
        optimal_qps_phase2 = HARDCODED_PHASE2_QPS

        self.logger.info(f"Phase 1 QPS: {optimal_qps_phase1:.2f} wf/s (HARDCODED)")
        self.logger.info(f"Phase 2 QPS: {optimal_qps_phase2:.2f} wf/s (HARDCODED)")

        # Store QPS values for external access
        self.phase1_qps = optimal_qps_phase1
        self.phase2_qps = optimal_qps_phase2
        self.qps_transition_index = large_phase_start_idx  # Workflow index where QPS changes

        self.logger.info(f"QPS transition: {optimal_qps_phase1:.2f} -> {optimal_qps_phase2:.2f} "
                        f"at workflow #{large_phase_start_idx}")

        # Create SINGLE migration plan: trigger after 5 large-fanout tasks have been submitted
        # Migration index = large_phase_start_idx + 5
        migration_trigger_idx = large_phase_start_idx + 5
        if migration_trigger_idx < len(a_submit_times):
            trigger_time = a_submit_times[migration_trigger_idx]

            # Only create migration if there's a significant change
            if abs(optimal_a_phase2 - optimal_a_phase1) >= self.min_change:
                plans.append(MigrationPlan(
                    trigger_time=trigger_time,
                    target_a_instances=optimal_a_phase2,
                    target_b_instances=optimal_b_phase2,
                    reason=f"two_phase: small_fanout({optimal_a_phase1}/{optimal_b_phase1})->large_fanout({optimal_a_phase2}/{optimal_b_phase2})"
                ))
                self.logger.info(f"Two-phase migration: trigger at t={trigger_time:.2f}s "
                               f"(after workflow #{migration_trigger_idx}), "
                               f"A: {optimal_a_phase1}->{optimal_a_phase2}, B: {optimal_b_phase1}->{optimal_b_phase2}")
            else:
                self.logger.info(f"Two-phase: No migration needed (change < {self.min_change} instances)")

        # Store initial optimal allocation for Phase 1 (this will be set at experiment start)
        self.initial_optimal_a = optimal_a_phase1
        self.initial_optimal_b = optimal_b_phase1

        self.logger.info(f"Two-phase mode: {len(plans)} migration(s) planned")
        return plans

    def get_qps_for_workflow(self, workflow_index: int) -> float:
        """
        Get the optimal QPS for a given workflow index.

        Args:
            workflow_index: Global workflow index (including warmup)

        Returns:
            Optimal QPS for that phase
        """
        if not self.two_phase_mode:
            return None  # Not in two-phase mode

        if workflow_index < self.qps_transition_index:
            return self.phase1_qps
        else:
            return self.phase2_qps

    def _get_current_instance_assignments(self):
        """
        Query real instance assignments from Schedulers.

        CRITICAL: Must query real state before each migration because
        instance-model mappings change after migrations.
        """
        all_instances = []

        # Query Scheduler A
        try:
            resp = requests.get(f"{self.scheduler_a_url}/instance/list", timeout=5.0)
            if resp.ok:
                for inst in resp.json().get("instances", []):
                    all_instances.append({
                        "endpoint": inst.get("endpoint") or f"http://localhost:{inst.get('port', 0)}",
                        "current_model": inst.get("model_id", "unknown")
                    })
        except Exception as e:
            self.logger.error(f"Failed to query Scheduler A instances: {e}")

        # Query Scheduler B
        try:
            resp = requests.get(f"{self.scheduler_b_url}/instance/list", timeout=5.0)
            if resp.ok:
                for inst in resp.json().get("instances", []):
                    all_instances.append({
                        "endpoint": inst.get("endpoint") or f"http://localhost:{inst.get('port', 0)}",
                        "current_model": inst.get("model_id", "unknown")
                    })
        except Exception as e:
            self.logger.error(f"Failed to query Scheduler B instances: {e}")

        return all_instances

    def _trigger_migration(self, plan: MigrationPlan):
        """Execute a migration by calling the Planner API."""
        # Get current real instance state
        all_instances = self._get_current_instance_assignments()

        if not all_instances:
            self.logger.error("No instances found from schedulers, skipping migration")
            return

        # Count current allocations
        current_a = sum(1 for i in all_instances if i["current_model"] == "sleep_model_a")
        current_b = sum(1 for i in all_instances if i["current_model"] == "sleep_model_b")
        self.logger.info(f"Current: A={current_a}, B={current_b} -> Target: A={plan.target_a_instances}, B={plan.target_b_instances}")

        # Throughput coefficients (must match B matrix)
        # Model A: 0.0347 QPS per instance
        # Model B: 0.369 QPS per instance
        throughput_a = 0.0347
        throughput_b = 0.369

        # IMPORTANT: Planner expects target as CAPACITY ratios, not instance count ratios.
        # We must compute the capacity that would result from the desired instance counts,
        # then send that as the target so Planner allocates instances accordingly.
        target_a_capacity = plan.target_a_instances * throughput_a
        target_b_capacity = plan.target_b_instances * throughput_b

        self.logger.info(f"Target capacity: A={target_a_capacity:.3f}, B={target_b_capacity:.3f}")

        # Build migration request
        payload = {
            "instances": all_instances,
            "planner_input": {
                "M": len(all_instances),
                "N": 2,
                "B": [[throughput_a, throughput_b]] * len(all_instances),
                "a": 1.0,
                "target": [target_a_capacity, target_b_capacity],  # Capacity ratios, not instance ratios
                "algorithm": "simulated_annealing",
                "objective_method": "ratio_difference"
            },
            "scheduler_mapping": {
                "sleep_model_a": self.scheduler_a_url,
                "sleep_model_b": self.scheduler_b_url
            }
        }

        try:
            response = requests.post(
                f"{self.planner_url}/deploy/migration",
                json=payload,
                timeout=60.0
            )
            if response.ok:
                result = response.json()
                self.logger.info(f"Migration completed: changes={result.get('changes_count', 'N/A')}")
            else:
                self.logger.error(f"Migration failed: HTTP {response.status_code} - {response.text[:200]}")
        except Exception as e:
            self.logger.error(f"Migration error: {e}")

    def _run(self):
        """Main loop: execute migrations according to pre-computed plan."""
        self.logger.info(f"Oracle deployer waiting for task submission to start...")
        self.logger.info(f"Pre-computed {len(self.migration_plan)} migration plans")

        # Wait for task submission to start (with timeout to avoid hanging)
        if not self._task_submission_started.wait(timeout=300):
            self.logger.warning("Timeout waiting for task submission signal, aborting")
            return

        # Now set the start time - synchronized with actual task submission
        self.experiment_start_time = time.time()
        self.logger.info(f"Task submission started, beginning migration schedule")

        # In two-phase mode, trigger initial deployment for Phase 1 optimal allocation
        if self.two_phase_mode and hasattr(self, 'initial_optimal_a'):
            self.logger.info(f"Two-phase mode: triggering initial deployment for Phase 1 "
                           f"(A={self.initial_optimal_a}, B={self.initial_optimal_b})")
            initial_plan = MigrationPlan(
                trigger_time=0.0,
                target_a_instances=self.initial_optimal_a,
                target_b_instances=self.initial_optimal_b,
                reason="two_phase_initial: optimal for small fanout"
            )
            self._trigger_migration(initial_plan)

        while self.running and self.plan_index < len(self.migration_plan):
            elapsed = time.time() - self.experiment_start_time
            next_plan = self.migration_plan[self.plan_index]

            if elapsed >= next_plan.trigger_time:
                self.logger.info(f"Executing migration {self.plan_index + 1}/{len(self.migration_plan)}: "
                               f"A={next_plan.target_a_instances}, B={next_plan.target_b_instances} "
                               f"({next_plan.reason})")
                self._trigger_migration(next_plan)
                self.plan_index += 1
            else:
                # Wait for next migration time
                wait_time = min(0.5, next_plan.trigger_time - elapsed)
                time.sleep(wait_time)

        self.logger.info("Oracle deployer completed all planned migrations")

    def signal_task_submission_started(self):
        """
        Signal that task submission has started.

        Call this method when the A task submitter actually begins submitting tasks.
        This synchronizes the migration timeline with the actual task submission.
        """
        self.logger.info("Received signal: task submission started")
        self._task_submission_started.set()

    def start(self):
        """Start the oracle deployer thread."""
        self.running = True
        self.plan_index = 0
        self._task_submission_started.clear()  # Reset the event
        self.thread = threading.Thread(target=self._run, name="Thread8-OracleDeployer")
        self.thread.start()
        self.logger.info("Oracle deployer thread started (waiting for task submission signal)")

    def stop(self):
        """Stop the oracle deployer thread."""
        self.running = False
        self._task_submission_started.set()  # Unblock if waiting
        if self.thread:
            self.thread.join(timeout=10.0)
            self.logger.info("Oracle deployer thread stopped")


# ============================================================================
# Thread 1: Poisson Task Submitter for A Tasks
# ============================================================================

class PoissonTaskSubmitter:
    """
    Thread 1: Submit A tasks following Poisson process with QPS control.

    Submits tasks to Scheduler A with exponentially distributed inter-arrival
    times to simulate realistic workload patterns.
    """

    def __init__(self,
                 scheduler_url: str,
                 tasks: List[WorkflowTaskData],
                 qps: float,
                 workflow_states: Dict[str, WorkflowState],
                 model_id: str = "sleep_model_a",
                 strategy: str = "default",
                 rate_limiter: Optional[RateLimiter] = None,
                 inter_arrival_times: Optional[np.ndarray] = None):
        """
        Initialize Poisson task submitter.

        Args:
            scheduler_url: Scheduler A URL (e.g., http://localhost:8100)
            tasks: List of pre-generated A task data
            qps: Target queries per second (e.g., 8.0) - used to generate inter-arrival times if not provided
            workflow_states: Shared workflow state dictionary
            model_id: Model ID to use for tasks
            strategy: Scheduling strategy name (for MAPE error simulation)
            rate_limiter: Optional shared rate limiter for global QPS control
            inter_arrival_times: Optional pre-computed inter-arrival times (for synchronization with OracleDeployer)
        """
        self.scheduler_url = scheduler_url
        self.tasks = tasks
        self.qps = qps
        self.workflow_states = workflow_states
        self.model_id = model_id
        self.strategy = strategy
        self.rate_limiter = rate_limiter
        self.inter_arrival_times = inter_arrival_times  # Pre-computed times for oracle sync
        self.logger = logging.getLogger("Thread1.ATaskSubmitter")

        # Tracking
        self.submitted_tasks: List[TaskRecord] = []
        self.submission_start_time: Optional[float] = None
        self.submission_end_time: Optional[float] = None
        self.failed_submissions: int = 0  # Track failed A task submissions

        # Thread control
        self.thread: Optional[threading.Thread] = None
        self.running = False

    def _submit_task(self, task_data: WorkflowTaskData) -> TaskRecord:
        """
        Submit a single A task to Scheduler A.

        Args:
            task_data: Pre-generated task data

        Returns:
            TaskRecord with submission info
        """
        # Apply MAPE error to exp_runtime for min_time strategy
        exp_runtime_to_send = task_data.exp_runtime
        if self.strategy == "min_time":
            exp_runtime_to_send = apply_mape_error(task_data.exp_runtime)

        payload = {
            "task_id": task_data.task_id,
            "model_id": self.model_id,
            "task_input": {
                "sleep_time": task_data.sleep_time
            },
            "metadata": {
                "exp_runtime": exp_runtime_to_send,
                "workflow_id": task_data.workflow_id,
                "task_type": "A",
                "is_warmup": task_data.is_warmup
            }
        }

        submit_time = time.time()

        # CRITICAL: Set a_submit_time BEFORE attempting submission
        # This ensures we have a timestamp even if submission fails
        workflow_id = task_data.workflow_id
        if workflow_id in self.workflow_states:
            self.workflow_states[workflow_id].a_submit_time = submit_time

        try:
            response = requests.post(
                f"{self.scheduler_url}/task/submit",
                json=payload,
                timeout=5.0
            )

            # Check for HTTP errors
            if not response.ok:
                # Try to extract error details from response body
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", response.text)
                except:
                    error_msg = response.text or f"HTTP {response.status_code}"

                error_detail = f"HTTP {response.status_code}: {error_msg}"
                self.logger.error(f"Failed to submit A task {task_data.task_id}: {error_detail}")
                self.failed_submissions += 1

                return TaskRecord(
                    task_id=task_data.task_id,
                    workflow_id=task_data.workflow_id,
                    task_type="A",
                    sleep_time=task_data.sleep_time,
                    exp_runtime=task_data.exp_runtime,
                    submit_time=submit_time,
                    status="submit_failed",
                    error=error_detail,
                    is_warmup=task_data.is_warmup
                )

            result = response.json()

            # Check if submission was successful
            if not result.get("success", True):
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Scheduler rejected A task {task_data.task_id}: {error_msg}")
                self.failed_submissions += 1

                return TaskRecord(
                    task_id=task_data.task_id,
                    workflow_id=task_data.workflow_id,
                    task_type="A",
                    sleep_time=task_data.sleep_time,
                    exp_runtime=task_data.exp_runtime,
                    submit_time=submit_time,
                    status="submit_failed",
                    error=f"Rejected: {error_msg}",
                    is_warmup=task_data.is_warmup
                )

            # Note: a_submit_time already set before try block

            # Create task record
            record = TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="A",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                assigned_instance=result.get("task", {}).get("assigned_instance"),
                is_warmup=task_data.is_warmup
            )

            return record

        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after 5.0s"
            self.logger.error(f"Failed to submit A task {task_data.task_id}: {error_msg}")
            self.failed_submissions += 1
            return TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="A",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                status="submit_failed",
                error=error_msg,
                is_warmup=task_data.is_warmup
            )
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            self.logger.error(f"Failed to submit A task {task_data.task_id}: {error_msg}")
            self.failed_submissions += 1
            return TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="A",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                status="submit_failed",
                error=error_msg,
                is_warmup=task_data.is_warmup
            )
        except Exception as e:
            error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
            self.logger.error(f"Failed to submit A task {task_data.task_id}: {error_msg}")
            self.failed_submissions += 1
            return TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="A",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                status="submit_failed",
                error=error_msg,
                is_warmup=task_data.is_warmup
            )

    def _run(self):
        """Main submission loop with Poisson inter-arrival times and optional global rate limiting."""
        if self.rate_limiter is not None:
            self.logger.debug(f"Starting Poisson submission: {len(self.tasks)} A tasks at {self.qps} QPS (with global rate limiter)")
        else:
            self.logger.debug(f"Starting Poisson submission: {len(self.tasks)} A tasks at {self.qps} QPS")
        self._run_poisson()

    def _run_poisson(self):
        """
        Poisson submission with optional global rate limiter.

        Behavior:
        1. Wait for Poisson inter-arrival time (controlled by --qps)
        2. Acquire token from global rate limiter if present (controlled by --gqps)
        3. Submit task

        This allows --qps and --gqps to work together:
        - --qps controls A task arrival pattern (Poisson process)
        - --gqps controls global submission rate across all tasks
        """
        # Use pre-computed inter-arrival times if provided (for OracleDeployer synchronization)
        # Otherwise generate new times using exponential distribution
        if self.inter_arrival_times is not None:
            inter_arrival_times = self.inter_arrival_times
            # Calculate effective QPS from pre-computed times
            total_time = np.sum(inter_arrival_times)
            effective_qps = len(inter_arrival_times) / total_time if total_time > 0 else 0
            self.logger.info("=" * 60)
            self.logger.info("USING PRE-COMPUTED INTER-ARRIVAL TIMES (HARDCODED QPS)")
            self.logger.info(f"  User-provided QPS: {self.qps:.2f} (IGNORED)")
            self.logger.info(f"  Effective avg QPS: {effective_qps:.2f} wf/s")
            self.logger.info(f"  Total tasks: {len(inter_arrival_times)}")
            self.logger.info(f"  Expected duration: {total_time:.1f}s")
            self.logger.info("=" * 60)
        else:
            lambda_rate = self.qps
            inter_arrival_times = np.random.exponential(1.0 / lambda_rate, len(self.tasks))
            self.logger.info(f"Using user-provided QPS: {self.qps:.2f} wf/s")

        self.submission_start_time = time.time()

        for i, (task_data, wait_time) in enumerate(zip(self.tasks, inter_arrival_times)):
            if not self.running:
                self.logger.warning("Submission stopped early")
                break

            # Step 1: Wait for Poisson inter-arrival time (controlled by --qps)
            if i > 0:
                time.sleep(wait_time)

            # Step 2: Acquire token from global rate limiter if present (controlled by --gqps)
            if self.rate_limiter is not None:
                self.rate_limiter.acquire(1)

            # Step 3: Submit task
            record = self._submit_task(task_data)
            self.submitted_tasks.append(record)

            if (i + 1) % 20 == 0:
                self.logger.debug(f"Submitted {i + 1}/{len(self.tasks)} A tasks")

        self.submission_end_time = time.time()
        submission_duration = self.submission_end_time - self.submission_start_time
        actual_qps = len(self.tasks) / submission_duration

        self.logger.debug(f"Submission complete: {len(self.submitted_tasks)} tasks in {submission_duration:.2f}s "
                        f"(actual QPS: {actual_qps:.2f})")
        if self.failed_submissions > 0:
            self.logger.error(f"⚠️  A TASK SUBMISSION FAILURES: {self.failed_submissions}/{len(self.tasks)} tasks failed to submit!")
            self.logger.error(f"⚠️  This means {self.failed_submissions} workflows will NEVER complete!")

    def _run_with_rate_limiter(self):
        """Submission with shared rate limiter."""
        self.submission_start_time = time.time()

        for i, task_data in enumerate(self.tasks):
            if not self.running:
                self.logger.warning("Submission stopped early")
                break

            # Acquire token from rate limiter (blocks if necessary)
            self.rate_limiter.acquire(1)

            # Submit task
            record = self._submit_task(task_data)
            self.submitted_tasks.append(record)

            if (i + 1) % 20 == 0:
                self.logger.debug(f"Submitted {i + 1}/{len(self.tasks)} A tasks")

        self.submission_end_time = time.time()
        submission_duration = self.submission_end_time - self.submission_start_time
        actual_qps = len(self.tasks) / submission_duration

        self.logger.debug(f"Submission complete: {len(self.submitted_tasks)} tasks in {submission_duration:.2f}s "
                        f"(actual QPS: {actual_qps:.2f})")
        if self.failed_submissions > 0:
            self.logger.error(f"⚠️  A TASK SUBMISSION FAILURES: {self.failed_submissions}/{len(self.tasks)} tasks failed to submit!")
            self.logger.error(f"⚠️  This means {self.failed_submissions} workflows will NEVER complete!")

    def start(self):
        """Start the submission thread."""
        if self.thread is not None:
            self.logger.warning("Submitter already started")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run, name="Thread1-ATaskSubmitter")
        self.thread.start()
        self.logger.debug("Submission thread started")

    def stop(self):
        """Stop the submission thread."""
        if self.thread is None:
            return

        self.running = False
        self.thread.join(timeout=10.0)
        self.logger.debug("Submission thread stopped")

    def is_alive(self) -> bool:
        """Check if submission thread is still running."""
        return self.thread is not None and self.thread.is_alive()


# ============================================================================
# Thread 2: A Task Result Receiver + B Task Submitter
# ============================================================================

class ATaskReceiver:
    """
    Thread 2: Receive A task results via WebSocket and submit B1 tasks.

    For each completed A task:
    1. Receive completion event from Scheduler A
    2. Look up the number of B1 tasks (n) for this workflow
    3. Submit n B1 tasks to Scheduler B
    4. Record B1 task submission info
    """

    def __init__(self,
                 scheduler_a_ws: str,
                 scheduler_b_url: str,
                 a_task_ids: List[str],
                 b1_tasks_by_workflow: Dict[str, List[WorkflowTaskData]],
                 workflow_states: Dict[str, WorkflowState],
                 model_id: str = "sleep_model_b",
                 strategy: str = "default",
                 rate_limiter: Optional[RateLimiter] = None):
        """
        Initialize A task receiver.

        Args:
            scheduler_a_ws: WebSocket URL for Scheduler A
            scheduler_b_url: HTTP URL for Scheduler B
            a_task_ids: List of all A task IDs to subscribe to
            b1_tasks_by_workflow: Map of workflow_id -> list of B1 task data
            workflow_states: Shared workflow state dictionary
            model_id: Model ID to use for B1 tasks
            rate_limiter: Optional shared rate limiter for global QPS control
        """
        self.scheduler_a_ws = scheduler_a_ws
        self.scheduler_b_url = scheduler_b_url
        self.a_task_ids = a_task_ids
        self.b1_tasks_by_workflow = b1_tasks_by_workflow
        self.workflow_states = workflow_states
        self.model_id = model_id
        self.rate_limiter = rate_limiter
        self.logger = logging.getLogger("Thread2.ATaskReceiver")

        # Tracking
        self.a_results: List[TaskRecord] = []
        self.b1_submitted: List[TaskRecord] = []
        self.received_a_task_count = 0  # Track number of A tasks received
        self.expected_a_task_count = len(a_task_ids)  # Total A tasks expected
        self.failed_b1_submissions: int = 0  # Track failed B1 task submissions

        # Thread control
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.raw_result_list: List[Dict[str, Any]] = []
        self.strategy_name = strategy

    def _submit_b1_task(self, task_data: WorkflowTaskData) -> TaskRecord:
        """
        Submit a single B1 task to Scheduler B.

        Args:
            task_data: Pre-generated B1 task data

        Returns:
            TaskRecord with submission info
        """
        # Apply MAPE error to exp_runtime for min_time strategy
        exp_runtime_to_send = task_data.exp_runtime
        if self.strategy_name == "min_time":
            exp_runtime_to_send = apply_mape_error(task_data.exp_runtime)

        payload = {
            "task_id": task_data.task_id,
            "model_id": self.model_id,
            "task_input": {
                "sleep_time": task_data.sleep_time
            },
            "metadata": {
                "exp_runtime": exp_runtime_to_send,
                "workflow_id": task_data.workflow_id,
                "task_type": "B1",
                "b_index": task_data.b_index,
                "is_warmup": task_data.is_warmup
            }
        }

        submit_time = time.time()

        try:
            response = requests.post(
                f"{self.scheduler_b_url}/task/submit",
                json=payload,
                timeout=5.0
            )

            # Check for HTTP errors
            if not response.ok:
                # Try to extract error details from response body
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", response.text)
                except:
                    error_msg = response.text or f"HTTP {response.status_code}"

                error_detail = f"HTTP {response.status_code}: {error_msg}"
                self.logger.error(f"Failed to submit B1 task {task_data.task_id}: {error_detail}")
                self.failed_b1_submissions += 1

                return TaskRecord(
                    task_id=task_data.task_id,
                    workflow_id=task_data.workflow_id,
                    task_type="B1",
                    sleep_time=task_data.sleep_time,
                    exp_runtime=task_data.exp_runtime,
                    submit_time=submit_time,
                    status="submit_failed",
                    error=error_detail,
                    b_index=task_data.b_index,
                    is_warmup=task_data.is_warmup
                )

            result = response.json()

            # Check if submission was successful
            if not result.get("success", True):
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Scheduler rejected B1 task {task_data.task_id}: {error_msg}")
                self.failed_b1_submissions += 1

                return TaskRecord(
                    task_id=task_data.task_id,
                    workflow_id=task_data.workflow_id,
                    task_type="B1",
                    sleep_time=task_data.sleep_time,
                    exp_runtime=task_data.exp_runtime,
                    submit_time=submit_time,
                    status="submit_failed",
                    error=f"Rejected: {error_msg}",
                    b_index=task_data.b_index,
                    is_warmup=task_data.is_warmup
                )

            record = TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="B1",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                assigned_instance=result.get("task", {}).get("assigned_instance"),
                b_index=task_data.b_index
            )

            return record

        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after 5.0s"
            self.logger.error(f"Failed to submit B1 task {task_data.task_id}: {error_msg}")
            self.failed_b1_submissions += 1
            return TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="B1",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                status="submit_failed",
                error=error_msg,
                b_index=task_data.b_index,
                    is_warmup=task_data.is_warmup
            )
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            self.logger.error(f"Failed to submit B1 task {task_data.task_id}: {error_msg}")
            self.failed_b1_submissions += 1
            return TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="B1",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                status="submit_failed",
                error=error_msg,
                b_index=task_data.b_index,
                    is_warmup=task_data.is_warmup
            )
        except Exception as e:
            error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
            self.logger.error(f"Failed to submit B1 task {task_data.task_id}: {error_msg}")
            self.failed_b1_submissions += 1
            return TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="B1",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                status="submit_failed",
                error=error_msg,
                b_index=task_data.b_index,
                    is_warmup=task_data.is_warmup
            )

    async def _run_async(self):
        """Main WebSocket loop for receiving A task results."""
        self.logger.debug(f"Connecting to Scheduler A WebSocket: {self.scheduler_a_ws}")

        try:
            async with websockets.connect(
                self.scheduler_a_ws,
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=10,   # Wait up to 10 seconds for pong
                close_timeout=10   # Wait up to 10 seconds for close handshake
            ) as websocket:
                # Subscribe to all A task IDs
                subscribe_msg = {
                    "type": "subscribe",
                    "task_ids": self.a_task_ids
                }
                await websocket.send(json.dumps(subscribe_msg))
                self.logger.info(f"Subscribed to {len(self.a_task_ids)} A tasks")

                # Wait for acknowledgment
                ack = await websocket.recv()
                ack_data = json.loads(ack)
                self.logger.info(f"Subscription confirmed: {ack_data.get('message')}")

                # Receive A task results
                while self.running:
                    # Check if all A tasks have been received
                    if self.received_a_task_count >= self.expected_a_task_count:
                        self.logger.info(
                            f"All {self.expected_a_task_count} A tasks received and B1 tasks submitted. "
                            f"Gracefully closing WebSocket connection."
                        )
                        break

                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)

                        if data["type"] == "result":
                            await self._handle_a_result(data)
                        elif data["type"] == "error":
                            self.logger.error(f"WebSocket error: {data.get('error')}")

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        self.logger.error(f"Error receiving message: {e}, \ntrackback: {traceback.format_exc()}")
                        break

                # Gracefully close the WebSocket connection
                await websocket.close(code=1000, reason="All tasks completed")
                self.logger.debug("WebSocket connection closed gracefully")

        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")

    async def _handle_a_result(self, data: Dict):
        """
        Handle A task completion result and submit B1 tasks.

        Args:
            data: WebSocket result message
        """
        
        task_id = data["task_id"]
        status = data["status"]
        raw_result = data.get("result") or {}  # Handle None result
        workflow_id = None

        # Extract workflow_id from task_id (format: task-A-{strategy}-workflow-{i:04d}-A-{experiment_id})
        parts = task_id.split("-")
        if len(parts) >= 7 and parts[3] == "workflow":
            workflow_id = f"wf-{parts[2]}-{parts[4]}-{parts[6]}"
            raw_result["workflow_id"] = workflow_id
            
        self.raw_result_list.append(raw_result)

        if not workflow_id:
            self.logger.error(f"Could not extract workflow_id from task_id: {task_id}")
            return

        # Extract sleep_time and execution_time from result
        result = data.get("result", {})
        sleep_time = result.get("sleep_time", 0.0)
        execution_time = data.get("execution_time", 0.0)

        # Create A task record
        a_record = TaskRecord(
            task_id=task_id,
            workflow_id=workflow_id,
            task_type="A",
            sleep_time=sleep_time,
            exp_runtime=execution_time * 1000,  # Convert seconds to milliseconds
            complete_time=time.time(),
            status=status,
            execution_time_ms=data.get("execution_time_ms", execution_time * 1000),
            result=data.get("result"),
            error=data.get("error")
        )
        self.a_results.append(a_record)

        # Update workflow state
        if workflow_id in self.workflow_states:
            self.workflow_states[workflow_id].a_complete_time = a_record.complete_time

        # If A task succeeded, submit B1 tasks
        if status == "completed":
            await self._submit_b1_tasks_for_workflow(workflow_id)
        else:
            self.logger.warning(f"A task {task_id} failed, skipping B1 task submission")

        # Increment received count after handling the result
        self.received_a_task_count += 1
        self.logger.debug(
            f"Received A task {self.received_a_task_count}/{self.expected_a_task_count}"
        )

    async def _submit_b1_tasks_for_workflow(self, workflow_id: str):
        """
        Submit all B1 tasks for a given workflow.

        Args:
            workflow_id: Workflow ID
        """
        if workflow_id not in self.b1_tasks_by_workflow:
            self.logger.error(f"No B1 tasks found for workflow {workflow_id}")
            return

        b1_tasks = self.b1_tasks_by_workflow[workflow_id]

        self.logger.debug(f"Submitting {len(b1_tasks)} B1 tasks for workflow {workflow_id}")

        # Submit all B1 tasks (in current thread, not async)
        # Using synchronous submission for simplicity
        for b1_task_data in b1_tasks:
            # If using rate limiter, acquire token before submitting
            if self.rate_limiter is not None:
                self.rate_limiter.acquire(1)

            record = self._submit_b1_task(b1_task_data)
            self.b1_submitted.append(record)

        self.logger.debug(f"Completed B1 task submission for workflow {workflow_id}")

    def _run(self):
        """Thread entry point."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._run_async())
        except Exception as e:
            self.logger.error(f"Error in async loop: {e}")
        finally:
            try:
                # Cancel any pending tasks
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()
                # Wait for all tasks to be cancelled
                if pending:
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                self.loop.close()


    def start(self):
        """Start the receiver thread."""
        if self.thread is not None:
            self.logger.warning("Receiver already started")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run, name="Thread2-ATaskReceiver")
        self.thread.start()
        self.logger.debug("A task receiver thread started")

    def stop(self):
        """Stop the receiver thread."""
        os.makedirs(f"raw_results_{self.strategy_name}", exist_ok=True)
        with open(f"raw_results_{self.strategy_name}/a1_task_results.jsonl", "w") as f:
            for record in self.raw_result_list:
                f.write(f'{json.dumps(record)}\n')
                
        if self.thread is None:
            return

        self.running = False
        self.thread.join(timeout=10.0)
        self.logger.debug("A task receiver thread stopped")


    def is_alive(self) -> bool:
        """Check if receiver thread is still running."""
        return self.thread is not None and self.thread.is_alive()


# ============================================================================
# Thread 3: B1 Task Result Receiver + B2 Task Submitter
# ============================================================================

class B1TaskReceiver:
    """
    Thread 3: Receive B1 task results via WebSocket and submit B2 tasks.

    For each completed B1 task:
    1. Receive completion event from Scheduler B
    2. Update workflow state (mark B1 task complete)
    3. Submit corresponding B2 task to Scheduler B
    4. Record B2 task submission info
    """

    def __init__(self,
                 scheduler_b_ws: str,
                 scheduler_b_url: str,
                 b1_task_ids: List[str],
                 b2_tasks_by_b1: Dict[str, WorkflowTaskData],
                 workflow_states: Dict[str, WorkflowState],
                 model_id: str = "sleep_model_b",
                 strategy: str = "default",
                 rate_limiter: Optional[RateLimiter] = None):
        """
        Initialize B1 task receiver.

        Args:
            scheduler_b_ws: WebSocket URL for Scheduler B
            scheduler_b_url: HTTP URL for Scheduler B
            b1_task_ids: List of all B1 task IDs to subscribe to
            b2_tasks_by_b1: Map of b1_task_id -> B2 task data
            workflow_states: Shared workflow state dictionary
            model_id: Model ID to use for B2 tasks
            rate_limiter: Optional shared rate limiter for global QPS control
        """
        self.scheduler_b_ws = scheduler_b_ws
        self.scheduler_b_url = scheduler_b_url
        self.b1_task_ids = b1_task_ids
        self.b2_tasks_by_b1 = b2_tasks_by_b1
        self.workflow_states = workflow_states
        self.model_id = model_id
        self.rate_limiter = rate_limiter
        self.logger = logging.getLogger("Thread3.B1TaskReceiver")

        # Tracking
        self.b1_results: List[TaskRecord] = []
        self.b2_submitted: List[TaskRecord] = []

        # Thread control
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        self.raw_result_list = []
        self.strategy_name = strategy

    def _submit_b2_task(self, task_data: WorkflowTaskData) -> TaskRecord:
        """
        Submit a single B2 task to Scheduler B.

        Args:
            task_data: Pre-generated B2 task data

        Returns:
            TaskRecord with submission info
        """
        # Apply MAPE error to exp_runtime for min_time strategy
        exp_runtime_to_send = task_data.exp_runtime
        if self.strategy_name == "min_time":
            exp_runtime_to_send = apply_mape_error(task_data.exp_runtime)

        payload = {
            "task_id": task_data.task_id,
            "model_id": self.model_id,
            "task_input": {
                "sleep_time": task_data.sleep_time
            },
            "metadata": {
                "exp_runtime": exp_runtime_to_send,
                "workflow_id": task_data.workflow_id,
                "task_type": "B2",
                "b_index": task_data.b_index,
                "is_warmup": task_data.is_warmup
            }
        }

        submit_time = time.time()

        try:
            response = requests.post(
                f"{self.scheduler_b_url}/task/submit",
                json=payload,
                timeout=5.0
            )

            if not response.ok:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", response.text)
                except:
                    error_msg = response.text or f"HTTP {response.status_code}"

                error_detail = f"HTTP {response.status_code}: {error_msg}"
                self.logger.error(f"Failed to submit B2 task {task_data.task_id}: {error_detail}")

                return TaskRecord(
                    task_id=task_data.task_id,
                    workflow_id=task_data.workflow_id,
                    task_type="B2",
                    sleep_time=task_data.sleep_time,
                    exp_runtime=task_data.exp_runtime,
                    submit_time=submit_time,
                    status="submit_failed",
                    error=error_detail,
                    b_index=task_data.b_index,
                    is_warmup=task_data.is_warmup
                )

            result = response.json()

            if not result.get("success", True):
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Scheduler rejected B2 task {task_data.task_id}: {error_msg}")

                return TaskRecord(
                    task_id=task_data.task_id,
                    workflow_id=task_data.workflow_id,
                    task_type="B2",
                    sleep_time=task_data.sleep_time,
                    exp_runtime=task_data.exp_runtime,
                    submit_time=submit_time,
                    status="submit_failed",
                    error=f"Rejected: {error_msg}",
                    b_index=task_data.b_index,
                    is_warmup=task_data.is_warmup
                )

            record = TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="B2",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                assigned_instance=result.get("task", {}).get("assigned_instance"),
                b_index=task_data.b_index
            )

            return record

        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after 5.0s"
            self.logger.error(f"Failed to submit B2 task {task_data.task_id}: {error_msg}")
            return TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="B2",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                status="submit_failed",
                error=error_msg,
                b_index=task_data.b_index,
                    is_warmup=task_data.is_warmup
            )
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            self.logger.error(f"Failed to submit B2 task {task_data.task_id}: {error_msg}")
            return TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="B2",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                status="submit_failed",
                error=error_msg,
                b_index=task_data.b_index,
                    is_warmup=task_data.is_warmup
            )
        except Exception as e:
            error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
            self.logger.error(f"Failed to submit B2 task {task_data.task_id}: {error_msg}")
            return TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="B2",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                status="submit_failed",
                error=error_msg,
                b_index=task_data.b_index,
                    is_warmup=task_data.is_warmup
            )

    async def _run_async(self):
        """Main WebSocket loop for receiving B1 task results."""
        self.logger.debug(f"Connecting to Scheduler B WebSocket: {self.scheduler_b_ws}")

        try:
            async with websockets.connect(self.scheduler_b_ws) as websocket:
                subscribe_msg = {
                    "type": "subscribe",
                    "task_ids": self.b1_task_ids
                }
                await websocket.send(json.dumps(subscribe_msg))
                self.logger.debug(f"Subscribed to {len(self.b1_task_ids)} B1 tasks")

                ack = await websocket.recv()
                ack_data = json.loads(ack)
                self.logger.debug(f"Subscription confirmed: {ack_data.get('message')}")

                while self.running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)

                        if data["type"] == "result":
                            await self._handle_b1_result(data)
                        elif data["type"] == "error":
                            self.logger.error(f"WebSocket error: {data.get('error')}")

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        self.logger.error(f"Error receiving message: {e}, \ntrackback: {traceback.format_exc()}")
                        break

        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")

    async def _handle_b1_result(self, data: Dict):
        """
        Handle B1 task completion result and submit corresponding B2 task.

        Args:
            data: WebSocket result message
        """
        task_id = data["task_id"]
        status = data["status"]
        raw_b_result = data.get("result") or {}  # Handle None result
        complete_time = time.time()
        workflow_id = None

        # Extract workflow_id from task_id (format: task-B1-{strategy}-workflow-{i:04d}-B1-{j:02d}-{experiment_id})
        parts = task_id.split("-")
        if len(parts) >= 8 and parts[3] == "workflow":
            workflow_id = f"wf-{parts[2]}-{parts[4]}-{parts[7]}"
            raw_b_result["workflow_id"] = workflow_id
        
        self.raw_result_list.append(raw_b_result)

        if not workflow_id:
            self.logger.error(f"Could not extract workflow_id from task_id: {task_id}")
            return

        # Extract sleep_time and execution_time from result
        result = data.get("result", {})
        sleep_time = result.get("sleep_time", 0.0)
        execution_time = data.get("execution_time", 0.0)

        # Create B1 task record
        b1_record = TaskRecord(
            task_id=task_id,
            workflow_id=workflow_id,
            task_type="B1",
            sleep_time=sleep_time,
            exp_runtime=execution_time * 1000,  # Convert seconds to milliseconds
            complete_time=complete_time,
            status=status,
            execution_time_ms=data.get("execution_time_ms", execution_time * 1000),
            result=data.get("result"),
            error=data.get("error")
        )
        self.b1_results.append(b1_record)

        # Update workflow state
        if workflow_id not in self.workflow_states:
            self.logger.error(f"Unknown workflow_id: {workflow_id}")
            return

        workflow = self.workflow_states[workflow_id]

        # Mark B1 task as complete
        if status == "completed":
            workflow.mark_b1_task_complete(task_id, complete_time)

            # Submit corresponding B2 task
            if task_id in self.b2_tasks_by_b1:
                # If using rate limiter, acquire token before submitting
                if self.rate_limiter is not None:
                    self.rate_limiter.acquire(1)

                b2_task_data = self.b2_tasks_by_b1[task_id]
                record = self._submit_b2_task(b2_task_data)
                self.b2_submitted.append(record)
                self.logger.debug(f"Submitted B2 task {b2_task_data.task_id} for completed B1 {task_id}")
            else:
                self.logger.error(f"No B2 task found for B1 task {task_id}")
        else:
            self.logger.warning(f"B1 task {task_id} failed for workflow {workflow_id}")

    def _run(self):
        """Thread entry point."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._run_async())
        except Exception as e:
            self.logger.error(f"Error in async loop: {e}")
        finally:
            try:
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                self.loop.close()

    def start(self):
        """Start the receiver thread."""
        if self.thread is not None:
            self.logger.warning("Receiver already started")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run, name="Thread3-B1TaskReceiver")
        self.thread.start()
        self.logger.debug("B1 task receiver thread started")

    def stop(self):
        """Stop the receiver thread."""
        os.makedirs(f"raw_results_{self.strategy_name}", exist_ok=True)
        with open(f"raw_results_{self.strategy_name}/b1_task_results.jsonl", "w") as f:
            for record in self.raw_result_list:
                f.write(f"{json.dumps(record)}\n")
        if self.thread is None:
            return

        self.running = False
        self.thread.join(timeout=10.0)
        self.logger.debug("B1 task receiver thread stopped")
        

    def is_alive(self) -> bool:
        """Check if receiver thread is still running."""
        return self.thread is not None and self.thread.is_alive()


# ============================================================================
# Thread 4: B2 Task Result Receiver + Workflow State Updater
# ============================================================================

class B2TaskReceiver:
    """
    Thread 4: Receive B2 task results via WebSocket and update workflow state.

    For each completed B2 task:
    1. Receive completion event from Scheduler B
    2. Update workflow state (add to b2_complete_times dict)
    3. Check if all B2 tasks are done (using atomic len() operation)
    4. If all B2 tasks done, push to merge_ready_queue for Thread 5 to submit merge task
    """

    def __init__(self,
                 scheduler_b_ws: str,
                 b2_task_ids: List[str],
                 workflow_states: Dict[str, WorkflowState],
                 merge_ready_queue: Queue,
                 strategy: str = "default"
                 ):
        """
        Initialize B2 task receiver.

        Args:
            scheduler_b_ws: WebSocket URL for Scheduler B
            b2_task_ids: List of all B2 task IDs to subscribe to
            workflow_states: Shared workflow state dictionary
            merge_ready_queue: Queue to push "all B2 tasks complete" events
        """
        self.scheduler_b_ws = scheduler_b_ws
        self.b2_task_ids = b2_task_ids
        self.workflow_states = workflow_states
        self.merge_ready_queue = merge_ready_queue
        self.logger = logging.getLogger("Thread4.B2TaskReceiver")

        # Tracking
        self.b2_results: List[TaskRecord] = []

        # Thread control
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Track workflow completion to avoid duplicates
        self.completed_workflows: Set[str] = set()
        
        self.raw_result_list = []
        self.strategy_name = strategy

    async def _run_async(self):
        """Main WebSocket loop for receiving B2 task results."""
        self.logger.debug(f"Connecting to Scheduler B WebSocket: {self.scheduler_b_ws}")

        try:
            async with websockets.connect(self.scheduler_b_ws) as websocket:
                # Subscribe to all B2 task IDs
                subscribe_msg = {
                    "type": "subscribe",
                    "task_ids": self.b2_task_ids
                }
                await websocket.send(json.dumps(subscribe_msg))
                self.logger.debug(f"Subscribed to {len(self.b2_task_ids)} B2 tasks")

                # Wait for acknowledgment
                ack = await websocket.recv()
                ack_data = json.loads(ack)
                self.logger.debug(f"Subscription confirmed: {ack_data.get('message')}")

                # Receive B2 task results
                while self.running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)

                        if data["type"] == "result":
                            await self._handle_b2_result(data)
                        elif data["type"] == "error":
                            self.logger.error(f"WebSocket error: {data.get('error')}")

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        self.logger.error(f"Error receiving message: {e}, \ntrackback: {traceback.format_exc()}")
                        break

        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")

    async def _handle_b2_result(self, data: Dict):
        """
        Handle B2 task completion result and update workflow state.

        Args:
            data: WebSocket result message
        """
        task_id = data["task_id"]
        status = data["status"]
        raw_result = data.get("result") or {}  # Handle None result
        complete_time = time.time()
        workflow_id = None

        # Extract workflow_id from task_id (format: task-B2-{strategy}-workflow-{i:04d}-B2-{j:02d}-{experiment_id})
        parts = task_id.split("-")
        if len(parts) >= 8 and parts[3] == "workflow":
            workflow_id = f"wf-{parts[2]}-{parts[4]}-{parts[7]}"
            raw_result["workflow_id"] = workflow_id
            
        self.raw_result_list.append(raw_result)

        if not workflow_id:
            self.logger.error(f"Could not extract workflow_id from task_id: {task_id}")
            return

        # Extract sleep_time and execution_time from result
        result = data.get("result", {})
        sleep_time = result.get("sleep_time", 0.0)
        execution_time = data.get("execution_time", 0.0)

        # Create B2 task record
        b2_record = TaskRecord(
            task_id=task_id,
            workflow_id=workflow_id,
            task_type="B2",
            sleep_time=sleep_time,
            exp_runtime=execution_time * 1000,  # Convert seconds to milliseconds
            complete_time=complete_time,
            status=status,
            execution_time_ms=data.get("execution_time_ms", execution_time * 1000),
            result=data.get("result"),
            error=data.get("error")
        )
        self.b2_results.append(b2_record)

        # Update workflow state
        if workflow_id not in self.workflow_states:
            self.logger.error(f"Unknown workflow_id: {workflow_id}")
            return

        workflow = self.workflow_states[workflow_id]

        # Mark B2 task as complete
        if status == "completed":
            workflow.mark_b2_task_complete(task_id, complete_time)

            # Check if all B2 tasks are now complete (ready for merge task)
            if workflow.are_all_b2_tasks_complete() and workflow_id not in self.completed_workflows:
                self.completed_workflows.add(workflow_id)
                self._push_merge_ready(workflow)
                self.logger.debug(f"Workflow {workflow_id} B2 tasks all completed "
                               f"({len(workflow.b2_complete_times)}/{workflow.total_b2_tasks} B2 tasks), "
                               f"ready for merge task")
        else:
            self.logger.warning(f"B2 task {task_id} failed for workflow {workflow_id}")

    def _push_merge_ready(self, workflow: WorkflowState):
        """
        Push merge-ready event to queue for Thread 5 (MergeTaskSubmitter).

        Args:
            workflow: Workflow with all B tasks completed
        """
        # Just push the workflow_id, Thread 5 will submit the merge task
        self.merge_ready_queue.put(workflow.workflow_id)
        self.logger.debug(f"Pushed merge-ready event for workflow {workflow.workflow_id}")

    def _run(self):
        """Thread entry point."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._run_async())
        except Exception as e:
            self.logger.error(f"Error in async loop: {e}")
        finally:
            try:
                # Cancel any pending tasks
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()
                # Wait for all tasks to be cancelled
                if pending:
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                self.loop.close()

    def start(self):
        """Start the receiver thread."""
        if self.thread is not None:
            self.logger.warning("Receiver already started")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run, name="Thread4-B2TaskReceiver")
        self.thread.start()
        self.logger.info("B2 task receiver thread started")

    def stop(self):
        """Stop the receiver thread."""
        os.makedirs(f"raw_results_{self.strategy_name}", exist_ok=True)
        with open(f"raw_results_{self.strategy_name}/b2_task_results.jsonl", "w") as f:
            for record in self.raw_result_list:
                f.write(f"{json.dumps(record)}\n")
        if self.thread is None:
            return

        self.running = False
        self.thread.join(timeout=10.0)
        self.logger.info("B2 task receiver thread stopped")

    def is_alive(self) -> bool:
        """Check if receiver thread is still running."""
        return self.thread is not None and self.thread.is_alive()


# ============================================================================
# Thread 5: Merge Task Submitter
# ============================================================================

class MergeTaskSubmitter:
    """
    Thread 5: Submit merge A tasks when all B tasks complete.

    Listens to merge_ready_queue and:
    1. Receives workflow_id when all B tasks are complete
    2. Submits merge A task to Scheduler A
    3. Updates workflow state with merge_submit_time
    """

    def __init__(self,
                 scheduler_a_url: str,
                 merge_tasks: List[WorkflowTaskData],
                 workflow_states: Dict[str, WorkflowState],
                 merge_ready_queue: Queue,
                 strategy: str = "default",
                 rate_limiter: Optional[RateLimiter] = None):
        """
        Initialize merge task submitter.

        Args:
            scheduler_a_url: HTTP URL for Scheduler A
            merge_tasks: List of all merge task data (indexed by workflow)
            workflow_states: Shared workflow state dictionary
            merge_ready_queue: Queue to receive merge-ready events
            strategy: Scheduling strategy name (for MAPE error simulation)
            rate_limiter: Optional shared rate limiter for global QPS control
        """
        self.scheduler_a_url = scheduler_a_url
        self.merge_tasks = merge_tasks
        self.workflow_states = workflow_states
        self.merge_ready_queue = merge_ready_queue
        self.strategy = strategy
        self.rate_limiter = rate_limiter
        self.logger = logging.getLogger("Thread5.MergeTaskSubmitter")
        self.model_id = "sleep_model_a"

        # Thread control
        self.thread: Optional[threading.Thread] = None
        self.running = False

        # Track submissions
        self.submitted_count = 0

    def _submit_merge_task(self, workflow_id: str):
        """
        Submit merge task for a workflow.

        Args:
            workflow_id: Workflow ID ready for merge
        """
        if workflow_id not in self.workflow_states:
            self.logger.error(f"Unknown workflow_id: {workflow_id}")
            return

        workflow = self.workflow_states[workflow_id]

        # Find the merge task for this workflow
        merge_task = None
        for task in self.merge_tasks:
            if task.workflow_id == workflow_id:
                merge_task = task
                break

        if not merge_task:
            self.logger.error(f"Could not find merge task for workflow {workflow_id}")
            return

        # Submit merge task to Scheduler A
        try:
            # Apply MAPE error to exp_runtime for min_time strategy
            exp_runtime_to_send = merge_task.exp_runtime
            if self.strategy == "min_time":
                exp_runtime_to_send = apply_mape_error(merge_task.exp_runtime)

            payload = {
                "task_id": merge_task.task_id,
                "model_id": self.model_id,
                "task_input": {
                    "sleep_time": merge_task.sleep_time
                },
                "metadata": {
                    "exp_runtime": exp_runtime_to_send,
                    "workflow_id": merge_task.workflow_id,
                    "task_type": "B",
                    "is_warmup": merge_task.is_warmup
                }
            }

            response = requests.post(
                f"{self.scheduler_a_url}/task/submit",
                json=payload,
                timeout=5.0
            )

            submit_time = time.time()
            workflow.merge_submit_time = submit_time
            self.submitted_count += 1

            if response.status_code == 200:
                self.logger.info(
                    f"Submitted merge task {merge_task.task_id} for workflow {workflow_id} "
                    f"(sleep={merge_task.sleep_time:.2f}s, {self.submitted_count} total)"
                )
            else:
                self.logger.error(
                    f"Failed to submit merge task {merge_task.task_id}: "
                    f"HTTP {response.status_code}"
                )

        except Exception as e:
            self.logger.error(f"Error submitting merge task {merge_task.task_id}: {e}")

    def _run(self):
        """Thread entry point - wait for merge-ready events and submit tasks."""
        self.logger.info("Merge task submitter started")

        while self.running:
            try:
                # Wait for merge-ready event (workflow_id)
                workflow_id = self.merge_ready_queue.get(timeout=1.0)

                # If using rate limiter, acquire token before submitting
                if self.rate_limiter is not None:
                    self.rate_limiter.acquire(1)

                self._submit_merge_task(workflow_id)

            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in merge task submitter: {e}")

        self.logger.info("Merge task submitter stopped")

    def start(self):
        """Start the merge task submitter thread."""
        if self.thread is not None and self.thread.is_alive():
            self.logger.warning("Merge task submitter already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.logger.debug("Merge task submitter thread started")

    def stop(self):
        """Stop the merge task submitter thread."""

                
        self.running = False
        if self.thread:
            self.thread.join(timeout=10.0)
        self.logger.debug("Merge task submitter thread stopped")

    def is_alive(self) -> bool:
        """Check if submitter thread is still running."""
        return self.thread is not None and self.thread.is_alive()


# ============================================================================
# Thread 6: Merge Task Result Receiver
# ============================================================================

class MergeTaskReceiver:
    """
    Thread 6: Receive merge A task results via WebSocket and finalize workflows.

    For each completed merge task:
    1. Receive completion event from Scheduler A
    2. Update workflow state (set merge_complete_time)
    3. Push final workflow completion event to queue for Thread 4
    """

    def __init__(self,
                 scheduler_a_ws: str,
                 merge_task_ids: List[str],
                 workflow_states: Dict[str, WorkflowState],
                 completion_queue: Queue,
                 strategy: str = "default"):
        """
        Initialize merge task receiver.

        Args:
            scheduler_a_ws: WebSocket URL for Scheduler A
            merge_task_ids: List of all merge task IDs to subscribe to
            workflow_states: Shared workflow state dictionary
            completion_queue: Queue to push final workflow completion events
        """
        self.scheduler_a_ws = scheduler_a_ws
        self.merge_task_ids = merge_task_ids
        self.workflow_states = workflow_states
        self.completion_queue = completion_queue
        self.logger = logging.getLogger("Thread6.MergeTaskReceiver")

        # Tracking
        self.merge_results: List[TaskRecord] = []

        # Thread control
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Track workflow completion to avoid duplicates
        self.completed_workflows: Set[str] = set()
        self.raw_result_list: List[Dict[str, Any]] = []
        self.strategy_name = strategy

    async def _run_async(self):
        """Main WebSocket loop for receiving merge task results."""
        self.logger.debug(f"Connecting to Scheduler A WebSocket: {self.scheduler_a_ws}")

        try:
            async with websockets.connect(self.scheduler_a_ws) as websocket:
                # Subscribe to all merge task IDs
                subscribe_msg = {
                    "type": "subscribe",
                    "task_ids": self.merge_task_ids
                }
                await websocket.send(json.dumps(subscribe_msg))
                self.logger.debug(f"Subscribed to {len(self.merge_task_ids)} merge tasks")

                # Wait for acknowledgment
                ack = await websocket.recv()
                ack_data = json.loads(ack)
                self.logger.debug(f"Subscription confirmed: {ack_data.get('message')}")

                # Receive merge task results
                while self.running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)

                        if data["type"] == "result":
                            await self._handle_merge_result(data)
                        elif data["type"] == "error":
                            self.logger.error(f"WebSocket error: {data.get('error')}")

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        self.logger.error(f"Error receiving message: {e}, \ntraceback{traceback.format_exc()}")
                        break

        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")

    async def _handle_merge_result(self, data: Dict):
        """
        Handle merge task completion result and finalize workflow.

        Args:
            data: WebSocket result message
        """
        task_id = data["task_id"]
        status = data["status"]
        raw_result = data.get("result") or {}  # Handle None result
        complete_time = time.time()
        workflow_id = None

        # Extract workflow_id from task_id (format: task-A-{strategy}-workflow-{i:04d}-merge-{experiment_id})
        parts = task_id.split("-")
        if len(parts) >= 7 and parts[3] == "workflow" and parts[5] == "merge":
            workflow_id = f"wf-{parts[2]}-{parts[4]}-{parts[6]}"
            raw_result["workflow_id"] = workflow_id

        self.raw_result_list.append(raw_result)
        if not workflow_id:
            self.logger.error(f"Could not extract workflow_id from merge task_id: {task_id}")
            return

        # Extract sleep_time and execution_time from result
        result = data.get("result", {})
        sleep_time = result.get("sleep_time", 0.0)
        execution_time = data.get("execution_time", 0.0)

        # Get workflow state to extract merge_submit_time
        if workflow_id not in self.workflow_states:
            self.logger.error(f"Unknown workflow_id: {workflow_id}")
            return

        workflow = self.workflow_states[workflow_id]

        # Create merge task record WITH submit_time
        merge_record = TaskRecord(
            task_id=task_id,
            workflow_id=workflow_id,
            task_type="merge",
            sleep_time=sleep_time,
            exp_runtime=execution_time * 1000,  # Convert seconds to milliseconds
            submit_time=workflow.merge_submit_time,  # ✅ FIX: Add submit_time from workflow
            complete_time=complete_time,
            status=status,
            execution_time_ms=data.get("execution_time_ms", execution_time * 1000),
            result=data.get("result"),
            error=data.get("error")
        )
        self.merge_results.append(merge_record)

        # Mark merge task as complete and finalize workflow
        if status == "completed":
            workflow.mark_merge_task_complete(complete_time)

            # Enhanced logging for debugging
            self.logger.debug(
                f"Processing merge completion for {workflow_id}: "
                f"a_submit_time={workflow.a_submit_time}, "
                f"merge_complete_time={complete_time}, "
                f"already_completed={workflow_id in self.completed_workflows}"
            )

            # Push final workflow completion event
            # CRITICAL: Only mark as completed AFTER successful event push
            if workflow_id not in self.completed_workflows:
                success = self._push_workflow_completion(workflow)
                if success:
                    self.completed_workflows.add(workflow_id)
                    self.logger.debug(f"Workflow {workflow_id} FULLY COMPLETED "
                                   f"(merge task finished)")
                else:
                    self.logger.warning(
                        f"Failed to push completion event for workflow {workflow_id}, "
                        f"will retry if merge result received again"
                    )
        else:
            self.logger.warning(f"Merge task {task_id} failed for workflow {workflow_id}")

    def _push_workflow_completion(self, workflow: WorkflowState) -> bool:
        """
        Push final workflow completion event to queue for Thread 4.

        Args:
            workflow: Fully completed workflow state

        Returns:
            True if event was successfully pushed, False otherwise
        """
        if workflow.a_submit_time is None or workflow.merge_complete_time is None:
            self.logger.error(
                f"Incomplete workflow timing data for {workflow.workflow_id}: "
                f"a_submit_time={workflow.a_submit_time}, "
                f"merge_complete_time={workflow.merge_complete_time}"
            )
            return False

        event = WorkflowCompletionEvent(
            workflow_id=workflow.workflow_id,
            workflow_time=workflow.merge_complete_time - workflow.a_submit_time,
            a_task_id=workflow.a_task_id,
            b1_task_ids=workflow.b1_task_ids,
            b2_task_ids=workflow.b2_task_ids,
            merge_task_id=workflow.merge_task_id,
            total_b1_tasks=workflow.total_b1_tasks,
            total_b2_tasks=workflow.total_b2_tasks,
            a_submit_time=workflow.a_submit_time,
            a_complete_time=workflow.a_complete_time or 0.0,
            all_b1_complete_time=workflow.all_b1_complete_time or 0.0,
            all_b2_complete_time=workflow.all_b2_complete_time or 0.0,
            merge_submit_time=workflow.merge_submit_time or 0.0,
            merge_complete_time=workflow.merge_complete_time,
            workflow_complete_time=workflow.workflow_complete_time,
            is_warmup=workflow.is_warmup,
            is_target_for_stats=workflow.is_target_for_stats
        )

        self.completion_queue.put(event)
        return True

    def _run(self):
        """Thread entry point."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._run_async())
        except Exception as e:
            self.logger.error(f"Error in async loop: {e}")
        finally:
            try:
                # Cancel any pending tasks
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()
                # Wait for all tasks to be cancelled
                if pending:
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception as e:
                self.logger.error(f"Error cancelling tasks: {e}")
            finally:
                self.loop.close()

    def start(self):
        """Start the merge task receiver thread."""
        if self.thread is not None and self.thread.is_alive():
            self.logger.warning("Merge task receiver already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.logger.debug("Merge task receiver thread started")

    def stop(self):
        """Stop the merge task receiver thread."""
        os.makedirs(f"raw_results_{self.strategy_name}", exist_ok=True)
        with open(f"raw_results_{self.strategy_name}/merge_task_result.jsonl", "w") as f:
            for record in self.raw_result_list:
                f.write(f"{json.dumps(record)}\n")
        self.running = False
        self.thread.join(timeout=10.0)
        self.logger.debug("Merge task receiver thread stopped")

    def is_alive(self) -> bool:
        """Check if receiver thread is still running."""
        return self.thread is not None and self.thread.is_alive()


# ============================================================================
# Thread 7: Workflow Monitor
# ============================================================================

class WorkflowMonitor:
    """
    Thread 7: Monitor workflow completion and calculate statistics.

    Polls the completion queue from Thread 6 (merge task receiver) and:
    1. Receives workflow completion events
    2. Calculates per-workflow statistics
    3. Aggregates overall statistics
    4. Detects experiment completion
    """

    def __init__(self,
                 completion_queue: Queue,
                 expected_workflows: int,
                 target_workflows: int):
        """
        Initialize workflow monitor.

        Args:
            completion_queue: Queue to receive workflow completion events
            expected_workflows: Total number of workflows to expect
            target_workflows: Number of target workflows to complete before stopping
        """
        self.completion_queue = completion_queue
        self.expected_workflows = expected_workflows
        self.target_workflows = target_workflows
        self.logger = logging.getLogger("Thread4.WorkflowMonitor")

        # Tracking
        self.completed_workflows: List[WorkflowCompletionEvent] = []

        # Thread control
        self.thread: Optional[threading.Thread] = None
        self.running = False

    def _run(self):
        """Main monitoring loop."""
        self.logger.info(
            f"Starting workflow monitor (total: {self.expected_workflows}, "
            f"target for stats: {self.target_workflows})"
        )

        while self.running:
            try:
                # Poll queue with timeout
                event = self.completion_queue.get(timeout=0.5)
                self.completed_workflows.append(event)

                completed_count = len(self.completed_workflows)

                # Count target workflows (excluding warmup and non-target)
                target_completed = sum(1 for e in self.completed_workflows
                                      if not e.is_warmup and e.is_target_for_stats)

                # Enhanced progress logging
                if completed_count % 10 == 0 or target_completed == self.target_workflows:
                    percentage = (target_completed / self.target_workflows * 100) if self.target_workflows > 0 else 0
                    self.logger.info(
                        f"Target workflows completed: {target_completed}/{self.target_workflows} "
                        f"({percentage:.1f}%), total: {completed_count}/{self.expected_workflows}"
                    )

                # EARLY TERMINATION: Stop when target workflows are complete
                if target_completed >= self.target_workflows:
                    self.logger.info(
                        f"Target workflows completed! Stopping early. "
                        f"Target: {target_completed}/{self.target_workflows}, "
                        f"Total: {completed_count}/{self.expected_workflows}"
                    )
                    break

            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in workflow monitor: {e}")

        final_count = len(self.completed_workflows)
        final_target_count = sum(1 for e in self.completed_workflows
                                if not e.is_warmup and e.is_target_for_stats)

        self.logger.info(
            f"Workflow monitor stopped: {final_target_count}/{self.target_workflows} target workflows, "
            f"{final_count}/{self.expected_workflows} total workflows"
        )

    def start(self):
        """Start the monitor thread."""
        if self.thread is not None:
            self.logger.warning("Monitor already started")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run, name="Thread4-WorkflowMonitor")
        self.thread.start()
        self.logger.debug("Workflow monitor thread started")

    def stop(self):
        """Stop the monitor thread."""
        if self.thread is None:
            return

        self.running = False
        self.thread.join(timeout=10.0)
        self.logger.debug("Workflow monitor thread stopped")

    def is_alive(self) -> bool:
        """Check if monitor thread is still running."""
        return self.thread is not None and self.thread.is_alive()


# ============================================================================
# Utility Functions
# ============================================================================

def clear_scheduler_tasks(scheduler_url: str):
    """Clear all tasks from a scheduler."""
    logger = logging.getLogger("Utils")
    try:
        response = requests.post(f"{scheduler_url}/task/clear")
        response.raise_for_status()
        logger.debug(f"Cleared tasks from {scheduler_url}")
    except Exception as e:
        logger.error(f"Failed to clear tasks from {scheduler_url}: {e}")


def set_scheduling_strategy(scheduler_url: str, strategy: str):
    """Set the scheduling strategy for a scheduler."""
    logger = logging.getLogger("Utils")
    payload = {"strategy_name": strategy}
    if "strategy" == "probabilistic":
        payload["quantiles"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    try:
        response = requests.post(
            f"{scheduler_url}/strategy/set",
            json={"strategy_name": strategy}
        )
        response.raise_for_status()
        logger.debug(f"Set strategy to '{strategy}' on {scheduler_url}")
    except Exception as e:
        logger.error(f"Failed to set strategy on {scheduler_url}: {e}")


def clear_planner_timeline(planner_url: str = PLANNER_URL):
    """
    Clear the instance count timeline in the Planner service.

    This should be called at the start of each experiment to ensure
    clean timeline data for the current run.

    Args:
        planner_url: Planner service URL (default: PLANNER_URL constant)

    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger("Utils")
    try:
        response = requests.post(f"{planner_url}/timeline/clear", timeout=10.0)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            logger.info(f"Cleared timeline in Planner: {data.get('message', 'OK')}")
            return True
        else:
            logger.warning(f"Timeline clear returned success=False: {data}")
            return False
    except requests.exceptions.ConnectionError as e:
        logger.warning(f"Could not connect to Planner at {planner_url}: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to clear timeline from Planner: {e}")
        return False


def get_planner_timeline(planner_url: str = PLANNER_URL) -> Optional[Dict]:
    """
    Retrieve the instance count timeline from the Planner service.

    This should be called at the end of each experiment to collect
    the deployment history data.

    Args:
        planner_url: Planner service URL (default: PLANNER_URL constant)

    Returns:
        Dictionary with timeline data, or None if request failed
        Format: {
            "success": bool,
            "entry_count": int,
            "entries": List[Dict] - each entry has:
                - timestamp: float
                - timestamp_iso: str
                - event_type: str ("deploy_migration" or "auto_optimize")
                - instance_counts: Dict[model_id -> count]
                - total_instances: int
                - changes_count: int
                - success: bool
                - target_distribution: Optional[List[float]]
                - score: Optional[float]
        }
    """
    logger = logging.getLogger("Utils")
    try:
        response = requests.get(f"{planner_url}/timeline", timeout=10.0)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            entry_count = data.get("entry_count", 0)
            logger.info(f"Retrieved timeline from Planner: {entry_count} entries")
            return data
        else:
            logger.warning(f"Timeline retrieval returned success=False: {data}")
            return None
    except requests.exceptions.ConnectionError as e:
        logger.warning(f"Could not connect to Planner at {planner_url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to retrieve timeline from Planner: {e}")
        return None


def generate_task_ids(num_workflows: int, fanout_values: List[int], strategy: str, experiment_id: str) -> tuple[
    List[str], List[str], List[str], List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Pre-generate all task IDs for WebSocket subscription.

    Args:
        num_workflows: Number of workflows
        fanout_values: List of fanout values (number of B1/B2 task pairs per workflow)
        strategy: Scheduling strategy name
        experiment_id: Unique experiment identifier to prevent task ID collisions

    Returns:
        Tuple of (a_task_ids, merge_task_ids, all_b1_task_ids, all_b2_task_ids, b1_task_ids_by_workflow, b2_task_ids_by_workflow)
    """
    a_task_ids = []
    merge_task_ids = []
    all_b1_task_ids = []
    all_b2_task_ids = []
    b1_task_ids_by_workflow = {}
    b2_task_ids_by_workflow = {}

    for i in range(num_workflows):
        workflow_id = f"wf-{strategy}-{i:04d}-{experiment_id}"

        # A task ID with experiment_id suffix
        a_task_id = f"task-A-{strategy}-workflow-{i:04d}-A-{experiment_id}"
        a_task_ids.append(a_task_id)

        # Merge A task ID with experiment_id suffix
        merge_task_id = f"task-A-{strategy}-workflow-{i:04d}-merge-{experiment_id}"
        merge_task_ids.append(merge_task_id)

        # B1 and B2 task IDs for this workflow with experiment_id suffix
        n = fanout_values[i]
        b1_task_ids = []
        b2_task_ids = []
        for j in range(n):
            b1_task_id = f"task-B1-{strategy}-workflow-{i:04d}-B1-{j:02d}-{experiment_id}"
            b2_task_id = f"task-B2-{strategy}-workflow-{i:04d}-B2-{j:02d}-{experiment_id}"
            b1_task_ids.append(b1_task_id)
            b2_task_ids.append(b2_task_id)
            all_b1_task_ids.append(b1_task_id)
            all_b2_task_ids.append(b2_task_id)

        b1_task_ids_by_workflow[workflow_id] = b1_task_ids
        b2_task_ids_by_workflow[workflow_id] = b2_task_ids

    return a_task_ids, merge_task_ids, all_b1_task_ids, all_b2_task_ids, b1_task_ids_by_workflow, b2_task_ids_by_workflow


# ============================================================================
# Metrics Calculation
# ============================================================================

def merge_task_records(records: List[TaskRecord]) -> List[TaskRecord]:
    """
    Merge submit and completion records for the same task.

    When a task is submitted, one record is created with submit_time.
    When it completes, another record is created with complete_time/status.
    This function merges them into complete records with both timestamps.

    Args:
        records: List of task records (mix of submit and completion records)

    Returns:
        List of merged task records
    """
    # Build lookup maps
    submit_map = {}  # task_id -> submit record
    complete_map = {}  # task_id -> completion record

    for r in records:
        if r.submit_time is not None and r.status is None:
            # This is a submit record
            submit_map[r.task_id] = r
        elif r.complete_time is not None and r.status is not None:
            # This is a completion record
            complete_map[r.task_id] = r
        elif r.submit_time is not None and r.complete_time is not None:
            # This is already a merged record
            complete_map[r.task_id] = r

    # Merge records
    merged = []
    for task_id, complete_rec in complete_map.items():
        if complete_rec.submit_time is None and task_id in submit_map:
            # Need to merge submit_time from submit record
            submit_rec = submit_map[task_id]
            merged_rec = TaskRecord(
                task_id=complete_rec.task_id,
                workflow_id=complete_rec.workflow_id,
                task_type=complete_rec.task_type,
                sleep_time=complete_rec.sleep_time or submit_rec.sleep_time,
                exp_runtime=complete_rec.exp_runtime or submit_rec.exp_runtime,
                submit_time=submit_rec.submit_time,
                assigned_instance=submit_rec.assigned_instance,
                complete_time=complete_rec.complete_time,
                status=complete_rec.status,
                execution_time_ms=complete_rec.execution_time_ms,
                result=complete_rec.result,
                error=complete_rec.error,
                b_index=complete_rec.b_index,
                is_warmup=complete_rec.is_warmup or submit_rec.is_warmup
            )
            merged.append(merged_rec)
        else:
            # Already has submit_time or no submit record found
            merged.append(complete_rec)

    # Also include submit records that never completed
    for task_id, submit_rec in submit_map.items():
        if task_id not in complete_map:
            merged.append(submit_rec)

    return merged


def calculate_task_metrics(records: List[TaskRecord], task_type: str) -> Dict:
    """
    Calculate metrics for a list of task records.

    NOTE: Warmup tasks (is_warmup=True) are excluded from statistics.

    Args:
        records: List of task records
        task_type: "A" or "B"

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

    # First merge submit and completion records
    merged_records = merge_task_records(records)

    # Filter records by task type and exclude warmup tasks
    filtered = [r for r in merged_records if r.task_type == task_type and not r.is_warmup]
    num_warmup = sum(1 for r in merged_records if r.task_type == task_type and r.is_warmup)

    # Count statuses
    num_generated = len(filtered)
    num_submitted = sum(1 for r in filtered if r.submit_time is not None)
    completed = [r for r in filtered if r.status == "completed"]
    num_completed = len(completed)
    num_failed = sum(1 for r in filtered if r.status and r.status != "completed")

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

    NOTE: Warmup workflows (is_warmup=True) and non-target workflows
    (is_target_for_stats=False) are excluded from statistics.

    Args:
        completed_workflows: List of workflow completion events

    Returns:
        Dictionary of workflow metrics
    """
    if not completed_workflows:
        return {
            "num_completed": 0,
            "num_warmup": 0,
            "num_excluded": 0,
            "workflow_times": [],
            "avg_workflow_time": 0.0,
            "median_workflow_time": 0.0,
            "p50_workflow_time": 0.0,
            "p95_workflow_time": 0.0,
            "p99_workflow_time": 0.0,
            "fanout_distribution": {},
            "avg_fanout": 0.0
        }

    # Filter to only include workflows that are targets for statistics
    # Exclude: warmup workflows AND workflows not marked as targets (last 50%)
    actual_workflows = [e for e in completed_workflows if not e.is_warmup and e.is_target_for_stats]
    num_warmup = sum(1 for e in completed_workflows if e.is_warmup)
    num_excluded = sum(1 for e in completed_workflows if not e.is_warmup and not e.is_target_for_stats)

    if not actual_workflows:
        return {
            "num_completed": 0,
            "num_warmup": num_warmup,
            "num_excluded": num_excluded,
            "workflow_times": [],
            "avg_workflow_time": 0.0,
            "median_workflow_time": 0.0,
            "p50_workflow_time": 0.0,
            "p95_workflow_time": 0.0,
            "p99_workflow_time": 0.0,
            "fanout_distribution": {},
            "avg_fanout": 0.0
        }

    # Extract workflow times
    workflow_times = [event.workflow_time for event in actual_workflows]
    workflow_times_arr = np.array(workflow_times)

    # Extract fanout values (total B1/B2 task pairs per workflow)
    fanout_values = [event.total_b1_tasks for event in actual_workflows]  # B1 and B2 have same count
    fanout_arr = np.array(fanout_values)

    # Calculate fanout distribution
    unique_fanouts, counts = np.unique(fanout_arr, return_counts=True)
    fanout_distribution = {int(f): int(c) for f, c in zip(unique_fanouts, counts)}

    return {
        "num_completed": len(actual_workflows),
        "num_warmup": num_warmup,
        "num_excluded": num_excluded,
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
    print(f"  Completed:  {wf_metrics['num_completed']} (excl. {wf_metrics['num_warmup']} warmup, {wf_metrics['num_excluded']} tail)")
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


def print_metrics_summary_exp07(
    strategy: str,
    a_metrics: Dict,
    b1_metrics: Dict,
    b2_metrics: Dict,
    merge_metrics: Dict,
    wf_metrics: Dict
):
    """
    Print a summary of metrics for exp07 test run with B1/B2/merge split.

    Args:
        strategy: Strategy name
        a_metrics: A task metrics
        b1_metrics: B1 task metrics
        b2_metrics: B2 task metrics
        merge_metrics: Merge task metrics
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

    print("\nB1 Tasks:")
    print(f"  Generated:  {b1_metrics['num_generated']} (excl. {b1_metrics['num_warmup']} warmup)")
    print(f"  Submitted:  {b1_metrics['num_submitted']}")
    print(f"  Completed:  {b1_metrics['num_completed']}")
    print(f"  Failed:     {b1_metrics['num_failed']}")
    if b1_metrics['avg_completion_time'] > 0:
        print(f"  Avg time:   {b1_metrics['avg_completion_time']:.2f}s")
        print(f"  Median:     {b1_metrics['median_completion_time']:.2f}s")
        print(f"  P95:        {b1_metrics['p95_completion_time']:.2f}s")

    print("\nB2 Tasks:")
    print(f"  Generated:  {b2_metrics['num_generated']} (excl. {b2_metrics['num_warmup']} warmup)")
    print(f"  Submitted:  {b2_metrics['num_submitted']}")
    print(f"  Completed:  {b2_metrics['num_completed']}")
    print(f"  Failed:     {b2_metrics['num_failed']}")
    if b2_metrics['avg_completion_time'] > 0:
        print(f"  Avg time:   {b2_metrics['avg_completion_time']:.2f}s")
        print(f"  Median:     {b2_metrics['median_completion_time']:.2f}s")
        print(f"  P95:        {b2_metrics['p95_completion_time']:.2f}s")

    print("\nMerge Tasks:")
    print(f"  Generated:  {merge_metrics['num_generated']} (excl. {merge_metrics['num_warmup']} warmup)")
    print(f"  Submitted:  {merge_metrics['num_submitted']}")
    print(f"  Completed:  {merge_metrics['num_completed']}")
    print(f"  Failed:     {merge_metrics['num_failed']}")
    if merge_metrics['avg_completion_time'] > 0:
        print(f"  Avg time:   {merge_metrics['avg_completion_time']:.2f}s")
        print(f"  Median:     {merge_metrics['median_completion_time']:.2f}s")
        print(f"  P95:        {merge_metrics['p95_completion_time']:.2f}s")

    print("\nWorkflows:")
    print(f"  Completed:  {wf_metrics['num_completed']} (excl. {wf_metrics['num_warmup']} warmup, {wf_metrics['num_excluded']} tail)")
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


# ============================================================================
# Main Test Orchestration
# ============================================================================

def test_strategy_workflow(
    strategy: str,
    num_workflows: int,
    task_times_a: List[float],
    task_times_a2: List[float],
    task_times_b1: List[float],
    task_times_b2: List[float],
    fanout_values: List[int],
    qps_a: float,
    experiment_id: str,
    timeout_minutes: int = 10,
    gqps: Optional[float] = None,
    warmup_ratio: float = 0.0,
    warmup_task_times_a: Optional[List[float]] = None,
    warmup_task_times_a2: Optional[List[float]] = None,
    warmup_task_times_b1: Optional[List[float]] = None,
    warmup_task_times_b2: Optional[List[float]] = None,
    warmup_fanout_values: Optional[List[int]] = None,
    continuous_mode: bool = False,
    target_workflows: Optional[int] = None,
    metric_portion: float = 0.5,
    oracle_deployer: Optional['OracleDeployer'] = None,
    inter_arrival_times: Optional[np.ndarray] = None
) -> Dict:
    """
    Test a single scheduling strategy with dynamic workflow fanout (B1/B2 split).

    Args:
        strategy: Scheduling strategy ("min_time", "round_robin", "probabilistic")
        num_workflows: Number of workflows to test (actual workflows, not including warmup)
        task_times_a: List of task execution times for A tasks
        task_times_b1: List of task execution times for B1 tasks (slow peak)
        task_times_b2: List of task execution times for B2 tasks (fast peak)
        fanout_values: List of fanout values (number of B1/B2 task pairs per workflow)
        qps_a: Target QPS for A task submission
        experiment_id: Unique identifier for this experiment run (prevents task ID collisions)
        timeout_minutes: Maximum time to wait for completion
        gqps: Optional global QPS limit for all task submissions
        warmup_ratio: Ratio of warmup workflows (0.0-1.0)
        warmup_task_times_a: Pre-generated warmup A task times (optional)
        warmup_task_times_b1: Pre-generated warmup B1 task times (optional)
        warmup_task_times_b2: Pre-generated warmup B2 task times (optional)
        warmup_fanout_values: Pre-generated warmup fanout values (optional)
        continuous_mode: Enable continuous request mode (bool)
        target_workflows: Target number of workflows for continuous mode (optional)
        metric_portion: Portion of non-warmup workflows to include in statistics (0.0-1.0, default: 0.5)
        oracle_deployer: Optional OracleDeployer for predictive redeployment (will be signaled when task submission starts)
        inter_arrival_times: Optional pre-computed inter-arrival times (for OracleDeployer synchronization)

    Returns:
        Dictionary of test results
    """
    logger = logging.getLogger(f"Test.{strategy}")
    logger.debug(f"Starting test for strategy: {strategy}")

    # Step 1: Clear tasks from both schedulers and Planner timeline
    logger.info("Step 1: Clearing tasks from schedulers and Planner timeline")
    clear_scheduler_tasks(SCHEDULER_A_URL)
    clear_scheduler_tasks(SCHEDULER_B_URL)
    clear_planner_timeline(PLANNER_URL)
    time.sleep(1.0)

    # Step 2: Set scheduling strategy
    logger.info(f"Step 2: Setting strategy to '{strategy}'")
    set_scheduling_strategy(SCHEDULER_A_URL, strategy)
    set_scheduling_strategy(SCHEDULER_B_URL, strategy)
    time.sleep(0.5)

    # Step 2.5: Calculate warmup workflows
    num_warmup_workflows = int(num_workflows * warmup_ratio)
    total_workflows = num_warmup_workflows + num_workflows
    logger.info(f"Total workflows: {total_workflows} ({num_warmup_workflows} warmup + {num_workflows} actual)")
    logger.info(f"Will generate {total_workflows} A tasks, {total_workflows} B1 tasks, {total_workflows} B2 tasks, and {total_workflows} merge tasks")

    # Use pre-generated warmup fanout values or fall back to empty list
    if num_warmup_workflows > 0:
        if warmup_fanout_values is None:
            logger.warning("warmup_fanout_values not provided, using empty list")
            warmup_fanout_values = []
        all_fanout_values = list(warmup_fanout_values) + list(fanout_values)
    else:
        all_fanout_values = fanout_values

    # Step 3: Pre-generate all task IDs (B1 and B2 separately) with experiment_id
    logger.info("Step 3: Pre-generating task IDs")
    a_task_ids, merge_task_ids, all_b1_task_ids, all_b2_task_ids, b1_task_ids_by_workflow, b2_task_ids_by_workflow = generate_task_ids(
        total_workflows, all_fanout_values, strategy, experiment_id
    )
    logger.info(f"Generated {len(a_task_ids)} A task IDs ({num_warmup_workflows} warmup + {num_workflows} actual), "
               f"{len(merge_task_ids)} merge task IDs, {len(all_b1_task_ids)} B1 task IDs, {len(all_b2_task_ids)} B2 task IDs")

    # Step 4: Generate task data (A, B1, B2, merge)
    logger.info("Step 4: Generating task data")
    a_tasks: List[WorkflowTaskData] = []
    merge_tasks: List[WorkflowTaskData] = []
    b1_tasks_by_workflow: Dict[str, List[WorkflowTaskData]] = {}
    b2_tasks_by_b1: Dict[str, WorkflowTaskData] = {}  # Map B1 task_id -> B2 task data

    # Calculate mean runtime for each task type (for min_time strategy)
    # Combine warmup and actual task times for accurate mean calculation
    if strategy == "min_time":
        # Combine warmup and actual A1 times
        all_task_times_a = list(warmup_task_times_a) + list(task_times_a) if warmup_task_times_a is not None else list(task_times_a)
        mean_runtime_a = np.mean(all_task_times_a) * 1000  # Convert to ms

        # Combine warmup and actual A2 times
        all_task_times_a2 = list(warmup_task_times_a2) + list(task_times_a2) if warmup_task_times_a2 is not None else list(task_times_a2)
        mean_runtime_a2 = np.mean(all_task_times_a2) * 1000  # Convert to ms

        # Combine warmup and actual B1 times
        all_task_times_b1 = list(warmup_task_times_b1) + list(task_times_b1) if warmup_task_times_b1 is not None else list(task_times_b1)
        mean_runtime_b1 = np.mean(all_task_times_b1) * 1000  # Convert to ms

        # Combine warmup and actual B2 times
        all_task_times_b2 = list(warmup_task_times_b2) + list(task_times_b2) if warmup_task_times_b2 is not None else list(task_times_b2)
        mean_runtime_b2 = np.mean(all_task_times_b2) * 1000  # Convert to ms

        logger.info(f"Strategy is min_time: using mean runtimes - A1={mean_runtime_a:.2f}ms, A2={mean_runtime_a2:.2f}ms, B1={mean_runtime_b1:.2f}ms, B2={mean_runtime_b2:.2f}ms")
    else:
        # For other strategies, use actual task time
        mean_runtime_a = None
        mean_runtime_a2 = None
        mean_runtime_b1 = None
        mean_runtime_b2 = None
        logger.info(f"Strategy is {strategy}: using actual task times for exp_runtime")

    # Use pre-generated warmup task times or fall back to empty arrays
    if num_warmup_workflows > 0:
        if warmup_task_times_a is None or warmup_task_times_a2 is None or warmup_task_times_b1 is None or warmup_task_times_b2 is None:
            logger.warning("warmup_task_times not provided, using empty arrays")
            warmup_task_times_a = np.array([])
            warmup_task_times_a2 = np.array([])
            warmup_task_times_b1 = np.array([])
            warmup_task_times_b2 = np.array([])
        else:
            warmup_task_times_a = np.array(warmup_task_times_a)
            warmup_task_times_a2 = np.array(warmup_task_times_a2)
            warmup_task_times_b1 = np.array(warmup_task_times_b1)
            warmup_task_times_b2 = np.array(warmup_task_times_b2)
    else:
        warmup_task_times_a = np.array([])
        warmup_task_times_a2 = np.array([])
        warmup_task_times_b1 = np.array([])
        warmup_task_times_b2 = np.array([])

    # Generate all workflows (warmup first, then actual)
    for i in range(total_workflows):
        workflow_id = f"wf-{strategy}-{i:04d}-{experiment_id}"
        a_task_id = a_task_ids[i]
        merge_task_id = merge_task_ids[i]
        is_warmup = i < num_warmup_workflows

        # Determine task times based on warmup status
        if is_warmup:
            sleep_time_a = warmup_task_times_a[i]
            sleep_time_a2 = warmup_task_times_a2[i]
            fanout = all_fanout_values[i]
        else:
            actual_idx = i - num_warmup_workflows
            sleep_time_a = task_times_a[actual_idx]
            sleep_time_a2 = task_times_a2[actual_idx]
            fanout = fanout_values[actual_idx]

        # Create A task
        # Use mean runtime for min_time strategy, actual time for others
        exp_runtime_a = mean_runtime_a if strategy == "min_time" else sleep_time_a * 1000
        a_task = WorkflowTaskData(
            task_id=a_task_id,
            workflow_id=workflow_id,
            task_type="A",
            sleep_time=sleep_time_a,
            exp_runtime=exp_runtime_a,
            is_warmup=is_warmup
        )
        a_tasks.append(a_task)

        # Create merge A task using A2 time from external data
        # Use mean runtime for min_time strategy, actual time for others
        merge_sleep_time = sleep_time_a2
        exp_runtime_a2 = mean_runtime_a2 if strategy == "min_time" else merge_sleep_time * 1000
        merge_task = WorkflowTaskData(
            task_id=merge_task_id,
            workflow_id=workflow_id,
            task_type="A",  # Submitted to Scheduler A
            sleep_time=merge_sleep_time,
            exp_runtime=exp_runtime_a2,
            is_warmup=is_warmup
        )
        merge_tasks.append(merge_task)

        # Create B1 and B2 tasks for this workflow
        n = fanout
        b1_tasks = []
        b1_task_ids_wf = b1_task_ids_by_workflow[workflow_id]
        b2_task_ids_wf = b2_task_ids_by_workflow[workflow_id]

        for j in range(n):
            # Calculate index into task_times_b1 and task_times_b2
            if is_warmup:
                b_task_index = sum(all_fanout_values[:i]) + j
                sleep_time_b1 = warmup_task_times_b1[b_task_index]
                sleep_time_b2 = warmup_task_times_b2[b_task_index]
            else:
                actual_idx = i - num_warmup_workflows
                b_task_index = sum(fanout_values[:actual_idx]) + j
                sleep_time_b1 = task_times_b1[b_task_index]
                sleep_time_b2 = task_times_b2[b_task_index]

            # Create B1 task (slow peak)
            # Use mean runtime for min_time strategy, actual time for others
            exp_runtime_b1 = mean_runtime_b1 if strategy == "min_time" else sleep_time_b1 * 1000
            b1_task = WorkflowTaskData(
                task_id=b1_task_ids_wf[j],
                workflow_id=workflow_id,
                task_type="B1",
                sleep_time=sleep_time_b1,
                exp_runtime=exp_runtime_b1,
                b_index=j,  # Track pairing index
                is_warmup=is_warmup
            )
            b1_tasks.append(b1_task)

            # Create B2 task (fast peak) paired with this B1 task
            # Use mean runtime for min_time strategy, actual time for others
            exp_runtime_b2 = mean_runtime_b2 if strategy == "min_time" else sleep_time_b2 * 1000
            b2_task = WorkflowTaskData(
                task_id=b2_task_ids_wf[j],
                workflow_id=workflow_id,
                task_type="B2",
                sleep_time=sleep_time_b2,
                exp_runtime=exp_runtime_b2,
                b_index=j,  # Track pairing index
                is_warmup=is_warmup
            )
            # Map B1 task_id -> B2 task data for Thread 3 to look up
            b2_tasks_by_b1[b1_task_ids_wf[j]] = b2_task

        b1_tasks_by_workflow[workflow_id] = b1_tasks

    # Step 5: Initialize workflow states (with B1 and B2 separation)
    logger.info("Step 5: Initializing workflow states")

    # Calculate how many non-warmup workflows should be included in statistics
    # Default: first metric_portion (e.g., 50%) of non-warmup workflows are targets for statistics
    if continuous_mode and target_workflows is not None:
        # In continuous mode, use the specified target_workflows count
        stats_target_count = target_workflows
        logger.info(f"Continuous mode: marking first {stats_target_count} non-warmup workflows as targets")
    else:
        # In normal mode, use first metric_portion of non-warmup workflows
        stats_target_count = int(num_workflows * metric_portion)
        logger.info(f"Normal mode: marking first {metric_portion:.0%} ({stats_target_count}/{num_workflows}) of non-warmup workflows as targets for statistics")

    workflow_states: Dict[str, WorkflowState] = {}
    target_count = 0  # Count of non-warmup workflows marked as targets
    for i in range(total_workflows):
        workflow_id = f"wf-{strategy}-{i:04d}-{experiment_id}"
        is_warmup = i < num_warmup_workflows

        # Determine if this is a target workflow for statistics
        if is_warmup:
            # Warmup workflows are never targets for statistics
            is_target_for_stats = False
        else:
            # Mark first stats_target_count non-warmup workflows as targets
            if target_count < stats_target_count:
                is_target_for_stats = True
                target_count += 1
            else:
                is_target_for_stats = False  # Exclude last 50% (or overflow in continuous mode)

        workflow_states[workflow_id] = WorkflowState(
            workflow_id=workflow_id,
            strategy=strategy,
            a_task_id=a_task_ids[i],
            b1_task_ids=b1_task_ids_by_workflow[workflow_id],
            b2_task_ids=b2_task_ids_by_workflow[workflow_id],
            merge_task_id=merge_task_ids[i],
            total_b1_tasks=all_fanout_values[i],
            total_b2_tasks=all_fanout_values[i],
            is_warmup=is_warmup,
            is_target_for_stats=is_target_for_stats
        )

    # Step 6: Create queues and rate limiter (if using global QPS)
    logger.info("Step 6: Creating queues")
    merge_ready_queue = Queue()  # NEW: For B->merge transition
    completion_queue = Queue()   # For merge->monitor (final completion)
    rate_limiter = RateLimiter(gqps) if gqps is not None else None

    # Step 7: Start Thread 6 (Merge Task Receiver) - must start before merge tasks exist
    logger.info("Step 7: Starting Thread 6 (Merge Task Receiver)")
    merge_receiver = MergeTaskReceiver(
        scheduler_a_ws=SCHEDULER_A_WS,
        merge_task_ids=merge_task_ids,
        workflow_states=workflow_states,
        completion_queue=completion_queue,
        strategy=strategy
    )
    merge_receiver.start()
    time.sleep(2.0)  # Wait for WebSocket connection

    # Step 8: Start Thread 5 (Merge Task Submitter)
    logger.info("Step 8: Starting Thread 5 (Merge Task Submitter)")
    merge_submitter = MergeTaskSubmitter(
        scheduler_a_url=SCHEDULER_A_URL,
        merge_tasks=merge_tasks,
        workflow_states=workflow_states,
        merge_ready_queue=merge_ready_queue,
        strategy=strategy,
        rate_limiter=rate_limiter
    )
    merge_submitter.start()

    # Step 8.5: Start Thread 4 (B2 Task Receiver) - must start before B2 tasks are submitted
    logger.debug("Step 8.5: Starting Thread 4 (B2 Task Receiver)")
    b2_receiver = B2TaskReceiver(
        scheduler_b_ws=SCHEDULER_B_WS,
        b2_task_ids=all_b2_task_ids,
        workflow_states=workflow_states,
        merge_ready_queue=merge_ready_queue,
        strategy=strategy
    )
    b2_receiver.start()
    time.sleep(2.0)  # Wait for WebSocket connection

    # Step 9: Start Thread 3 (B1 Task Receiver) - must start before B1 tasks are submitted
    logger.info("Step 9: Starting Thread 3 (B1 Task Receiver)")
    b1_receiver = B1TaskReceiver(
        scheduler_b_ws=SCHEDULER_B_WS,
        scheduler_b_url=SCHEDULER_B_URL,
        b1_task_ids=all_b1_task_ids,
        b2_tasks_by_b1=b2_tasks_by_b1,
        workflow_states=workflow_states,
        rate_limiter=rate_limiter,
        strategy=strategy
    )
    b1_receiver.start()
    time.sleep(2.0)  # Wait for WebSocket connection

    # Step 10: Start Thread 7 (Workflow Monitor)
    logger.info("Step 10: Starting Thread 7 (Workflow Monitor)")
    # Monitor will stop when stats_target_count workflows (marked as is_target_for_stats=True) complete
    logger.info(
        f"Monitor configured: will stop after {stats_target_count} target workflows complete "
        f"(out of {total_workflows} total)"
    )
    monitor = WorkflowMonitor(
        completion_queue=completion_queue,
        expected_workflows=total_workflows,
        target_workflows=stats_target_count
    )
    monitor.start()

    # Step 11: Start Thread 2 (A Task Receiver + B1 Task Submitter)
    logger.info("Step 11: Starting Thread 2 (A Task Receiver + B1 Submitter)")
    a_receiver = ATaskReceiver(
        scheduler_a_ws=SCHEDULER_A_WS,
        scheduler_b_url=SCHEDULER_B_URL,
        a_task_ids=a_task_ids,
        b1_tasks_by_workflow=b1_tasks_by_workflow,
        workflow_states=workflow_states,
        rate_limiter=rate_limiter,
        strategy=strategy
    )
    a_receiver.start()
    time.sleep(2.0)  # Wait for WebSocket connection

    # Step 12: Start Thread 1 (A Task Submitter)
    logger.info("Step 12: Starting Thread 1 (A Task Submitter)")
    a_submitter = PoissonTaskSubmitter(
        scheduler_url=SCHEDULER_A_URL,
        tasks=a_tasks,
        qps=qps_a,
        workflow_states=workflow_states,
        strategy=strategy,
        rate_limiter=rate_limiter,
        inter_arrival_times=inter_arrival_times  # For OracleDeployer synchronization
    )
    a_submitter.start()

    # Signal oracle deployer that task submission has started
    if oracle_deployer is not None:
        oracle_deployer.signal_task_submission_started()

    # Step 13: Wait for A task submission to complete (with early stop support)
    logger.info("Step 13: Waiting for A task submission to complete (or early stop when target workflows done)")
    early_stopped = False
    while a_submitter.is_alive():
        # Check if monitor has stopped (target workflows completed)
        if not monitor.is_alive():
            logger.info("=" * 60)
            logger.info("EARLY STOP: Target workflows completed, stopping A task submission")
            logger.info("=" * 60)
            a_submitter.stop()
            early_stopped = True
            break
        time.sleep(0.5)

    if early_stopped:
        logger.info(f"A task submission early stopped after {len(a_submitter.submitted_tasks)}/{len(a_tasks)} tasks")
    else:
        logger.debug("A task submission complete")

    # Step 14: Wait for workflow completion with timeout
    logger.info(f"Step 14: Waiting for target workflows to complete (timeout: {timeout_minutes} minutes)")
    logger.info(f"Will stop when {stats_target_count} target workflows complete (out of {total_workflows} total)")
    timeout_seconds = timeout_minutes * 60
    start_wait = time.time()

    while monitor.is_alive() and (time.time() - start_wait) < timeout_seconds:
        time.sleep(1.0)
        elapsed = time.time() - start_wait
        if int(elapsed) % 10 == 0:
            target_completed = sum(1 for e in monitor.completed_workflows
                                  if not e.is_warmup and e.is_target_for_stats)
            logger.info(
                f"Waiting... {target_completed}/{stats_target_count} target workflows completed, "
                f"{len(monitor.completed_workflows)}/{total_workflows} total"
            )

    # Step 15: Stop all threads
    logger.info("Step 15: Stopping all threads")
    a_submitter.stop()
    a_receiver.stop()
    b1_receiver.stop()
    b2_receiver.stop()
    merge_submitter.stop()
    merge_receiver.stop()
    monitor.stop()

    # Step 15.0: Check if target workflows completed
    target_completed = sum(1 for e in monitor.completed_workflows
                          if not e.is_warmup and e.is_target_for_stats)

    if target_completed >= stats_target_count:
        logger.info(f"✓ Target workflows completed: {target_completed}/{stats_target_count}")
        logger.info(f"  Total workflows processed: {len(monitor.completed_workflows)}/{total_workflows}")
    else:
        logger.warning("\n" + "="*80)
        logger.warning(f"TIMEOUT: Target workflows incomplete ({target_completed}/{stats_target_count})")
        logger.warning("="*80)

        # Only attempt recovery for target workflows that didn't complete
        logger.warning("Possible causes:")
        logger.warning("  - A tasks failed to submit or complete")
        logger.warning("  - B1/B2 tasks failed")
        logger.warning("  - Merge tasks failed to submit or complete")
        logger.warning("  - Timeout occurred before target workflows could complete")
        logger.warning("="*80 + "\n")

    # Step 15.1: Diagnostic Summary - Task Submission Failures
    logger.debug("\n" + "="*80)
    logger.info("DIAGNOSTIC SUMMARY - Task Submission Failures")
    logger.debug("="*80)

    total_failed = a_submitter.failed_submissions + a_receiver.failed_b1_submissions
    if total_failed > 0:
        logger.error(f"⚠️  TOTAL TASK SUBMISSION FAILURES: {total_failed}")
        logger.error(f"   - A task failures: {a_submitter.failed_submissions}/{len(a_tasks)}")
        logger.error(f"   - B1 task failures: {a_receiver.failed_b1_submissions}")
        logger.error(f"⚠️  Expected missing workflows: ~{total_failed} (if all failures were A tasks)")
    else:
        logger.debug("✓ No task submission failures detected")

    logger.info(f"\nWorkflow Completion Status:")
    logger.info(f"   - Total workflows: {total_workflows} (submitted)")
    logger.info(f"   - Target workflows: {stats_target_count} (for statistics)")
    logger.info(f"   - Completed (total): {len(monitor.completed_workflows)} workflows")
    logger.info(f"   - Completed (target): {target_completed}/{stats_target_count} workflows")

    if target_completed < stats_target_count:
        missing_count = stats_target_count - target_completed
        logger.warning(f"\n⚠️  ANALYSIS: {missing_count} target workflows did not complete")
        if a_submitter.failed_submissions > 0:
            logger.warning(f"   - {a_submitter.failed_submissions} A task submission failures")
        if missing_count > a_submitter.failed_submissions:
            logger.warning(f"   - Additional failures may have occurred at B1/B2/merge stages")
            logger.warning(f"   - Check for B2 task failures, merge task failures, or race conditions")
    else:
        logger.info(f"✓ All target workflows completed successfully")

    logger.debug("="*80 + "\n")

    # Step 15.5: Continuous mode cleanup
    if continuous_mode:
        logger.debug("Step 15.5: Continuous mode cleanup")
        logger.debug("Waiting 5 seconds before force-clearing schedulers...")
        time.sleep(5.0)

        from common import force_clear_scheduler_tasks
        logger.debug("Force-clearing Scheduler A...")
        force_clear_scheduler_tasks(SCHEDULER_A_URL)
        logger.debug("Force-clearing Scheduler B...")
        force_clear_scheduler_tasks(SCHEDULER_B_URL)
        logger.debug("Schedulers cleared successfully")

    # Step 16: Collect results
    logger.info("Step 16: Collecting results")

    # Combine all task records (separate B1 and B2)
    all_a_records = a_submitter.submitted_tasks + a_receiver.a_results
    all_b1_records = a_receiver.b1_submitted + b1_receiver.b1_results
    all_b2_records = b1_receiver.b2_submitted + b2_receiver.b2_results
    all_merge_records = merge_receiver.merge_results

    # Calculate metrics (separate for B1 and B2)
    a_metrics = calculate_task_metrics(all_a_records, "A")
    b1_metrics = calculate_task_metrics(all_b1_records, "B1")
    b2_metrics = calculate_task_metrics(all_b2_records, "B2")
    merge_metrics = calculate_task_metrics(all_merge_records, "merge")
    wf_metrics = calculate_workflow_metrics(monitor.completed_workflows)

    # Print summary
    logger.debug("="*80)
    logger.info(f"EXPERIMENT RESULTS - Strategy: {strategy}")
    logger.debug("="*80)
    logger.debug(f"A Tasks: {a_metrics}")
    logger.debug(f"B1 Tasks: {b1_metrics}")
    logger.debug(f"B2 Tasks: {b2_metrics}")
    logger.debug(f"Merge Tasks: {merge_metrics}")
    logger.debug(f"Workflows: {wf_metrics}")
    logger.debug("="*80)

    # Print formatted summary
    if continuous_mode:
        from common import calculate_makespan, print_continuous_mode_summary
        makespan_metrics = calculate_makespan(monitor.completed_workflows)
        # For experiment 07 with B1/B2 split, pass all metrics separately
        print_continuous_mode_summary(strategy, makespan_metrics, a_metrics, b1_metrics, b2_metrics, merge_metrics, wf_metrics)
    else:
        # Use exp07-specific print function that supports B1/B2/merge split
        print_metrics_summary_exp07(strategy, a_metrics, b1_metrics, b2_metrics, merge_metrics, wf_metrics)

    # Calculate actual QPS
    actual_qps = 0.0
    if a_submitter.submission_end_time and a_submitter.submission_start_time:
        duration = a_submitter.submission_end_time - a_submitter.submission_start_time
        actual_qps = len(a_tasks) / duration if duration > 0 else 0.0

    # Step 17: Retrieve instance deployment timeline from Planner
    logger.info("Step 17: Retrieving instance deployment timeline from Planner")
    timeline_data = get_planner_timeline(PLANNER_URL)
    if timeline_data:
        logger.info(f"Retrieved {timeline_data.get('entry_count', 0)} timeline entries from Planner")
    else:
        logger.warning("Failed to retrieve timeline data from Planner")

    return {
        "strategy": strategy,
        "num_workflows": num_workflows,
        "target_qps": qps_a,
        "actual_qps": actual_qps,
        "a_tasks": a_metrics,
        "b1_tasks": b1_metrics,
        "b2_tasks": b2_metrics,
        "merge_tasks": merge_metrics,
        "workflows": wf_metrics,
        "submission_time": a_submitter.submission_end_time - a_submitter.submission_start_time
                          if a_submitter.submission_end_time and a_submitter.submission_start_time else 0.0,
        "planner_timeline": timeline_data  # Include timeline data in results
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main(num_workflows: int = 100, qps_a: float = 8.0, seed: int = 42,
         strategies: List[str] = None, gqps: Optional[float] = None, warmup_ratio: float = 0.0,
         continuous_mode: bool = False, metric_portion: float = 0.5, timeout_minutes: int = 20,
         oracle_deploy: bool = False, two_phase_mode: bool = False):
    """
    Main entry point for experiment 07.

    Args:
        num_workflows: Number of workflows to generate and execute per strategy
        qps_a: Target QPS for A task submission
        seed: Random seed for reproducibility
        strategies: List of strategies to test (default: all three)
        gqps: Optional global QPS limit for all task submissions
        warmup_ratio: Warmup task ratio (0.0-1.0)
        continuous_mode: Enable continuous request mode (2x workflows, track first num_workflows)
        metric_portion: Portion of non-warmup workflows to use for statistics (0.0-1.0, default: 0.5)
        timeout_minutes: Maximum time in minutes to wait for workflows to complete (default: 20)
        oracle_deploy: Enable oracle-based predictive redeployment
        two_phase_mode: Enable two-phase fanout (20% small, 80% large) with single migration
    """
    if strategies is None:
        strategies = ["min_time", "round_robin", "probabilistic"]

    # Generate unique experiment ID (not affected by seed)
    # This prevents task ID collisions when running multiple experiments
    import uuid
    experiment_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID

    logger = logging.getLogger("Main")
    logger.info("Starting Experiment 07: Multi-Model Workflow with B1/B2 Split and Merge")
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Configuration: {num_workflows} workflows, QPS={qps_a}, seed={seed}")
    if gqps is not None:
        logger.debug(f"Global QPS: {gqps}")
    if warmup_ratio > 0:
        logger.debug(f"Warmup ratio: {warmup_ratio}")
    logger.info(f"Strategies to test: {', '.join(strategies)}")

    # Experiment parameters
    if continuous_mode:
        NUM_WORKFLOWS = 2 * num_workflows  # Generate 2x workflows in continuous mode
        TARGET_WORKFLOWS = num_workflows  # Track first num_workflows
        logger.info(f"CONTINUOUS MODE: Generating {NUM_WORKFLOWS} workflows, tracking first {TARGET_WORKFLOWS}")
    else:
        NUM_WORKFLOWS = num_workflows
        TARGET_WORKFLOWS = None

    QPS_A = qps_a
    SEED = seed

    # Generate workloads using four normal distributions
    logger.debug(f"Generating workloads using four normal distributions...")

    # Generate workflow workload using four normal distributions
    workflow_workload, workflow_config = generate_workflow_from_traces(
        num_workflows=NUM_WORKFLOWS,
        seed=SEED,
        use_pattern_fanout=True  # Enable pattern-based fanout (25% segments alternating N(6, 0.6²) and N(14, 0.6²))
    )
    print_workflow_stats(workflow_workload)

    # Validate fanout pattern distribution
    validate_fanout_pattern(workflow_workload.fanout_values, NUM_WORKFLOWS)

    logger.debug(f"Generated {len(workflow_workload.a1_times)} workflows from traces")

    # Extract individual task time lists and fanout values
    # A1 (boot) times are used for the initial A task submission
    # A2 (summary) times are used for the merge A task submission
    task_times_a = workflow_workload.a1_times
    task_times_a2 = workflow_workload.a2_times
    fanout_values = workflow_workload.fanout_values

    # Two-phase fanout constants (defined here for use in warmup too)
    SMALL_FANOUT_MEAN = 6
    LARGE_FANOUT_MEAN = 14
    FANOUT_STD = 0.6
    two_phase_split_index = 0  # Will be set if two_phase_mode is enabled

    # Two-phase fanout mode: override fanout values
    # Pattern: first 20% = small fanout (mean=6), remaining 80% = large fanout (mean=14)
    if two_phase_mode:
        logger.info("TWO-PHASE MODE: Generating custom fanout pattern (20% small, 80% large)")
        rng = np.random.RandomState(SEED + 7777)

        # Calculate split point (first 20% use small fanout)
        small_fanout_count = int(NUM_WORKFLOWS * 0.2)
        large_fanout_count = NUM_WORKFLOWS - small_fanout_count

        small_fanouts = np.clip(
            rng.normal(SMALL_FANOUT_MEAN, FANOUT_STD, small_fanout_count).astype(int),
            5, 15
        ).tolist()
        large_fanouts = np.clip(
            rng.normal(LARGE_FANOUT_MEAN, FANOUT_STD, large_fanout_count).astype(int),
            5, 15
        ).tolist()

        fanout_values = small_fanouts + large_fanouts
        logger.info(f"Two-phase fanout: {small_fanout_count} small (mean={np.mean(small_fanouts):.1f}), "
                   f"{large_fanout_count} large (mean={np.mean(large_fanouts):.1f})")

        # Store the split point for OracleDeployer
        two_phase_split_index = small_fanout_count
        logger.info(f"Phase transition at workflow index {two_phase_split_index}")

        # IMPORTANT: Regenerate B task times to match new fanout values
        # B1 (query): N(3, 0.3²) - mean=3ms, std=0.3ms
        # B2 (criteria): N(2, 0.2²) - mean=2ms, std=0.2ms
        logger.info("TWO-PHASE MODE: Regenerating B task times to match new fanout values")
        b_rng = np.random.RandomState(SEED + 7779)  # Different seed for B task times
        total_b_tasks = sum(fanout_values)
        b1_times_pool = np.abs(b_rng.normal(3.0, 0.3, total_b_tasks)).tolist()  # abs to avoid negative
        b2_times_pool = np.abs(b_rng.normal(2.0, 0.2, total_b_tasks)).tolist()

        # Rebuild B1/B2 times as nested lists (per workflow)
        workflow_b1_times = []
        workflow_b2_times = []
        b_idx = 0
        for fanout in fanout_values:
            workflow_b1_times.append(b1_times_pool[b_idx:b_idx + fanout])
            workflow_b2_times.append(b2_times_pool[b_idx:b_idx + fanout])
            b_idx += fanout

        # Override workflow_workload's B times
        workflow_workload.b1_times = workflow_b1_times
        workflow_workload.b2_times = workflow_b2_times
        logger.info(f"Regenerated {total_b_tasks} B1 and B2 task times for two-phase mode")

    # Flatten B1 and B2 task times for the task time pools
    # These will be indexed by the b_task_index when submitting
    task_times_b1 = [t for workflow_b1 in workflow_workload.b1_times for t in workflow_b1]
    task_times_b2 = [t for workflow_b2 in workflow_workload.b2_times for t in workflow_b2]

    logger.debug(f"Extracted {len(task_times_a)} A1 task times (initial A tasks)")
    logger.debug(f"Extracted {len(task_times_a2)} A2 task times (merge A tasks)")
    logger.debug(f"Extracted {len(task_times_b1)} B1 task times (query tasks)")
    logger.debug(f"Extracted {len(task_times_b2)} B2 task times (criteria tasks)")
    logger.debug(f"Fanout distribution: min={min(fanout_values)}, max={max(fanout_values)}, mean={np.mean(fanout_values):.2f}")

    # Generate warmup data once for all strategies (if warmup is enabled)
    num_warmup_workflows = int(NUM_WORKFLOWS * warmup_ratio)
    if num_warmup_workflows > 0:
        logger.debug(f"Generating {num_warmup_workflows} warmup workflows from traces (warmup_ratio={warmup_ratio:.1%})")

        # Generate warmup workflow using four normal distributions with a different seed
        warmup_workflow, warmup_config = generate_workflow_from_traces(
            num_workflows=num_warmup_workflows,
            seed=SEED + 1000  # Different seed for warmup data
        )

        # Extract warmup task times and fanout values
        warmup_task_times_a = warmup_workflow.a1_times
        warmup_task_times_a2 = warmup_workflow.a2_times
        warmup_fanout_values = warmup_workflow.fanout_values

        logger.debug(f"Generated {len(warmup_task_times_a)} warmup A1 task times from traces")
        logger.debug(f"Generated {len(warmup_task_times_a2)} warmup A2 task times from traces")

        # In two-phase mode, override warmup fanout to use small fanout (mean=6)
        if two_phase_mode:
            logger.info("TWO-PHASE MODE: Overriding warmup fanout to use small fanout (mean=6)")
            warmup_rng = np.random.RandomState(SEED + 7778)  # Different seed from main fanout
            warmup_fanout_values = np.clip(
                warmup_rng.normal(SMALL_FANOUT_MEAN, FANOUT_STD, num_warmup_workflows).astype(int),
                5, 15
            ).tolist()
            logger.info(f"Warmup fanout (two-phase): mean={np.mean(warmup_fanout_values):.1f}")

            # IMPORTANT: Regenerate warmup B task times to match new fanout values
            logger.info("TWO-PHASE MODE: Regenerating warmup B task times to match new fanout values")
            warmup_b_rng = np.random.RandomState(SEED + 7780)  # Different seed for warmup B task times
            warmup_total_b_tasks = sum(warmup_fanout_values)
            warmup_b1_pool = np.abs(warmup_b_rng.normal(3.0, 0.3, warmup_total_b_tasks)).tolist()
            warmup_b2_pool = np.abs(warmup_b_rng.normal(2.0, 0.2, warmup_total_b_tasks)).tolist()

            # Rebuild warmup B1/B2 times as nested lists
            warmup_workflow_b1_times = []
            warmup_workflow_b2_times = []
            warmup_b_idx = 0
            for fanout in warmup_fanout_values:
                warmup_workflow_b1_times.append(warmup_b1_pool[warmup_b_idx:warmup_b_idx + fanout])
                warmup_workflow_b2_times.append(warmup_b2_pool[warmup_b_idx:warmup_b_idx + fanout])
                warmup_b_idx += fanout

            warmup_workflow.b1_times = warmup_workflow_b1_times
            warmup_workflow.b2_times = warmup_workflow_b2_times
            logger.info(f"Regenerated {warmup_total_b_tasks} warmup B1 and B2 task times for two-phase mode")

        # Flatten warmup B1 and B2 task times (after potential override)
        warmup_task_times_b1 = [t for workflow_b1 in warmup_workflow.b1_times for t in workflow_b1]
        warmup_task_times_b2 = [t for workflow_b2 in warmup_workflow.b2_times for t in workflow_b2]

        logger.debug(f"Generated {len(warmup_task_times_b1)} warmup B1 task times")
        logger.debug(f"Generated {len(warmup_task_times_b2)} warmup B2 task times")
        logger.debug(f"Warmup fanout distribution: min={min(warmup_fanout_values)}, max={max(warmup_fanout_values)}, mean={np.mean(warmup_fanout_values):.2f}")
    else:
        warmup_fanout_values = None
        warmup_task_times_a = None
        warmup_task_times_a2 = None
        warmup_task_times_b1 = None
        warmup_task_times_b2 = None
        logger.info("No warmup workflows (warmup_ratio=0.0)")

    # Run tests for specified strategies
    all_results = []

    # Query total instances if oracle deploy is enabled
    total_instances = 0
    inter_arrival_times = None  # Pre-computed inter-arrival times for oracle sync
    combined_fanout_values = None  # Combined fanout values for oracle sync
    phase1_qps = None
    phase2_qps = None
    qps_transition_index = 0
    if oracle_deploy:
        logger.info("Oracle deploy enabled, querying total instances from schedulers...")
        try:
            resp_a = requests.get(f"{SCHEDULER_A_URL}/health", timeout=5.0)
            resp_b = requests.get(f"{SCHEDULER_B_URL}/health", timeout=5.0)
            if resp_a.ok and resp_b.ok:
                instances_a = resp_a.json().get("stats", {}).get("total_instances", 0)
                instances_b = resp_b.json().get("stats", {}).get("total_instances", 0)
                total_instances = instances_a + instances_b
                logger.info(f"Total instances: {total_instances} (A: {instances_a}, B: {instances_b})")

                # Create combined fanout values (warmup + actual) for OracleDeployer
                if warmup_fanout_values is not None:
                    combined_fanout_values = list(warmup_fanout_values) + list(fanout_values)
                else:
                    combined_fanout_values = list(fanout_values)
                logger.info(f"Combined fanout values: {len(combined_fanout_values)} workflows")

                # In two-phase mode, use HARDCODED optimal QPS values
                # These are pre-calculated based on actual trace data and 48 instances
                # User-provided QPS values are IGNORED in two-phase mode
                if two_phase_mode:
                    logger.info("=" * 60)
                    logger.info("TWO-PHASE MODE: Using HARDCODED optimal QPS values")
                    logger.info("=" * 60)
                    logger.warning(f"Ignoring user-provided QPS={QPS_A:.2f}")

                    # ================================================================
                    # HARDCODED QPS VALUES
                    # Pre-calculated based on actual workload_generator.py data:
                    #   - 48 total instances
                    #   - 15% B model bias
                    #   - 15% overload factor
                    #
                    # Task durations (from workload_generator.py):
                    #   A1 (boot): N(30, 2²) = 30 seconds
                    #   A2 (summary): N(25, 2²) = 25 seconds
                    #   B1 (query): N(3, 0.3²) = 3 seconds
                    #   B2 (criteria): N(2, 0.2²) = 2 seconds
                    #
                    # Total A work per workflow = 55 seconds (A is always bottleneck)
                    # Total B work per B task = 5 seconds
                    #
                    # Phase 1 (small fanout = 6):
                    #   Instance allocation: A=24, B=24
                    #   A capacity: 24/55 = 0.436 wf/s
                    #   Target QPS (115% overload): 0.50 wf/s
                    #
                    # Phase 2 (large fanout = 14):
                    #   Instance allocation: A=14, B=34
                    #   A capacity: 14/55 = 0.255 wf/s
                    #   Target QPS (115% overload): 0.29 wf/s
                    # ================================================================
                    HARDCODED_PHASE1_QPS = 0.50  # 1 workflow every 2.0 seconds
                    HARDCODED_PHASE2_QPS = 0.29  # 1 workflow every 3.4 seconds

                    phase1_qps = HARDCODED_PHASE1_QPS
                    phase2_qps = HARDCODED_PHASE2_QPS

                    # Phase transition index = warmup + small_fanout_count (first 20%)
                    qps_transition_index = num_warmup_workflows + two_phase_split_index

                    logger.info(f"Phase 1 (fanout=6):  QPS={phase1_qps:.2f} wf/s (interval={1/phase1_qps:.2f}s)")
                    logger.info(f"Phase 2 (fanout=14): QPS={phase2_qps:.2f} wf/s (interval={1/phase2_qps:.2f}s)")
                    logger.info(f"QPS transition at workflow #{qps_transition_index}")
                    logger.info("=" * 60)

                # Pre-compute inter-arrival times with phase-aware QPS
                rng = np.random.RandomState(SEED + 9999)
                total_tasks = NUM_WORKFLOWS + num_warmup_workflows

                if two_phase_mode and phase1_qps and phase2_qps:
                    # Generate inter-arrival times with different QPS for each phase
                    inter_arrival_times = []
                    for i in range(total_tasks):
                        if i < qps_transition_index:
                            qps = phase1_qps
                        else:
                            qps = phase2_qps
                        inter_arrival_times.append(rng.exponential(1.0 / qps))
                    inter_arrival_times = np.array(inter_arrival_times)
                    logger.info(f"Pre-computed {len(inter_arrival_times)} inter-arrival times "
                               f"(phase1: {qps_transition_index} @ {phase1_qps:.2f} QPS, "
                               f"phase2: {total_tasks - qps_transition_index} @ {phase2_qps:.2f} QPS)")
                else:
                    # Single QPS for entire experiment
                    inter_arrival_times = rng.exponential(1.0 / QPS_A, total_tasks)
                    logger.info(f"Pre-computed {len(inter_arrival_times)} inter-arrival times @ {QPS_A:.2f} QPS")

            else:
                logger.error("Failed to query scheduler health, disabling oracle deploy")
                oracle_deploy = False
        except Exception as e:
            logger.error(f"Failed to query schedulers: {e}, disabling oracle deploy")
            oracle_deploy = False

    for strategy in strategies:
        logger.info(f"\n{'=' * 80}\nTesting strategy: {strategy}\n{'=' * 80}")

        # Create and start OracleDeployer for this strategy run
        oracle_deployer = None
        if oracle_deploy:
            logger.info("Creating OracleDeployer for predictive redeployment...")
            oracle_deployer = OracleDeployer(
                planner_url=PLANNER_URL,
                scheduler_a_url=SCHEDULER_A_URL,
                scheduler_b_url=SCHEDULER_B_URL,
                total_instances=total_instances,
                fanout_values=combined_fanout_values,  # Combined warmup + actual fanout
                inter_arrival_times=inter_arrival_times,  # Use pre-computed times
                window_size_s=5.0,
                lookahead_s=3.0,
                min_change_threshold=2,
                two_phase_mode=two_phase_mode,
                two_phase_split_index=two_phase_split_index,
                num_warmup_workflows=num_warmup_workflows
            )
            oracle_deployer.start()

        try:
            results = test_strategy_workflow(
                strategy=strategy,
                num_workflows=NUM_WORKFLOWS,
                task_times_a=task_times_a,
                task_times_a2=task_times_a2,
                task_times_b1=task_times_b1,
                task_times_b2=task_times_b2,
                fanout_values=fanout_values,
                qps_a=QPS_A,
                experiment_id=experiment_id,
                timeout_minutes=timeout_minutes,
                gqps=gqps,
                warmup_ratio=warmup_ratio,
                warmup_task_times_a=warmup_task_times_a,
                warmup_task_times_a2=warmup_task_times_a2,
                warmup_task_times_b1=warmup_task_times_b1,
                warmup_task_times_b2=warmup_task_times_b2,
                warmup_fanout_values=warmup_fanout_values,
                continuous_mode=continuous_mode,
                target_workflows=TARGET_WORKFLOWS,
                metric_portion=metric_portion,
                oracle_deployer=oracle_deployer,
                inter_arrival_times=inter_arrival_times  # Pass to PoissonTaskSubmitter
            )
            all_results.append(results)
        finally:
            # Stop OracleDeployer after strategy completes
            if oracle_deployer:
                oracle_deployer.stop()

        # Brief pause between strategies
        time.sleep(5.0)

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/results_workflow_b1b2_{timestamp}.json"

    output_data = {
        "experiment": "07.multi_model_workflow_b1b2_merge",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_workflows": NUM_WORKFLOWS,
            "qps_a": QPS_A,
            "seed": SEED,
            "workflow_config": asdict(workflow_config),
            "warmup_ratio": warmup_ratio,
            "gqps": gqps,
            "continuous_mode": continuous_mode
        },
        "results": all_results
    }
    
    if not os.path.exists("results"):
        os.makedirs("results", exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")

    # Print comparison table
    print("\n" + "=" * 100)
    print("Strategy Comparison")
    print("=" * 100)
    print(f"{'Strategy':<15} {'A Avg (s)':<12} {'B1 Avg (s)':<12} {'B2 Avg (s)':<12} {'WF Avg (s)':<12} {'WF P95 (s)':<12} {'Completed':<12}")
    print("-" * 100)

    for result in all_results:
        strategy = result['strategy']
        a_avg = result['a_tasks']['avg_completion_time']
        b1_avg = result['b1_tasks']['avg_completion_time']
        b2_avg = result['b2_tasks']['avg_completion_time']
        wf_avg = result['workflows']['avg_workflow_time']
        wf_p95 = result['workflows']['p95_workflow_time']
        wf_completed = result['workflows']['num_completed']

        print(f"{strategy:<15} {a_avg:<12.2f} {b1_avg:<12.2f} {b2_avg:<12.2f} {wf_avg:<12.2f} {wf_p95:<12.2f} {wf_completed:<12}")

    print("=" * 100)
    logger.info("Experiment complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment 04: Multi-Model Workflow with Dynamic Fanout (1-to-n)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--num-workflows",
        type=int,
        default=100,
        help="Number of workflows to generate and execute per strategy"
    )

    parser.add_argument(
        "--qps",
        type=float,
        default=8.0,
        help="Target queries per second (QPS) for A task submission"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["probabilistic", "random", "round_robin", "min_time", "po2"],
        choices=["min_time", "round_robin", "probabilistic", "random", "po2"],
        help="Scheduling strategies to test"
    )

    parser.add_argument(
        "--gqps",
        type=float,
        default=None,
        help="Global QPS limit for both A and B task submissions (overrides --qps if set)"
    )

    parser.add_argument(
        "--warmup",
        type=float,
        default=0.0,
        help="Warmup task ratio (0.0-1.0). E.g., 0.2 means 20%% warmup tasks before actual workload. Warmup tasks are excluded from statistics."
    )

    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Enable continuous request mode. In this mode, tasks are submitted continuously at target QPS until warmup + num_workflows tasks complete. Statistics are calculated only for the first num_workflows (excluding warmup)."
    )

    parser.add_argument(
        "--metric-portion",
        type=float,
        default=0.5,
        help="Portion of non-warmup workflows to include in final statistics (0.0-1.0, default: 0.5). For example, 0.5 means only the first 50%% of workflows are used for metrics calculation."
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Maximum time in minutes to wait for workflows to complete (default: 20)"
    )

    parser.add_argument(
        "--oracle-deploy",
        action="store_true",
        help="Enable oracle-based predictive redeployment. Uses pre-generated workflow data to predict future load and proactively trigger migrations BEFORE load changes occur."
    )

    parser.add_argument(
        "--two-phase",
        action="store_true",
        dest="two_phase",
        default=True,
        help="(DEFAULT=ON) Enable two-phase fanout mode: warmup+first 20%% use small fanout (mean=6), remaining 80%% use large fanout (mean=14). Oracle performs ONE migration after 5 large-fanout tasks."
    )
    parser.add_argument(
        "--no-two-phase",
        action="store_false",
        dest="two_phase",
        help="Disable two-phase fanout mode, use uniform fanout distribution instead."
    )

    args = parser.parse_args()

    main(
        num_workflows=args.num_workflows,
        qps_a=args.qps,
        seed=args.seed,
        strategies=args.strategies,
        gqps=args.gqps,
        warmup_ratio=args.warmup,
        continuous_mode=args.continuous,
        metric_portion=args.metric_portion,
        timeout_minutes=args.timeout,
        oracle_deploy=args.oracle_deploy,
        two_phase_mode=args.two_phase
    )
