#!/usr/bin/env python3
"""
Experiment 08: Multi-Model Workflow with Dynamic Instance Migration

This experiment extends Experiment 04 by adding dynamic instance migration
between schedulers during phase transitions.

Key Features:
- Three phases with different fanout: n=3, 8, 1
- Dynamic instance distribution matching A:B task ratio
- Safe instance migration using drain/remove API
- Continuous task submission during migration
- Per-phase metrics collection

Author: Claude Code
Date: 2025-11-03
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
import websockets
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from migration_controller import MigrationController, PhaseConfig
from workload_generator import generate_bimodal_distribution, generate_b_task_bimodal_distribution


# ============================================================================
# Configuration
# ============================================================================

SCHEDULER_A_URL = "http://localhost:8100"
SCHEDULER_B_URL = "http://localhost:8200"

# Default phase configs (16 instances total)
# Can be overridden via --phase-configs command line argument
DEFAULT_PHASE_CONFIGS = [
    PhaseConfig(phase_id=1, fanout=3, scheduler_a_instances=4, scheduler_b_instances=12, num_workflows=0),
    PhaseConfig(phase_id=2, fanout=8, scheduler_a_instances=2, scheduler_b_instances=14, num_workflows=0),
    PhaseConfig(phase_id=3, fanout=1, scheduler_a_instances=8, scheduler_b_instances=8, num_workflows=0),
]

# Global variable to hold the current phase configs
PHASE_CONFIGS = DEFAULT_PHASE_CONFIGS.copy()


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
        import threading
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
class WorkflowState:
    """Tracks the state of a single workflow."""
    workflow_id: str
    phase: int
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

    def mark_b_task_complete(self, b_task_id: str, complete_time: float) -> bool:
        """Mark a B task as complete. Returns True if workflow is now complete."""
        if b_task_id not in self.b_complete_times:
            self.b_complete_times[b_task_id] = complete_time
            self.completed_b_tasks += 1

            if self.is_complete():
                self.workflow_complete_time = max(self.b_complete_times.values())
                return True
        return False

    def is_complete(self) -> bool:
        """Check if workflow is complete."""
        return self.completed_b_tasks == self.total_b_tasks

    def get_latency_ms(self) -> Optional[float]:
        """Calculate total workflow latency in milliseconds."""
        if self.a_submit_time and self.workflow_complete_time:
            return (self.workflow_complete_time - self.a_submit_time) * 1000
        return None


@dataclass
class PhaseResults:
    """Results for a single phase."""
    phase_id: int
    fanout: int
    num_workflows: int
    workflows_completed: int = 0

    # Timing
    phase_start_time: Optional[float] = None
    phase_end_time: Optional[float] = None

    # Workflow latencies
    workflow_latencies: List[float] = field(default_factory=list)

    def add_workflow_latency(self, latency_ms: float):
        """Add a workflow latency."""
        self.workflow_latencies.append(latency_ms)
        self.workflows_completed += 1

    def get_statistics(self) -> Dict:
        """Calculate phase statistics."""
        if not self.workflow_latencies:
            return {}

        latencies = sorted(self.workflow_latencies)
        return {
            "phase_id": self.phase_id,
            "fanout": self.fanout,
            "num_workflows": self.num_workflows,
            "completed": self.workflows_completed,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": latencies[0],
            "max_latency_ms": latencies[-1],
            "p50_latency_ms": latencies[len(latencies) // 2],
            "p90_latency_ms": latencies[int(len(latencies) * 0.9)],
            "p99_latency_ms": latencies[int(len(latencies) * 0.99)],
        }


# ============================================================================
# Task Result Monitor
# ============================================================================

class TaskResultMonitor:
    """Monitors task completion via WebSocket connections to schedulers."""

    def __init__(self, scheduler_a_url: str, scheduler_b_url: str, workflow_states: Dict[str, WorkflowState]):
        self.scheduler_a_ws_url = scheduler_a_url.replace("http://", "ws://") + "/task/get_result"
        self.scheduler_b_ws_url = scheduler_b_url.replace("http://", "ws://") + "/task/get_result"
        self.workflow_states = workflow_states

        self.ws_a: Optional[websockets.WebSocketClientProtocol] = None
        self.ws_b: Optional[websockets.WebSocketClientProtocol] = None

        # Track completed tasks
        self.completed_a_tasks = set()
        self.completed_b_tasks = set()

        # Track completed workflows
        self.completed_workflows = set()

    async def connect_and_subscribe(self, a_task_ids: List[str], b_task_ids: List[str]):
        """Connect to both schedulers and subscribe to task IDs."""
        try:
            # Connect to Scheduler A
            logger.info(f"Connecting to Scheduler A WebSocket: {self.scheduler_a_ws_url}")
            self.ws_a = await websockets.connect(self.scheduler_a_ws_url)

            # Subscribe to A tasks
            subscribe_msg_a = {
                "type": "subscribe",
                "task_ids": a_task_ids
            }
            await self.ws_a.send(json.dumps(subscribe_msg_a))
            ack_a = await self.ws_a.recv()
            logger.info(f"Scheduler A: {json.loads(ack_a)['message']}")

            # Connect to Scheduler B
            logger.info(f"Connecting to Scheduler B WebSocket: {self.scheduler_b_ws_url}")
            self.ws_b = await websockets.connect(self.scheduler_b_ws_url)

            # Subscribe to B tasks
            subscribe_msg_b = {
                "type": "subscribe",
                "task_ids": b_task_ids
            }
            await self.ws_b.send(json.dumps(subscribe_msg_b))
            ack_b = await self.ws_b.recv()
            logger.info(f"Scheduler B: {json.loads(ack_b)['message']}")

        except Exception as e:
            logger.error(f"Failed to connect/subscribe to WebSocket: {e}")
            raise

    async def listen_for_results(self):
        """Listen for task results from both schedulers."""
        async def listen_scheduler(ws, scheduler_name):
            try:
                while True:
                    message = await ws.recv()
                    data = json.loads(message)

                    if data["type"] == "result":
                        await self.handle_task_result(data, scheduler_name)
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"{scheduler_name} WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error listening to {scheduler_name}: {e}")

        # Listen to both schedulers concurrently
        await asyncio.gather(
            listen_scheduler(self.ws_a, "Scheduler A"),
            listen_scheduler(self.ws_b, "Scheduler B"),
            return_exceptions=True
        )

    async def handle_task_result(self, data: Dict, scheduler_name: str):
        """Handle a task result message."""
        task_id = data["task_id"]
        status = data["status"]

        if status == "completed":
            complete_time = time.time()

            # Check if this is an A task or B task
            if task_id.startswith("task-A-"):
                self.completed_a_tasks.add(task_id)
                # Find the workflow and mark A task complete
                for workflow_state in self.workflow_states.values():
                    if workflow_state.a_task_id == task_id:
                        workflow_state.a_complete_time = complete_time
                        logger.debug(f"A task completed: {task_id}")
                        break

            elif task_id.startswith("task-B-"):
                self.completed_b_tasks.add(task_id)
                # Find the workflow and mark B task complete
                for workflow_state in self.workflow_states.values():
                    if task_id in workflow_state.b_task_ids:
                        is_workflow_complete = workflow_state.mark_b_task_complete(task_id, complete_time)
                        logger.debug(f"B task completed: {task_id} (workflow complete: {is_workflow_complete})")

                        if is_workflow_complete:
                            self.completed_workflows.add(workflow_state.workflow_id)
                            logger.info(f"✓ Workflow completed: {workflow_state.workflow_id} "
                                      f"({len(self.completed_workflows)} workflows done)")
                        break
        elif status == "failed":
            logger.warning(f"Task failed: {task_id} - {data.get('error', 'Unknown error')}")

    async def wait_for_phase_completion(self, expected_workflows: int, timeout: float = 600.0) -> int:
        """Wait for all workflows in the phase to complete."""
        start_time = time.time()

        logger.info(f"Waiting for {expected_workflows} workflows to complete (timeout: {timeout}s)...")

        while len(self.completed_workflows) < expected_workflows:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(f"Timeout reached! Completed {len(self.completed_workflows)}/{expected_workflows} workflows")
                break

            # Progress update every 10 seconds
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                logger.info(f"  Progress: {len(self.completed_workflows)}/{expected_workflows} workflows "
                          f"({len(self.completed_a_tasks)} A tasks, {len(self.completed_b_tasks)} B tasks)")

            await asyncio.sleep(0.5)

        logger.info(f"✓ Phase complete: {len(self.completed_workflows)}/{expected_workflows} workflows")
        return len(self.completed_workflows)

    async def close(self):
        """Close WebSocket connections."""
        if self.ws_a:
            await self.ws_a.close()
        if self.ws_b:
            await self.ws_b.close()


# ============================================================================
# Main Experiment Class
# ============================================================================

class DynamicMigrationExperiment:
    """Runs the dynamic instance migration experiment."""

    def __init__(
        self,
        num_workflows_per_phase: int,
        strategy: str,
        qps: float,
        total_instances: int = 16,
        instance_start_port: int = 8210,
        enable_migration: bool = True,
    ):
        self.num_workflows_per_phase = num_workflows_per_phase
        self.strategy = strategy
        self.qps = qps
        self.total_instances = total_instances
        self.instance_start_port = instance_start_port
        self.enable_migration = enable_migration

        # Update phase configs with workflow counts
        for phase_config in PHASE_CONFIGS:
            phase_config.num_workflows = num_workflows_per_phase

        # Migration controller
        self.migration_controller = MigrationController(
            scheduler_a_url=SCHEDULER_A_URL,
            scheduler_b_url=SCHEDULER_B_URL,
            instance_port_range=(instance_start_port, instance_start_port + total_instances - 1),
        )

        # Experiment state
        self.workflow_states: Dict[str, WorkflowState] = {}
        self.phase_results: List[PhaseResults] = []
        self.current_phase = 0

        # Task ID tracking
        self.all_a_task_ids: List[str] = []
        self.all_b_task_ids: List[str] = []

        # Instance tracking
        self.current_instances_a: List[str] = []
        self.current_instances_b: List[str] = []

    def initialize_instances(self):
        """Get initial instance distribution from Phase 1 config."""
        logger.info("Initializing instance distribution...")

        phase1 = PHASE_CONFIGS[0]

        # Get instances on each scheduler
        resp_a = requests.get(f"{SCHEDULER_A_URL}/instance/list")
        resp_b = requests.get(f"{SCHEDULER_B_URL}/instance/list")

        instances_a = [inst["instance_id"] for inst in resp_a.json()["instances"]]
        instances_b = [inst["instance_id"] for inst in resp_b.json()["instances"]]

        self.current_instances_a = instances_a[:phase1.scheduler_a_instances]
        self.current_instances_b = instances_b[:phase1.scheduler_b_instances]

        logger.info(f"Scheduler A: {len(self.current_instances_a)} instances")
        logger.info(f"Scheduler B: {len(self.current_instances_b)} instances")

    async def run_phase(self, phase_idx: int, next_phase_idx: Optional[int] = None) -> Tuple[PhaseResults, Optional[asyncio.Task]]:
        """
        Run a single phase of the experiment.

        Args:
            phase_idx: Index of the current phase
            next_phase_idx: Index of the next phase (for migration triggering)

        Returns:
            Tuple of (PhaseResults, migration_task or None)
        """
        phase_config = PHASE_CONFIGS[phase_idx]
        self.current_phase = phase_idx

        logger.info(f"\n{'='*70}")
        logger.info(f"Starting Phase {phase_config.phase_id}: n={phase_config.fanout}")
        logger.info(f"  Workflows: {phase_config.num_workflows}")
        logger.info(f"  Scheduler A: {phase_config.scheduler_a_instances} instances")
        logger.info(f"  Scheduler B: {phase_config.scheduler_b_instances} instances")
        logger.info(f"{'='*70}\n")

        phase_result = PhaseResults(
            phase_id=phase_config.phase_id,
            fanout=phase_config.fanout,
            num_workflows=phase_config.num_workflows,
        )
        phase_result.phase_start_time = time.time()

        # Pre-generate task IDs for this phase
        phase_start_idx = sum(PHASE_CONFIGS[i].num_workflows for i in range(phase_idx))
        self._generate_task_ids_for_phase(phase_config)

        # Get task IDs for this phase only
        phase_a_task_ids = self.all_a_task_ids[phase_start_idx:phase_start_idx + phase_config.num_workflows]
        num_b_tasks_per_workflow = phase_config.fanout
        phase_b_start_idx = sum(PHASE_CONFIGS[i].num_workflows * PHASE_CONFIGS[i].fanout for i in range(phase_idx))
        phase_b_task_ids = self.all_b_task_ids[phase_b_start_idx:phase_b_start_idx + phase_config.num_workflows * num_b_tasks_per_workflow]

        # Create and connect task monitor
        monitor = TaskResultMonitor(SCHEDULER_A_URL, SCHEDULER_B_URL, self.workflow_states)
        await monitor.connect_and_subscribe(phase_a_task_ids, phase_b_task_ids)

        # Start listening for results in background
        listen_task = asyncio.create_task(monitor.listen_for_results())

        # Submit all workflows
        await self._submit_workflows_for_phase(phase_config, phase_result)

        # Trigger migration for next phase (if applicable) immediately after B tasks submitted
        # This allows migration to run in parallel with task execution
        migration_task = None
        if self.enable_migration and next_phase_idx is not None:
            logger.info(f"\n→ All B tasks submitted - triggering migration to Phase {next_phase_idx + 1}")
            migration_task = await self.migrate_instances(phase_idx, next_phase_idx)

        # Wait for all workflows to complete
        completed_count = await monitor.wait_for_phase_completion(
            expected_workflows=phase_config.num_workflows,
            timeout=600.0
        )

        # Collect workflow latencies
        for i in range(phase_config.num_workflows):
            workflow_id = f"wf-{self.strategy}-p{phase_config.phase_id}-{i:04d}"
            workflow_state = self.workflow_states[workflow_id]

            if workflow_state.is_complete():
                latency_ms = workflow_state.get_latency_ms()
                if latency_ms is not None:
                    phase_result.add_workflow_latency(latency_ms)

        # Clean up monitor
        listen_task.cancel()
        await monitor.close()

        phase_result.phase_end_time = time.time()
        logger.info(f"✓ Phase {phase_config.phase_id} complete: {phase_result.workflows_completed}/{phase_config.num_workflows} workflows\n")

        return phase_result, migration_task

    def _generate_task_ids_for_phase(self, phase_config: PhaseConfig):
        """Pre-generate all task IDs for a phase."""
        phase_id = phase_config.phase_id

        for i in range(phase_config.num_workflows):
            workflow_id = f"wf-{self.strategy}-p{phase_id}-{i:04d}"

            # A task ID
            a_task_id = f"task-A-{self.strategy}-p{phase_id}-wf{i:04d}"
            self.all_a_task_ids.append(a_task_id)

            # B task IDs
            b_task_ids = []
            for j in range(phase_config.fanout):
                b_task_id = f"task-B-{self.strategy}-p{phase_id}-wf{i:04d}-b{j:02d}"
                b_task_ids.append(b_task_id)
                self.all_b_task_ids.append(b_task_id)

            # Create workflow state
            self.workflow_states[workflow_id] = WorkflowState(
                workflow_id=workflow_id,
                phase=phase_id,
                strategy=self.strategy,
                a_task_id=a_task_id,
                b_task_ids=b_task_ids,
                total_b_tasks=phase_config.fanout,
            )

    async def _submit_workflows_for_phase(self, phase_config: PhaseConfig, phase_result: PhaseResults):
        """Submit all workflows for a phase."""
        # Simple sequential submission for now
        inter_arrival_time = 1.0 / self.qps

        for i in range(phase_config.num_workflows):
            workflow_id = f"wf-{self.strategy}-p{phase_config.phase_id}-{i:04d}"
            workflow_state = self.workflow_states[workflow_id]

            # Submit A task
            await self._submit_a_task(workflow_state)

            # Submit all B tasks for this workflow
            await self._submit_b_tasks(workflow_state)

            # Wait for inter-arrival time
            await asyncio.sleep(inter_arrival_time)

            # Progress indicator
            if (i + 1) % 10 == 0:
                logger.info(f"  Submitted {i+1}/{phase_config.num_workflows} workflows")

        logger.info(f"✓ All workflows submitted for Phase {phase_config.phase_id}")

    async def _submit_a_task(self, workflow_state: WorkflowState):
        """Submit an A task."""
        # Generate task execution time
        times_list, _ = generate_bimodal_distribution(1)
        exec_time = times_list[0]

        task_data = {
            "task_id": workflow_state.a_task_id,
            "model_id": "sleep_model",
            "task_input": {"sleep_time": exec_time},
            "metadata": {
                "exp_runtime": int(exec_time * 1000),
                "workflow_id": workflow_state.workflow_id,
                "task_type": "A",
                "phase": workflow_state.phase,
                "is_warmup": workflow_state.is_warmup
            }
        }

        workflow_state.a_submit_time = time.time()

        try:
            response = requests.post(f"{SCHEDULER_A_URL}/task/submit", json=task_data, timeout=5.0)
            if response.status_code != 200:
                logger.warning(f"Failed to submit A task {workflow_state.a_task_id}: {response.status_code}")
        except Exception as e:
            logger.error(f"Error submitting A task: {e}")

    async def _submit_b_tasks(self, workflow_state: WorkflowState):
        """Submit all B tasks for a workflow."""
        for b_task_id in workflow_state.b_task_ids:
            # Generate task execution time for each B task (using B task bimodal distribution)
            times_list, _ = generate_b_task_bimodal_distribution(1)
            exec_time = times_list[0]

            task_data = {
                "task_id": b_task_id,
                "model_id": "sleep_model",
                "task_input": {"sleep_time": exec_time},
                "metadata": {
                    "exp_runtime": int(exec_time * 1000),
                    "workflow_id": workflow_state.workflow_id,
                    "task_type": "B",
                    "phase": workflow_state.phase,
                    "is_warmup": workflow_state.is_warmup
                }
            }

            try:
                response = requests.post(f"{SCHEDULER_B_URL}/task/submit", json=task_data, timeout=5.0)
                if response.status_code != 200:
                    logger.warning(f"Failed to submit B task {b_task_id}: {response.status_code}")
            except Exception as e:
                logger.error(f"Error submitting B task {b_task_id}: {e}")

    async def migrate_instances(self, from_phase_idx: int, to_phase_idx: int):
        """
        Initiate non-blocking instance migration between phases.

        Returns a background task that monitors migration progress.
        This allows the next phase to begin task submission immediately
        while migrations are still in progress.
        """
        from_phase = PHASE_CONFIGS[from_phase_idx]
        to_phase = PHASE_CONFIGS[to_phase_idx]

        logger.info(f"\n{'='*70}")
        logger.info(f"Initiating Migration: Phase {from_phase.phase_id} → Phase {to_phase.phase_id}")
        logger.info(f"{'='*70}\n")

        # Plan migration
        instances_a_to_b, instances_b_to_a = self.migration_controller.plan_migration(
            from_phase,
            to_phase,
            self.current_instances_a,
            self.current_instances_b,
        )

        # Initiate migrations (non-blocking - calls /model/restart and returns immediately)
        if instances_a_to_b:
            logger.info(f"Initiating restart for {len(instances_a_to_b)} instances: A → B")
            self.migration_controller.initiate_migration(
                instances_a_to_b,
                SCHEDULER_A_URL,
                SCHEDULER_B_URL,
            )

        if instances_b_to_a:
            logger.info(f"Initiating restart for {len(instances_b_to_a)} instances: B → A")
            self.migration_controller.initiate_migration(
                instances_b_to_a,
                SCHEDULER_B_URL,
                SCHEDULER_A_URL,
            )

        # Create background task to monitor migrations
        async def monitor_and_update():
            """Background task to monitor migration and update instance tracking."""
            logger.info("Starting background migration monitor...")
            migration_stats = await self.migration_controller.monitor_migrations_async()

            # Update instance tracking after migrations complete
            for inst_id in instances_a_to_b:
                if inst_id in self.current_instances_a:
                    self.current_instances_a.remove(inst_id)
                    self.current_instances_b.append(inst_id)

            for inst_id in instances_b_to_a:
                if inst_id in self.current_instances_b:
                    self.current_instances_b.remove(inst_id)
                    self.current_instances_a.append(inst_id)

            logger.info(f"✓ Migration complete: {migration_stats['completed']}/{migration_stats['total_migrations']} successful")
            return migration_stats

        # Start background task and return it
        migration_task = asyncio.create_task(monitor_and_update())
        logger.info("✓ Migration initiated - task submission will continue in parallel")

        return migration_task

    async def run_experiment(self):
        """Run the complete experiment."""
        logger.info(f"\n{'='*70}")
        logger.info(f"Experiment 08: Dynamic Instance Migration")
        logger.info(f"{'='*70}\n")
        logger.info(f"Strategy: {self.strategy}")
        logger.info(f"Migration enabled: {self.enable_migration}")
        logger.info(f"Workflows per phase: {self.num_workflows_per_phase}")
        logger.info(f"QPS: {self.qps}")
        logger.info(f"Total instances: {self.total_instances}")

        # Initialize
        self.initialize_instances()

        # Run phases
        migration_tasks = []  # Background tasks for migration monitoring

        for phase_idx in range(len(PHASE_CONFIGS)):
            # Determine next phase index for migration
            next_phase_idx = phase_idx + 1 if phase_idx < len(PHASE_CONFIGS) - 1 else None

            # Run phase - migration will be triggered after all B tasks are submitted
            phase_result, migration_task = await self.run_phase(phase_idx, next_phase_idx)
            self.phase_results.append(phase_result)

            # Collect migration task if one was created
            if migration_task is not None:
                migration_tasks.append(migration_task)
            elif not self.enable_migration and next_phase_idx is not None:
                logger.info(f"\n{'='*70}")
                logger.info(f"Migration disabled - keeping static instance distribution")
                logger.info(f"{'='*70}\n")

        # Wait for all background migrations to complete
        all_migration_stats = []
        if migration_tasks:
            logger.info(f"\n{'='*70}")
            logger.info(f"Waiting for {len(migration_tasks)} background migrations to complete...")
            logger.info(f"{'='*70}\n")

            migration_results = await asyncio.gather(*migration_tasks, return_exceptions=True)

            for idx, result in enumerate(migration_results):
                if isinstance(result, Exception):
                    logger.error(f"Migration {idx + 1} failed with exception: {result}")
                else:
                    all_migration_stats.append(result)

            logger.info("✓ All migrations completed")

        # Print results
        self.print_results(all_migration_stats)

        # Save results to JSON
        self.save_results_to_json(all_migration_stats)

    def print_results(self, migration_stats: List[Dict]):
        """Print experiment results."""
        logger.info(f"\n{'='*70}")
        logger.info("Experiment Results")
        logger.info(f"{'='*70}\n")

        # Phase results
        for phase_result in self.phase_results:
            stats = phase_result.get_statistics()
            if stats:
                logger.info(f"Phase {stats['phase_id']} (n={stats['fanout']}):")
                logger.info(f"  Completed: {stats['completed']}/{stats['num_workflows']}")
                logger.info(f"  Avg latency: {stats['avg_latency_ms']:.1f}ms")
                logger.info(f"  Min/Max latency: {stats['min_latency_ms']:.1f}/{stats['max_latency_ms']:.1f}ms")
                logger.info(f"  P50/P90/P99: {stats['p50_latency_ms']:.1f}/{stats['p90_latency_ms']:.1f}/{stats['p99_latency_ms']:.1f}ms")
                logger.info("")

        # Migration results
        for idx, stats in enumerate(migration_stats):
            logger.info(f"Migration {idx+1} → {idx+2}:")
            logger.info(f"  Total: {stats['total_migrations']}")
            logger.info(f"  Successful: {stats['completed']}")
            logger.info(f"  Failed: {stats['failed']}")
            logger.info(f"  Duration: {stats['total_duration_ms']:.1f}ms")
            logger.info("")

    def save_results_to_json(self, migration_stats: List[Dict]):
        """Save experiment results to a JSON file."""
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        migration_mode = "migration" if self.enable_migration else "static"
        filename = f"exp08_{migration_mode}_{self.strategy}_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)

        # Collect all data
        results = {
            "experiment": "08_dynamic_migration",
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "strategy": self.strategy,
                "migration_enabled": self.enable_migration,
                "qps": self.qps,
                "total_instances": self.total_instances,
                "num_workflows_per_phase": self.num_workflows_per_phase,
                "instance_start_port": self.instance_start_port,
            },
            "phases": [],
            "migrations": migration_stats,
            "workflows": [],
        }

        # Add phase statistics
        for phase_result in self.phase_results:
            stats = phase_result.get_statistics()
            if stats:
                phase_data = {
                    "phase_id": stats['phase_id'],
                    "fanout": stats['fanout'],
                    "num_workflows": stats['num_workflows'],
                    "completed_workflows": stats['completed'],
                    "phase_duration_s": phase_result.phase_end_time - phase_result.phase_start_time if phase_result.phase_end_time else None,
                    "latency_stats": {
                        "avg_ms": stats['avg_latency_ms'],
                        "min_ms": stats['min_latency_ms'],
                        "max_ms": stats['max_latency_ms'],
                        "p50_ms": stats['p50_latency_ms'],
                        "p90_ms": stats['p90_latency_ms'],
                        "p99_ms": stats['p99_latency_ms'],
                    },
                    "all_latencies_ms": phase_result.workflow_latencies,
                }
                results["phases"].append(phase_data)

        # Add individual workflow data
        for workflow_id, workflow_state in self.workflow_states.items():
            if workflow_state.is_complete():
                workflow_data = {
                    "workflow_id": workflow_id,
                    "phase": workflow_state.phase,
                    "fanout": workflow_state.total_b_tasks,
                    "latency_ms": workflow_state.get_latency_ms(),
                    "a_task_id": workflow_state.a_task_id,
                    "b_task_ids": workflow_state.b_task_ids,
                }
                results["workflows"].append(workflow_data)

        # Write to file
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"✓ Results saved to: {filepath}")
        return filepath


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 08: Multi-Model Workflow with Dynamic Instance Migration"
    )
    parser.add_argument(
        "--num-workflows-per-phase",
        type=int,
        default=10,
        help="Number of workflows to submit in each phase (default: 10)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="min_time",
        choices=["min_time", "round_robin", "probabilistic"],
        help="Scheduling strategy to use (default: min_time)"
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=2.0,
        help="A task submission rate in queries per second (default: 2.0)"
    )
    parser.add_argument(
        "--total-instances",
        type=int,
        default=16,
        help="Total number of instances in the pool (default: 16)"
    )
    parser.add_argument(
        "--instance-start-port",
        type=int,
        default=8210,
        help="Starting port for instances (default: 8210)"
    )
    parser.add_argument(
        "--enable-migration",
        action="store_true",
        default=True,
        help="Enable dynamic instance migration between phases (default: True)"
    )
    parser.add_argument(
        "--disable-migration",
        dest="enable_migration",
        action="store_false",
        help="Disable migration and use static instance distribution"
    )
    parser.add_argument(
        "--phase-configs",
        type=str,
        default=None,
        help="Phase configurations as comma-separated string: 'phase_id:fanout:a_inst:b_inst,...' "
             "(e.g., '1:3:4:12,2:8:2:14,3:1:8:8'). If not provided, uses default 16-instance config."
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

    args = parser.parse_args()

    # Parse phase configurations if provided
    global PHASE_CONFIGS
    if args.phase_configs:
        try:
            PHASE_CONFIGS = []
            for phase_str in args.phase_configs.split(','):
                phase_id, fanout, a_inst, b_inst = map(int, phase_str.split(':'))
                PHASE_CONFIGS.append(
                    PhaseConfig(
                        phase_id=phase_id,
                        fanout=fanout,
                        scheduler_a_instances=a_inst,
                        scheduler_b_instances=b_inst,
                        num_workflows=0  # Will be set by experiment
                    )
                )
            logger.info(f"Using custom phase configurations:")
            for pc in PHASE_CONFIGS:
                logger.info(f"  Phase {pc.phase_id}: fanout={pc.fanout}, A={pc.scheduler_a_instances}, B={pc.scheduler_b_instances}")
        except Exception as e:
            logger.error(f"Failed to parse --phase-configs: {e}")
            logger.error("Expected format: 'phase_id:fanout:a_inst:b_inst,...'")
            sys.exit(1)
    else:
        PHASE_CONFIGS = DEFAULT_PHASE_CONFIGS.copy()
        logger.info("Using default phase configurations (16 instances)")

    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Create and run experiment
    experiment = DynamicMigrationExperiment(
        num_workflows_per_phase=args.num_workflows_per_phase,
        strategy=args.strategy,
        qps=args.qps,
        total_instances=args.total_instances,
        instance_start_port=args.instance_start_port,
        enable_migration=args.enable_migration,
    )

    # Run experiment
    try:
        asyncio.run(experiment.run_experiment())
        logger.info("\n✅ Experiment completed successfully!")
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Experiment interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
