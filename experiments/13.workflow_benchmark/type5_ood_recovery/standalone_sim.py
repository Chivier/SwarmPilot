#!/usr/bin/env python3
"""
Standalone OOD Recovery Simulation - Predictor Only

This script simulates the OOD recovery experiment using only the predictor service.
It uses a Discrete Event Simulation (DES) engine with virtual time to simulate
arbitrary numbers of instances without requiring the scheduler or instance services.

Features:
- Virtual time simulation (no real waiting)
- Configurable number of simulated instances
- Predictor integration for runtime estimation
- Three-phase OOD pattern (same as test_ood_sim.py)
- Gantt chart and throughput visualization output

Usage:
    # Start predictor first
    uv run predictor/src/main.py &

    # Run simulation
    uv run standalone_sim.py \\
        --predictor-url http://127.0.0.1:8000 \\
        --num-instances 48 \\
        --num-tasks 500 \\
        --qps 2.0 \\
        --phase23-distribution four_peak \\
        --output-dir output_standalone
"""

import argparse
import heapq
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from config import OODRecoveryConfig
from task_data import OODTaskData, TaskGenerator
from plot_gantt import TaskExecution, plot_gantt_chart, plot_gantt_with_queuing
from plot_throughput import plot_single, plot_recovery_vs_baseline, load_metrics


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for the simulation."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


# =============================================================================
# Event Types
# =============================================================================

class EventType(Enum):
    """Types of events in the simulation."""
    TASK_SUBMIT = "submit"
    TASK_COMPLETE = "complete"


@dataclass(order=True)
class SimEvent:
    """
    A simulation event scheduled at a specific virtual time.
    
    Events are ordered by time for the priority queue.
    """
    time: float
    event_type: EventType = field(compare=False)
    payload: Any = field(compare=False)


# =============================================================================
# Simulated Instance
# =============================================================================

@dataclass
class SimulatedTask:
    """A task being executed or queued in a simulated instance."""
    task_data: OODTaskData
    predicted_runtime: float  # Predicted runtime from predictor (seconds)
    quantiles: Dict[float, float]  # Runtime quantiles from predictor
    actual_runtime: float     # Actual runtime (from task_data.actual_sleep_time)
    submit_time: float        # Virtual time when submitted
    start_time: Optional[float] = None   # Virtual time when started
    complete_time: Optional[float] = None  # Virtual time when completed


class SimulatedInstance:
    """
    Simulates an instance that can execute tasks.
    
    Each instance has a queue and processes tasks one at a time.
    """
    
    def __init__(self, instance_id: str):
        self.id = instance_id
        self.queue: List[SimulatedTask] = []
        self.current_task: Optional[SimulatedTask] = None
        self.finish_time: float = 0.0  # When current task finishes
        self.completed_tasks: List[SimulatedTask] = []
        
        # Probabilistic queue state
        self.queue_quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        self.queue_values: List[float] = [0.0] * len(self.queue_quantiles)
    
    def update_queue_state(self, task: SimulatedTask):
        """
        Update queue probabilistic state using Monte Carlo sampling.
        Matches scheduler.py `update_queue` logic.
        """
        num_samples = 1000
        random_percentiles = np.random.random(num_samples)
        
        # Sample from current queue distribution
        queue_samples = np.interp(
            random_percentiles,
            self.queue_quantiles,
            self.queue_values
        )
        
        # Sample from task distribution
        task_qs = sorted(task.quantiles.keys())
        task_vals = [task.quantiles[q] for q in task_qs]
        
        task_samples = np.interp(
            random_percentiles,
            task_qs,
            task_vals
        )
        
        # Compute new total time samples
        total_samples = queue_samples + task_samples
        
        # Compute new quantiles
        self.queue_values = list(np.quantile(total_samples, self.queue_quantiles))
    
    def is_idle(self) -> bool:
        """Check if instance is not executing any task."""
        return self.current_task is None
    
    
    def get_predicted_queue_time(self) -> float:
        """Get expected queue time (median) for metrics."""
        # Use median (0.5 quantile) as the representative value
        idx_median = self.queue_quantiles.index(0.5)
        return self.queue_values[idx_median]
    
    def assign_task(self, task: SimulatedTask, current_time: float) -> Optional[float]:
        """
        Assign a task to this instance.
        
        If instance is idle, task starts immediately.
        Otherwise, task is queued.
        
        Returns the completion time if task starts immediately, None otherwise.
        """
        if self.is_idle():
            # Start immediately
            self.current_task = task
            task.start_time = current_time
            task.complete_time = current_time + task.actual_runtime
            self.finish_time = task.complete_time
            return task.complete_time
        else:
             # Add to queue
            self.queue.append(task)
            self.update_queue_state(task)
            return None
    
    def complete_current_task(self, current_time: float) -> Tuple[SimulatedTask, Optional[float]]:
        """
        Complete the current task and start the next one if available.
        
        Returns: (completed_task, next_completion_time or None)
        """
        completed = self.current_task
        self.completed_tasks.append(completed)
        self.current_task = None
        
        # Start next task from queue if available
        if self.queue:
            next_task = self.queue.pop(0)
            self.current_task = next_task
            next_task.start_time = current_time
            next_task.complete_time = current_time + next_task.actual_runtime
            self.finish_time = next_task.complete_time
            return completed, next_task.complete_time
        else:
            self.finish_time = current_time
            return completed, None


# =============================================================================
# Predictor Client
# =============================================================================

class PredictorClient:
    """
    HTTP client for the predictor service.
    
    Calls the /predict endpoint to get runtime predictions.
    """
    
    def __init__(self, predictor_url: str, model_id: str = "sleep_model_a"):
        self.predictor_url = predictor_url.rstrip("/")
        self.model_id = model_id
        self.client = httpx.Client(timeout=30.0)
        self._cache: Dict[str, float] = {}  # Simple cache for predictions
    
    def predict(self, features: Dict[str, Any], prediction_type: str = "quantile") -> Dict[str, Any]:
        """
        Get runtime prediction for given features.
        
        Args:
            features: Task features (e.g., token_length, task_type)
            prediction_type: "quantile" or "expect_error"
        
        Returns:
            Prediction result dict with 'expected_runtime_ms' or 'quantiles'
        """
        # Create cache key
        cache_key = json.dumps(features, sort_keys=True)
        if cache_key in self._cache:
            return {"expected_runtime_ms": self._cache[cache_key]}
        
        request_body = {
            "model_id": self.model_id,
            "platform_info": {
                "gpu_model": "simulation",
                "driver_version": "sim",
                "cuda_version": "sim"
            },
            "features": features,
            "prediction_type": prediction_type,
        }
        
        try:
            response = self.client.post(
                f"{self.predictor_url}/predict",
                json=request_body
            )
            response.raise_for_status()
            result = response.json()
            
            # Cache the result
            if "result" in result and "expected_runtime_ms" in result["result"]:
                self._cache[cache_key] = result["result"]["expected_runtime_ms"]
            
            return result.get("result", {})
        except Exception as e:
            logging.warning(f"Predictor call failed: {e}")
            # Return fallback prediction
            return {"expected_runtime_ms": 10000.0}
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()


# =============================================================================
# Simulation Engine
# =============================================================================

class SimulationEngine:
    """
    Discrete Event Simulation engine for OOD recovery experiment.
    
    Uses virtual time (no real waiting) to simulate task scheduling
    and execution across multiple instances.
    """
    
    def __init__(
        self,
        config: OODRecoveryConfig,
        num_instances: int,
        predictor_client: Optional[PredictorClient] = None,
        logger: Optional[logging.Logger] = None,
        seed: int = 42,
        phase1_qps: Optional[float] = None,
        phase23_qps: Optional[float] = None,
        runtime_scale: float = 1.0,
    ):
        self.config = config
        self.num_instances = num_instances
        self.predictor = predictor_client
        self.logger = logger or logging.getLogger(__name__)
        self.seed = seed
        self.runtime_scale = runtime_scale
        
        # Phase-specific QPS
        self.phase1_qps = phase1_qps if phase1_qps is not None else config.qps
        self.phase23_qps = phase23_qps if phase23_qps is not None else config.qps
        
        # Virtual time
        self.virtual_time: float = 0.0

        # Simulated delays (in seconds)
        self.prediction_delay: float = 0.001  # 1ms for prediction
        self.scheduling_delay: float = 0.002  # 2ms for scheduling

        # Event queue (min-heap by time)
        self.event_queue: List[SimEvent] = []
        
        # Instances
        self.instances: List[SimulatedInstance] = [
            SimulatedInstance(f"instance_{i:02d}")
            for i in range(num_instances)
        ]
        
        # Task generator
        self.task_generator = TaskGenerator(seed=seed, config=config)
        
        # Phase tracking
        self.current_phase = 1
        self.phase1_submitted = 0
        self.phase2_submitted = 0
        self.phase2_completed = 0
        self.phase3_submitted = 0
        self.phase_transition_time: Optional[float] = None
        
        # Metrics collection
        self.submitted_tasks: List[OODTaskData] = []
        self.completed_tasks: List[SimulatedTask] = []
        self.throughput_samples: List[Dict] = []
        
        # Counters
        self.total_submitted = 0
        self.total_completed = 0
        self.task_index = 0
    
    def schedule_event(self, event: SimEvent):
        """Add an event to the priority queue."""
        heapq.heappush(self.event_queue, event)
    
    def run(self) -> Dict:
        """
        Run the simulation.
        
        Returns:
            Metrics dictionary compatible with plot_throughput.py
        """
        self.logger.info("=" * 70)
        self.logger.info("Starting Standalone OOD Simulation")
        self.logger.info(f"Instances: {self.num_instances}")
        self.logger.info(f"Target Tasks: {self.config.num_tasks}")
        self.logger.info(f"Phase 1 QPS: {self.phase1_qps}")
        self.logger.info(f"Phase 2/3 QPS: {self.phase23_qps}")
        self.logger.info(f"Phase 1 Count: {self.config.phase1_count}")
        self.logger.info(f"Phase 2/3 Distribution: {self.config.phase23_distribution}")
        self.logger.info(f"No Recovery: {self.config.no_recovery}")
        self.logger.info("=" * 70)
        
        # Schedule task submissions using Poisson process (exponential inter-arrival times)
        # In a Poisson process with rate λ (QPS), inter-arrival times are exponentially
        # distributed with mean 1/λ

        # Create a separate RNG for submission times to ensure reproducibility
        submission_rng = np.random.default_rng(self.seed + 1000)

        current_time = 0.0
        for i in range(self.config.num_tasks):
            # Determine QPS for this task index (Phase 1 vs Phase 2/3)
            if i < self.config.phase1_count:
                qps = self.phase1_qps
            else:
                qps = self.phase23_qps

            self.schedule_event(SimEvent(
                time=current_time,
                event_type=EventType.TASK_SUBMIT,
                payload={"task_index": i}
            ))

            # Sample inter-arrival time from exponential distribution
            # Exponential distribution mean = 1/λ where λ = QPS
            inter_arrival = submission_rng.exponential(1.0 / qps)
            current_time += inter_arrival
        
        # Run event loop
        start_real_time = time.time()
        last_sample_time = 0.0
        sample_interval = 1.0  # Sample throughput every virtual second
        
        while self.event_queue and self.total_completed < self.config.num_tasks:
            # Get next event
            event = heapq.heappop(self.event_queue)
            self.virtual_time = event.time
            
            # Process event
            if event.event_type == EventType.TASK_SUBMIT:
                self._handle_submit(event)
            elif event.event_type == EventType.TASK_COMPLETE:
                self._handle_complete(event)
            
            # Sample throughput periodically
            if self.virtual_time - last_sample_time >= sample_interval:
                self._sample_throughput()
                last_sample_time = self.virtual_time
        
        # Final throughput sample
        self._sample_throughput()
        
        elapsed_real = time.time() - start_real_time
        self.logger.info("=" * 70)
        self.logger.info(f"Simulation Complete")
        self.logger.info(f"Virtual Time: {self.virtual_time:.2f}s")
        self.logger.info(f"Real Time: {elapsed_real:.2f}s")
        self.logger.info(f"Speedup: {self.virtual_time / elapsed_real:.1f}x")
        self.logger.info(f"Tasks Completed: {self.total_completed}")
        self.logger.info("=" * 70)
        
        return self._generate_metrics()
    
    def _get_phase_for_task(self) -> int:
        """Determine the phase for the next task based on counts."""
        if self.phase1_submitted < self.config.phase1_count:
            return 1
        elif self.config.no_recovery:
            return 2  # Stay in phase 2 for baseline
        elif self.phase_transition_time is not None:
            return 3  # Recovery phase
        else:
            return 2  # OOD phase
    
    def _handle_submit(self, event: SimEvent):
        """Handle a task submission event."""
        # Generate task
        task_data = self.task_generator.generate_task(self.task_index)
        task_data.phase = self._get_phase_for_task()
        task_data.calculate_times(self.config)
        task_data.submit_time = self.virtual_time
        
        self.submitted_tasks.append(task_data)
        self.task_index += 1
        self.total_submitted += 1
        
        # Update phase counters
        if task_data.phase == 1:
            self.phase1_submitted += 1
        elif task_data.phase == 2:
            self.phase2_submitted += 1
        else:
            self.phase3_submitted += 1
        
        # Get prediction (use exp_runtime_ms from task_data which simulates predictor)
        # NOTE: runtime_scale is already applied in calculate_times(), don't apply again!
        predicted_runtime_s = task_data.exp_runtime_ms / 1000.0
        actual_runtime_s = task_data.actual_sleep_time

        # Simulate prediction delay (1ms)
        self.virtual_time += self.prediction_delay

        # Generate quantiles (simulate predictor behavior)
        # Uses CV=0.40 to match exp_cv from submitter.py
        # Quantile keys: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        quantiles = self._generate_quantiles(predicted_runtime_s)

        # Simulate scheduling delay (2ms)
        self.virtual_time += self.scheduling_delay

        # Create simulated task
        sim_task = SimulatedTask(
            task_data=task_data,
            predicted_runtime=predicted_runtime_s,
            quantiles=quantiles,
            actual_runtime=actual_runtime_s,
            submit_time=self.virtual_time,
        )

        # Select best instance (Probabilistic Strategy)
        best_instance = self._select_instance_probabilistic(sim_task)

        # Assign task to instance
        completion_time = best_instance.assign_task(sim_task, self.virtual_time)
        task_data.instance_id = best_instance.id
        
        # Schedule completion event if task started immediately
        if completion_time is not None:
            self.schedule_event(SimEvent(
                time=completion_time,
                event_type=EventType.TASK_COMPLETE,
                payload={"instance_id": best_instance.id}
            ))
        
        if self.total_submitted % 100 == 0:
            self.logger.debug(
                f"[T={self.virtual_time:.2f}s] Submitted task {self.total_submitted}, "
                f"Phase={task_data.phase}, Instance={best_instance.id}"
            )
    
    def _handle_complete(self, event: SimEvent):
        """Handle a task completion event."""
        instance_id = event.payload["instance_id"]
        instance = next(i for i in self.instances if i.id == instance_id)
        
        # Complete current task
        completed_task, next_completion = instance.complete_current_task(self.virtual_time)
        self.completed_tasks.append(completed_task)
        self.total_completed += 1
        
        # Update phase completion counters
        phase = completed_task.task_data.phase
        if phase == 2:
            self.phase2_completed += 1
            # Check for phase transition
            if (self.phase_transition_time is None and 
                not self.config.no_recovery and
                self.phase2_completed >= self.config.phase2_transition_count):
                self._trigger_phase_transition()
        
        # Schedule next task completion if there's a queued task
        if next_completion is not None:
            self.schedule_event(SimEvent(
                time=next_completion,
                event_type=EventType.TASK_COMPLETE,
                payload={"instance_id": instance_id}
            ))
        
        if self.total_completed % 100 == 0:
            self.logger.info(
                f"[T={self.virtual_time:.2f}s] Completed {self.total_completed}/{self.config.num_tasks} tasks"
            )
    
    def _generate_quantiles(self, predicted_runtime_s: float) -> Dict[float, float]:
        """
        Generate quantile distribution for a task prediction.

        Uses normal approximation with CV=0.40 (matching exp_cv from submitter.py).
        Quantile keys: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

        Args:
            predicted_runtime_s: Mean predicted runtime in seconds

        Returns:
            Dict mapping quantile (e.g., 0.5) to runtime value in seconds
        """
        from scipy.stats import norm

        sigma = predicted_runtime_s * 0.4  # CV = 0.40 (match exp_cv from submitter.py)
        quantiles = {}
        for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
            value = max(0.1, norm.ppf(q, loc=predicted_runtime_s, scale=sigma))
            quantiles[q] = value
        return quantiles

    def _recalculate_instance_queue_state(self, instance: SimulatedInstance):
        """
        Recalculate instance queue state from scratch using Monte Carlo sampling.

        This matches scheduler.py's update_queue() logic:
        - Start with zero queue state
        - For each queued task, combine queue distribution with task distribution
        - Use 1000 Monte Carlo samples to compute new quantiles
        """
        # Reset queue state to zeros
        instance.queue_values = [0.0] * len(instance.queue_quantiles)

        # Re-add each queued task to rebuild queue state
        for task in instance.queue:
            instance.update_queue_state(task)

    def _trigger_phase_transition(self):
        """
        Trigger transition from Phase 2 to Phase 3 with full queue recalculation.

        This matches the behavior in receiver.py's _update_pending_phase2_metadata():
        1. Update quantiles for all queued Phase 2 tasks with corrected predictions
        2. Recalculate each instance's queue state from scratch
        """
        self.phase_transition_time = self.virtual_time
        self.logger.info(
            f"[T={self.virtual_time:.2f}s] *** PHASE TRANSITION: Phase 2 -> Phase 3 ***"
        )
        self.logger.info(
            f"Phase 2 completed: {self.phase2_completed}, "
            f"Threshold: {self.config.phase2_transition_count}"
        )

        # Step 1: Update quantiles for all queued Phase 2 tasks
        updated_count = 0
        for instance in self.instances:
            for task in instance.queue:
                if task.task_data.phase == 2:
                    # Recalculate quantiles using CORRECTED exp_runtime (actual_runtime)
                    corrected_runtime = task.actual_runtime
                    task.predicted_runtime = corrected_runtime
                    task.quantiles = self._generate_quantiles(corrected_runtime)
                    updated_count += 1

        self.logger.info(f"Updated {updated_count} pending Phase 2 tasks with corrected predictions")

        # Step 2: Full queue state recalculation for each instance
        for instance in self.instances:
            self._recalculate_instance_queue_state(instance)
    
    def _select_instance_probabilistic(self, task: SimulatedTask) -> SimulatedInstance:
        """
        Select instance using Monte Carlo simulation matching ProbabilisticSchedulingStrategy.
        """
        num_samples = 10
        num_instances = len(self.instances)
        
        # Generate random percentiles (shared across instances)
        random_percentiles = np.random.random(num_samples)
        
        # Build combined time matrix: shape (num_instances, num_samples)
        total_times_matrix = np.zeros((num_instances, num_samples))
        
        # Vectorized sampling for task (same for all instances)
        task_qs = sorted(task.quantiles.keys())
        task_vals = [task.quantiles[q] for q in task_qs]
        task_samples = np.interp(random_percentiles, task_qs, task_vals)
        
        for i, instance in enumerate(self.instances):
            # Sample queue times
            if instance.is_idle():
                queue_samples = np.zeros(num_samples)
            else:
                queue_samples = np.interp(
                    random_percentiles,
                    instance.queue_quantiles,
                    instance.queue_values
                )
            
            # Combine prediction + queue time
            # Note: In real scheduler, we add task_samples to queue_samples
            # BUT for selection, we are comparing (Queue + Task) finish time
            # So we add task time to queue time
            total_times_matrix[i, :] = queue_samples
            
        # Find winner for each sample (argmin along axis 0)
        winners = np.argmin(total_times_matrix, axis=0)
        
        # Count wins
        win_counts = np.bincount(winners, minlength=num_instances)
        
        # Select instance with most wins
        best_instance_idx = np.argmax(win_counts)
        return self.instances[best_instance_idx]
    
    def _sample_throughput(self):
        """Record throughput sample at current virtual time."""
        # Count completions by phase
        phase_completed = {1: 0, 2: 0, 3: 0}
        for task in self.completed_tasks:
            phase_completed[task.task_data.phase] += 1

        # Count submissions by phase
        phase_submitted = {1: 0, 2: 0, 3: 0}
        for task in self.submitted_tasks:
            phase_submitted[task.phase] += 1

        # Calculate real-time throughput (completions in last second)
        # This is the PRIMARY metric - real-time completion rate
        realtime_throughput = sum(
            1 for t in self.completed_tasks
            if t.complete_time >= self.virtual_time - 1.0
        )

        # Calculate real-time submission rate (for reference)
        realtime_submit_rate = sum(
            1 for t in self.submitted_tasks
            if t.submit_time >= self.virtual_time - 1.0
        )

        sample = {
            "elapsed_s": self.virtual_time,
            "total_submitted": self.total_submitted,
            "total_completed": self.total_completed,
            # Primary metrics: real-time completion rate
            "realtime_throughput": realtime_throughput,  # Real-time completion rate (tasks/sec)
            "cumulative_throughput": self.total_completed / max(self.virtual_time, 0.001),
            # Secondary metrics: submission rate
            "realtime_submit_rate": realtime_submit_rate,
            "cumulative_submit_rate": self.total_submitted / max(self.virtual_time, 0.001),
            "phase_submitted": {
                "1": phase_submitted[1],
                "2": phase_submitted[2],
                "3": phase_submitted[3],
            },
            "phase_completed": {
                "1": phase_completed[1],
                "2": phase_completed[2],
                "3": phase_completed[3],
            }
        }
        self.throughput_samples.append(sample)
    
    def _generate_metrics(self) -> Dict:
        """Generate metrics dictionary compatible with plotting scripts."""
        # Build task executions for Gantt chart
        task_executions = []
        for task in self.completed_tasks:
            td = task.task_data
            task_executions.append({
                "task_id": td.task_id,
                "task_index": td.task_index,
                "phase": td.phase,
                "instance_id": td.instance_id,
                "submit_time": task.submit_time,
                "complete_time": task.complete_time,
                "sleep_time_s": task.actual_runtime,
                "exec_start_time": task.start_time,
            })
        
        # Calculate real-time throughput statistics from samples (PRIMARY METRIC)
        realtime_throughput_values = [s.get("realtime_throughput", 0) for s in self.throughput_samples if s.get("realtime_throughput") is not None]
        avg_realtime_throughput = np.mean(realtime_throughput_values) if realtime_throughput_values else 0.0
        max_realtime_throughput = max(realtime_throughput_values) if realtime_throughput_values else 0.0
        min_realtime_throughput = min(realtime_throughput_values) if realtime_throughput_values else 0.0
        std_realtime_throughput = np.std(realtime_throughput_values) if realtime_throughput_values else 0.0

        # Summary statistics
        summary = {
            "total_submitted": self.total_submitted,
            "total_completed": self.total_completed,
            "experiment_duration_s": self.virtual_time,
            "phase_transition_delay_s": self.phase_transition_time,
            "phase1_submitted": self.phase1_submitted,
            "phase2_submitted": self.phase2_submitted,
            "phase3_submitted": self.phase3_submitted,
            # Primary metrics: real-time completion rate
            "realtime_throughput_avg": avg_realtime_throughput,
            "realtime_throughput_max": max_realtime_throughput,
            "realtime_throughput_min": min_realtime_throughput,
            "realtime_throughput_std": std_realtime_throughput,
            # Secondary metrics: cumulative averages
            "average_throughput": self.total_completed / max(self.virtual_time, 0.001),
            "average_submit_rate": self.total_submitted / max(self.virtual_time, 0.001),
        }
        
        return {
            "config": {
                "mode": "Baseline" if self.config.no_recovery else "Recovery",
                "num_instances": self.num_instances,
                "num_tasks": self.config.num_tasks,
                "qps": self.config.qps,
                "phase23_distribution": self.config.phase23_distribution,
            },
            "summary": summary,
            "throughput_trend": self.throughput_samples,
            "task_executions": task_executions,
        }
    
    def get_task_executions(self) -> List[TaskExecution]:
        """Convert completed tasks to TaskExecution objects for plotting."""
        executions = []
        for task in self.completed_tasks:
            td = task.task_data
            executions.append(TaskExecution(
                task_id=td.task_id,
                task_index=td.task_index,
                phase=td.phase,
                instance_id=td.instance_id,
                submit_time=task.submit_time,
                complete_time=task.complete_time,
                sleep_time_s=task.actual_runtime,
                exec_start_time=task.start_time,
            ))
        return executions


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Standalone OOD Recovery Simulation (Predictor Only)"
    )
    
    # Predictor settings
    parser.add_argument(
        "--predictor-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Predictor service URL (default: http://127.0.0.1:8000)"
    )
    
    # Instance settings
    parser.add_argument(
        "--num-instances",
        type=int,
        default=48,
        help="Number of simulated instances (default: 48)"
    )
    
    # Task settings
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=500,
        help="Number of tasks to simulate (default: 500)"
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=2.0,
        help="Default task submission rate (queries per second, default: 2.0)"
    )
    parser.add_argument(
        "--phase1-qps",
        type=float,
        default=None,
        help="Phase 1 QPS (default: same as --qps)"
    )
    parser.add_argument(
        "--phase23-qps",
        type=float,
        default=None,
        help="Phase 2/3 QPS (default: same as --qps)"
    )
    
    # Scaling
    parser.add_argument(
        "--runtime-scale",
        type=float,
        default=1.0,
        help="Global scaling factor for task runtime (default: 1.0)"
    )
    
    # Phase settings
    parser.add_argument(
        "--phase1-count",
        type=int,
        default=100,
        help="Number of Phase 1 tasks (default: 100)"
    )
    parser.add_argument(
        "--phase2-transition-count",
        type=int,
        default=20,
        help="Phase 2 completions before transition (default: 20)"
    )
    parser.add_argument(
        "--phase23-distribution",
        type=str,
        default="weighted_bimodal",
        choices=["normal", "uniform", "peak_dependent", "four_peak", "weighted_bimodal"],
        help="Distribution for Phase 2/3 runtimes (default: weighted_bimodal)"
    )
    parser.add_argument(
        "--phase23-bimodal-scale",
        type=float,
        default=2.0,
        help="Scale factor for Phase 2/3 weighted bimodal samples (default: 2.0)"
    )
    parser.add_argument(
        "--phase23-small-peak-ratio",
        type=float,
        default=0.2,
        help="Ratio of small peak samples in Phase 2/3 (default: 0.2)"
    )
    parser.add_argument(
        "--phase1-scale",
        type=float,
        default=0.1,
        help="Scale factor for Phase 1 runtime (default: 0.1)"
    )
    parser.add_argument(
        "--phase1-small-peak-ratio",
        type=float,
        default=0.8,
        help="Ratio of small peak samples in Phase 1 (default: 0.8, i.e., 80%% small peak)"
    )

    # Mode
    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help="Baseline mode (no Phase 2->3 transition)"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_standalone",
        help="Output directory for results (default: output_standalone)"
    )
    
    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    logger = setup_logging(args.verbose)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if args.no_recovery:
        output_dir = output_dir / "baseline"
    else:
        output_dir = output_dir / "recovery"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine phase-specific QPS
    phase1_qps = args.phase1_qps if args.phase1_qps is not None else args.qps
    phase23_qps = args.phase23_qps if args.phase23_qps is not None else args.qps
    
    # Create configuration
    config = OODRecoveryConfig(
        num_tasks=args.num_tasks,
        qps=args.qps,  # Default QPS for config
        phase1_count=args.phase1_count,
        phase2_transition_count=args.phase2_transition_count,
        phase23_distribution=args.phase23_distribution,
        phase23_bimodal_scale=args.phase23_bimodal_scale,
        phase23_small_peak_ratio=args.phase23_small_peak_ratio,
        phase1_scale=args.phase1_scale,  # Phase 1 runtime scale factor
        phase1_small_peak_ratio=args.phase1_small_peak_ratio,  # Phase 1 distribution ratio
        no_recovery=args.no_recovery,
        runtime_scale=args.runtime_scale,  # Pass runtime_scale to config for calculate_times()
    )
    
    # Create predictor client (optional - we use task_data.exp_runtime_ms directly)
    predictor = None
    # Uncomment to use real predictor:
    # predictor = PredictorClient(args.predictor_url)
    
    # Create and run simulation
    engine = SimulationEngine(
        config=config,
        num_instances=args.num_instances,
        predictor_client=predictor,
        logger=logger,
        seed=args.seed,
        phase1_qps=phase1_qps,
        phase23_qps=phase23_qps,
        runtime_scale=args.runtime_scale,
    )
    
    metrics = engine.run()
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")
    
    # Generate Gantt chart
    task_executions = engine.get_task_executions()
    gantt_path = output_dir / "gantt.png"
    plot_gantt_chart(
        task_executions,
        gantt_path,
        title=f"OOD Simulation Gantt Chart ({args.num_instances} instances)"
    )
    
    # Generate throughput plot
    throughput_path = output_dir / "throughput.png"
    plot_single(
        metrics,
        output_path=str(throughput_path),
        title=f"OOD Simulation Throughput ({config.phase23_distribution})"
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    print(f"Mode: {'Baseline' if config.no_recovery else 'Recovery'}")
    print(f"Instances: {args.num_instances}")
    print(f"Tasks: {metrics['summary']['total_completed']}")
    print(f"Duration: {metrics['summary']['experiment_duration_s']:.2f}s (virtual)")
    print(f"\nReal-time Throughput (PRIMARY METRIC):")
    print(f"  Average: {metrics['summary']['realtime_throughput_avg']:.2f} tasks/s")
    print(f"  Max:     {metrics['summary']['realtime_throughput_max']:.2f} tasks/s")
    print(f"  Min:     {metrics['summary']['realtime_throughput_min']:.2f} tasks/s")
    print(f"  Std:     {metrics['summary']['realtime_throughput_std']:.2f} tasks/s")
    print(f"\nCumulative Average Throughput: {metrics['summary']['average_throughput']:.2f} tasks/s")
    if metrics['summary']['phase_transition_delay_s']:
        print(f"Phase Transition: {metrics['summary']['phase_transition_delay_s']:.2f}s")
    print(f"\nOutputs:")
    print(f"  Metrics: {metrics_path}")
    print(f"  Gantt:   {gantt_path}")
    print(f"  Throughput: {throughput_path}")
    print("=" * 70)

    # Check if both Recovery and Baseline results exist, generate comparison plot
    base_output_dir = Path(args.output_dir)
    recovery_metrics_path = base_output_dir / "recovery" / "metrics.json"
    baseline_metrics_path = base_output_dir / "baseline" / "metrics.json"

    if recovery_metrics_path.exists() and baseline_metrics_path.exists():
        try:
            recovery_metrics = load_metrics(str(recovery_metrics_path))
            baseline_metrics = load_metrics(str(baseline_metrics_path))

            comparison_path = base_output_dir / "comparison.png"
            plot_recovery_vs_baseline(
                recovery_metrics,
                baseline_metrics,
                output_path=str(comparison_path),
                title=f"Recovery vs Baseline Comparison ({config.phase23_distribution})"
            )
            print(f"\nComparison plot generated: {comparison_path}")
        except Exception as e:
            logger.warning(f"Failed to generate comparison plot: {e}")

    if predictor:
        predictor.close()


if __name__ == "__main__":
    main()
