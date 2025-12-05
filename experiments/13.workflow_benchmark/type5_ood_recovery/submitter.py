"""
OOD Task Submitter with Three-Phase Logic and Continuous Submission.

This module implements the task submitter that handles:
- Phase 1: Correct prediction (warmup) - fixed count
- Phase 2: OOD - wrong prediction - initial fixed count
- Phase 3 (Recovery) / Phase 2 continued (Baseline): Continuous submission until target completions

The continuous submission mode simulates real-world scenarios where the system
keeps processing new tasks after the predictor is corrected.
"""

import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.base_classes import BaseTaskSubmitter
from common.rate_limiter import RateLimiter
from .config import OODRecoveryConfig
from .task_data import OODTaskData, TaskGenerator, get_qps_scale_factor


class OODTaskSubmitter(BaseTaskSubmitter):
    """
    Task submitter with three-phase OOD pattern and continuous submission.

    Submission modes:
    - Phase 1: Submit phase1_count tasks with correct predictions
    - Phase 2: Submit tasks with wrong predictions
    - Continuous mode (Phase 3 for Recovery, Phase 2 for Baseline):
      Keep submitting new tasks until num_tasks completions are received

    Phase transitions:
    - Phase 1 → Phase 2: After phase1_count tasks submitted
    - Phase 2 → Phase 3: When phase_transition_event is set (by receiver)

    In baseline mode (no_recovery=True), never transitions to Phase 3,
    but still uses continuous submission in Phase 2.
    """

    def __init__(
        self,
        config: OODRecoveryConfig,
        tasks: List[OODTaskData],
        task_lookup: Dict[str, OODTaskData],
        phase_transition_event: threading.Event,
        task_generator: TaskGenerator,
        get_total_completed: Callable[[], int],
        rate_limiter: Optional[RateLimiter] = None,
        metrics: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize OOD task submitter.

        Args:
            config: OODRecoveryConfig instance
            tasks: Pre-generated list of OODTaskData for initial phases
            task_lookup: Dict mapping task_id to OODTaskData (shared with receiver)
            phase_transition_event: Threading event set when Phase 2 transition count reached
            task_generator: TaskGenerator for creating new tasks in continuous mode
            get_total_completed: Callable to get current total completed count from receiver
            rate_limiter: Optional RateLimiter for QPS scaling at phase transitions
            metrics: Optional MetricsCollector for task tracking
            **kwargs: Additional arguments for BaseTaskSubmitter
        """
        super().__init__(
            name="OODSubmitter",
            scheduler_url=config.scheduler_url,
            qps=config.qps,
            duration=config.duration,
            metrics=metrics,
            **kwargs
        )

        self.config = config
        self.tasks = tasks
        self.task_lookup = task_lookup
        self.phase_transition_event = phase_transition_event
        self.task_generator = task_generator
        self.get_total_completed = get_total_completed
        self.rate_limiter = rate_limiter

        # QPS scaling info (calculated once at init)
        self._qps_scale_factor = get_qps_scale_factor(config)
        self._original_qps = config.qps
        self._scaled_qps = self._original_qps * self._qps_scale_factor

        # Phase tracking
        self.phase1_count = config.phase1_count
        self.current_index = 0
        self.current_phase = 1
        self.phase2_started = False
        self.continuous_mode = False

        # Phase counters
        self.phase_submit_counts = {1: 0, 2: 0, 3: 0}
        self._phase_lock = threading.Lock()

    def _should_continue_submitting(self) -> bool:
        """
        Check if we should continue submitting tasks.

        Continue until num_tasks completions are received.

        Returns:
            True if more tasks should be submitted
        """
        total_completed = self.get_total_completed()
        return total_completed < self.config.num_tasks

    def _get_next_task_data(self) -> Optional[OODTaskData]:
        """
        Get the next task to submit, determining its phase.

        Phase logic:
        - Tasks 0 to (phase1_count - 1): Phase 1
        - Tasks phase1_count onwards: Phase 2 or 3
        - In continuous mode: Generate new tasks dynamically

        Phase transitions:
        - Phase 1 → Phase 2: After phase1_count tasks submitted
        - Phase 2 → Phase 3: When phase_transition_event is set (Recovery only)
        - Enter continuous mode: After initial tasks exhausted in final phase

        Returns:
            OODTaskData for the next task, or None if should stop
        """
        if not self._should_continue_submitting():
            return None

        # Check if we need to transition to continuous mode
        if self.current_index >= len(self.tasks) and not self.continuous_mode:
            # All pre-generated tasks submitted, enter continuous mode
            self.continuous_mode = True
            self.logger.info(
                f"Entering continuous submission mode at index {self.current_index} "
                f"(Phase {self.current_phase})"
            )

        # Get or generate task
        if self.continuous_mode:
            # Generate new task dynamically
            task = self.task_generator.generate_task(self.current_index)
        else:
            # Use pre-generated task
            task = self.tasks[self.current_index]

        # Determine phase for this task
        if self.current_index < self.phase1_count:
            # Phase 1 tasks
            task.phase = 1
        else:
            # Phase 2 or Phase 3 tasks
            if not self.phase2_started:
                self.phase2_started = True
                self.current_phase = 2
                self.logger.info(f"Entering Phase 2 at task index {self.current_index}")

                # Scale QPS at Phase 1 → Phase 2 boundary
                # QPS scales inversely with task duration: longer tasks = lower QPS
                if self.rate_limiter is not None:
                    self.logger.info(
                        f"Scaling QPS at Phase 1→2 boundary: "
                        f"{self._original_qps:.2f} → {self._scaled_qps:.2f} "
                        f"(factor: {self._qps_scale_factor:.4f})"
                    )
                    self.rate_limiter.set_rate(self._scaled_qps)

            # Check for Phase 2 → Phase 3 transition
            if (self.current_phase == 2 and
                self.phase_transition_event.is_set() and
                not self.config.no_recovery):
                self.current_phase = 3
                self.logger.info(f"Transitioning to Phase 3 at task index {self.current_index}")

            task.phase = self.current_phase

        # Calculate actual sleep time and exp_runtime based on phase
        task.calculate_times(self.config)

        self.current_index += 1

        # Update phase counters
        with self._phase_lock:
            self.phase_submit_counts[task.phase] += 1

            # Check for submission-based Phase 2→3 transition trigger
            if (self.config.transition_on_submit and
                task.phase == 2 and
                not self.phase_transition_event.is_set() and
                not self.config.no_recovery and
                self.phase_submit_counts[2] >= self.config.phase2_transition_count):
                self.phase_transition_event.set()
                self.logger.info(
                    f"Phase 2→3 transition triggered after {self.phase_submit_counts[2]} "
                    f"Phase 2 tasks SUBMITTED (task index {self.current_index})"
                )

        return task

    def _prepare_task_payload(self, task_data: OODTaskData) -> Dict[str, Any]:
        """
        Prepare the JSON payload for task submission.

        Args:
            task_data: OODTaskData instance

        Returns:
            Task submission payload with sleep_time and exp_runtime
        """
        # Record submit time
        task_data.submit_time = time.time()

        # Store in lookup for receiver
        self.task_lookup[task_data.task_id] = task_data

        # Record workflow start in metrics
        if self.metrics:
            self.metrics.record_workflow_start(
                workflow_id=task_data.task_id,
                workflow_type="ood_task"
            )
            self.metrics.record_task_submit(
                task_id=task_data.task_id,
                workflow_id=task_data.task_id,
                task_type=f"phase{task_data.phase}"
            )

        return {
            "task_id": task_data.task_id,
            "model_id": self.config.model_id,
            "task_input": {
                "sleep_time": task_data.actual_sleep_time,
            },
            "metadata": {
                "exp_runtime": task_data.exp_runtime_ms,
                "exp_cv": 0.40,
                "phase": task_data.phase,
                "task_index": task_data.task_index,
                "base_sleep_time": task_data.base_sleep_time,
                "strategy": self.config.strategy,
            }
        }

    def get_phase_submit_counts(self) -> Dict[int, int]:
        """Get the number of tasks submitted per phase."""
        with self._phase_lock:
            return dict(self.phase_submit_counts)

    def is_continuous_mode(self) -> bool:
        """Check if submitter is in continuous mode."""
        return self.continuous_mode

    def run(self):
        """Override run to add phase tracking logging."""
        trigger_mode = "submit" if self.config.transition_on_submit else "complete"
        self.logger.info(
            f"Starting OOD task submission: "
            f"Phase1={self.phase1_count} tasks, "
            f"Initial Phase2+3={self.config.phase23_count} tasks, "
            f"Target completions={self.config.num_tasks}, "
            f"QPS={self.config.qps}, "
            f"NoRecovery={self.config.no_recovery}, "
            f"TransitionTrigger={trigger_mode}(count={self.config.phase2_transition_count})"
        )

        # Call parent run
        super().run()

        # Log final phase counts
        counts = self.get_phase_submit_counts()
        total_submitted = sum(counts.values())
        self.logger.info(
            f"OOD task submission complete: "
            f"Phase1={counts.get(1, 0)}, "
            f"Phase2={counts.get(2, 0)}, "
            f"Phase3={counts.get(3, 0)}, "
            f"Total={total_submitted}, "
            f"ContinuousMode={self.continuous_mode}"
        )
