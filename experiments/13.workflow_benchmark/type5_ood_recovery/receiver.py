"""
OOD Task Receiver with Phase Transition Detection and Metadata Update.

This module implements the task receiver that:
- Receives task completion results via WebSocket
- Detects first Phase 2 completion to trigger Phase 3 transition
- Updates pending Phase 2 tasks with correct exp_runtime when transition occurs
- Tracks per-phase completion metrics
"""

import asyncio
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.base_classes import BaseTaskReceiver
from .config import OODRecoveryConfig
from .task_data import OODTaskData


class OODTaskReceiver(BaseTaskReceiver):
    """
    Task receiver with phase transition detection.

    Detects when Phase 2 tasks complete to trigger Phase 3 transition.
    Tracks total completions for the continuous submission mode.
    """

    # Maximum number of tasks to subscribe to (for continuous submission mode)
    MAX_TASK_IDS = 10000

    def __init__(
        self,
        config: OODRecoveryConfig,
        task_lookup: Dict[str, OODTaskData],
        phase_transition_event: threading.Event,
        metrics: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize OOD task receiver.

        Args:
            config: OODRecoveryConfig instance
            task_lookup: Dict mapping task_id to OODTaskData (shared with submitter)
            phase_transition_event: Threading event to set when phase2_transition_count Phase 2 tasks complete
            metrics: Optional MetricsCollector for task tracking
            **kwargs: Additional arguments for BaseTaskReceiver
        """
        super().__init__(
            name="OODReceiver",
            scheduler_url=config.scheduler_url,
            model_id=config.model_id,
            metrics=metrics,
            **kwargs
        )

        self.config = config
        self.task_lookup = task_lookup
        self.phase_transition_event = phase_transition_event

        # Pre-generate task IDs for subscription (supports continuous submission)
        # Generate more than needed to handle dynamic task generation
        self.task_ids = [f"task-ood-{i:04d}" for i in range(self.MAX_TASK_IDS)]

        # Phase tracking
        self.phase_transition_triggered = False
        self.phase_transition_time: Optional[float] = None
        self.phase_complete_counts = {1: 0, 2: 0, 3: 0}
        self._phase_lock = threading.Lock()

        # Completion tracking
        self.completed_tasks: List[OODTaskData] = []
        self._completed_lock = threading.Lock()

    def _get_subscription_payload(self) -> Dict[str, Any]:
        """
        Get the subscription payload for WebSocket connection.

        Subscribes to a large pool of pre-generated task IDs to support
        continuous/dynamic task submission.

        Returns:
            Subscription message with task IDs
        """
        return {
            "type": "subscribe",
            "model_id": self.config.model_id,
            "task_ids": self.task_ids,
        }

    async def _update_pending_phase2_metadata(self) -> Dict[str, Any]:
        """
        Update metadata for all pending Phase 2 tasks with correct exp_runtime.

        When Phase 2→3 transition occurs, this method:
        1. Finds all submitted but not completed Phase 2 tasks
        2. Calculates correct exp_runtime (actual_sleep_time * 1000.0)
        3. Calls /task/update_metadata API to trigger re-prediction

        This allows the scheduler to re-predict and re-schedule already-submitted
        Phase 2 tasks using the correct runtime estimate.

        Returns:
            Dict with update results from the API
        """
        # Find all pending Phase 2 tasks (submitted but not completed)
        pending_phase2_tasks = []
        for task_id, task in self.task_lookup.items():
            if task.phase == 2 and not task.is_complete:
                pending_phase2_tasks.append(task)

        if not pending_phase2_tasks:
            self.logger.info("No pending Phase 2 tasks to update")
            return {"success": True, "total": 0, "updated": 0}

        # Prepare update payload
        updates = []
        for task in pending_phase2_tasks:
            # Calculate correct exp_runtime (same as Phase 3 calculation)
            correct_exp_runtime_ms = task.actual_sleep_time * 1000.0
            updates.append({
                "task_id": task.task_id,
                "metadata": {
                    "exp_runtime": correct_exp_runtime_ms,
                    "exp_cv": 0.40,
                    "phase": task.phase,  # Keep as phase 2 (task already submitted)
                    "task_index": task.task_index,
                    "base_sleep_time": task.base_sleep_time,
                    "strategy": self.config.strategy,
                    "metadata_updated_at_transition": True,  # Mark as updated
                }
            })

        self.logger.info(
            f"Updating {len(updates)} pending Phase 2 tasks with correct exp_runtime"
        )

        # Call the update_metadata API
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.config.scheduler_url}/task/update_metadata",
                    json={"updates": updates},
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()

                self.logger.info(
                    f"Metadata update result: {result.get('succeeded', 0)}/{result.get('total', 0)} "
                    f"succeeded, {result.get('failed', 0)} failed, {result.get('skipped', 0)} skipped"
                )

                # Log sample of updated predictions
                if result.get("results"):
                    sample_results = result["results"][:3]
                    for r in sample_results:
                        if r.get("queue_updated"):
                            self.logger.debug(
                                f"  {r['task_id']}: "
                                f"{r.get('old_prediction_ms', 'N/A')}ms → "
                                f"{r.get('new_prediction_ms', 'N/A')}ms"
                            )

                return result

        except httpx.HTTPError as e:
            self.logger.error(f"Failed to update Phase 2 metadata: {e}")
            return {"success": False, "error": str(e)}

    async def _process_result(self, data: Dict[str, Any]):
        """
        Process a task completion result.

        Args:
            data: Result data from WebSocket message
        """
        task_id = data.get("task_id")
        if not task_id:
            self.logger.warning("Received result without task_id")
            return

        # Look up task data
        task = self.task_lookup.get(task_id)
        if not task:
            self.logger.warning(f"Unknown task_id: {task_id}")
            return

        # Record completion
        task.complete_time = time.time()
        task.is_complete = True

        # Update phase complete counts
        with self._phase_lock:
            self.phase_complete_counts[task.phase] += 1

        # Track completed tasks
        with self._completed_lock:
            self.completed_tasks.append(task)

        # Record in metrics
        if self.metrics:
            self.metrics.record_task_complete(
                task_id=task_id,
                success=True
            )
            self.metrics.record_workflow_complete(
                workflow_id=task_id,
                successful_tasks=1,
                failed_tasks=0
            )

        # Check for Phase 2 → Phase 3 transition trigger
        # Transition occurs after phase2_transition_count Phase 2 tasks complete
        if (task.phase == 2 and
            not self.phase_transition_triggered and
            not self.config.no_recovery):
            phase2_count = self.phase_complete_counts[2]
            if phase2_count >= self.config.phase2_transition_count:
                self.phase_transition_triggered = True
                self.phase_transition_time = time.time()
                self.phase_transition_event.set()
                self.logger.info(
                    f"Phase 2 transition triggered after {phase2_count} Phase 2 tasks completed "
                    f"(last: {task_id}, duration: {task.duration:.2f}s) - Starting Phase 3"
                )

                # Update pending Phase 2 tasks with correct exp_runtime
                # This allows the scheduler to re-predict already-submitted tasks
                try:
                    await self._update_pending_phase2_metadata()
                except Exception as e:
                    self.logger.error(f"Error updating pending Phase 2 tasks: {e}")

    def get_phase_complete_counts(self) -> Dict[int, int]:
        """Get the number of tasks completed per phase."""
        with self._phase_lock:
            return dict(self.phase_complete_counts)

    def get_completed_tasks(self) -> List[OODTaskData]:
        """Get list of completed tasks."""
        with self._completed_lock:
            return list(self.completed_tasks)

    def get_total_completed(self) -> int:
        """Get total number of completed tasks."""
        with self._completed_lock:
            return len(self.completed_tasks)

    def get_phase_transition_time(self) -> Optional[float]:
        """Get timestamp when phase transition was triggered."""
        return self.phase_transition_time

    def run(self):
        """Override run to add completion tracking logging."""
        self.logger.info(
            f"Starting OOD task receiver: "
            f"Subscribing to {len(self.task_ids)} task IDs for continuous task reception"
        )

        # Call parent run
        super().run()

        # Log final phase counts
        counts = self.get_phase_complete_counts()
        self.logger.info(
            f"OOD task receiver stopped: "
            f"Phase1={counts.get(1, 0)}, "
            f"Phase2={counts.get(2, 0)}, "
            f"Phase3={counts.get(3, 0)} completed"
        )
