"""Deep Research workflow task submitters."""

import random
import threading
from queue import Queue, Empty
from typing import Any, Dict, List, Optional

from common import BaseTaskSubmitter
from .workflow_data import DeepResearchWorkflowData


class ATaskSubmitter(BaseTaskSubmitter):
    """
    Submits A tasks (initial query) with Poisson arrivals.

    Pre-generates all workflow data upfront, then submits A tasks
    following a Poisson process controlled by the rate limiter.
    """

    def __init__(self, config, workflow_states: Dict,
                 state_lock: threading.Lock, **kwargs):
        """
        Initialize A task submitter.

        Args:
            config: DeepResearchConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            **kwargs: Passed to BaseTaskSubmitter (name, scheduler_url, rate_limiter, etc.)
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock

        # Pre-generate all workflow data
        self.workflows = [
            DeepResearchWorkflowData(
                workflow_id=f"workflow-{i:04d}",
                fanout_count=config.fanout_count
            )
            for i in range(config.num_workflows)
        ]

        # Populate shared workflow_states
        with state_lock:
            for workflow in self.workflows:
                workflow_states[workflow.workflow_id] = workflow

        self.index = 0
        self.logger.info(f"Pre-generated {len(self.workflows)} workflows")

    def _prepare_task_payload(self, workflow_data: DeepResearchWorkflowData) -> Dict[str, Any]:
        """
        Prepare task submission payload for A.

        Args:
            workflow_data: Workflow data

        Returns:
            JSON payload for /task/submit
        """
        sleep_time = random.uniform(
            self.config.sleep_time_min,
            self.config.sleep_time_max
        )

        return {
            "task_id": f"{workflow_data.workflow_id}-A",
            "model_id": self.config.model_a_id,
            "task_input": {"sleep_time": sleep_time},
            "metadata": {
                "workflow_id": workflow_data.workflow_id,
                "fanout_count": workflow_data.fanout_count,
                "task_type": "A"
            }
        }

    def _get_next_task_data(self) -> Optional[DeepResearchWorkflowData]:
        """
        Get next workflow to submit.

        Returns:
            Next workflow data, or None if all submitted
        """
        if self.index < len(self.workflows):
            workflow = self.workflows[self.index]
            self.index += 1
            return workflow
        return None


class B1TaskSubmitter(BaseTaskSubmitter):
    """
    Submits B1 tasks (first level parallel processing).

    Receives A completion events from queue and submits fanout_count B1 tasks.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 a_result_queue: Queue, **kwargs):
        """
        Initialize B1 task submitter.

        Args:
            config: DeepResearchConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            a_result_queue: Queue receiving (workflow_id, a_result) tuples from A receiver
            **kwargs: Passed to BaseTaskSubmitter
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.a_result_queue = a_result_queue

    def _prepare_task_payload(self, task_data: tuple) -> Dict[str, Any]:
        """
        Prepare task submission payload for B1.

        Args:
            task_data: Tuple of (workflow_id, a_result, b1_index)

        Returns:
            JSON payload for /task/submit
        """
        workflow_id, a_result, b1_index = task_data

        sleep_time = random.uniform(
            self.config.sleep_time_min,
            self.config.sleep_time_max
        )

        task_id = f"{workflow_id}-B1-{b1_index}"

        return {
            "task_id": task_id,
            "model_id": self.config.model_b_id,
            "task_input": {"sleep_time": sleep_time},
            "metadata": {
                "workflow_id": workflow_id,
                "a_result": a_result,
                "b1_index": b1_index,
                "task_type": "B1"
            }
        }

    def _get_next_task_data(self) -> Optional[tuple]:
        """
        Get next B1 task from queue (blocking until available).

        Returns:
            Tuple of (workflow_id, a_result, b1_index), or None if stopped
        """
        try:
            return self.a_result_queue.get(timeout=0.1)
        except Empty:
            if self.stop_event.is_set():
                return None
            return self._get_next_task_data()


class B2TaskSubmitter(BaseTaskSubmitter):
    """
    Submits B2 tasks (second level parallel processing).

    Receives B1 completion events from queue and submits corresponding B2 tasks.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 b1_result_queue: Queue, **kwargs):
        """
        Initialize B2 task submitter.

        Args:
            config: DeepResearchConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            b1_result_queue: Queue receiving (workflow_id, b1_result, b1_index) tuples
            **kwargs: Passed to BaseTaskSubmitter
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.b1_result_queue = b1_result_queue

    def _prepare_task_payload(self, task_data: tuple) -> Dict[str, Any]:
        """
        Prepare task submission payload for B2.

        Args:
            task_data: Tuple of (workflow_id, b1_result, b1_index)

        Returns:
            JSON payload for /task/submit
        """
        workflow_id, b1_result, b1_index = task_data

        sleep_time = random.uniform(
            self.config.sleep_time_min,
            self.config.sleep_time_max
        )

        task_id = f"{workflow_id}-B2-{b1_index}"

        return {
            "task_id": task_id,
            "model_id": self.config.model_b_id,
            "task_input": {"sleep_time": sleep_time},
            "metadata": {
                "workflow_id": workflow_id,
                "b1_result": b1_result,
                "b1_index": b1_index,
                "task_type": "B2"
            }
        }

    def _get_next_task_data(self) -> Optional[tuple]:
        """
        Get next B2 task from queue (blocking until available).

        Returns:
            Tuple of (workflow_id, b1_result, b1_index), or None if stopped
        """
        try:
            return self.b1_result_queue.get(timeout=0.1)
        except Empty:
            if self.stop_event.is_set():
                return None
            return self._get_next_task_data()


class MergeTaskSubmitter(BaseTaskSubmitter):
    """
    Submits Merge tasks (aggregation/finalization).

    Receives workflow_id from queue when all B2 tasks complete.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 merge_trigger_queue: Queue, **kwargs):
        """
        Initialize Merge task submitter.

        Args:
            config: DeepResearchConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            merge_trigger_queue: Queue receiving workflow_id when all B2 complete
            **kwargs: Passed to BaseTaskSubmitter
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.merge_trigger_queue = merge_trigger_queue

    def _prepare_task_payload(self, workflow_id: str) -> Dict[str, Any]:
        """
        Prepare task submission payload for Merge.

        Args:
            workflow_id: Workflow identifier

        Returns:
            JSON payload for /task/submit
        """
        sleep_time = random.uniform(
            self.config.sleep_time_min,
            self.config.sleep_time_max
        )

        # Collect all B2 results for aggregation
        with self.state_lock:
            workflow_data = self.workflow_states.get(workflow_id)
            b2_results = []
            if workflow_data:
                # In real mode, would collect actual B2 outputs
                # For simulation, just create placeholder
                b2_results = [f"b2_result_{i}" for i in range(workflow_data.fanout_count)]

        return {
            "task_id": f"{workflow_id}-Merge",
            "model_id": self.config.model_merge_id,
            "task_input": {"sleep_time": sleep_time},
            "metadata": {
                "workflow_id": workflow_id,
                "b2_results": b2_results,
                "task_type": "Merge"
            }
        }

    def _get_next_task_data(self) -> Optional[str]:
        """
        Get next Merge task from queue (blocking until available).

        Returns:
            workflow_id, or None if stopped
        """
        try:
            return self.merge_trigger_queue.get(timeout=0.1)
        except Empty:
            if self.stop_event.is_set():
                return None
            return self._get_next_task_data()
