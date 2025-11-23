"""Text2Video workflow task submitters."""

import random
import threading
from queue import Queue, Empty
from typing import Any, Dict, List, Optional

from common import BaseTaskSubmitter
from .workflow_data import Text2VideoWorkflowData


class A1TaskSubmitter(BaseTaskSubmitter):
    """
    Submits A1 tasks (caption → positive prompt) with Poisson arrivals.

    Pre-generates all workflow data upfront, then submits A1 tasks
    following a Poisson process controlled by the rate limiter.
    """

    def __init__(self, captions: List[str], config, workflow_states: Dict,
                 state_lock: threading.Lock, **kwargs):
        """
        Initialize A1 task submitter.

        Args:
            captions: List of video captions
            config: Text2VideoConfig instance
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
            Text2VideoWorkflowData(
                workflow_id=f"workflow-{i:04d}",
                caption=captions[i % len(captions)],
                max_b_loops=config.max_b_loops
            )
            for i in range(config.num_workflows)
        ]

        # Populate shared workflow_states
        with state_lock:
            for workflow in self.workflows:
                workflow_states[workflow.workflow_id] = workflow

        self.index = 0
        self.logger.info(f"Pre-generated {len(self.workflows)} workflows")

    def _prepare_task_payload(self, workflow_data: Text2VideoWorkflowData) -> Dict[str, Any]:
        """
        Prepare task submission payload for A1.

        Args:
            workflow_data: Workflow data with caption

        Returns:
            JSON payload for /task/submit
        """
        # Generate random sleep time for simulation mode
        sleep_time = random.uniform(
            self.config.sleep_time_min,
            self.config.sleep_time_max
        )

        return {
            "task_id": f"{workflow_data.workflow_id}-A1",
            "model_id": self.config.model_a_id,
            "task_input": {"sleep_time": sleep_time},
            "metadata": {
                "workflow_id": workflow_data.workflow_id,
                "caption": workflow_data.caption,
                "task_type": "A1"
            }
        }

    def _get_next_task_data(self) -> Optional[Text2VideoWorkflowData]:
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


class A2TaskSubmitter(BaseTaskSubmitter):
    """
    Submits A2 tasks (positive prompt → negative prompt).

    Receives A1 completion events from queue and submits corresponding A2 tasks.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 a1_result_queue: Queue, **kwargs):
        """
        Initialize A2 task submitter.

        Args:
            config: Text2VideoConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            a1_result_queue: Queue receiving (workflow_id, a1_result) tuples from A1 receiver
            **kwargs: Passed to BaseTaskSubmitter
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.a1_result_queue = a1_result_queue

    def _prepare_task_payload(self, task_data: tuple) -> Dict[str, Any]:
        """
        Prepare task submission payload for A2.

        Args:
            task_data: Tuple of (workflow_id, a1_result)

        Returns:
            JSON payload for /task/submit
        """
        workflow_id, a1_result = task_data

        sleep_time = random.uniform(
            self.config.sleep_time_min,
            self.config.sleep_time_max
        )

        return {
            "task_id": f"{workflow_id}-A2",
            "model_id": self.config.model_a_id,
            "task_input": {"sleep_time": sleep_time},
            "metadata": {
                "workflow_id": workflow_id,
                "positive_prompt": a1_result,
                "task_type": "A2"
            }
        }

    def _get_next_task_data(self) -> Optional[tuple]:
        """
        Get next A2 task from queue (blocking until available).

        Returns:
            Tuple of (workflow_id, a1_result), or None if stopped
        """
        try:
            # Block for 0.1s to allow graceful shutdown
            return self.a1_result_queue.get(timeout=0.1)
        except Empty:
            # Queue empty, check if we should stop
            if self.stop_event.is_set():
                return None
            # Otherwise keep waiting
            return self._get_next_task_data()


class BTaskSubmitter(BaseTaskSubmitter):
    """
    Submits B tasks (video generation) with loop control.

    Receives A2 completion events from queue and submits B tasks.
    Implements B-loop logic: re-submits B task if loop count < max_b_loops.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 a2_result_queue: Queue, **kwargs):
        """
        Initialize B task submitter.

        Args:
            config: Text2VideoConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            a2_result_queue: Queue receiving (workflow_id, a2_result) tuples from A2 receiver
            **kwargs: Passed to BaseTaskSubmitter
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.a2_result_queue = a2_result_queue

    def _prepare_task_payload(self, task_data: tuple) -> Dict[str, Any]:
        """
        Prepare task submission payload for B.

        Args:
            task_data: Tuple of (workflow_id, a2_result, loop_iteration)

        Returns:
            JSON payload for /task/submit
        """
        workflow_id, a2_result, loop_iteration = task_data

        sleep_time = random.uniform(
            self.config.sleep_time_min,
            self.config.sleep_time_max
        )

        return {
            "task_id": f"{workflow_id}-B{loop_iteration}",
            "model_id": self.config.model_b_id,
            "task_input": {"sleep_time": sleep_time},
            "metadata": {
                "workflow_id": workflow_id,
                "negative_prompt": a2_result,
                "task_type": "B",
                "loop_iteration": loop_iteration
            }
        }

    def _get_next_task_data(self) -> Optional[tuple]:
        """
        Get next B task from queue (blocking until available).

        Returns:
            Tuple of (workflow_id, a2_result, loop_iteration), or None if stopped
        """
        try:
            # Block for 0.1s to allow graceful shutdown
            return self.a2_result_queue.get(timeout=0.1)
        except Empty:
            if self.stop_event.is_set():
                return None
            return self._get_next_task_data()

    def add_task(self, workflow_id: str, a2_result: str, loop_iteration: int):
        """
        Add a B task to the submission queue.

        Called by B receiver after completion to trigger next iteration.

        Args:
            workflow_id: Workflow identifier
            a2_result: Negative prompt from A2
            loop_iteration: Current loop iteration number (1-based)
        """
        self.a2_result_queue.put((workflow_id, a2_result, loop_iteration))
