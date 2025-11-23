"""Deep Research workflow task receivers."""

import threading
import time
from queue import Queue
from typing import Any, Dict

from common import BaseTaskReceiver


class ATaskReceiver(BaseTaskReceiver):
    """
    Receives A task results and triggers B1 task fan-out.

    Listens to Scheduler A WebSocket, extracts A results,
    updates workflow state, and enqueues fanout_count B1 tasks.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 b1_submitter, a_result_queue: Queue, **kwargs):
        """
        Initialize A task receiver.

        Args:
            config: DeepResearchConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            b1_submitter: B1TaskSubmitter instance
            a_result_queue: Queue to send (workflow_id, a_result, b1_index) tuples
            **kwargs: Passed to BaseTaskReceiver (name, scheduler_url, model_id, etc.)
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.b1_submitter = b1_submitter
        self.a_result_queue = a_result_queue

    def _get_subscription_payload(self) -> Dict[str, Any]:
        """Get WebSocket subscription payload."""
        return {
            "type": "subscribe",
            "model_id": self.model_id
        }

    async def _process_result(self, data: Dict[str, Any]):
        """
        Process A task completion result.

        Extracts A result, updates workflow state, and fans out to n B1 tasks.

        Args:
            data: Result data from WebSocket message
        """
        try:
            # Extract metadata
            metadata = data.get("metadata", {})
            workflow_id = metadata.get("workflow_id")
            fanout_count = metadata.get("fanout_count", self.config.fanout_count)

            if not workflow_id:
                self.logger.warning("Received A result without workflow_id")
                return

            # Extract result
            result = data.get("result", {})
            a_result = result.get("output", "")

            # Update workflow state
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    workflow_data.a_result = a_result
                    # Pre-allocate B1 task IDs
                    workflow_data.b1_task_ids = [
                        f"{workflow_id}-B1-{i}" for i in range(fanout_count)
                    ]
                    workflow_data.b2_task_ids = [
                        f"{workflow_id}-B2-{i}" for i in range(fanout_count)
                    ]
                else:
                    self.logger.warning(f"Workflow {workflow_id} not found in state")
                    return

            # Fan out to n B1 tasks
            for b1_index in range(fanout_count):
                self.a_result_queue.put((workflow_id, a_result, b1_index))

            self.logger.debug(
                f"A result processed for {workflow_id}, "
                f"triggered {fanout_count} B1 tasks"
            )

        except Exception as e:
            self.logger.error(f"Error processing A result: {e}", exc_info=True)


class B1TaskReceiver(BaseTaskReceiver):
    """
    Receives B1 task results and triggers corresponding B2 tasks.

    Listens to Scheduler B WebSocket, extracts B1 results,
    updates workflow state, and enqueues B2 tasks (1:1 mapping).
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 b2_submitter, b1_result_queue: Queue, **kwargs):
        """
        Initialize B1 task receiver.

        Args:
            config: DeepResearchConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            b2_submitter: B2TaskSubmitter instance
            b1_result_queue: Queue to send (workflow_id, b1_result, b1_index) tuples
            **kwargs: Passed to BaseTaskReceiver
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.b2_submitter = b2_submitter
        self.b1_result_queue = b1_result_queue

    def _get_subscription_payload(self) -> Dict[str, Any]:
        """Get WebSocket subscription payload."""
        return {
            "type": "subscribe",
            "model_id": self.model_id
        }

    async def _process_result(self, data: Dict[str, Any]):
        """
        Process B1 task completion result.

        Extracts B1 result, updates workflow state, and triggers corresponding B2 task.

        Args:
            data: Result data from WebSocket message
        """
        try:
            # Extract metadata
            metadata = data.get("metadata", {})
            workflow_id = metadata.get("workflow_id")
            b1_index = metadata.get("b1_index", 0)
            task_id = data.get("task_id")

            if not workflow_id or task_id is None:
                self.logger.warning("Received B1 result without workflow_id or task_id")
                return

            # Extract result
            result = data.get("result", {})
            b1_result = result.get("output", "")

            complete_time = time.time()

            # Update workflow state (atomic dict operation)
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    workflow_data.b1_complete_times[task_id] = complete_time
                else:
                    self.logger.warning(f"Workflow {workflow_id} not found in state")
                    return

            # Trigger corresponding B2 task (1:1 mapping)
            self.b1_result_queue.put((workflow_id, b1_result, b1_index))

            self.logger.debug(
                f"B1-{b1_index} complete for {workflow_id}, triggered B2-{b1_index}"
            )

        except Exception as e:
            self.logger.error(f"Error processing B1 result: {e}", exc_info=True)


class B2TaskReceiver(BaseTaskReceiver):
    """
    Receives B2 task results and triggers Merge when all B2 complete.

    Listens to Scheduler B WebSocket, tracks B2 completion,
    and triggers Merge task when all B2 tasks for a workflow are done.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 merge_submitter, merge_trigger_queue: Queue, **kwargs):
        """
        Initialize B2 task receiver.

        Args:
            config: DeepResearchConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            merge_submitter: MergeTaskSubmitter instance
            merge_trigger_queue: Queue to send workflow_id when all B2 complete
            **kwargs: Passed to BaseTaskReceiver
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.merge_submitter = merge_submitter
        self.merge_trigger_queue = merge_trigger_queue

    def _get_subscription_payload(self) -> Dict[str, Any]:
        """Get WebSocket subscription payload."""
        return {
            "type": "subscribe",
            "model_id": self.model_id
        }

    async def _process_result(self, data: Dict[str, Any]):
        """
        Process B2 task completion result.

        Checks if all B2 tasks complete, and if so, triggers Merge task.

        Args:
            data: Result data from WebSocket message
        """
        try:
            # Extract metadata
            metadata = data.get("metadata", {})
            workflow_id = metadata.get("workflow_id")
            task_id = data.get("task_id")

            if not workflow_id or task_id is None:
                self.logger.warning("Received B2 result without workflow_id or task_id")
                return

            complete_time = time.time()
            should_trigger_merge = False

            # Update workflow state and check for completion (atomic operations)
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if not workflow_data:
                    self.logger.warning(f"Workflow {workflow_id} not found in state")
                    return

                # Record completion time
                workflow_data.b2_complete_times[task_id] = complete_time

                # Check if all B2 tasks complete
                if workflow_data.all_b2_complete():
                    should_trigger_merge = True

            # Trigger Merge outside lock to avoid deadlock
            if should_trigger_merge:
                self.merge_trigger_queue.put(workflow_id)
                self.logger.debug(
                    f"All B2 complete for {workflow_id}, triggered Merge"
                )

        except Exception as e:
            self.logger.error(f"Error processing B2 result: {e}", exc_info=True)


class MergeTaskReceiver(BaseTaskReceiver):
    """
    Receives Merge task results and marks workflow complete.

    Listens to Scheduler A WebSocket, updates workflow completion time.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 **kwargs):
        """
        Initialize Merge task receiver.

        Args:
            config: DeepResearchConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            **kwargs: Passed to BaseTaskReceiver
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock

        # Track workflow completions
        self.completed_workflows = 0

    def _get_subscription_payload(self) -> Dict[str, Any]:
        """Get WebSocket subscription payload."""
        return {
            "type": "subscribe",
            "model_id": self.model_id
        }

    async def _process_result(self, data: Dict[str, Any]):
        """
        Process Merge task completion result.

        Marks workflow as complete.

        Args:
            data: Result data from WebSocket message
        """
        try:
            # Extract metadata
            metadata = data.get("metadata", {})
            workflow_id = metadata.get("workflow_id")

            if not workflow_id:
                self.logger.warning("Received Merge result without workflow_id")
                return

            complete_time = time.time()

            # Update workflow state
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    workflow_data.merge_complete_time = complete_time
                    self.completed_workflows += 1
                    self.logger.info(
                        f"Workflow {workflow_id} complete "
                        f"({self.completed_workflows} total, "
                        f"fanout={workflow_data.fanout_count})"
                    )
                else:
                    self.logger.warning(f"Workflow {workflow_id} not found in state")

        except Exception as e:
            self.logger.error(f"Error processing Merge result: {e}", exc_info=True)
