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
                 b1_submitter, a_result_queue: Queue, task_ids: list = None, **kwargs):
        """
        Initialize A task receiver.

        Args:
            config: DeepResearchConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            b1_submitter: B1TaskSubmitter instance
            a_result_queue: Queue to send (workflow_id, a_result, b1_index) tuples
            task_ids: List of task IDs to subscribe to
            **kwargs: Passed to BaseTaskReceiver (name, scheduler_url, model_id, etc.)
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.b1_submitter = b1_submitter
        self.a_result_queue = a_result_queue
        self.task_ids = task_ids or []

    def _get_subscription_payload(self) -> Dict[str, Any]:
        """Get WebSocket subscription payload."""
        return {
            "type": "subscribe",
            "task_ids": self.task_ids
        }

    async def _process_result(self, data: Dict[str, Any]):
        """
        Process A task completion result.

        Extracts A result, updates workflow state, and fans out to n B1 tasks.

        Args:
            data: Result data from WebSocket message
        """
        try:
            # Extract task ID for identification
            task_id = data.get("task_id")
            if not task_id:
                self.logger.warning("Received A result without task_id")
                return

            # Parse workflow_id from task_id (works for both simulation and real mode)
            # Format: task-A-{strategy}-workflow-{num}
            parts = task_id.split('-')
            workflow_id = None

            # Find "workflow" in parts and extract the number after it
            for i, part in enumerate(parts):
                if part == "workflow" and i + 1 < len(parts):
                    workflow_num = parts[i + 1]
                    workflow_id = f"workflow-{workflow_num}"
                    break

            if not workflow_id:
                self.logger.warning(f"Cannot parse workflow_id from task_id: {task_id}")
                return

            # Extract result
            result = data.get("result", {})
            a_result = result.get("output", "")

            # Record completion time
            complete_time = time.time()

            # Update workflow state
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    workflow_data.a_result = a_result
                    workflow_data.a_complete_time = complete_time

                    # Pre-allocate B1 task IDs with correct format
                    workflow_num = workflow_id.split('-')[-1]
                    workflow_data.b1_task_ids = [
                        f"task-B1-{workflow_data.strategy}-workflow-{workflow_num}-{i}"
                        for i in range(workflow_data.fanout_count)
                    ]
                    workflow_data.b2_task_ids = [
                        f"task-B2-{workflow_data.strategy}-workflow-{workflow_num}-{i}"
                        for i in range(workflow_data.fanout_count)
                    ]

                    fanout_count = workflow_data.fanout_count
                else:
                    self.logger.warning(f"Workflow {workflow_id} not found in state")
                    return

            # Fan out to n B1 tasks
            for b1_index in range(fanout_count):
                self.a_result_queue.put((workflow_id, a_result, b1_index))

                # Record B1 submit time
                with self.state_lock:
                    if b1_index < len(workflow_data.b1_submit_times):
                        workflow_data.b1_submit_times[b1_index] = time.time()
                    else:
                        workflow_data.b1_submit_times.append(time.time())

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
                 b2_submitter, b1_result_queue: Queue, task_ids: list = None, **kwargs):
        """
        Initialize B1 task receiver.

        Args:
            config: DeepResearchConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            b2_submitter: B2TaskSubmitter instance
            b1_result_queue: Queue to send (workflow_id, b1_result, b1_index) tuples
            task_ids: List of task IDs to subscribe to
            **kwargs: Passed to BaseTaskReceiver
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.b2_submitter = b2_submitter
        self.b1_result_queue = b1_result_queue
        self.task_ids = task_ids or []

    def _get_subscription_payload(self) -> Dict[str, Any]:
        """Get WebSocket subscription payload."""
        return {
            "type": "subscribe",
            "task_ids": self.task_ids
        }

    async def _process_result(self, data: Dict[str, Any]):
        """
        Process B1 task completion result.

        Extracts B1 result, updates workflow state, and triggers corresponding B2 task.

        Args:
            data: Result data from WebSocket message
        """
        try:
            # Extract task ID for identification
            task_id = data.get("task_id")
            if not task_id:
                self.logger.warning("Received B1 result without task_id")
                return

            # Parse task_id to extract workflow_id and b_index
            # Format: task-B1-{strategy}-workflow-{num}-{b_index}
            parts = task_id.split('-')
            workflow_id = None
            b_index = None

            # Find "workflow" in parts and extract the number and index after it
            for i, part in enumerate(parts):
                if part == "workflow" and i + 2 < len(parts):
                    workflow_num = parts[i + 1]
                    workflow_id = f"workflow-{workflow_num}"
                    # The b_index is the part after workflow_num
                    try:
                        b_index = int(parts[i + 2])
                    except (ValueError, IndexError):
                        self.logger.warning(f"Cannot parse b_index from task_id: {task_id}")
                        return
                    break

            if not workflow_id:
                self.logger.warning(f"Cannot parse workflow_id from task_id: {task_id}")
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

                    # Record B2 submit time
                    if b_index < len(workflow_data.b2_submit_times):
                        workflow_data.b2_submit_times[b_index] = time.time()
                    else:
                        workflow_data.b2_submit_times.append(time.time())
                else:
                    self.logger.warning(f"Workflow {workflow_id} not found in state")
                    return

            # Trigger corresponding B2 task (1:1 mapping)
            self.b1_result_queue.put((workflow_id, b1_result, b_index))

            self.logger.debug(
                f"B1-{b_index} complete for {workflow_id}, triggered B2-{b_index}"
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
                 merge_submitter, merge_trigger_queue: Queue, task_ids: list = None, **kwargs):
        """
        Initialize B2 task receiver.

        Args:
            config: DeepResearchConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            merge_submitter: MergeTaskSubmitter instance
            merge_trigger_queue: Queue to send workflow_id when all B2 complete
            task_ids: List of task IDs to subscribe to
            **kwargs: Passed to BaseTaskReceiver
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.merge_submitter = merge_submitter
        self.merge_trigger_queue = merge_trigger_queue
        self.task_ids = task_ids or []

    def _get_subscription_payload(self) -> Dict[str, Any]:
        """Get WebSocket subscription payload."""
        return {
            "type": "subscribe",
            "task_ids": self.task_ids
        }

    async def _process_result(self, data: Dict[str, Any]):
        """
        Process B2 task completion result.

        Checks if all B2 tasks complete, and if so, triggers Merge task.

        Args:
            data: Result data from WebSocket message
        """
        try:
            # Extract task ID for identification
            task_id = data.get("task_id")
            if not task_id:
                self.logger.warning("Received B2 result without task_id")
                return

            # Parse task_id to extract workflow_id
            # Format: task-B2-{strategy}-workflow-{num}-{b_index}
            parts = task_id.split('-')
            workflow_id = None

            # Find "workflow" in parts and extract the number after it
            for i, part in enumerate(parts):
                if part == "workflow" and i + 1 < len(parts):
                    workflow_num = parts[i + 1]
                    workflow_id = f"workflow-{workflow_num}"
                    break

            if not workflow_id:
                self.logger.warning(f"Cannot parse workflow_id from task_id: {task_id}")
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
                    # Record merge submit time
                    workflow_data.merge_submit_time = time.time()

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
                 task_ids: list = None, **kwargs):
        """
        Initialize Merge task receiver.

        Args:
            config: DeepResearchConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            task_ids: List of task IDs to subscribe to
            **kwargs: Passed to BaseTaskReceiver
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.task_ids = task_ids or []

        # Track workflow completions
        self.completed_workflows = 0

    def _get_subscription_payload(self) -> Dict[str, Any]:
        """Get WebSocket subscription payload."""
        return {
            "type": "subscribe",
            "task_ids": self.task_ids
        }

    async def _process_result(self, data: Dict[str, Any]):
        """
        Process Merge task completion result.

        Marks workflow as complete.

        Args:
            data: Result data from WebSocket message
        """
        try:
            # Extract task ID for identification
            task_id = data.get("task_id")
            if not task_id:
                self.logger.warning("Received Merge result without task_id")
                return

            # Parse task_id to extract workflow_id
            # Format: task-merge-{strategy}-workflow-{num}
            parts = task_id.split('-')
            workflow_id = None

            # Find "workflow" in parts and extract the number after it
            for i, part in enumerate(parts):
                if part == "workflow" and i + 1 < len(parts):
                    workflow_num = parts[i + 1]
                    workflow_id = f"workflow-{workflow_num}"
                    break

            if not workflow_id:
                self.logger.warning(f"Cannot parse workflow_id from task_id: {task_id}")
                return

            complete_time = time.time()

            # Update workflow state
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    workflow_data.merge_complete_time = complete_time
                    workflow_data.workflow_complete_time = complete_time
                    self.completed_workflows += 1

                    # Record workflow completion in metrics
                    if self.metrics:
                        self.metrics.record_workflow_complete(
                            workflow_id=workflow_id,
                            successful_tasks=workflow_data.fanout_count * 2 + 2,  # A + B1*n + B2*n + Merge
                            failed_tasks=0
                        )

                    self.logger.info(
                        f"Workflow {workflow_id} complete "
                        f"({self.completed_workflows} total, "
                        f"fanout={workflow_data.fanout_count})"
                    )
                else:
                    self.logger.warning(f"Workflow {workflow_id} not found in state")

        except Exception as e:
            self.logger.error(f"Error processing Merge result: {e}", exc_info=True)
