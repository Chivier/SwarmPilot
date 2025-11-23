"""Text2Video workflow task receivers."""

import threading
import time
from queue import Queue
from typing import Any, Dict

from common import BaseTaskReceiver


class A1TaskReceiver(BaseTaskReceiver):
    """
    Receives A1 task results and triggers A2 task submission.

    Listens to Scheduler A WebSocket, extracts positive prompts,
    updates workflow state, and enqueues A2 tasks.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 a2_submitter, a1_result_queue: Queue, **kwargs):
        """
        Initialize A1 task receiver.

        Args:
            config: Text2VideoConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            a2_submitter: A2TaskSubmitter instance (for metrics tracking)
            a1_result_queue: Queue to send (workflow_id, a1_result) tuples
            **kwargs: Passed to BaseTaskReceiver (name, ws_url, model_id, etc.)
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.a2_submitter = a2_submitter
        self.a1_result_queue = a1_result_queue

    def _get_subscription_payload(self) -> Dict[str, Any]:
        """
        Get WebSocket subscription payload.

        Returns:
            Subscription message for scheduler WebSocket
        """
        return {
            "type": "subscribe",
            "model_id": self.model_id
        }

    async def _process_result(self, data: Dict[str, Any]):
        """
        Process A1 task completion result.

        Extracts positive prompt, updates workflow state, and triggers A2.

        Args:
            data: Result data from WebSocket message
        """
        try:
            # Extract metadata
            metadata = data.get("metadata", {})
            workflow_id = metadata.get("workflow_id")

            if not workflow_id:
                self.logger.warning("Received A1 result without workflow_id")
                return

            # Extract result (positive prompt)
            result = data.get("result", {})
            a1_result = result.get("output", "")

            # Update workflow state
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    workflow_data.a1_result = a1_result
                else:
                    self.logger.warning(f"Workflow {workflow_id} not found in state")
                    return

            # Trigger A2 task submission
            self.a1_result_queue.put((workflow_id, a1_result))

            self.logger.debug(f"A1 result processed for {workflow_id}, triggered A2")

        except Exception as e:
            self.logger.error(f"Error processing A1 result: {e}", exc_info=True)


class A2TaskReceiver(BaseTaskReceiver):
    """
    Receives A2 task results and triggers B task submission.

    Listens to Scheduler A WebSocket, extracts negative prompts,
    updates workflow state, and enqueues first B task.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 b_submitter, a2_result_queue: Queue, **kwargs):
        """
        Initialize A2 task receiver.

        Args:
            config: Text2VideoConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            b_submitter: BTaskSubmitter instance
            a2_result_queue: Queue to send (workflow_id, a2_result, loop_iteration) tuples
            **kwargs: Passed to BaseTaskReceiver
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.b_submitter = b_submitter
        self.a2_result_queue = a2_result_queue

    def _get_subscription_payload(self) -> Dict[str, Any]:
        """Get WebSocket subscription payload."""
        return {
            "type": "subscribe",
            "model_id": self.model_id
        }

    async def _process_result(self, data: Dict[str, Any]):
        """
        Process A2 task completion result.

        Extracts negative prompt, updates workflow state, and triggers first B task.

        Args:
            data: Result data from WebSocket message
        """
        try:
            # Extract metadata
            metadata = data.get("metadata", {})
            workflow_id = metadata.get("workflow_id")

            if not workflow_id:
                self.logger.warning("Received A2 result without workflow_id")
                return

            # Extract result (negative prompt)
            result = data.get("result", {})
            a2_result = result.get("output", "")

            # Update workflow state
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    workflow_data.a2_result = a2_result
                    # Initialize B loop counter
                    workflow_data.b_loop_count = 1
                else:
                    self.logger.warning(f"Workflow {workflow_id} not found in state")
                    return

            # Trigger first B task (loop_iteration = 1)
            self.a2_result_queue.put((workflow_id, a2_result, 1))

            self.logger.debug(f"A2 result processed for {workflow_id}, triggered B1")

        except Exception as e:
            self.logger.error(f"Error processing A2 result: {e}", exc_info=True)


class BTaskReceiver(BaseTaskReceiver):
    """
    Receives B task results and implements loop logic.

    Listens to Scheduler B WebSocket, tracks B iterations, and either:
    - Re-submits B task if loop_count < max_b_loops
    - Marks workflow complete if all iterations done
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 b_submitter, **kwargs):
        """
        Initialize B task receiver.

        Args:
            config: Text2VideoConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            b_submitter: BTaskSubmitter instance (for loop re-submission)
            **kwargs: Passed to BaseTaskReceiver
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.b_submitter = b_submitter

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
        Process B task completion result.

        Implements B-loop logic: re-submit if more iterations needed,
        otherwise mark workflow complete.

        Args:
            data: Result data from WebSocket message
        """
        try:
            # Extract metadata
            metadata = data.get("metadata", {})
            workflow_id = metadata.get("workflow_id")
            loop_iteration = metadata.get("loop_iteration", 1)

            if not workflow_id:
                self.logger.warning("Received B result without workflow_id")
                return

            complete_time = time.time()

            # Update workflow state
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if not workflow_data:
                    self.logger.warning(f"Workflow {workflow_id} not found in state")
                    return

                # Record completion time
                workflow_data.b_complete_times.append(complete_time)

                # Check if we should continue B loop
                if workflow_data.should_continue_b_loop():
                    # Increment loop counter for next iteration
                    workflow_data.b_loop_count += 1
                    next_iteration = workflow_data.b_loop_count

                    # Get a2_result for re-submission
                    a2_result = workflow_data.a2_result

                    self.logger.debug(
                        f"B{loop_iteration} complete for {workflow_id}, "
                        f"triggering B{next_iteration}"
                    )

                    # Trigger next B iteration (outside lock to avoid deadlock)
                    # Use b_submitter.add_task() instead of queue.put()
                    self.b_submitter.add_task(workflow_id, a2_result, next_iteration)

                elif workflow_data.is_complete():
                    # All B iterations complete
                    self.completed_workflows += 1
                    self.logger.info(
                        f"Workflow {workflow_id} complete "
                        f"({self.completed_workflows} total, "
                        f"{workflow_data.max_b_loops} B iterations)"
                    )

        except Exception as e:
            self.logger.error(f"Error processing B result: {e}", exc_info=True)
