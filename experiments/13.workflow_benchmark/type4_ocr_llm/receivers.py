"""OCR+LLM workflow task receivers."""

import threading
import time
from queue import Queue
from typing import Any, Dict, List

from common import BaseTaskReceiver


class ATaskReceiver(BaseTaskReceiver):
    """
    Receives A task (OCR) results and triggers B task submission.

    Listens to Scheduler A WebSocket, extracts OCR text,
    updates workflow state, and enqueues B tasks.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 b_submitter, a_result_queue: Queue, task_ids: list = None, **kwargs):
        """
        Initialize A task receiver (OCR).

        Args:
            config: OCRLLMConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            b_submitter: BTaskSubmitter instance (for metrics tracking)
            a_result_queue: Queue to send (workflow_id, a_result) tuples
            task_ids: List of task IDs to subscribe to
            **kwargs: Passed to BaseTaskReceiver (name, ws_url, model_id, etc.)
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.b_submitter = b_submitter
        self.a_result_queue = a_result_queue
        self.task_ids = task_ids or []

    def _get_subscription_payload(self) -> Dict[str, Any]:
        """
        Get WebSocket subscription payload.

        Returns:
            Subscription message for scheduler WebSocket
        """
        return {
            "type": "subscribe",
            "task_ids": self.task_ids
        }

    async def _process_result(self, data: Dict[str, Any]):
        """
        Process A task (OCR) completion result.

        Extracts OCR text, updates workflow state, and triggers B task.

        Args:
            data: Result data from WebSocket message
        """
        try:
            # Extract task_id
            task_id = data.get("task_id")
            if not task_id:
                self.logger.warning("Received result without task_id")
                return

            # Extract workflow_id from task_id
            # Format: task-A-{strategy}-workflow-XXXX
            workflow_id = None
            if task_id:
                parts = task_id.split("-")
                if "workflow" in parts:
                    idx = parts.index("workflow")
                    if idx + 1 < len(parts):
                        workflow_id = f"workflow-{parts[idx + 1]}"

                if not workflow_id:
                    self.logger.warning(f"Cannot extract workflow_id from task_id: {task_id}")
                    return

            # Check task status
            status = data.get("status")
            if status != "completed":
                self.logger.warning(f"A task {task_id} failed with status: {status}")
                return

            # Extract result (OCR text)
            # Support multiple result formats:
            # - Simulation mode: {"output": "..."}
            # - Real OCR mode: {"result": {"text": "...", "raw_results": [...]}}
            result = data.get("result", {})
            a_result = ""

            if isinstance(result, dict):
                # Try nested format first: result.result
                if "result" in result and isinstance(result.get("result"), dict):
                    inner = result.get("result", {})
                    # OCR returns "text" field
                    a_result = inner.get("text", inner.get("output", ""))
                # Then try direct format
                elif "text" in result:
                    a_result = result.get("text", "")
                elif "output" in result:
                    a_result = result.get("output", "")
                # Fallback: convert result to string
                else:
                    a_result = str(result) if result else ""
            else:
                a_result = str(result) if result else ""

            # Update workflow state
            complete_time = time.time()
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    workflow_data.a_result = a_result
                    workflow_data.a_complete_time = complete_time

                    # Record task completion in metrics
                    if self.metrics:
                        workflow_num = workflow_id.split('-')[-1]
                        a_task_id = f"task-A-{workflow_data.strategy}-workflow-{workflow_num}"
                        self.metrics.record_task_complete(
                            task_id=a_task_id,
                            success=True
                        )
                else:
                    self.logger.warning(f"Workflow {workflow_id} not found in state")
                    return

            # Trigger B task submission
            self.a_result_queue.put((workflow_id, a_result))

            self.logger.debug(f"A task {task_id} processed for {workflow_id}, triggered B")

        except Exception as e:
            self.logger.error(f"Error processing A result: {e}", exc_info=True)


class BTaskReceiver(BaseTaskReceiver):
    """
    Receives B task (LLM) results and marks workflow complete.

    Listens to Scheduler B WebSocket and tracks workflow completion.
    This is the final stage of the OCR+LLM workflow.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 task_ids: list = None, **kwargs):
        """
        Initialize B task receiver (LLM).

        Args:
            config: OCRLLMConfig instance
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
        Process B task (LLM) completion result.

        Marks workflow as complete since this is the final stage.

        Args:
            data: Result data from WebSocket message
        """
        try:
            # Extract task_id
            task_id = data.get("task_id")
            if not task_id:
                self.logger.warning("Received result without task_id")
                return

            # Extract workflow_id from task_id
            # Format: task-B-{strategy}-workflow-XXXX
            workflow_id = None
            if task_id:
                parts = task_id.split("-")
                if "workflow" in parts:
                    idx = parts.index("workflow")
                    if idx + 1 < len(parts):
                        workflow_id = f"workflow-{parts[idx + 1]}"

                if not workflow_id:
                    self.logger.warning(f"Cannot extract workflow_id from task_id: {task_id}")
                    return

            # Check task status
            status = data.get("status")
            if status != "completed":
                self.logger.warning(f"B task {task_id} failed with status: {status}")
                # Mark workflow as failed/complete even on failure
                with self.state_lock:
                    workflow_data = self.workflow_states.get(workflow_id)
                    if workflow_data:
                        workflow_data.workflow_complete_time = time.time()

                        # Record workflow completion in metrics (failed)
                        if self.metrics:
                            self.metrics.record_workflow_complete(
                                workflow_id=workflow_id,
                                successful_tasks=1,  # A task completed
                                failed_tasks=1  # B task failed
                            )
                return

            # Extract result (LLM output)
            result = data.get("result", {})
            b_result = ""

            if isinstance(result, dict):
                # Try nested format first: result.result.output
                if "result" in result and isinstance(result.get("result"), dict):
                    b_result = result.get("result", {}).get("output", "")
                elif "output" in result:
                    b_result = result.get("output", "")
                else:
                    b_result = str(result) if result else ""
            else:
                b_result = str(result) if result else ""

            complete_time = time.time()

            # Update workflow state
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if not workflow_data:
                    self.logger.warning(f"Workflow {workflow_id} not found in state")
                    return

                # Record completion
                workflow_data.b_result = b_result
                workflow_data.b_complete_time = complete_time
                workflow_data.workflow_complete_time = complete_time

                # Record task completion in metrics
                workflow_num = workflow_id.split('-')[-1]
                b_task_id = f"task-B-{workflow_data.strategy}-workflow-{workflow_num}"
                if self.metrics:
                    self.metrics.record_task_complete(
                        task_id=b_task_id,
                        success=True
                    )

                    # Record workflow completion in metrics
                    # Total tasks: A + B = 2
                    self.metrics.record_workflow_complete(
                        workflow_id=workflow_id,
                        successful_tasks=2,
                        failed_tasks=0
                    )

                self.completed_workflows += 1

                self.logger.info(
                    f"Workflow {workflow_id} complete "
                    f"({self.completed_workflows} total)"
                )

        except Exception as e:
            self.logger.error(f"Error processing B result: {e}", exc_info=True)
