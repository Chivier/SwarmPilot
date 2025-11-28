"""Text2Video workflow task receivers."""

import threading
import time
from queue import Queue
from typing import Any, Dict, List

from common import BaseTaskReceiver


class A1TaskReceiver(BaseTaskReceiver):
    """
    Receives A1 task results and triggers A2 task submission.

    Listens to Scheduler A WebSocket, extracts positive prompts,
    updates workflow state, and enqueues A2 tasks.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 a2_submitter, a1_result_queue: Queue, task_ids: list = None, **kwargs):
        """
        Initialize A1 task receiver.

        Args:
            config: Text2VideoConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            a2_submitter: A2TaskSubmitter instance (for metrics tracking)
            a1_result_queue: Queue to send (workflow_id, a1_result) tuples
            task_ids: List of task IDs to subscribe to
            **kwargs: Passed to BaseTaskReceiver (name, ws_url, model_id, etc.)
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.a2_submitter = a2_submitter
        self.a1_result_queue = a1_result_queue
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
        Process A1 task completion result.

        Extracts positive prompt, updates workflow state, and triggers A2.

        Args:
            data: Result data from WebSocket message
        """
        try:
            # Extract task_id
            task_id = data.get("task_id")
            if not task_id:
                self.logger.warning("Received result without task_id")
                return

            # Extract workflow_id from task_id (scheduler does NOT return metadata)
            # Format: task-A1-{strategy}-workflow-{num} (old) or task-A1-{strategy}-workflow-{prefix}-{num} (new)
            workflow_id = None
            if task_id:
                # Extract workflow ID from task_id
                # Old format: "task-A1-{strategy}-workflow-{num}"
                # New format: "task-A1-{strategy}-workflow-{prefix}-{num}"
                parts = task_id.split("-")
                if "workflow" in parts:
                    idx = parts.index("workflow")
                    # Check format: new (workflow-{prefix}-{num}) vs old (workflow-{num})
                    if idx + 2 < len(parts) and not parts[idx + 1].isdigit():
                        # New format: parts[idx+1] is prefix (alphanumeric), parts[idx+2] is num
                        workflow_id = f"workflow-{parts[idx + 1]}-{parts[idx + 2]}"
                    elif idx + 1 < len(parts):
                        # Old format: parts[idx+1] is the num
                        workflow_id = f"workflow-{parts[idx + 1]}"

                if not workflow_id:
                    self.logger.warning(f"Cannot extract workflow_id from task_id: {task_id}")
                    return

            # Check task status
            status = data.get("status")
            if status != "completed":
                # Extract detailed error information
                error_msg = data.get("error", "")
                error_details = data.get("error_details", "")
                result_info = data.get("result", {})
                if isinstance(result_info, dict):
                    error_msg = error_msg or result_info.get("error", "")
                    error_details = error_details or result_info.get("details", "")

                self.logger.warning(
                    f"A1 task {task_id} failed with status: {status}"
                    + (f", error: {error_msg}" if error_msg else "")
                    + (f", details: {error_details}" if error_details else "")
                    + (f", raw data: {data}" if not error_msg and not error_details else "")
                )
                return

            # Extract result (positive prompt)
            # Support multiple result formats from different model services:
            # - Simulation mode: {"output": "..."}
            # - Real mode LLM: {"result": {"output": "..."}} or {"output": "..."}
            result = data.get("result", {})
            if isinstance(result, dict):
                # Try nested format first: result.result.output
                if "result" in result and isinstance(result.get("result"), dict):
                    a1_result = result.get("result", {}).get("output", "")
                # Then try direct format: result.output
                elif "output" in result:
                    a1_result = result.get("output", "")
                # Fallback: convert result to string
                else:
                    a1_result = str(result) if result else ""
            else:
                a1_result = str(result) if result else ""

            # Get execution time for metrics
            execution_time_ms = data.get("execution_time_ms")

            # Update workflow state
            complete_time = time.time()
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    workflow_data.a1_result = a1_result
                    workflow_data.a1_complete_time = complete_time

                    # Record task completion in metrics
                    if self.metrics:
                        # Use replace to preserve full suffix (e.g., "a7b3-0001" from "workflow-a7b3-0001")
                        workflow_suffix = workflow_id.replace("workflow-", "")
                        a1_task_id = f"task-A1-{workflow_data.strategy}-workflow-{workflow_suffix}"
                        self.metrics.record_task_complete(
                            task_id=a1_task_id,
                            success=True
                        )
                else:
                    self.logger.warning(f"Workflow {workflow_id} not found in state")
                    return

            # Trigger A2 task submission
            self.a1_result_queue.put((workflow_id, a1_result))

            self.logger.debug(f"A1 task {task_id} processed for {workflow_id}, triggered A2")

        except Exception as e:
            self.logger.error(f"Error processing A1 result: {e}", exc_info=True)


class A2TaskReceiver(BaseTaskReceiver):
    """
    Receives A2 task results and triggers B task submission.

    Listens to Scheduler A WebSocket, extracts negative prompts,
    updates workflow state, and enqueues first B task.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 b_submitter, a2_result_queue: Queue, task_ids: list = None, **kwargs):
        """
        Initialize A2 task receiver.

        Args:
            config: Text2VideoConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            b_submitter: BTaskSubmitter instance
            a2_result_queue: Queue to send (workflow_id, a2_result, loop_iteration) tuples
            task_ids: List of task IDs to subscribe to
            **kwargs: Passed to BaseTaskReceiver
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.b_submitter = b_submitter
        self.a2_result_queue = a2_result_queue
        self.task_ids = task_ids or []

    def _get_subscription_payload(self) -> Dict[str, Any]:
        """Get WebSocket subscription payload."""
        return {
            "type": "subscribe",
            "task_ids": self.task_ids
        }

    async def _process_result(self, data: Dict[str, Any]):
        """
        Process A2 task completion result.

        Extracts negative prompt, updates workflow state, and triggers first B task.

        Args:
            data: Result data from WebSocket message
        """
        try:
            # Extract task_id
            task_id = data.get("task_id")
            if not task_id:
                self.logger.warning("Received result without task_id")
                return

            # Extract workflow_id from task_id (scheduler does NOT return metadata)
            # Format: task-A2-{strategy}-workflow-{num} (old) or task-A2-{strategy}-workflow-{prefix}-{num} (new)
            workflow_id = None
            if task_id:
                # Extract workflow ID from task_id
                # Old format: "task-A2-{strategy}-workflow-{num}"
                # New format: "task-A2-{strategy}-workflow-{prefix}-{num}"
                parts = task_id.split("-")
                if "workflow" in parts:
                    idx = parts.index("workflow")
                    # Check format: new (workflow-{prefix}-{num}) vs old (workflow-{num})
                    if idx + 2 < len(parts) and not parts[idx + 1].isdigit():
                        # New format: parts[idx+1] is prefix (alphanumeric), parts[idx+2] is num
                        workflow_id = f"workflow-{parts[idx + 1]}-{parts[idx + 2]}"
                    elif idx + 1 < len(parts):
                        # Old format: parts[idx+1] is the num
                        workflow_id = f"workflow-{parts[idx + 1]}"

                if not workflow_id:
                    self.logger.warning(f"Cannot extract workflow_id from task_id: {task_id}")
                    return

            # Check task status
            status = data.get("status")
            if status != "completed":
                # Extract detailed error information
                error_msg = data.get("error", "")
                error_details = data.get("error_details", "")
                result_info = data.get("result", {})
                if isinstance(result_info, dict):
                    error_msg = error_msg or result_info.get("error", "")
                    error_details = error_details or result_info.get("details", "")

                self.logger.warning(
                    f"A2 task {task_id} failed with status: {status}"
                    + (f", error: {error_msg}" if error_msg else "")
                    + (f", details: {error_details}" if error_details else "")
                    + (f", raw data: {data}" if not error_msg and not error_details else "")
                )
                return

            # Extract result (negative prompt)
            # Support multiple result formats from different model services:
            # - Simulation mode: {"output": "..."}
            # - Real mode LLM: {"result": {"output": "..."}} or {"output": "..."}
            result = data.get("result", {})
            if isinstance(result, dict):
                # Try nested format first: result.result.output
                if "result" in result and isinstance(result.get("result"), dict):
                    a2_result = result.get("result", {}).get("output", "")
                # Then try direct format: result.output
                elif "output" in result:
                    a2_result = result.get("output", "")
                # Fallback: convert result to string
                else:
                    a2_result = str(result) if result else ""
            else:
                a2_result = str(result) if result else ""

            # Get execution time for metrics
            execution_time_ms = data.get("execution_time_ms")

            # Update workflow state
            complete_time = time.time()
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    workflow_data.a2_result = a2_result
                    workflow_data.a2_complete_time = complete_time
                    # Initialize B loop counter
                    workflow_data.b_loop_count = 1

                    # Record task completion in metrics
                    if self.metrics:
                        # Use replace to preserve full suffix (e.g., "a7b3-0001" from "workflow-a7b3-0001")
                        workflow_suffix = workflow_id.replace("workflow-", "")
                        a2_task_id = f"task-A2-{workflow_data.strategy}-workflow-{workflow_suffix}"
                        self.metrics.record_task_complete(
                            task_id=a2_task_id,
                            success=True
                        )
                else:
                    self.logger.warning(f"Workflow {workflow_id} not found in state")
                    return

            # Trigger first B task (loop_iteration = 1)
            self.a2_result_queue.put((workflow_id, a2_result, 1))

            self.logger.debug(f"A2 task {task_id} processed for {workflow_id}, triggered B1")

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
                 b_submitter, task_ids: list = None, **kwargs):
        """
        Initialize B task receiver.

        Args:
            config: Text2VideoConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            b_submitter: BTaskSubmitter instance (for loop re-submission)
            task_ids: List of task IDs to subscribe to
            **kwargs: Passed to BaseTaskReceiver
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.b_submitter = b_submitter
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
        Process B task completion result.

        Implements B-loop logic: re-submit if more iterations needed,
        otherwise mark workflow complete.

        Args:
            data: Result data from WebSocket message
        """
        try:
            # Extract task_id
            task_id = data.get("task_id")
            if not task_id:
                self.logger.warning("Received result without task_id")
                return

            # Extract workflow_id and loop_iteration from task_id (scheduler does NOT return metadata)
            # Format: task-B{N}-{strategy}-workflow-{num} (old) or task-B{N}-{strategy}-workflow-{prefix}-{num} (new)
            import re
            workflow_id = None
            loop_iteration = 1

            # Extract workflow ID from task_id
            # Old format: "task-B{N}-{strategy}-workflow-{num}"
            # New format: "task-B{N}-{strategy}-workflow-{prefix}-{num}"
            parts = task_id.split("-")
            if "workflow" in parts:
                idx = parts.index("workflow")
                # Check format: new (workflow-{prefix}-{num}) vs old (workflow-{num})
                if idx + 2 < len(parts) and not parts[idx + 1].isdigit():
                    # New format: parts[idx+1] is prefix (alphanumeric), parts[idx+2] is num
                    workflow_id = f"workflow-{parts[idx + 1]}-{parts[idx + 2]}"
                elif idx + 1 < len(parts):
                    # Old format: parts[idx+1] is the num
                    workflow_id = f"workflow-{parts[idx + 1]}"

                # Extract loop iteration from task_id (B1, B2, etc.)
                match = re.match(r"task-B(\d+)", task_id)
                if match:
                    loop_iteration = int(match.group(1))

            if not workflow_id:
                self.logger.warning(f"Cannot extract workflow_id from task_id: {task_id}")
                return

            # Check task status
            status = data.get("status")
            if status != "completed":
                # Extract detailed error information
                error_msg = data.get("error", "")
                error_details = data.get("error_details", "")
                result_info = data.get("result", {})
                if isinstance(result_info, dict):
                    error_msg = error_msg or result_info.get("error", "")
                    error_details = error_details or result_info.get("details", "")

                self.logger.warning(
                    f"B task {task_id} failed with status: {status}"
                    + (f", error: {error_msg}" if error_msg else "")
                    + (f", details: {error_details}" if error_details else "")
                    + (f", raw data: {data}" if not error_msg and not error_details else "")
                )
                # Mark workflow as failed/complete even on failure
                with self.state_lock:
                    workflow_data = self.workflow_states.get(workflow_id)
                    if workflow_data:
                        workflow_data.workflow_complete_time = time.time()

                        # Record workflow completion in metrics (failed)
                        if self.metrics:
                            self.metrics.record_workflow_complete(
                                workflow_id=workflow_id,
                                successful_tasks=len(workflow_data.b_complete_times),  # Tasks completed so far
                                failed_tasks=1  # This failed B task
                            )
                return

            # Extract result (video generation result)
            # Handle nested result structure: data.result.result.output
            result = data.get("result", {})
            if isinstance(result, dict) and "result" in result:
                result = result.get("result", {})

            # Get execution time for metrics
            execution_time_ms = data.get("execution_time_ms")

            complete_time = time.time()

            # Update workflow state
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if not workflow_data:
                    self.logger.warning(f"Workflow {workflow_id} not found in state")
                    return

                # Record completion time
                workflow_data.b_complete_times.append(complete_time)

                # Record task completion in metrics
                # Use replace to preserve full suffix (e.g., "a7b3-0001" from "workflow-a7b3-0001")
                workflow_suffix = workflow_id.replace("workflow-", "")
                b_task_id = f"task-B{loop_iteration}-{workflow_data.strategy}-workflow-{workflow_suffix}"
                if self.metrics:
                    self.metrics.record_task_complete(
                        task_id=b_task_id,
                        success=True
                    )

                # Check if we should continue B loop
                if workflow_data.should_continue_b_loop():
                    # Increment loop counter for next iteration
                    workflow_data.b_loop_count += 1
                    next_iteration = workflow_data.b_loop_count

                    # Record submit time for next iteration
                    workflow_data.b_submit_times.append(time.time())

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
                    workflow_data.workflow_complete_time = complete_time
                    self.completed_workflows += 1

                    # Record workflow completion in metrics
                    if self.metrics:
                        # Total tasks: A1 + A2 + B*max_b_loops
                        total_tasks = 2 + workflow_data.max_b_loops
                        self.metrics.record_workflow_complete(
                            workflow_id=workflow_id,
                            successful_tasks=total_tasks,
                            failed_tasks=0
                        )

                    self.logger.info(
                        f"Workflow {workflow_id} complete "
                        f"({self.completed_workflows} total, "
                        f"{workflow_data.max_b_loops} B iterations)"
                    )

        except Exception as e:
            self.logger.error(f"Error processing B result: {e}", exc_info=True)
