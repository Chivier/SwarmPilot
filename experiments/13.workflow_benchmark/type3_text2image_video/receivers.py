"""Text2Image+Video workflow task receivers.

Workflow pattern: LLM (A) -> FLUX (C) -> T2VID (B loops)

This module provides three receivers:
- ATaskReceiver: Receives LLM results and triggers C (FLUX) submission
- CTaskReceiver: Receives FLUX results and triggers first B (T2VID) submission
- BTaskReceiver: Receives T2VID results and implements loop logic
"""

import re
import threading
import time
from queue import Queue
from typing import Any, Dict

from common import BaseTaskReceiver


class ATaskReceiver(BaseTaskReceiver):
    """
    Receives A task (LLM) results and triggers C task (FLUX) submission.

    Listens to Scheduler A WebSocket, extracts positive prompts,
    updates workflow state, and enqueues C tasks.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 c_submitter, a_result_queue: Queue, task_ids: list = None, **kwargs):
        """
        Initialize A task receiver.

        Args:
            config: Text2ImageVideoConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            c_submitter: CTaskSubmitter instance (for metrics tracking)
            a_result_queue: Queue to send (workflow_id, a_result) tuples
            task_ids: List of task IDs to subscribe to
            **kwargs: Passed to BaseTaskReceiver (name, ws_url, model_id, etc.)
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.c_submitter = c_submitter
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
        Process A task completion result.

        Extracts positive prompt, updates workflow state, and triggers C (FLUX).

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
            # Format: task-A-{strategy}-workflow-XXXX
            workflow_id = None
            if task_id:
                # Extract workflow ID from task_id
                # Expected format: "task-A-{strategy}-workflow-{num}"
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

            # Extract result (positive prompt)
            # Support multiple result formats from different model services:
            # - Simulation mode: {"output": "..."}
            # - Real mode LLM: {"result": {"output": "..."}} or {"output": "..."}
            result = data.get("result", {})
            if isinstance(result, dict):
                # Try nested format first: result.result.output
                if "result" in result and isinstance(result.get("result"), dict):
                    a_result = result.get("result", {}).get("output", "")
                # Then try direct format: result.output
                elif "output" in result:
                    a_result = result.get("output", "")
                # Fallback: convert result to string
                else:
                    a_result = str(result) if result else ""
            else:
                a_result = str(result) if result else ""

            # Get execution time for metrics
            execution_time_ms = data.get("execution_time_ms")

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

            # Trigger C (FLUX) task submission
            self.a_result_queue.put((workflow_id, a_result))

            self.logger.debug(f"A task {task_id} processed for {workflow_id}, triggered C")

        except Exception as e:
            self.logger.error(f"Error processing A result: {e}", exc_info=True)


class CTaskReceiver(BaseTaskReceiver):
    """
    Receives C task (FLUX) results and triggers first B task (T2VID) submission.

    Listens to Scheduler C WebSocket, extracts image metadata,
    updates workflow state, and enqueues first B task.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 b_submitter, c_result_queue: Queue, task_ids: list = None, **kwargs):
        """
        Initialize C task receiver.

        Args:
            config: Text2ImageVideoConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            b_submitter: BTaskSubmitter instance
            c_result_queue: Queue to send (workflow_id, positive_prompt, loop_iteration) tuples
            task_ids: List of task IDs to subscribe to
            **kwargs: Passed to BaseTaskReceiver
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.b_submitter = b_submitter
        self.c_result_queue = c_result_queue
        self.task_ids = task_ids or []

    def _get_subscription_payload(self) -> Dict[str, Any]:
        """Get WebSocket subscription payload."""
        return {
            "type": "subscribe",
            "task_ids": self.task_ids
        }

    async def _process_result(self, data: Dict[str, Any]):
        """
        Process C task (FLUX) completion result.

        Extracts image result metadata, updates workflow state, and triggers first B task.

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
            # Format: task-C-{strategy}-workflow-XXXX
            workflow_id = None
            if task_id:
                # Extract workflow ID from task_id
                # Expected format: "task-C-{strategy}-workflow-{num}"
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
                self.logger.warning(f"C task {task_id} failed with status: {status}")
                return

            # Extract result (image metadata/reference)
            # For simulation: {"output": "simulated_image_result"}
            # For real mode: {"image_ref": "...", "metadata": {...}}
            result = data.get("result", {})
            if isinstance(result, dict):
                if "result" in result and isinstance(result.get("result"), dict):
                    c_result = result.get("result", {}).get("output", "")
                elif "output" in result:
                    c_result = result.get("output", "")
                elif "image_ref" in result:
                    c_result = result.get("image_ref", "")
                else:
                    c_result = str(result) if result else ""
            else:
                c_result = str(result) if result else ""

            # Get execution time for metrics
            execution_time_ms = data.get("execution_time_ms")

            # Update workflow state
            complete_time = time.time()
            positive_prompt = None  # Will be retrieved from workflow_data
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    workflow_data.c_result = c_result
                    workflow_data.c_complete_time = complete_time
                    # Initialize B loop counter
                    workflow_data.b_loop_count = 1
                    # Get the positive prompt from A task result
                    positive_prompt = workflow_data.a_result

                    # Record task completion in metrics
                    if self.metrics:
                        workflow_num = workflow_id.split('-')[-1]
                        c_task_id = f"task-C-{workflow_data.strategy}-workflow-{workflow_num}"
                        self.metrics.record_task_complete(
                            task_id=c_task_id,
                            success=True
                        )
                else:
                    self.logger.warning(f"Workflow {workflow_id} not found in state")
                    return

            # Trigger first B task (loop_iteration = 1)
            # Pass positive_prompt from A task (not C result)
            self.c_result_queue.put((workflow_id, positive_prompt, 1))

            self.logger.debug(f"C task {task_id} processed for {workflow_id}, triggered B1")

        except Exception as e:
            self.logger.error(f"Error processing C result: {e}", exc_info=True)


class BTaskReceiver(BaseTaskReceiver):
    """
    Receives B task (T2VID) results and implements loop logic.

    Listens to Scheduler B WebSocket, tracks B iterations, and either:
    - Re-submits B task if loop_count < max_b_loops
    - Marks workflow complete if all iterations done
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 b_submitter, task_ids: list = None, **kwargs):
        """
        Initialize B task receiver.

        Args:
            config: Text2ImageVideoConfig instance
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
        Process B task (T2VID) completion result.

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
            # Format: task-B{N}-{strategy}-workflow-XXXX
            workflow_id = None
            loop_iteration = 1

            # Extract workflow ID from task_id
            # Expected format: "task-B{N}-{strategy}-workflow-{num}"
            parts = task_id.split("-")
            if "workflow" in parts:
                idx = parts.index("workflow")
                if idx + 1 < len(parts):
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
                workflow_num = workflow_id.split('-')[-1]
                b_task_id = f"task-B{loop_iteration}-{workflow_data.strategy}-workflow-{workflow_num}"
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

                    # Get positive_prompt from A task result (not C result)
                    # B tasks use the LLM output as prompt, not the FLUX output
                    positive_prompt = workflow_data.a_result

                    self.logger.debug(
                        f"B{loop_iteration} complete for {workflow_id}, "
                        f"triggering B{next_iteration}"
                    )

                    # Trigger next B iteration (outside lock to avoid deadlock)
                    # Use b_submitter.add_task() instead of queue.put()
                    self.b_submitter.add_task(workflow_id, positive_prompt, next_iteration)

                elif workflow_data.is_complete():
                    # All B iterations complete
                    workflow_data.workflow_complete_time = complete_time
                    self.completed_workflows += 1

                    # Record workflow completion in metrics
                    if self.metrics:
                        # Total tasks: A + C + B*max_b_loops
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
