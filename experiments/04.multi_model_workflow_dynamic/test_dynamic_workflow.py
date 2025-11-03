#!/usr/bin/env python3
"""
Experiment 04: Multi-Model Workflow with Dynamic Fanout (1-to-n)

This experiment tests workflow dependencies where each A task generates
a variable number (n) of B tasks that execute in parallel. A workflow
is complete when all B tasks for that workflow are complete.

Architecture:
- Thread 1: Submit A tasks (Poisson process, QPS-controlled)
- Thread 2: Receive A task results, submit n B tasks per A task
- Thread 3: Receive B task results, update workflow state
- Thread 4: Monitor workflow completion, calculate statistics

Key differences from experiment 03 (1-to-1):
- Variable fanout: Each A task generates 3-8 B tasks (uniform distribution)
- Workflow completion tracking: Must wait for all B tasks to complete
- Pre-calculated task IDs: All task IDs generated upfront for WebSocket subscription
"""

import asyncio
import websockets
import requests
import json
import time
import numpy as np
import threading
from queue import Queue, Empty
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Set
from datetime import datetime
import logging

# Import workload generators
from workload_generator import (
    generate_bimodal_distribution,
    generate_fanout_distribution,
    WorkloadConfig,
    FanoutConfig,
    print_distribution_stats,
    print_fanout_stats
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Scheduler endpoints
SCHEDULER_A_URL = "http://localhost:8100"
SCHEDULER_B_URL = "http://localhost:8200"
SCHEDULER_A_WS = "ws://localhost:8100/task/get_result"
SCHEDULER_B_WS = "ws://localhost:8200/task/get_result"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class WorkflowTaskData:
    """Pre-generated task data for workflow execution."""
    task_id: str
    workflow_id: str
    task_type: str  # "A" or "B"
    sleep_time: float
    exp_runtime: float  # Expected runtime in milliseconds


@dataclass
class TaskRecord:
    """Complete record of a task's lifecycle."""
    task_id: str
    workflow_id: str
    task_type: str
    sleep_time: float
    exp_runtime: float

    # Tracked at submission
    submit_time: Optional[float] = None
    assigned_instance: Optional[str] = None

    # Tracked at completion
    complete_time: Optional[float] = None
    status: Optional[str] = None
    execution_time_ms: Optional[float] = None

    # Results
    result: Optional[Dict] = None
    error: Optional[str] = None


@dataclass
class WorkflowState:
    """State tracking for a single workflow."""
    workflow_id: str
    strategy: str
    a_task_id: str
    b_task_ids: List[str]
    total_b_tasks: int
    completed_b_tasks: int = 0

    # Timestamps
    a_submit_time: Optional[float] = None
    a_complete_time: Optional[float] = None
    b_complete_times: Dict[str, float] = field(default_factory=dict)
    workflow_complete_time: Optional[float] = None

    def is_complete(self) -> bool:
        """Check if workflow is complete (all B tasks done)."""
        return self.completed_b_tasks >= self.total_b_tasks

    def mark_b_task_complete(self, b_task_id: str, complete_time: float):
        """Mark a B task as complete and update workflow state."""
        if b_task_id not in self.b_complete_times:
            self.b_complete_times[b_task_id] = complete_time
            self.completed_b_tasks += 1

            # Update workflow complete time if this is the last B task
            if self.is_complete():
                self.workflow_complete_time = max(self.b_complete_times.values())


@dataclass
class WorkflowCompletionEvent:
    """Event indicating a workflow has completed."""
    workflow_id: str
    workflow_time: float  # A submit → last B complete time (seconds)
    a_task_id: str
    b_task_ids: List[str]
    total_b_tasks: int
    a_submit_time: float
    a_complete_time: float
    workflow_complete_time: float


# ============================================================================
# Thread 1: Poisson Task Submitter for A Tasks
# ============================================================================

class PoissonTaskSubmitter:
    """
    Thread 1: Submit A tasks following Poisson process with QPS control.

    Submits tasks to Scheduler A with exponentially distributed inter-arrival
    times to simulate realistic workload patterns.
    """

    def __init__(self,
                 scheduler_url: str,
                 tasks: List[WorkflowTaskData],
                 qps: float,
                 workflow_states: Dict[str, WorkflowState],
                 model_id: str = "sleep_model"):
        """
        Initialize Poisson task submitter.

        Args:
            scheduler_url: Scheduler A URL (e.g., http://localhost:8100)
            tasks: List of pre-generated A task data
            qps: Target queries per second (e.g., 8.0)
            workflow_states: Shared workflow state dictionary
            model_id: Model ID to use for tasks
        """
        self.scheduler_url = scheduler_url
        self.tasks = tasks
        self.qps = qps
        self.workflow_states = workflow_states
        self.model_id = model_id
        self.logger = logging.getLogger("Thread1.ATaskSubmitter")

        # Tracking
        self.submitted_tasks: List[TaskRecord] = []
        self.submission_start_time: Optional[float] = None
        self.submission_end_time: Optional[float] = None

        # Thread control
        self.thread: Optional[threading.Thread] = None
        self.running = False

    def _submit_task(self, task_data: WorkflowTaskData) -> TaskRecord:
        """
        Submit a single A task to Scheduler A.

        Args:
            task_data: Pre-generated task data

        Returns:
            TaskRecord with submission info
        """
        payload = {
            "task_id": task_data.task_id,
            "model_id": self.model_id,
            "task_input": {
                "sleep_time": task_data.sleep_time
            },
            "metadata": {
                "exp_runtime": task_data.exp_runtime,
                "workflow_id": task_data.workflow_id,
                "task_type": "A"
            }
        }

        submit_time = time.time()

        try:
            response = requests.post(
                f"{self.scheduler_url}/task/submit",
                json=payload,
                timeout=5.0
            )

            # Check for HTTP errors
            if not response.ok:
                # Try to extract error details from response body
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", response.text)
                except:
                    error_msg = response.text or f"HTTP {response.status_code}"

                error_detail = f"HTTP {response.status_code}: {error_msg}"
                self.logger.error(f"Failed to submit A task {task_data.task_id}: {error_detail}")

                return TaskRecord(
                    task_id=task_data.task_id,
                    workflow_id=task_data.workflow_id,
                    task_type="A",
                    sleep_time=task_data.sleep_time,
                    exp_runtime=task_data.exp_runtime,
                    submit_time=submit_time,
                    status="submit_failed",
                    error=error_detail
                )

            result = response.json()

            # Check if submission was successful
            if not result.get("success", True):
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Scheduler rejected A task {task_data.task_id}: {error_msg}")

                return TaskRecord(
                    task_id=task_data.task_id,
                    workflow_id=task_data.workflow_id,
                    task_type="A",
                    sleep_time=task_data.sleep_time,
                    exp_runtime=task_data.exp_runtime,
                    submit_time=submit_time,
                    status="submit_failed",
                    error=f"Rejected: {error_msg}"
                )

            # Update workflow state with A task submit time
            workflow_id = task_data.workflow_id
            if workflow_id in self.workflow_states:
                self.workflow_states[workflow_id].a_submit_time = submit_time

            # Create task record
            record = TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="A",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                assigned_instance=result.get("task", {}).get("assigned_instance")
            )

            return record

        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after 5.0s"
            self.logger.error(f"Failed to submit A task {task_data.task_id}: {error_msg}")
            return TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="A",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                status="submit_failed",
                error=error_msg
            )
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            self.logger.error(f"Failed to submit A task {task_data.task_id}: {error_msg}")
            return TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="A",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                status="submit_failed",
                error=error_msg
            )
        except Exception as e:
            error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
            self.logger.error(f"Failed to submit A task {task_data.task_id}: {error_msg}")
            return TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="A",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                status="submit_failed",
                error=error_msg
            )

    def _run(self):
        """Main submission loop with Poisson inter-arrival times."""
        self.logger.info(f"Starting Poisson submission: {len(self.tasks)} A tasks at {self.qps} QPS")

        # Generate inter-arrival times (exponential distribution)
        lambda_rate = self.qps
        inter_arrival_times = np.random.exponential(1.0 / lambda_rate, len(self.tasks))

        self.submission_start_time = time.time()

        for i, (task_data, wait_time) in enumerate(zip(self.tasks, inter_arrival_times)):
            if not self.running:
                self.logger.warning("Submission stopped early")
                break

            # Wait for inter-arrival time
            if i > 0:
                time.sleep(wait_time)

            # Submit task
            record = self._submit_task(task_data)
            self.submitted_tasks.append(record)

            if (i + 1) % 20 == 0:
                self.logger.info(f"Submitted {i + 1}/{len(self.tasks)} A tasks")

        self.submission_end_time = time.time()
        submission_duration = self.submission_end_time - self.submission_start_time
        actual_qps = len(self.tasks) / submission_duration

        self.logger.info(f"Submission complete: {len(self.submitted_tasks)} tasks in {submission_duration:.2f}s "
                        f"(actual QPS: {actual_qps:.2f})")

    def start(self):
        """Start the submission thread."""
        if self.thread is not None:
            self.logger.warning("Submitter already started")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run, name="Thread1-ATaskSubmitter")
        self.thread.start()
        self.logger.info("Submission thread started")

    def stop(self):
        """Stop the submission thread."""
        if self.thread is None:
            return

        self.running = False
        self.thread.join(timeout=10.0)
        self.logger.info("Submission thread stopped")

    def is_alive(self) -> bool:
        """Check if submission thread is still running."""
        return self.thread is not None and self.thread.is_alive()


# ============================================================================
# Thread 2: A Task Result Receiver + B Task Submitter
# ============================================================================

class ATaskReceiver:
    """
    Thread 2: Receive A task results via WebSocket and submit B tasks.

    For each completed A task:
    1. Receive completion event from Scheduler A
    2. Look up the number of B tasks (n) for this workflow
    3. Submit n B tasks to Scheduler B in parallel
    4. Record B task submission info
    """

    def __init__(self,
                 scheduler_a_ws: str,
                 scheduler_b_url: str,
                 a_task_ids: List[str],
                 b_tasks_by_workflow: Dict[str, List[WorkflowTaskData]],
                 workflow_states: Dict[str, WorkflowState],
                 model_id: str = "sleep_model"):
        """
        Initialize A task receiver.

        Args:
            scheduler_a_ws: WebSocket URL for Scheduler A
            scheduler_b_url: HTTP URL for Scheduler B
            a_task_ids: List of all A task IDs to subscribe to
            b_tasks_by_workflow: Map of workflow_id -> list of B task data
            workflow_states: Shared workflow state dictionary
            model_id: Model ID to use for B tasks
        """
        self.scheduler_a_ws = scheduler_a_ws
        self.scheduler_b_url = scheduler_b_url
        self.a_task_ids = a_task_ids
        self.b_tasks_by_workflow = b_tasks_by_workflow
        self.workflow_states = workflow_states
        self.model_id = model_id
        self.logger = logging.getLogger("Thread2.ATaskReceiver")

        # Tracking
        self.a_results: List[TaskRecord] = []
        self.b_submitted: List[TaskRecord] = []

        # Thread control
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def _submit_b_task(self, task_data: WorkflowTaskData) -> TaskRecord:
        """
        Submit a single B task to Scheduler B.

        Args:
            task_data: Pre-generated B task data

        Returns:
            TaskRecord with submission info
        """
        payload = {
            "task_id": task_data.task_id,
            "model_id": self.model_id,
            "task_input": {
                "sleep_time": task_data.sleep_time
            },
            "metadata": {
                "exp_runtime": task_data.exp_runtime,
                "workflow_id": task_data.workflow_id,
                "task_type": "B"
            }
        }

        submit_time = time.time()

        try:
            response = requests.post(
                f"{self.scheduler_b_url}/task/submit",
                json=payload,
                timeout=5.0
            )

            # Check for HTTP errors
            if not response.ok:
                # Try to extract error details from response body
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", response.text)
                except:
                    error_msg = response.text or f"HTTP {response.status_code}"

                error_detail = f"HTTP {response.status_code}: {error_msg}"
                self.logger.error(f"Failed to submit B task {task_data.task_id}: {error_detail}")

                return TaskRecord(
                    task_id=task_data.task_id,
                    workflow_id=task_data.workflow_id,
                    task_type="B",
                    sleep_time=task_data.sleep_time,
                    exp_runtime=task_data.exp_runtime,
                    submit_time=submit_time,
                    status="submit_failed",
                    error=error_detail
                )

            result = response.json()

            # Check if submission was successful
            if not result.get("success", True):
                error_msg = result.get("error", "Unknown error")
                self.logger.error(f"Scheduler rejected B task {task_data.task_id}: {error_msg}")

                return TaskRecord(
                    task_id=task_data.task_id,
                    workflow_id=task_data.workflow_id,
                    task_type="B",
                    sleep_time=task_data.sleep_time,
                    exp_runtime=task_data.exp_runtime,
                    submit_time=submit_time,
                    status="submit_failed",
                    error=f"Rejected: {error_msg}"
                )

            record = TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="B",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                assigned_instance=result.get("task", {}).get("assigned_instance")
            )

            return record

        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after 5.0s"
            self.logger.error(f"Failed to submit B task {task_data.task_id}: {error_msg}")
            return TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="B",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                status="submit_failed",
                error=error_msg
            )
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            self.logger.error(f"Failed to submit B task {task_data.task_id}: {error_msg}")
            return TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="B",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                status="submit_failed",
                error=error_msg
            )
        except Exception as e:
            error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
            self.logger.error(f"Failed to submit B task {task_data.task_id}: {error_msg}")
            return TaskRecord(
                task_id=task_data.task_id,
                workflow_id=task_data.workflow_id,
                task_type="B",
                sleep_time=task_data.sleep_time,
                exp_runtime=task_data.exp_runtime,
                submit_time=submit_time,
                status="submit_failed",
                error=error_msg
            )

    async def _run_async(self):
        """Main WebSocket loop for receiving A task results."""
        self.logger.info(f"Connecting to Scheduler A WebSocket: {self.scheduler_a_ws}")

        try:
            async with websockets.connect(self.scheduler_a_ws) as websocket:
                # Subscribe to all A task IDs
                subscribe_msg = {
                    "type": "subscribe",
                    "task_ids": self.a_task_ids
                }
                await websocket.send(json.dumps(subscribe_msg))
                self.logger.info(f"Subscribed to {len(self.a_task_ids)} A tasks")

                # Wait for acknowledgment
                ack = await websocket.recv()
                ack_data = json.loads(ack)
                self.logger.info(f"Subscription confirmed: {ack_data.get('message')}")

                # Receive A task results
                while self.running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)

                        if data["type"] == "result":
                            await self._handle_a_result(data)
                        elif data["type"] == "error":
                            self.logger.error(f"WebSocket error: {data.get('error')}")

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        self.logger.error(f"Error receiving message: {e}")
                        break

        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")

    async def _handle_a_result(self, data: Dict):
        """
        Handle A task completion result and submit B tasks.

        Args:
            data: WebSocket result message
        """
        task_id = data["task_id"]
        status = data["status"]
        workflow_id = None

        # Extract workflow_id from task_id (format: task-A-{strategy}-workflow-{i:04d}-A)
        parts = task_id.split("-")
        if len(parts) >= 5 and parts[3] == "workflow":
            workflow_id = f"wf-{parts[2]}-{parts[4]}"

        if not workflow_id:
            self.logger.error(f"Could not extract workflow_id from task_id: {task_id}")
            return

        # Create A task record
        a_record = TaskRecord(
            task_id=task_id,
            workflow_id=workflow_id,
            task_type="A",
            sleep_time=0.0,  # Will be filled from original data if needed
            exp_runtime=0.0,
            complete_time=time.time(),
            status=status,
            execution_time_ms=data.get("execution_time_ms"),
            result=data.get("result"),
            error=data.get("error")
        )
        self.a_results.append(a_record)

        # Update workflow state
        if workflow_id in self.workflow_states:
            self.workflow_states[workflow_id].a_complete_time = a_record.complete_time

        # If A task succeeded, submit B tasks
        if status == "completed":
            await self._submit_b_tasks_for_workflow(workflow_id)
        else:
            self.logger.warning(f"A task {task_id} failed, skipping B task submission")

    async def _submit_b_tasks_for_workflow(self, workflow_id: str):
        """
        Submit all B tasks for a given workflow.

        Args:
            workflow_id: Workflow ID
        """
        if workflow_id not in self.b_tasks_by_workflow:
            self.logger.error(f"No B tasks found for workflow {workflow_id}")
            return

        b_tasks = self.b_tasks_by_workflow[workflow_id]

        self.logger.info(f"Submitting {len(b_tasks)} B tasks for workflow {workflow_id}")

        # Submit all B tasks (in current thread, not async)
        # Using synchronous submission for simplicity
        for b_task_data in b_tasks:
            record = self._submit_b_task(b_task_data)
            self.b_submitted.append(record)

        self.logger.debug(f"Completed B task submission for workflow {workflow_id}")

    def _run(self):
        """Thread entry point."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._run_async())
        except Exception as e:
            self.logger.error(f"Error in async loop: {e}")
        finally:
            try:
                # Cancel any pending tasks
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()
                # Wait for all tasks to be cancelled
                if pending:
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                self.loop.close()

    def start(self):
        """Start the receiver thread."""
        if self.thread is not None:
            self.logger.warning("Receiver already started")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run, name="Thread2-ATaskReceiver")
        self.thread.start()
        self.logger.info("A task receiver thread started")

    def stop(self):
        """Stop the receiver thread."""
        if self.thread is None:
            return

        self.running = False
        self.thread.join(timeout=10.0)
        self.logger.info("A task receiver thread stopped")

    def is_alive(self) -> bool:
        """Check if receiver thread is still running."""
        return self.thread is not None and self.thread.is_alive()


# ============================================================================
# Thread 3: B Task Result Receiver + Workflow State Updater
# ============================================================================

class BTaskReceiver:
    """
    Thread 3: Receive B task results via WebSocket and update workflow state.

    For each completed B task:
    1. Receive completion event from Scheduler B
    2. Update workflow state (increment completed_b_tasks counter)
    3. Check if workflow is complete (all B tasks done)
    4. If complete, push workflow completion event to queue for Thread 4
    """

    def __init__(self,
                 scheduler_b_ws: str,
                 b_task_ids: List[str],
                 workflow_states: Dict[str, WorkflowState],
                 completion_queue: Queue):
        """
        Initialize B task receiver.

        Args:
            scheduler_b_ws: WebSocket URL for Scheduler B
            b_task_ids: List of all B task IDs to subscribe to
            workflow_states: Shared workflow state dictionary
            completion_queue: Queue to push workflow completion events
        """
        self.scheduler_b_ws = scheduler_b_ws
        self.b_task_ids = b_task_ids
        self.workflow_states = workflow_states
        self.completion_queue = completion_queue
        self.logger = logging.getLogger("Thread3.BTaskReceiver")

        # Tracking
        self.b_results: List[TaskRecord] = []

        # Thread control
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Track workflow completion to avoid duplicates
        self.completed_workflows: Set[str] = set()

    async def _run_async(self):
        """Main WebSocket loop for receiving B task results."""
        self.logger.info(f"Connecting to Scheduler B WebSocket: {self.scheduler_b_ws}")

        try:
            async with websockets.connect(self.scheduler_b_ws) as websocket:
                # Subscribe to all B task IDs
                subscribe_msg = {
                    "type": "subscribe",
                    "task_ids": self.b_task_ids
                }
                await websocket.send(json.dumps(subscribe_msg))
                self.logger.info(f"Subscribed to {len(self.b_task_ids)} B tasks")

                # Wait for acknowledgment
                ack = await websocket.recv()
                ack_data = json.loads(ack)
                self.logger.info(f"Subscription confirmed: {ack_data.get('message')}")

                # Receive B task results
                while self.running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)

                        if data["type"] == "result":
                            await self._handle_b_result(data)
                        elif data["type"] == "error":
                            self.logger.error(f"WebSocket error: {data.get('error')}")

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        self.logger.error(f"Error receiving message: {e}")
                        break

        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")

    async def _handle_b_result(self, data: Dict):
        """
        Handle B task completion result and update workflow state.

        Args:
            data: WebSocket result message
        """
        task_id = data["task_id"]
        status = data["status"]
        complete_time = time.time()
        workflow_id = None

        # Extract workflow_id from task_id (format: task-B-{strategy}-workflow-{i:04d}-B-{j:02d})
        parts = task_id.split("-")
        if len(parts) >= 5 and parts[3] == "workflow":
            workflow_id = f"wf-{parts[2]}-{parts[4]}"

        if not workflow_id:
            self.logger.error(f"Could not extract workflow_id from task_id: {task_id}")
            return

        # Create B task record
        b_record = TaskRecord(
            task_id=task_id,
            workflow_id=workflow_id,
            task_type="B",
            sleep_time=0.0,
            exp_runtime=0.0,
            complete_time=complete_time,
            status=status,
            execution_time_ms=data.get("execution_time_ms"),
            result=data.get("result"),
            error=data.get("error")
        )
        self.b_results.append(b_record)

        # Update workflow state
        if workflow_id not in self.workflow_states:
            self.logger.error(f"Unknown workflow_id: {workflow_id}")
            return

        workflow = self.workflow_states[workflow_id]

        # Mark B task as complete
        if status == "completed":
            workflow.mark_b_task_complete(task_id, complete_time)

            # Check if workflow is now complete
            if workflow.is_complete() and workflow_id not in self.completed_workflows:
                self.completed_workflows.add(workflow_id)
                self._push_workflow_completion(workflow)
                self.logger.info(f"Workflow {workflow_id} completed "
                               f"({workflow.completed_b_tasks}/{workflow.total_b_tasks} B tasks)")
        else:
            self.logger.warning(f"B task {task_id} failed for workflow {workflow_id}")

    def _push_workflow_completion(self, workflow: WorkflowState):
        """
        Push workflow completion event to queue for Thread 4.

        Args:
            workflow: Completed workflow state
        """
        if workflow.a_submit_time is None or workflow.workflow_complete_time is None:
            self.logger.error(f"Incomplete workflow timing data for {workflow.workflow_id}")
            return

        event = WorkflowCompletionEvent(
            workflow_id=workflow.workflow_id,
            workflow_time=workflow.workflow_complete_time - workflow.a_submit_time,
            a_task_id=workflow.a_task_id,
            b_task_ids=workflow.b_task_ids,
            total_b_tasks=workflow.total_b_tasks,
            a_submit_time=workflow.a_submit_time,
            a_complete_time=workflow.a_complete_time or 0.0,
            workflow_complete_time=workflow.workflow_complete_time
        )

        self.completion_queue.put(event)

    def _run(self):
        """Thread entry point."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._run_async())
        except Exception as e:
            self.logger.error(f"Error in async loop: {e}")
        finally:
            try:
                # Cancel any pending tasks
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()
                # Wait for all tasks to be cancelled
                if pending:
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                self.loop.close()

    def start(self):
        """Start the receiver thread."""
        if self.thread is not None:
            self.logger.warning("Receiver already started")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run, name="Thread3-BTaskReceiver")
        self.thread.start()
        self.logger.info("B task receiver thread started")

    def stop(self):
        """Stop the receiver thread."""
        if self.thread is None:
            return

        self.running = False
        self.thread.join(timeout=10.0)
        self.logger.info("B task receiver thread stopped")

    def is_alive(self) -> bool:
        """Check if receiver thread is still running."""
        return self.thread is not None and self.thread.is_alive()


# ============================================================================
# Thread 4: Workflow Monitor
# ============================================================================

class WorkflowMonitor:
    """
    Thread 4: Monitor workflow completion and calculate statistics.

    Polls the completion queue from Thread 3 and:
    1. Receives workflow completion events
    2. Calculates per-workflow statistics
    3. Aggregates overall statistics
    4. Detects experiment completion
    """

    def __init__(self,
                 completion_queue: Queue,
                 expected_workflows: int):
        """
        Initialize workflow monitor.

        Args:
            completion_queue: Queue to receive workflow completion events
            expected_workflows: Total number of workflows to expect
        """
        self.completion_queue = completion_queue
        self.expected_workflows = expected_workflows
        self.logger = logging.getLogger("Thread4.WorkflowMonitor")

        # Tracking
        self.completed_workflows: List[WorkflowCompletionEvent] = []

        # Thread control
        self.thread: Optional[threading.Thread] = None
        self.running = False

    def _run(self):
        """Main monitoring loop."""
        self.logger.info(f"Starting workflow monitor (expecting {self.expected_workflows} workflows)")

        while self.running:
            try:
                # Poll queue with timeout
                event = self.completion_queue.get(timeout=0.5)
                self.completed_workflows.append(event)

                completed_count = len(self.completed_workflows)

                if completed_count % 10 == 0 or completed_count == self.expected_workflows:
                    self.logger.info(f"Workflows completed: {completed_count}/{self.expected_workflows}")

                # Check if all workflows complete
                if completed_count >= self.expected_workflows:
                    self.logger.info("All workflows completed!")
                    break

            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in workflow monitor: {e}")

        self.logger.info(f"Workflow monitor stopped. Completed: {len(self.completed_workflows)}/{self.expected_workflows}")

    def start(self):
        """Start the monitor thread."""
        if self.thread is not None:
            self.logger.warning("Monitor already started")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run, name="Thread4-WorkflowMonitor")
        self.thread.start()
        self.logger.info("Workflow monitor thread started")

    def stop(self):
        """Stop the monitor thread."""
        if self.thread is None:
            return

        self.running = False
        self.thread.join(timeout=10.0)
        self.logger.info("Workflow monitor thread stopped")

    def is_alive(self) -> bool:
        """Check if monitor thread is still running."""
        return self.thread is not None and self.thread.is_alive()


# ============================================================================
# Utility Functions
# ============================================================================

def clear_scheduler_tasks(scheduler_url: str):
    """Clear all tasks from a scheduler."""
    logger = logging.getLogger("Utils")
    try:
        response = requests.post(f"{scheduler_url}/task/clear")
        response.raise_for_status()
        logger.info(f"Cleared tasks from {scheduler_url}")
    except Exception as e:
        logger.error(f"Failed to clear tasks from {scheduler_url}: {e}")


def set_scheduling_strategy(scheduler_url: str, strategy: str):
    """Set the scheduling strategy for a scheduler."""
    logger = logging.getLogger("Utils")
    try:
        response = requests.post(
            f"{scheduler_url}/strategy/set",
            json={"strategy_name": strategy}
        )
        response.raise_for_status()
        logger.info(f"Set strategy to '{strategy}' on {scheduler_url}")
    except Exception as e:
        logger.error(f"Failed to set strategy on {scheduler_url}: {e}")


def generate_task_ids(num_workflows: int, fanout_values: List[int], strategy: str) -> tuple[
    List[str], List[str], Dict[str, List[str]]]:
    """
    Pre-generate all task IDs for WebSocket subscription.

    Args:
        num_workflows: Number of workflows
        fanout_values: List of fanout values (number of B tasks per workflow)
        strategy: Scheduling strategy name

    Returns:
        Tuple of (a_task_ids, all_b_task_ids, b_task_ids_by_workflow)
    """
    a_task_ids = []
    all_b_task_ids = []
    b_task_ids_by_workflow = {}

    for i in range(num_workflows):
        workflow_id = f"wf-{strategy}-{i:04d}"

        # A task ID
        a_task_id = f"task-A-{strategy}-workflow-{i:04d}-A"
        a_task_ids.append(a_task_id)

        # B task IDs for this workflow
        n = fanout_values[i]
        b_task_ids = []
        for j in range(n):
            b_task_id = f"task-B-{strategy}-workflow-{i:04d}-B-{j:02d}"
            b_task_ids.append(b_task_id)
            all_b_task_ids.append(b_task_id)

        b_task_ids_by_workflow[workflow_id] = b_task_ids

    return a_task_ids, all_b_task_ids, b_task_ids_by_workflow


# ============================================================================
# Metrics Calculation
# ============================================================================

def calculate_task_metrics(records: List[TaskRecord], task_type: str) -> Dict:
    """
    Calculate metrics for a list of task records.

    Args:
        records: List of task records
        task_type: "A" or "B"

    Returns:
        Dictionary of metrics
    """
    if not records:
        return {
            "task_type": task_type,
            "num_generated": 0,
            "num_submitted": 0,
            "num_completed": 0,
            "num_failed": 0,
            "completion_times": [],
            "avg_completion_time": 0.0,
            "median_completion_time": 0.0,
            "p95_completion_time": 0.0,
            "p99_completion_time": 0.0
        }

    # Filter records by task type
    filtered = [r for r in records if r.task_type == task_type]

    # Count statuses
    num_generated = len(filtered)
    num_submitted = sum(1 for r in filtered if r.submit_time is not None)
    completed = [r for r in filtered if r.status == "completed"]
    num_completed = len(completed)
    num_failed = sum(1 for r in filtered if r.status and r.status != "completed")

    # Calculate completion times (submit → complete)
    completion_times = []
    for r in completed:
        if r.submit_time is not None and r.complete_time is not None:
            completion_times.append(r.complete_time - r.submit_time)

    # Calculate statistics
    if completion_times:
        completion_times_arr = np.array(completion_times)
        avg_completion = float(np.mean(completion_times_arr))
        median_completion = float(np.median(completion_times_arr))
        p95_completion = float(np.percentile(completion_times_arr, 95))
        p99_completion = float(np.percentile(completion_times_arr, 99))
    else:
        avg_completion = 0.0
        median_completion = 0.0
        p95_completion = 0.0
        p99_completion = 0.0

    return {
        "task_type": task_type,
        "num_generated": num_generated,
        "num_submitted": num_submitted,
        "num_completed": num_completed,
        "num_failed": num_failed,
        "completion_times": completion_times,
        "avg_completion_time": avg_completion,
        "median_completion_time": median_completion,
        "p95_completion_time": p95_completion,
        "p99_completion_time": p99_completion
    }


def calculate_workflow_metrics(completed_workflows: List[WorkflowCompletionEvent]) -> Dict:
    """
    Calculate workflow-level metrics.

    Args:
        completed_workflows: List of workflow completion events

    Returns:
        Dictionary of workflow metrics
    """
    if not completed_workflows:
        return {
            "num_completed": 0,
            "workflow_times": [],
            "avg_workflow_time": 0.0,
            "median_workflow_time": 0.0,
            "p50_workflow_time": 0.0,
            "p95_workflow_time": 0.0,
            "p99_workflow_time": 0.0,
            "fanout_distribution": {},
            "avg_fanout": 0.0
        }

    # Extract workflow times
    workflow_times = [event.workflow_time for event in completed_workflows]
    workflow_times_arr = np.array(workflow_times)

    # Extract fanout values
    fanout_values = [event.total_b_tasks for event in completed_workflows]
    fanout_arr = np.array(fanout_values)

    # Calculate fanout distribution
    unique_fanouts, counts = np.unique(fanout_arr, return_counts=True)
    fanout_distribution = {int(f): int(c) for f, c in zip(unique_fanouts, counts)}

    return {
        "num_completed": len(completed_workflows),
        "workflow_times": workflow_times,
        "avg_workflow_time": float(np.mean(workflow_times_arr)),
        "median_workflow_time": float(np.median(workflow_times_arr)),
        "p50_workflow_time": float(np.percentile(workflow_times_arr, 50)),
        "p95_workflow_time": float(np.percentile(workflow_times_arr, 95)),
        "p99_workflow_time": float(np.percentile(workflow_times_arr, 99)),
        "fanout_distribution": fanout_distribution,
        "avg_fanout": float(np.mean(fanout_arr))
    }


def print_metrics_summary(strategy: str, a_metrics: Dict, b_metrics: Dict, wf_metrics: Dict):
    """
    Print a summary of metrics for a test run.

    Args:
        strategy: Strategy name
        a_metrics: A task metrics
        b_metrics: B task metrics
        wf_metrics: Workflow metrics
    """
    print("\n" + "=" * 80)
    print(f"Results Summary: {strategy}")
    print("=" * 80)

    print("\nA Tasks:")
    print(f"  Generated:  {a_metrics['num_generated']}")
    print(f"  Submitted:  {a_metrics['num_submitted']}")
    print(f"  Completed:  {a_metrics['num_completed']}")
    print(f"  Failed:     {a_metrics['num_failed']}")
    if a_metrics['avg_completion_time'] > 0:
        print(f"  Avg time:   {a_metrics['avg_completion_time']:.2f}s")
        print(f"  Median:     {a_metrics['median_completion_time']:.2f}s")
        print(f"  P95:        {a_metrics['p95_completion_time']:.2f}s")

    print("\nB Tasks:")
    print(f"  Generated:  {b_metrics['num_generated']}")
    print(f"  Submitted:  {b_metrics['num_submitted']}")
    print(f"  Completed:  {b_metrics['num_completed']}")
    print(f"  Failed:     {b_metrics['num_failed']}")
    if b_metrics['avg_completion_time'] > 0:
        print(f"  Avg time:   {b_metrics['avg_completion_time']:.2f}s")
        print(f"  Median:     {b_metrics['median_completion_time']:.2f}s")
        print(f"  P95:        {b_metrics['p95_completion_time']:.2f}s")

    print("\nWorkflows:")
    print(f"  Completed:  {wf_metrics['num_completed']}")
    print(f"  Avg fanout: {wf_metrics['avg_fanout']:.1f} B tasks per A task")
    if wf_metrics['avg_workflow_time'] > 0:
        print(f"  Avg time:   {wf_metrics['avg_workflow_time']:.2f}s")
        print(f"  Median:     {wf_metrics['median_workflow_time']:.2f}s")
        print(f"  P95:        {wf_metrics['p95_workflow_time']:.2f}s")
        print(f"  P99:        {wf_metrics['p99_workflow_time']:.2f}s")

    print("\nFanout Distribution:")
    for fanout, count in sorted(wf_metrics['fanout_distribution'].items()):
        percentage = (count / wf_metrics['num_completed']) * 100 if wf_metrics['num_completed'] > 0 else 0
        print(f"  {fanout} B tasks: {count} workflows ({percentage:.1f}%)")

    print("=" * 80)


# ============================================================================
# Main Test Orchestration
# ============================================================================

def test_strategy_workflow(
    strategy: str,
    num_workflows: int,
    task_times_a: List[float],
    task_times_b: List[float],
    fanout_values: List[int],
    qps_a: float,
    timeout_minutes: int = 10
) -> Dict:
    """
    Test a single scheduling strategy with dynamic workflow fanout.

    Args:
        strategy: Scheduling strategy ("min_time", "round_robin", "probabilistic")
        num_workflows: Number of workflows to test
        task_times_a: List of task execution times for A tasks
        task_times_b: List of task execution times for B tasks
        fanout_values: List of fanout values (number of B tasks per workflow)
        qps_a: Target QPS for A task submission
        timeout_minutes: Maximum time to wait for completion

    Returns:
        Dictionary of test results
    """
    logger = logging.getLogger(f"Test.{strategy}")
    logger.info(f"Starting test for strategy: {strategy}")

    # Step 1: Clear tasks from both schedulers
    logger.info("Step 1: Clearing tasks from schedulers")
    clear_scheduler_tasks(SCHEDULER_A_URL)
    clear_scheduler_tasks(SCHEDULER_B_URL)
    time.sleep(1.0)

    # Step 2: Set scheduling strategy
    logger.info(f"Step 2: Setting strategy to '{strategy}'")
    set_scheduling_strategy(SCHEDULER_A_URL, strategy)
    set_scheduling_strategy(SCHEDULER_B_URL, strategy)
    time.sleep(0.5)

    # Step 3: Pre-generate all task IDs
    logger.info("Step 3: Pre-generating task IDs")
    a_task_ids, all_b_task_ids, b_task_ids_by_workflow = generate_task_ids(
        num_workflows, fanout_values, strategy
    )
    logger.info(f"Generated {len(a_task_ids)} A task IDs, {len(all_b_task_ids)} B task IDs")

    # Step 4: Generate task data
    logger.info("Step 4: Generating task data")
    a_tasks: List[WorkflowTaskData] = []
    b_tasks_by_workflow: Dict[str, List[WorkflowTaskData]] = {}

    for i in range(num_workflows):
        workflow_id = f"wf-{strategy}-{i:04d}"
        a_task_id = a_task_ids[i]

        # Create A task
        a_task = WorkflowTaskData(
            task_id=a_task_id,
            workflow_id=workflow_id,
            task_type="A",
            sleep_time=task_times_a[i],
            exp_runtime=task_times_a[i] * 1000
        )
        a_tasks.append(a_task)

        # Create B tasks for this workflow
        n = fanout_values[i]
        b_tasks = []
        b_task_ids = b_task_ids_by_workflow[workflow_id]

        for j in range(n):
            # Calculate index into task_times_b
            b_task_index = sum(fanout_values[:i]) + j
            b_task = WorkflowTaskData(
                task_id=b_task_ids[j],
                workflow_id=workflow_id,
                task_type="B",
                sleep_time=task_times_b[b_task_index],
                exp_runtime=task_times_b[b_task_index] * 1000
            )
            b_tasks.append(b_task)

        b_tasks_by_workflow[workflow_id] = b_tasks

    # Step 5: Initialize workflow states
    logger.info("Step 5: Initializing workflow states")
    workflow_states: Dict[str, WorkflowState] = {}
    for i in range(num_workflows):
        workflow_id = f"wf-{strategy}-{i:04d}"
        workflow_states[workflow_id] = WorkflowState(
            workflow_id=workflow_id,
            strategy=strategy,
            a_task_id=a_task_ids[i],
            b_task_ids=b_task_ids_by_workflow[workflow_id],
            total_b_tasks=fanout_values[i]
        )

    # Step 6: Create completion queue
    completion_queue = Queue()

    # Step 7: Start Thread 3 (B Task Receiver) - must start before B tasks are submitted
    logger.info("Step 7: Starting Thread 3 (B Task Receiver)")
    b_receiver = BTaskReceiver(
        scheduler_b_ws=SCHEDULER_B_WS,
        b_task_ids=all_b_task_ids,
        workflow_states=workflow_states,
        completion_queue=completion_queue
    )
    b_receiver.start()
    time.sleep(2.0)  # Wait for WebSocket connection

    # Step 8: Start Thread 4 (Workflow Monitor)
    logger.info("Step 8: Starting Thread 4 (Workflow Monitor)")
    monitor = WorkflowMonitor(
        completion_queue=completion_queue,
        expected_workflows=num_workflows
    )
    monitor.start()

    # Step 9: Start Thread 2 (A Task Receiver + B Task Submitter)
    logger.info("Step 9: Starting Thread 2 (A Task Receiver + B Submitter)")
    a_receiver = ATaskReceiver(
        scheduler_a_ws=SCHEDULER_A_WS,
        scheduler_b_url=SCHEDULER_B_URL,
        a_task_ids=a_task_ids,
        b_tasks_by_workflow=b_tasks_by_workflow,
        workflow_states=workflow_states
    )
    a_receiver.start()
    time.sleep(2.0)  # Wait for WebSocket connection

    # Step 10: Start Thread 1 (A Task Submitter)
    logger.info("Step 10: Starting Thread 1 (A Task Submitter)")
    a_submitter = PoissonTaskSubmitter(
        scheduler_url=SCHEDULER_A_URL,
        tasks=a_tasks,
        qps=qps_a,
        workflow_states=workflow_states
    )
    a_submitter.start()

    # Step 11: Wait for A task submission to complete
    logger.info("Step 11: Waiting for A task submission to complete")
    while a_submitter.is_alive():
        time.sleep(0.5)
    logger.info("A task submission complete")

    # Step 12: Wait for workflow completion with timeout
    logger.info(f"Step 12: Waiting for workflow completion (timeout: {timeout_minutes} minutes)")
    timeout_seconds = timeout_minutes * 60
    start_wait = time.time()

    while monitor.is_alive() and (time.time() - start_wait) < timeout_seconds:
        time.sleep(1.0)
        elapsed = time.time() - start_wait
        if int(elapsed) % 10 == 0:
            logger.info(f"Waiting... {len(monitor.completed_workflows)}/{num_workflows} workflows completed")

    # Step 13: Stop all threads
    logger.info("Step 13: Stopping all threads")
    a_submitter.stop()
    a_receiver.stop()
    b_receiver.stop()
    monitor.stop()

    # Step 14: Collect results
    logger.info("Step 14: Collecting results")

    # Combine all task records
    all_a_records = a_submitter.submitted_tasks + a_receiver.a_results
    all_b_records = a_receiver.b_submitted + b_receiver.b_results

    # Calculate metrics
    a_metrics = calculate_task_metrics(all_a_records, "A")
    b_metrics = calculate_task_metrics(all_b_records, "B")
    wf_metrics = calculate_workflow_metrics(monitor.completed_workflows)

    # Print summary
    print_metrics_summary(strategy, a_metrics, b_metrics, wf_metrics)

    # Calculate actual QPS
    actual_qps = 0.0
    if a_submitter.submission_end_time and a_submitter.submission_start_time:
        duration = a_submitter.submission_end_time - a_submitter.submission_start_time
        actual_qps = len(a_tasks) / duration if duration > 0 else 0.0

    return {
        "strategy": strategy,
        "num_workflows": num_workflows,
        "target_qps": qps_a,
        "actual_qps": actual_qps,
        "a_tasks": a_metrics,
        "b_tasks": b_metrics,
        "workflows": wf_metrics,
        "submission_time": a_submitter.submission_end_time - a_submitter.submission_start_time
                          if a_submitter.submission_end_time and a_submitter.submission_start_time else 0.0
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main(num_workflows: int = 100, qps_a: float = 8.0, seed: int = 42,
         strategies: List[str] = None):
    """
    Main entry point for experiment 04.

    Args:
        num_workflows: Number of workflows to generate and execute per strategy
        qps_a: Target QPS for A task submission
        seed: Random seed for reproducibility
        strategies: List of strategies to test (default: all three)
    """
    if strategies is None:
        strategies = ["min_time", "round_robin", "probabilistic"]

    logger = logging.getLogger("Main")
    logger.info("Starting Experiment 04: Multi-Model Workflow with Dynamic Fanout")
    logger.info(f"Configuration: {num_workflows} workflows, QPS={qps_a}, seed={seed}")
    logger.info(f"Strategies to test: {', '.join(strategies)}")

    # Experiment parameters
    NUM_WORKFLOWS = num_workflows
    QPS_A = qps_a
    SEED = seed

    # Generate workloads
    logger.info("Generating workloads...")

    # A tasks: Bimodal distribution - generate exactly num_workflows tasks
    task_times_a, workload_a_config = generate_bimodal_distribution(NUM_WORKFLOWS, seed=SEED)
    print_distribution_stats(task_times_a, workload_a_config)
    logger.info(f"Generated {len(task_times_a)} A tasks (will submit all)")

    # Fanout: Uniform distribution (3-8 B tasks per A task)
    fanout_values, fanout_config = generate_fanout_distribution(NUM_WORKFLOWS, seed=SEED)
    print_fanout_stats(fanout_values, fanout_config)

    # B tasks: Bimodal distribution - generate exactly sum(fanout_values) tasks
    total_b_tasks = sum(fanout_values)
    task_times_b, workload_b_config = generate_bimodal_distribution(total_b_tasks, seed=SEED + 1)
    print_distribution_stats(task_times_b, workload_b_config)
    logger.info(f"Generated {len(task_times_b)} B tasks total (will submit all)")

    # Run tests for specified strategies
    all_results = []

    for strategy in strategies:
        logger.info(f"\n{'=' * 80}\nTesting strategy: {strategy}\n{'=' * 80}")
        results = test_strategy_workflow(
            strategy=strategy,
            num_workflows=NUM_WORKFLOWS,
            task_times_a=task_times_a,
            task_times_b=task_times_b,
            fanout_values=fanout_values,
            qps_a=QPS_A,
            timeout_minutes=10
        )
        all_results.append(results)

        # Brief pause between strategies
        time.sleep(5.0)

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/results_workflow_dynamic_{timestamp}.json"

    output_data = {
        "experiment": "04.multi_model_workflow_dynamic",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_workflows": NUM_WORKFLOWS,
            "qps_a": QPS_A,
            "seed": SEED,
            "workload_a": asdict(workload_a_config),
            "workload_b": asdict(workload_b_config),
            "fanout": asdict(fanout_config)
        },
        "results": all_results
    }
    
    if os.path.exists("results"):
        os.makedirs("results", exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("Strategy Comparison")
    print("=" * 80)
    print(f"{'Strategy':<15} {'A Avg (s)':<12} {'B Avg (s)':<12} {'WF Avg (s)':<12} {'WF P95 (s)':<12} {'Completed':<12}")
    print("-" * 80)

    for result in all_results:
        strategy = result['strategy']
        a_avg = result['a_tasks']['avg_completion_time']
        b_avg = result['b_tasks']['avg_completion_time']
        wf_avg = result['workflows']['avg_workflow_time']
        wf_p95 = result['workflows']['p95_workflow_time']
        wf_completed = result['workflows']['num_completed']

        print(f"{strategy:<15} {a_avg:<12.2f} {b_avg:<12.2f} {wf_avg:<12.2f} {wf_p95:<12.2f} {wf_completed:<12}")

    print("=" * 80)
    logger.info("Experiment complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment 04: Multi-Model Workflow with Dynamic Fanout (1-to-n)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--num-workflows",
        type=int,
        default=100,
        help="Number of workflows to generate and execute per strategy"
    )

    parser.add_argument(
        "--qps",
        type=float,
        default=8.0,
        help="Target queries per second (QPS) for A task submission"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["min_time", "round_robin", "probabilistic"],
        choices=["min_time", "round_robin", "probabilistic"],
        help="Scheduling strategies to test"
    )

    args = parser.parse_args()

    main(
        num_workflows=args.num_workflows,
        qps_a=args.qps,
        seed=args.seed,
        strategies=args.strategies
    )
