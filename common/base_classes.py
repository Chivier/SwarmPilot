"""
Base abstract classes for task submitters and receivers.

This module provides reusable foundation classes for implementing workflow
experiment threads, following patterns from the reference implementation in
experiments/03.Exp4.Text2Video/test_dynamic_workflow_sim.py.
"""

import asyncio
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import requests
import websockets


class BaseTaskSubmitter(threading.Thread, ABC):
    """
    Abstract base class for task submitter threads.

    Submitters are responsible for creating and submitting tasks to the scheduler
    at a controlled rate. They run in their own thread and can be stopped gracefully.

    Subclasses must implement:
    - _prepare_task_payload(): Create the JSON payload for a specific task
    - _get_next_task_data(): Get the next task to submit (or None when done)
    """

    def __init__(self,
                 name: str,
                 scheduler_url: str,
                 qps: Optional[float] = None,
                 duration: Optional[float] = None,
                 rate_limiter: Optional[Any] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize task submitter.

        Args:
            name: Thread name for identification
            scheduler_url: Base URL of the scheduler (e.g., "http://localhost:8100")
            qps: Target queries per second (used for interval calculation if no rate_limiter)
            duration: Maximum duration to run in seconds (optional)
            rate_limiter: Optional RateLimiter instance for global rate control
            logger: Optional logger instance
        """
        super().__init__(name=name, daemon=True)
        self.scheduler_url = scheduler_url
        self.qps = qps
        self.duration = duration
        self.rate_limiter = rate_limiter
        self.logger = logger or logging.getLogger(name)

        # Thread control
        self.stop_event = threading.Event()
        self.running = False

        # Tracking
        self.submitted_count = 0
        self.failed_count = 0
        self.submission_start_time: Optional[float] = None
        self.submission_end_time: Optional[float] = None
        self._count_lock = threading.Lock()

    @abstractmethod
    def _prepare_task_payload(self, task_data: Any) -> Dict[str, Any]:
        """
        Prepare the JSON payload for a task submission.

        Args:
            task_data: Data for the task to submit

        Returns:
            Dictionary containing task_id, model_id, task_input, and metadata
        """
        pass

    @abstractmethod
    def _get_next_task_data(self) -> Optional[Any]:
        """
        Get the next task data to submit.

        Returns:
            Task data object, or None if no more tasks to submit
        """
        pass

    def _submit_task(self, task_data: Any) -> bool:
        """
        Submit a single task to the scheduler.

        Args:
            task_data: Task data to submit

        Returns:
            True if submission succeeded, False otherwise
        """
        try:
            payload = self._prepare_task_payload(task_data)

            response = requests.post(
                f"{self.scheduler_url}/task/submit",
                json=payload,
                timeout=5.0
            )
            response.raise_for_status()

            with self._count_lock:
                self.submitted_count += 1

            return True

        except Exception as e:
            self.logger.error(f"Failed to submit task: {e}")
            with self._count_lock:
                self.failed_count += 1
            return False

    def run(self):
        """Main thread execution loop."""
        self.running = True
        self.submission_start_time = time.time()
        start_time = self.submission_start_time

        self.logger.info(f"Starting task submission (QPS={self.qps}, duration={self.duration})")

        while not self.stop_event.is_set():
            # Check duration limit
            if self.duration and (time.time() - start_time) >= self.duration:
                self.logger.info("Duration limit reached, stopping submission")
                break

            # Get next task
            task_data = self._get_next_task_data()
            if task_data is None:
                self.logger.info("No more tasks to submit")
                break

            # Apply rate limiting
            if self.rate_limiter:
                self.rate_limiter.acquire()

            # Submit task
            self._submit_task(task_data)

        self.submission_end_time = time.time()
        total_time = self.submission_end_time - self.submission_start_time
        actual_qps = self.submitted_count / total_time if total_time > 0 else 0

        self.logger.info(
            f"Task submission completed: {self.submitted_count} submitted, "
            f"{self.failed_count} failed in {total_time:.2f}s (actual QPS: {actual_qps:.2f})"
        )
        self.running = False

    def stop(self):
        """Stop the submitter thread gracefully."""
        self.logger.info("Stopping task submitter...")
        self.stop_event.set()
        self.running = False


class BaseTaskReceiver(threading.Thread, ABC):
    """
    Abstract base class for task receiver threads.

    Receivers listen for task completion results via WebSocket and process them.
    They run in their own thread with an async event loop for WebSocket handling.

    Subclasses must implement:
    - _get_subscription_payload(): Create subscription message for WebSocket
    - _process_result(): Handle a task completion result
    """

    def __init__(self,
                 name: str,
                 scheduler_url: str,
                 model_id: str,
                 workflow_states: Optional[Dict[str, Any]] = None,
                 state_lock: Optional[threading.Lock] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize task receiver.

        Args:
            name: Thread name for identification
            scheduler_url: Base URL of the scheduler
            model_id: Model ID to subscribe to results for
            workflow_states: Optional shared dictionary of workflow states
            state_lock: Optional lock for workflow state access
            logger: Optional logger instance
        """
        super().__init__(name=name, daemon=True)
        self.scheduler_url = scheduler_url
        self.model_id = model_id
        self.workflow_states = workflow_states or {}
        self.state_lock = state_lock or threading.Lock()
        self.logger = logger or logging.getLogger(name)

        # Build WebSocket URL from HTTP URL
        ws_protocol = "wss" if scheduler_url.startswith("https") else "ws"
        http_base = scheduler_url.replace("http://", "").replace("https://", "")
        self.ws_url = f"{ws_protocol}://{http_base}/task/subscribe"

        # Thread control
        self.stop_event = threading.Event()
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Tracking
        self.received_count = 0
        self.error_count = 0
        self._count_lock = threading.Lock()

    @abstractmethod
    def _get_subscription_payload(self) -> Dict[str, Any]:
        """
        Get the subscription payload for WebSocket connection.

        Returns:
            Dictionary with subscription message (e.g., {"type": "subscribe", "model_id": "..."})
        """
        pass

    @abstractmethod
    async def _process_result(self, data: Dict[str, Any]):
        """
        Process a task completion result.

        Args:
            data: Result data from WebSocket message
        """
        pass

    def _update_workflow_state(self, workflow_id: str, update_fn: callable):
        """
        Thread-safe workflow state update.

        Args:
            workflow_id: ID of the workflow to update
            update_fn: Function to apply to the workflow state
        """
        with self.state_lock:
            workflow_state = self.workflow_states.get(workflow_id)
            if workflow_state:
                update_fn(workflow_state)
            else:
                self.logger.warning(f"Workflow state not found for {workflow_id}")

    def _handle_error(self, error_msg: str, exception: Optional[Exception] = None):
        """
        Handle error during result processing.

        Args:
            error_msg: Error message to log
            exception: Optional exception that was raised
        """
        with self._count_lock:
            self.error_count += 1

        if exception:
            self.logger.error(f"{error_msg}: {exception}")
        else:
            self.logger.error(error_msg)

    async def _run_async(self):
        """Main async loop for receiving results."""
        self.logger.info(f"Connecting to WebSocket: {self.ws_url}")

        try:
            async with websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            ) as websocket:
                # Send subscription message
                subscribe_msg = self._get_subscription_payload()
                await websocket.send(json.dumps(subscribe_msg))
                self.logger.info(f"Sent subscription request: {subscribe_msg}")

                # Wait for acknowledgment
                try:
                    ack = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    ack_data = json.loads(ack)
                    if ack_data.get("type") == "ack":
                        self.logger.info(f"Subscription confirmed")
                    else:
                        self.logger.warning(f"Unexpected acknowledgment: {ack_data}")
                except asyncio.TimeoutError:
                    self.logger.error("Timeout waiting for subscription acknowledgment")
                    return

                # Receive loop
                while self.running and not self.stop_event.is_set():
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)

                        # Handle different message types
                        if data.get("type") == "ack":
                            # Skip acknowledgment messages
                            continue
                        elif data.get("type") == "result":
                            # Process task result
                            await self._process_result(data)
                            with self._count_lock:
                                self.received_count += 1
                        elif data.get("type") == "error":
                            self._handle_error(f"WebSocket error: {data.get('error')}")

                    except asyncio.TimeoutError:
                        # Timeout is normal for polling, continue
                        continue
                    except websockets.ConnectionClosed:
                        self.logger.warning("WebSocket connection closed")
                        break
                    except Exception as e:
                        self._handle_error("Error receiving result", e)

        except Exception as e:
            self._handle_error("WebSocket connection error", e)

    def run(self):
        """Main thread execution loop."""
        self.running = True
        self.logger.info("Starting task receiver")

        # Create new event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            # Run async loop
            self.loop.run_until_complete(self._run_async())
        finally:
            self.loop.close()
            self.running = False
            self.logger.info(
                f"Task receiver stopped: {self.received_count} received, "
                f"{self.error_count} errors"
            )

    def stop(self):
        """Stop the receiver thread gracefully."""
        self.logger.info("Stopping task receiver...")
        self.stop_event.set()
        self.running = False
