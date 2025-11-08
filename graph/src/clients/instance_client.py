"""
Asynchronous HTTP/WebSocket client for the Instance Service.

Provides low-level API methods for interacting with instances, including:
- Model lifecycle management (start, stop, restart)
- Task submission and monitoring (HTTP and WebSocket)
- Instance health checking and status queries
- Real-time task result streaming via WebSocket

Features:
- Connection pooling and keep-alive
- Automatic retries with exponential backoff
- Configurable timeouts
- WebSocket support for real-time updates
- Strict response validation with Pydantic
- Full type hints

Example usage:
    ```python
    from graph.src.clients.instance_client import InstanceClient

    async with InstanceClient("http://localhost:5000") as client:
        # Start a model
        await client.start_model(
            model_id="easyocr/det",
            parameters={"gpu": 0}
        )

        # Submit a task via HTTP
        result = await client.submit_task(
            task_id="task-123",
            model_id="easyocr/det",
            task_input={"image": "base64..."}
        )

        # Or use WebSocket for real-time updates
        await client.connect_websocket()

        async def handle_result(data):
            print(f"Task completed: {data['task_id']}")

        # Start listening in background
        listen_task = await client.start_listening(handle_result)

        # Submit tasks via WebSocket
        await client.submit_task_ws(
            task_id="task-456",
            model_id="easyocr/det",
            task_input={"image": "base64..."}
        )
    ```
"""

import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import httpx
import websockets
from websockets.client import WebSocketClientProtocol

from .models import (
    ErrorResponse,
    HealthResponse,
    InstanceInfo,
    ModelStartResponse,
    ModelStopResponse,
    RestartInitResponse,
    RestartOperation,
    TaskCancelResponse,
    TaskClearResponse,
    TaskInfo,
    TaskListResponse,
    TaskSubmitResponse,
    WSError,
    WSPing,
    WSPong,
    WSTaskResult,
    WSTaskStatus,
    WSTaskSubmit,
)

logger = logging.getLogger(__name__)


# ==================== Exception Classes ====================


class InstanceClientError(Exception):
    """Base exception for all InstanceClient errors."""

    pass


class InstanceConnectionError(InstanceClientError):
    """Raised when connection to instance fails.

    Examples:
        - Cannot connect to instance HTTP endpoint
        - WebSocket connection failed
        - Connection timeout
    """

    pass


class InstanceAPIError(InstanceClientError):
    """Raised when API returns an error response.

    Includes the HTTP status code and response data for debugging.
    """

    def __init__(self, message: str, status_code: int, response_data: Optional[Dict[str, Any]] = None):
        """Initialize the API error.

        Args:
            message: Error message describing what went wrong
            status_code: HTTP status code returned by the API
            response_data: Optional response data from the server
        """
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}


class InstanceTimeoutError(InstanceClientError):
    """Raised when a request to the instance times out.

    Examples:
        - HTTP request exceeds timeout
        - WebSocket receive timeout
    """

    pass


# ==================== Instance Client ====================


class InstanceClient:
    """Asynchronous HTTP/WebSocket client for the Instance Service.

    The Instance Service manages model containers and task queues, processing
    inference requests sequentially. This client provides comprehensive methods
    for model management, task submission, and real-time monitoring.
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        websocket_url: Optional[str] = None,
        verify_ssl: bool = True,
        instance_module_path: Optional[str] = None,
    ):
        """Initialize the InstanceClient.

        Args:
            base_url: Base URL of the instance (e.g., "http://localhost:5000")
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum retry attempts for failed requests (default: 3)
            retry_delay: Initial delay between retries in seconds (default: 1.0)
            websocket_url: WebSocket URL (defaults to ws:// version of base_url)
            verify_ssl: Whether to verify SSL certificates (default: True)
            instance_module_path: Path to instance module directory (auto-detected if None)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.verify_ssl = verify_ssl

        # WebSocket URL
        if websocket_url:
            self.websocket_url = websocket_url
        else:
            # Convert http:// to ws:// or https:// to wss://
            ws_scheme = "wss" if base_url.startswith("https") else "ws"
            host = base_url.replace("http://", "").replace("https://", "")
            self.websocket_url = f"{ws_scheme}://{host}/ws"

        # HTTP client (initialized in __aenter__)
        self._http_client: Optional[httpx.AsyncClient] = None

        # WebSocket connection
        self._ws_connection: Optional[WebSocketClientProtocol] = None
        self._ws_listen_task: Optional[asyncio.Task] = None

        # Auto-start management
        self._instance_process: Optional[subprocess.Popen] = None
        self.instance_module_path = instance_module_path or self._find_instance_module()

    def _find_instance_module(self) -> str:
        """Find instance module path automatically.

        Returns:
            Path to instance module directory
        """
        # Try to find instance module relative to current file
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent  # graph/src/clients/instance_client.py -> project root
        instance_path = project_root / "instance"

        if instance_path.exists():
            return str(instance_path)

        # Fallback: check environment variable
        if "INSTANCE_MODULE_PATH" in os.environ:
            return os.environ["INSTANCE_MODULE_PATH"]

        # Fallback: assume instance is a sibling directory
        return str(Path.cwd() / "instance")

    @property
    def is_instance_running(self) -> bool:
        """Check if instance process is running.

        Returns:
            True if process is running, False otherwise
        """
        if self._instance_process is None:
            return False
        return self._instance_process.poll() is None

    async def __aenter__(self) -> "InstanceClient":
        """Enter the async context manager - initialize HTTP client."""
        await self._ensure_http_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager - cleanup resources."""
        await self.close()

    async def close(self) -> None:
        """Close all connections and cleanup resources.

        Closes WebSocket connection, cancels listening tasks, and closes HTTP client.
        This is called automatically when using the async context manager.
        """
        # Close WebSocket
        if self._ws_connection:
            await self.disconnect_websocket()

        # Cancel listen task
        if self._ws_listen_task and not self._ws_listen_task.done():
            self._ws_listen_task.cancel()
            try:
                await self._ws_listen_task
            except asyncio.CancelledError:
                pass

        # Close HTTP client
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        logger.debug(f"InstanceClient closed for {self.base_url}")

    # ==================== Internal Helper Methods ====================

    async def _ensure_http_client(self) -> None:
        """Ensure HTTP client is initialized."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=self.timeout,
                verify=self.verify_ssl,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Make HTTP request with retry logic and error handling.

        Automatically retries on network errors and timeouts with exponential backoff.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint (e.g., "/model/start")
            **kwargs: Additional arguments for httpx request (json, params, etc.)

        Returns:
            httpx.Response object for successful requests

        Raises:
            InstanceConnectionError: Connection to instance failed
            InstanceTimeoutError: Request timed out
            InstanceAPIError: API returned error response (4xx, 5xx)
        """
        await self._ensure_http_client()

        url = f"{self.base_url}{endpoint}"
        attempt = 0

        while attempt < self.max_retries:
            try:
                logger.debug(f"Request: {method} {url} (attempt {attempt + 1}/{self.max_retries})")

                response = await self._http_client.request(method, url, **kwargs)

                # Handle successful response
                if response.status_code < 400:
                    logger.debug(f"Response: {response.status_code} from {url}")
                    return response

                # Handle error response
                self._handle_error(response)

            except httpx.TimeoutException as e:
                attempt += 1
                if attempt >= self.max_retries:
                    raise InstanceTimeoutError(f"Request timed out after {self.timeout}s: {url}") from e

                delay = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(f"Timeout, retrying in {delay}s... ({attempt}/{self.max_retries})")
                await asyncio.sleep(delay)

            except httpx.ConnectError as e:
                attempt += 1
                if attempt >= self.max_retries:
                    raise InstanceConnectionError(f"Failed to connect to {url}: {e}") from e

                delay = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(f"Connection error, retrying in {delay}s... ({attempt}/{self.max_retries})")
                await asyncio.sleep(delay)

            except httpx.RequestError as e:
                attempt += 1
                if attempt >= self.max_retries:
                    raise InstanceConnectionError(f"Request failed for {url}: {e}") from e

                delay = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(f"Request error, retrying in {delay}s... ({attempt}/{self.max_retries})")
                await asyncio.sleep(delay)

        # Should not reach here, but just in case
        raise InstanceClientError(f"Failed to complete request after {self.max_retries} attempts")

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle error responses from API.

        Parses error message from response and raises appropriate exception.

        Args:
            response: httpx.Response object with error status

        Raises:
            InstanceAPIError: With error message and status code
        """
        try:
            error_data = response.json()
            error_msg = error_data.get("error", "Unknown error")
        except Exception:
            error_msg = response.text or f"HTTP {response.status_code}"

        raise InstanceAPIError(
            message=error_msg,
            status_code=response.status_code,
            response_data=error_data if "error_data" in locals() else {},
        )

    # ==================== Model Management API ====================

    async def start_model(
        self, model_id: str, parameters: Optional[Dict[str, Any]] = None, scheduler_url: Optional[str] = None
    ) -> ModelStartResponse:
        """Start a model/tool on this instance.

        Initializes and loads the specified model, making it ready to process tasks.
        Optionally registers with a scheduler for automatic task distribution.

        Args:
            model_id: Model identifier (e.g., "easyocr/det", "gpt-4")
            parameters: Model-specific initialization parameters (e.g., {"gpu": 0, "batch_size": 8})
            scheduler_url: Optional scheduler URL to register with for automatic task routing

        Returns:
            ModelStartResponse with:
                - model_id: Started model identifier
                - status: Model status (should be "running")
                - parameters: Initialization parameters used
                - started_at: ISO 8601 timestamp

        Raises:
            InstanceAPIError: If model already running, model not found, or initialization failed

        Example:
            ```python
            response = await client.start_model(
                model_id="easyocr/det",
                parameters={"gpu": 0, "workers": 4},
                scheduler_url="http://scheduler:8000"
            )
            print(f"Model {response.model_id} started at {response.started_at}")
            ```
        """
        payload = {
            "model_id": model_id,
            "parameters": parameters or {},
        }
        if scheduler_url:
            payload["scheduler_url"] = scheduler_url

        response = await self._make_request("POST", "/model/start", json=payload)
        return ModelStartResponse(**response.json())

    async def stop_model(self) -> ModelStopResponse:
        """Stop the currently running model.

        Gracefully shuts down the model, freeing resources. Any queued tasks
        will be marked as failed.

        Returns:
            ModelStopResponse with:
                - model_id: Stopped model identifier
                - status: Model status (should be "stopped")
                - stopped_at: ISO 8601 timestamp

        Raises:
            InstanceAPIError: If no model is currently running

        Example:
            ```python
            response = await client.stop_model()
            print(f"Model {response.model_id} stopped")
            ```
        """
        response = await self._make_request("POST", "/model/stop")
        return ModelStopResponse(**response.json())

    async def restart_model(
        self, model_id: str, parameters: Optional[Dict[str, Any]] = None, scheduler_url: Optional[str] = None
    ) -> RestartInitResponse:
        """Gracefully restart with a new model (or same model with new parameters).

        Initiates a multi-step restart operation:
        1. Drains the current task queue (completes all pending tasks)
        2. Stops the old model
        3. Starts the new model

        Use get_restart_status() to track progress.

        Args:
            model_id: New model identifier to start
            parameters: Model-specific initialization parameters
            scheduler_url: Optional scheduler URL to register with

        Returns:
            RestartInitResponse with:
                - operation_id: Unique ID for tracking this restart operation
                - status: Initial status ("initiated")
                - current_phase: Current phase of restart ("draining")
                - message: Status message

        Raises:
            InstanceAPIError: If restart cannot be initiated (e.g., another restart in progress)

        Example:
            ```python
            response = await client.restart_model(
                model_id="gpt-4-turbo",
                parameters={"max_tokens": 4096}
            )

            # Track progress
            while True:
                status = await client.get_restart_status(response.operation_id)
                print(f"Phase: {status.current_phase}, Status: {status.status}")
                if status.status in ("completed", "failed"):
                    break
                await asyncio.sleep(1)
            ```
        """
        payload = {
            "model_id": model_id,
            "parameters": parameters or {},
        }
        if scheduler_url:
            payload["scheduler_url"] = scheduler_url

        response = await self._make_request("POST", "/model/restart", json=payload)
        return RestartInitResponse(**response.json())

    async def get_restart_status(self, operation_id: str) -> RestartOperation:
        """Get status of a restart operation.

        Queries the current state of an ongoing or completed restart operation.

        Args:
            operation_id: Operation ID from restart_model() response

        Returns:
            RestartOperation with:
                - operation_id: Operation identifier
                - status: Overall status ("initiated", "in_progress", "completed", "failed")
                - current_phase: Current phase ("draining", "stopping", "starting", "completed")
                - message: Detailed status message
                - started_at: When restart was initiated
                - completed_at: When restart completed (if finished)

        Raises:
            InstanceAPIError: If operation ID not found

        Example:
            ```python
            status = await client.get_restart_status("op-12345")
            if status.status == "completed":
                print("Restart successful!")
            elif status.status == "failed":
                print(f"Restart failed: {status.message}")
            ```
        """
        response = await self._make_request("GET", f"/model/restart/{operation_id}")
        return RestartOperation(**response.json())

    # ==================== Task Management API (HTTP) ====================

    async def submit_task(self, task_id: str, model_id: str, task_input: Dict[str, Any]) -> TaskSubmitResponse:
        """Submit a task to the instance queue via HTTP.

        Adds a task to the instance's queue for processing. Tasks are processed
        sequentially in FIFO order. For real-time result updates, use WebSocket
        methods (submit_task_ws + listen_for_results) instead.

        Args:
            task_id: Unique task identifier
            model_id: Model to process this task (must match running model)
            task_input: Task input data in model-specific format

        Returns:
            TaskSubmitResponse with:
                - task_id: Task identifier
                - status: Task status ("queued" or "running")
                - submitted_at: ISO 8601 timestamp

        Raises:
            InstanceAPIError: If task_id already exists, model mismatch, or queue full

        Example:
            ```python
            response = await client.submit_task(
                task_id="task-001",
                model_id="easyocr/det",
                task_input={"image": "base64_encoded_image_data"}
            )
            print(f"Task {response.task_id} submitted with status: {response.status}")
            ```
        """
        payload = {
            "task_id": task_id,
            "model_id": model_id,
            "task_input": task_input,
        }

        response = await self._make_request("POST", "/task/submit", json=payload)
        return TaskSubmitResponse(**response.json())

    async def get_task(self, task_id: str) -> TaskInfo:
        """Get detailed information about a specific task.

        Retrieves complete task metadata including status, timestamps, result, and error.

        Args:
            task_id: Task identifier to query

        Returns:
            TaskInfo with:
                - task_id: Task identifier
                - model_id: Model ID
                - status: Task status ("queued", "running", "completed", "failed", "cancelled")
                - task_input: Original input data
                - result: Task result (if completed)
                - error: Error message (if failed)
                - submitted_at, started_at, completed_at: ISO 8601 timestamps
                - execution_time_ms: Execution duration in milliseconds

        Raises:
            InstanceAPIError: If task not found (404)

        Example:
            ```python
            task = await client.get_task("task-001")
            if task.status == "completed":
                print(f"Result: {task.result}")
            elif task.status == "failed":
                print(f"Error: {task.error}")
            ```
        """
        response = await self._make_request("GET", "/task/query", params={"task_id": task_id})
        data = response.json()

        # Handle both single task and list response
        if isinstance(data, dict) and "tasks" in data:
            tasks = data["tasks"]
            if tasks:
                return TaskInfo(**tasks[0])
            raise InstanceAPIError("Task not found", 404, data)

        return TaskInfo(**data)

    async def list_tasks(self, status: Optional[str] = None, limit: int = 100) -> TaskListResponse:
        """List all tasks with optional filtering.

        Retrieves a list of tasks from the instance, optionally filtered by status.

        Args:
            status: Filter by status ("queued", "running", "completed", "failed", "cancelled")
            limit: Maximum number of tasks to return (default: 100)

        Returns:
            TaskListResponse with:
                - tasks: List of TaskInfo objects
                - count: Number of tasks returned
                - total: Total number of tasks matching filter

        Example:
            ```python
            # Get all completed tasks
            response = await client.list_tasks(status="completed")
            for task in response.tasks:
                print(f"Task {task.task_id}: {task.execution_time_ms}ms")

            # Get all tasks
            all_tasks = await client.list_tasks(limit=1000)
            ```
        """
        params = {"limit": limit}
        if status:
            params["status"] = status

        response = await self._make_request("GET", "/task/query", params=params)
        return TaskListResponse(**response.json())

    async def cancel_task(self, task_id: str) -> TaskCancelResponse:
        """Cancel a queued task or remove a completed/failed task.

        Removes a task from the queue (if queued) or from history (if finished).
        Cannot cancel running tasks - they must complete or fail naturally.

        Args:
            task_id: Task identifier to cancel

        Returns:
            TaskCancelResponse with:
                - task_id: Cancelled task identifier
                - status: Previous task status
                - message: Confirmation message

        Raises:
            InstanceAPIError: If task cannot be cancelled (e.g., currently running)

        Example:
            ```python
            response = await client.cancel_task("task-001")
            print(f"Cancelled task with status: {response.status}")
            ```
        """
        response = await self._make_request("DELETE", f"/task/{task_id}")
        return TaskCancelResponse(**response.json())

    async def clear_tasks(self) -> TaskClearResponse:
        """Clear all tasks from the instance.

        Removes all queued, completed, failed, and cancelled tasks.
        Cannot clear running tasks - will skip those and clear the rest.

        Returns:
            TaskClearResponse with:
                - cleared_count: Number of tasks removed
                - message: Confirmation message

        Example:
            ```python
            response = await client.clear_tasks()
            print(f"Cleared {response.cleared_count} tasks")
            ```
        """
        response = await self._make_request("DELETE", "/task/clear")
        return TaskClearResponse(**response.json())

    # ==================== Instance Management API ====================

    async def get_info(self) -> InstanceInfo:
        """Get comprehensive instance information.

        Retrieves detailed information about the instance including running model,
        queue statistics, and uptime.

        Returns:
            InstanceInfo with:
                - instance_id: Instance identifier
                - status: Instance status ("ready", "busy", "idle")
                - model: Currently running model info (or None)
                - queue_stats: Task queue statistics (queued, running, completed, failed)
                - uptime_seconds: Instance uptime
                - version: Instance service version

        Example:
            ```python
            info = await client.get_info()
            print(f"Instance: {info.instance_id}")
            print(f"Model: {info.model.model_id if info.model else 'None'}")
            print(f"Queue: {info.queue_stats.queued} queued, {info.queue_stats.running} running")
            ```
        """
        response = await self._make_request("GET", "/info")
        return InstanceInfo(**response.json())

    async def health_check(self) -> HealthResponse:
        """Check instance health status.

        Performs a lightweight health check for monitoring and load balancing.
        Returns quickly with basic health indicators.

        Returns:
            HealthResponse with:
                - status: "healthy" or "unhealthy"
                - timestamp: ISO 8601 timestamp
                - model_running: Boolean indicating if a model is loaded
                - queue_size: Number of tasks in queue

        Example:
            ```python
            health = await client.health_check()
            if health.status == "healthy":
                print("Instance is healthy")
            ```
        """
        response = await self._make_request("GET", "/health")
        return HealthResponse(**response.json())

    # ==================== WebSocket API ====================

    async def connect_websocket(self) -> None:
        """Establish WebSocket connection to the instance.

        Creates a persistent WebSocket connection for real-time task submission
        and result streaming. Must be called before using WebSocket methods.

        Raises:
            InstanceConnectionError: If WebSocket connection fails

        Example:
            ```python
            await client.connect_websocket()
            # Now can use submit_task_ws() and listen_for_results()
            ```
        """
        try:
            logger.info(f"Connecting to WebSocket: {self.websocket_url}")
            self._ws_connection = await websockets.connect(
                self.websocket_url,
                ping_interval=20,
                ping_timeout=10,
            )
            logger.info(f"WebSocket connected: {self.websocket_url}")

        except Exception as e:
            raise InstanceConnectionError(f"Failed to connect WebSocket: {e}") from e

    async def disconnect_websocket(self) -> None:
        """Close WebSocket connection gracefully.

        Example:
            ```python
            await client.disconnect_websocket()
            ```
        """
        if self._ws_connection:
            await self._ws_connection.close()
            self._ws_connection = None
            logger.info("WebSocket disconnected")

    async def submit_task_ws(self, task_id: str, model_id: str, task_input: Dict[str, Any]) -> None:
        """Submit a task via WebSocket for real-time result delivery.

        Sends a task through the WebSocket connection. Results will be delivered
        via the WebSocket as soon as processing completes. Use listen_for_results()
        to receive results.

        Args:
            task_id: Unique task identifier
            model_id: Model to process this task
            task_input: Task input data in model-specific format

        Raises:
            InstanceConnectionError: If WebSocket not connected

        Example:
            ```python
            await client.connect_websocket()
            await client.submit_task_ws(
                task_id="task-ws-001",
                model_id="easyocr/det",
                task_input={"image": "base64..."}
            )
            # Result will be delivered via listen_for_results()
            ```
        """
        if not self._ws_connection:
            raise InstanceConnectionError("WebSocket not connected. Call connect_websocket() first.")

        message = WSTaskSubmit(task_id=task_id, model_id=model_id, task_input=task_input)

        await self._ws_connection.send(message.model_dump_json())
        logger.debug(f"Submitted task via WebSocket: {task_id}")

    async def listen_for_results(
        self, callback: Callable[[Dict[str, Any]], None], error_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Listen for task results and status updates from WebSocket.

        Blocking call that continuously receives and processes WebSocket messages.
        Invokes callback for each task result or status update. Run in a separate
        task for non-blocking operation (use start_listening() instead).

        Args:
            callback: Function called with task results/status updates
                Message format: {"type": "task_result", "task_id": "...", "status": "...", "result": {...}}
            error_callback: Optional function called with error messages

        Raises:
            InstanceConnectionError: If WebSocket not connected

        Example:
            ```python
            async def handle_result(data):
                if data["type"] == "task_result":
                    print(f"Task {data['task_id']}: {data['status']}")
                    if data['status'] == 'completed':
                        print(f"Result: {data['result']}")

            await client.connect_websocket()
            # This blocks until WebSocket closes
            await client.listen_for_results(handle_result)
            ```
        """
        if not self._ws_connection:
            raise InstanceConnectionError("WebSocket not connected. Call connect_websocket() first.")

        logger.info("Listening for WebSocket messages...")

        try:
            async for message in self._ws_connection:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "task_result":
                        result = WSTaskResult(**data)
                        callback(result.model_dump())

                    elif msg_type == "task_status":
                        status = WSTaskStatus(**data)
                        callback(status.model_dump())

                    elif msg_type == "error":
                        error = WSError(**data)
                        if error_callback:
                            error_callback(error.error)
                        else:
                            logger.error(f"WebSocket error: {error.error}")

                    elif msg_type == "ping":
                        # Respond to ping
                        pong = WSPong(timestamp=datetime.now())
                        await self._ws_connection.send(pong.model_dump_json())

                    else:
                        logger.warning(f"Unknown WebSocket message type: {msg_type}")

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            self._ws_connection = None

    async def start_listening(
        self, callback: Callable[[Dict[str, Any]], None], error_callback: Optional[Callable[[str], None]] = None
    ) -> asyncio.Task:
        """Start listening for WebSocket messages in background task.

        Non-blocking version of listen_for_results(). Creates a background task
        that continuously listens for messages.

        Args:
            callback: Function called with task results/status updates
            error_callback: Optional function called with error messages

        Returns:
            asyncio.Task that can be cancelled to stop listening

        Example:
            ```python
            async def handle_result(data):
                print(f"Received: {data}")

            await client.connect_websocket()

            # Start listening in background
            listen_task = await client.start_listening(handle_result)

            # Submit tasks
            for i in range(10):
                await client.submit_task_ws(f"task-{i}", "easyocr/det", {...})

            # Wait a bit, then stop listening
            await asyncio.sleep(30)
            listen_task.cancel()
            ```
        """
        self._ws_listen_task = asyncio.create_task(self.listen_for_results(callback, error_callback))
        return self._ws_listen_task

    async def stop_listening(self) -> None:
        """Stop listening for WebSocket messages.

        Cancels the background listening task created by start_listening().

        Example:
            ```python
            await client.stop_listening()
            ```
        """
        if self._ws_listen_task and not self._ws_listen_task.done():
            self._ws_listen_task.cancel()
            try:
                await self._ws_listen_task
            except asyncio.CancelledError:
                pass
            self._ws_listen_task = None
            logger.info("Stopped listening for WebSocket messages")

    # ==================== Port Management ====================

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use.

        Args:
            port: Port number to check

        Returns:
            True if port is in use, False otherwise
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return False
            except OSError:
                return True

    def _find_available_port(self, start_port: int, max_attempts: int = 100) -> Optional[int]:
        """Find an available port starting from the given port.

        Args:
            start_port: Starting port number
            max_attempts: Maximum number of ports to try

        Returns:
            Available port number, or None if no port found
        """
        for port in range(start_port, start_port + max_attempts):
            if not self._is_port_in_use(port):
                return port
        return None

    def _extract_port_from_url(self, url: str) -> int:
        """Extract port number from URL.

        Args:
            url: URL string (e.g., "http://localhost:5000")

        Returns:
            Port number (defaults to 80 for http, 443 for https if not specified)
        """
        parsed = urlparse(url)
        if parsed.port:
            return parsed.port
        return 443 if parsed.scheme == "https" else 80

    def _update_instance_url_port(self, new_port: int) -> None:
        """Update instance URL with new port.

        Args:
            new_port: New port number
        """
        parsed = urlparse(self.base_url)
        self.base_url = f"{parsed.scheme}://{parsed.hostname}:{new_port}{parsed.path}"

        # Update WebSocket URL as well
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        self.websocket_url = f"{ws_scheme}://{parsed.hostname}:{new_port}/ws"

    async def _wait_for_instance_ready(self, timeout: float = 30.0, check_interval: float = 1.0) -> bool:
        """Wait for instance to become ready.

        Args:
            timeout: Maximum time to wait in seconds
            check_interval: Interval between health checks in seconds

        Returns:
            True if instance is ready, False if timeout
        """
        print(f"Waiting for instance to become ready (timeout: {timeout}s)...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{self.base_url}/health")
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status") == "healthy":
                            elapsed = time.time() - start_time
                            print(f"✓ Instance is ready (took {elapsed:.1f}s)")
                            return True
            except Exception:
                pass

            await asyncio.sleep(check_interval)

        print(f"✗ Instance did not become ready within {timeout}s")
        return False

    # ==================== Auto-Start Methods ====================

    async def start_instance(
        self,
        auto_find_port: bool = True,
        max_port_search: int = 100,
        wait_timeout: float = 30.0,
        port: Optional[int] = None,
        instance_id: Optional[str] = None,
    ) -> bool:
        """Start instance service automatically with port auto-discovery.

        This method performs the following steps:
        1. Check if instance port is available
        2. If port is occupied and auto_find_port=True, find an available port
        3. Start instance process
        4. Wait for instance to become ready

        Args:
            auto_find_port: Automatically find available port if configured port is occupied
            max_port_search: Maximum number of ports to search for availability
            wait_timeout: Maximum time to wait for instance to become ready (seconds)
            port: Specific port to use (overrides URL port)
            instance_id: Instance identifier (defaults to "instance-default")

        Returns:
            True if instance started successfully, False otherwise

        Raises:
            RuntimeError: If instance module path is not found
            ConnectionError: If instance fails to start

        Example:
            ```python
            client = InstanceClient(base_url="http://localhost:5000")

            # Start instance with auto port discovery
            success = await client.start_instance(auto_find_port=True)
            if success:
                print(f"Instance running at {client.base_url}")
            ```
        """
        print("=" * 60)
        print("Starting Instance Service")
        print("=" * 60)

        # Step 1: Determine target port
        if port is not None:
            current_port = port
            self._update_instance_url_port(current_port)
        else:
            current_port = self._extract_port_from_url(self.base_url)

        print(f"Checking instance port {current_port}...")

        # Step 2: Check if port is available
        if self._is_port_in_use(current_port):
            # Port is in use, check if it's our instance
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(f"{self.base_url}/health")
                    if response.status_code == 200:
                        print(f"✓ Instance is already running at {self.base_url}")
                        return True
            except Exception:
                pass

            # Port is occupied by something else
            if auto_find_port:
                print(f"⚠ Port {current_port} is occupied, searching for available port...")
                available_port = self._find_available_port(current_port + 1, max_port_search)

                if available_port is None:
                    print(f"✗ Could not find available port in range {current_port + 1} to {current_port + max_port_search}")
                    raise ConnectionError(f"No available port found for instance service")

                print(f"✓ Found available port: {available_port}")
                self._update_instance_url_port(available_port)
                current_port = available_port
            else:
                print(f"✗ Port {current_port} is already in use (auto_find_port=False)")
                raise ConnectionError(f"Port {current_port} is already in use")

        # Step 3: Verify instance module exists
        if not os.path.exists(self.instance_module_path):
            raise RuntimeError(f"Instance module not found at: {self.instance_module_path}")

        # Step 4: Prepare environment variables
        env = os.environ.copy()
        env["INSTANCE_PORT"] = str(current_port)
        if instance_id:
            env["INSTANCE_ID"] = instance_id

        # Step 5: Start instance process
        print(f"Starting instance at port {current_port}...")
        print(f"  Module: {self.instance_module_path}")
        if instance_id:
            print(f"  Instance ID: {instance_id}")

        try:
            # Use uvicorn to run the instance
            cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "src.api:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(current_port),
                "--log-level",
                "info",
            ]

            self._instance_process = subprocess.Popen(
                cmd,
                cwd=self.instance_module_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            print(f"✓ Instance process started (PID: {self._instance_process.pid})")

        except Exception as e:
            print(f"✗ Failed to start instance process: {e}")
            raise ConnectionError(f"Failed to start instance: {e}")

        # Step 6: Wait for instance to become ready
        is_ready = await self._wait_for_instance_ready(timeout=wait_timeout)

        if not is_ready:
            print("✗ Instance failed to become ready")
            self.stop_instance()
            raise ConnectionError("Instance failed to become ready")

        print("=" * 60)
        print(f"✓ Instance is running at {self.base_url}")
        print("=" * 60)
        return True

    def stop_instance(self) -> bool:
        """Stop the instance process if it was started by this client.

        Returns:
            True if instance was stopped successfully, False otherwise
        """
        if self._instance_process is None:
            print("No instance process to stop")
            return False

        if not self.is_instance_running:
            print("Instance process is already stopped")
            self._instance_process = None
            return True

        print(f"Stopping instance process (PID: {self._instance_process.pid})...")

        try:
            self._instance_process.terminate()
            self._instance_process.wait(timeout=10.0)
            print("✓ Instance stopped successfully")
            self._instance_process = None
            return True
        except subprocess.TimeoutExpired:
            print("⚠ Instance did not stop gracefully, forcing termination...")
            self._instance_process.kill()
            self._instance_process.wait()
            print("✓ Instance killed")
            self._instance_process = None
            return True
        except Exception as e:
            print(f"✗ Error stopping instance: {e}")
            return False
