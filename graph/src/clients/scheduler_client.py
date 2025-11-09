"""
Asynchronous client for the Scheduler service.

Provides low-level API methods for interacting with the scheduler, including:
- Instance management (remove, drain, list, info)
  Note: Instance registration is handled automatically by instances when
  start_model() is called. Direct registration is not exposed.
- Task management (submit, list, info, clear)
- Strategy management (get, set)
- Health checking
- WebSocket integration for real-time task result notifications

Features:
- Connection pooling and keep-alive
- Automatic retries with exponential backoff
- Configurable timeouts
- Comprehensive error handling with structured exceptions
- Full type hints

Example usage:
    ```python
    async with SchedulerClient("http://localhost:8000") as client:
        # Submit a task
        await client.submit_task(
            task_id="task-001",
            model_id="gpt-4",
            task_input={"prompt": "Hello"},
            metadata={"user_id": "123"}
        )

        # List instances
        instances = await client.list_instances()
        print(f"Active instances: {instances['count']}")

        # Subscribe to task results via WebSocket
        async with client.subscribe_to_tasks(["task-001"]) as ws:
            result = await ws.receive_result()
            print(result)
    ```
"""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from urllib.parse import urljoin

import httpx
import websockets
from websockets.asyncio.client import ClientConnection

from .exceptions import (
    SchedulerClientError,
    SchedulerConnectionError,
    SchedulerNotFoundError,
    SchedulerServiceError,
    SchedulerTimeoutError,
    SchedulerValidationError,
    SchedulerWebSocketError,
)


class WebSocketResultStream:
    """Wrapper for WebSocket to receive task results."""

    def __init__(self, websocket: ClientConnection):
        """Initialize the WebSocket result stream.

        Args:
            websocket: Connected WebSocket client protocol
        """
        self._websocket = websocket

    async def receive_result(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Receive a single task result from the WebSocket.

        Args:
            timeout: Maximum time to wait for a result in seconds (None = no timeout)

        Returns:
            Dictionary containing the task result with keys:
                - type: Message type ("result", "ack", "error")
                - task_id: Task ID (for "result" type)
                - status: Task status ("completed" or "failed")
                - result: Task result dictionary (if completed)
                - error: Error message (if failed)
                - timestamps: Submission, start, and completion timestamps
                - execution_time_ms: Task execution time in milliseconds

        Raises:
            SchedulerTimeoutError: If timeout is exceeded
            SchedulerWebSocketError: If connection is closed or message is invalid
        """
        try:
            if timeout:
                message = await asyncio.wait_for(self._websocket.recv(), timeout=timeout)
            else:
                message = await self._websocket.recv()

            return json.loads(message)
        except asyncio.TimeoutError:
            raise SchedulerTimeoutError("WebSocket receive timeout", timeout_seconds=timeout or 0.0)
        except websockets.exceptions.ConnectionClosed as e:
            raise SchedulerWebSocketError(f"WebSocket connection closed: {e}")
        except json.JSONDecodeError as e:
            raise SchedulerWebSocketError(f"Invalid JSON received from WebSocket: {e}")

    async def receive_results(self, timeout: Optional[float] = None) -> AsyncIterator[Dict[str, Any]]:
        """Asynchronously iterate over task results from the WebSocket.

        Args:
            timeout: Maximum time to wait for each result in seconds (None = no timeout)

        Yields:
            Task result dictionaries

        Raises:
            SchedulerTimeoutError: If timeout is exceeded
            SchedulerWebSocketError: If connection is closed or message is invalid
        """
        while True:
            yield await self.receive_result(timeout=timeout)

    async def subscribe(self, task_ids: List[str]) -> None:
        """Subscribe to additional task IDs.

        Args:
            task_ids: List of task IDs to subscribe to

        Raises:
            SchedulerWebSocketError: If subscription fails
        """
        message = {"type": "subscribe", "task_ids": task_ids}
        try:
            await self._websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed as e:
            raise SchedulerWebSocketError(f"Cannot subscribe - WebSocket connection closed: {e}")

    async def unsubscribe(self, task_ids: List[str]) -> None:
        """Unsubscribe from task IDs.

        Args:
            task_ids: List of task IDs to unsubscribe from

        Raises:
            SchedulerWebSocketError: If unsubscription fails
        """
        message = {"type": "unsubscribe", "task_ids": task_ids}
        try:
            await self._websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed as e:
            raise SchedulerWebSocketError(f"Cannot unsubscribe - WebSocket connection closed: {e}")

    async def close(self) -> None:
        """Close the WebSocket connection gracefully."""
        await self._websocket.close()


class SchedulerClient:
    """Asynchronous client for the Scheduler service.

    Provides methods for all scheduler REST and WebSocket APIs with automatic
    retry, connection pooling, and comprehensive error handling.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff_base: float = 0.5,
        max_connections: int = 100,
    ):
        """Initialize the scheduler client.

        Args:
            base_url: Base URL of the scheduler service (e.g., "http://localhost:8000")
            timeout: Default timeout for REST API requests in seconds
            max_retries: Maximum number of retries for transient failures
            retry_backoff_base: Base delay for exponential backoff (in seconds)
            max_connections: Maximum number of concurrent HTTP connections
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base

        # HTTP client will be initialized in __aenter__
        self._client: Optional[httpx.AsyncClient] = None
        self._client_limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=20)

    async def __aenter__(self) -> "SchedulerClient":
        """Enter the async context manager - initialize HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            limits=self._client_limits,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager - cleanup HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized.

        Returns:
            The HTTP client instance

        Raises:
            RuntimeError: If client is not initialized (use async with SchedulerClient(...))
        """
        if self._client is None:
            raise RuntimeError("SchedulerClient must be used as an async context manager (async with)")
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        """Make an HTTP request with automatic retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., "/instance/register")
            json_data: JSON request body (for POST requests)
            params: Query parameters (for GET requests)
            retry_count: Current retry attempt number (used internally)

        Returns:
            Parsed JSON response

        Raises:
            SchedulerNotFoundError: Resource not found (404)
            SchedulerValidationError: Invalid request (400, 422)
            SchedulerServiceError: Service error (500, 502, 503)
            SchedulerConnectionError: Connection failed
            SchedulerTimeoutError: Request timeout
        """
        client = self._ensure_client()

        try:
            response = await client.request(method, path, json=json_data, params=params)

            # Handle error responses
            if response.status_code == 404:
                raise SchedulerNotFoundError(
                    "Resource not found", status_code=404, response_body=response.text
                )
            elif response.status_code in (400, 422):
                raise SchedulerValidationError(
                    "Request validation failed", status_code=response.status_code, response_body=response.text
                )
            elif response.status_code >= 500:
                # Retry on server errors if retries are available
                if retry_count < self.max_retries:
                    await asyncio.sleep(self.retry_backoff_base * (2**retry_count))
                    return await self._request(method, path, json_data, params, retry_count + 1)
                raise SchedulerServiceError(
                    "Scheduler service error", status_code=response.status_code, response_body=response.text
                )

            # Parse successful response
            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException as e:
            # Retry on timeout if retries are available
            if retry_count < self.max_retries:
                await asyncio.sleep(self.retry_backoff_base * (2**retry_count))
                return await self._request(method, path, json_data, params, retry_count + 1)
            raise SchedulerTimeoutError(f"Request timeout: {e}", timeout_seconds=self.timeout)

        except httpx.ConnectError as e:
            # Retry on connection errors if retries are available
            if retry_count < self.max_retries:
                await asyncio.sleep(self.retry_backoff_base * (2**retry_count))
                return await self._request(method, path, json_data, params, retry_count + 1)
            raise SchedulerConnectionError(f"Failed to connect to scheduler: {e}", original_exception=e)

        except httpx.HTTPError as e:
            # Other HTTP errors
            raise SchedulerClientError(f"HTTP error: {e}")

    # ==================== Instance Management API ====================
    # Note: Instance registration is handled automatically by the instance
    # when start_model() is called. Direct registration via SchedulerClient
    # is not supported.

    async def remove_instance(self, instance_id: str) -> Dict[str, Any]:
        """Remove an instance from the scheduler.

        The instance must be in DRAINING state with no pending tasks before removal.

        Args:
            instance_id: ID of the instance to remove

        Returns:
            Response dictionary with keys:
                - success: Boolean indicating success
                - message: Status message
                - instance_id: ID of the removed instance

        Raises:
            SchedulerNotFoundError: If instance does not exist
            SchedulerValidationError: If instance is not in DRAINING state or has pending tasks
        """
        return await self._request("POST", "/instance/remove", json_data={"instance_id": instance_id})

    async def drain_instance(self, instance_id: str) -> Dict[str, Any]:
        """Start draining an instance (stop assigning new tasks).

        Args:
            instance_id: ID of the instance to drain

        Returns:
            Response dictionary with keys:
                - success: Boolean indicating success
                - message: Status message
                - instance_id: ID of the instance
                - status: Current status (should be "draining")
                - pending_tasks: Number of pending tasks
                - running_tasks: Number of running tasks
                - estimated_completion_time_ms: Estimated time until all tasks complete

        Raises:
            SchedulerNotFoundError: If instance does not exist
            SchedulerValidationError: If instance is already draining or removed
        """
        return await self._request("POST", "/instance/drain", json_data={"instance_id": instance_id})

    async def get_drain_status(self, instance_id: str) -> Dict[str, Any]:
        """Check the draining status of an instance.

        Args:
            instance_id: ID of the instance to check

        Returns:
            Response dictionary with keys:
                - success: Boolean indicating success
                - instance_id: ID of the instance
                - status: Current status ("active", "draining", "removed")
                - pending_tasks: Number of pending tasks
                - running_tasks: Number of running tasks
                - can_remove: Boolean indicating if instance can be safely removed
                - drain_initiated_at: ISO 8601 timestamp when draining started (or null)

        Raises:
            SchedulerNotFoundError: If instance does not exist
        """
        return await self._request("GET", "/instance/drain/status", params={"instance_id": instance_id})

    async def list_instances(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """List all registered instances, optionally filtered by model ID.

        Args:
            model_id: Optional model ID to filter by

        Returns:
            Response dictionary with keys:
                - success: Boolean indicating success
                - count: Number of instances returned
                - instances: List of instance objects, each containing:
                    - instance_id: Instance identifier
                    - model_id: Model ID
                    - endpoint: Instance endpoint
                    - status: Instance status
                    - platform_info: Platform information
        """
        params = {"model_id": model_id} if model_id else None
        return await self._request("GET", "/instance/list", params=params)

    async def get_instance_info(self, instance_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific instance.

        Args:
            instance_id: ID of the instance

        Returns:
            Response dictionary with keys:
                - success: Boolean indicating success
                - instance: Instance object with full details
                - queue_info: Queue state (format depends on scheduling strategy)
                - stats: Statistics with pending_tasks, completed_tasks, failed_tasks

        Raises:
            SchedulerNotFoundError: If instance does not exist
        """
        return await self._request("GET", "/instance/info", params={"instance_id": instance_id})

    # ==================== Task Management API ====================

    async def submit_task(
        self, task_id: str, model_id: str, task_input: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Submit a new task to the scheduler.

        Note: In v2.0+, this API returns immediately (~5ms) with the task in PENDING status
        and assigned_instance empty. The background scheduler assigns an instance within
        50-100ms. Use WebSocket subscriptions or polling to detect when the task is assigned.

        Args:
            task_id: Unique identifier for the task
            model_id: Model ID to use for this task
            task_input: Task input data (model-specific format)
            metadata: Optional metadata dictionary

        Returns:
            Response dictionary with keys:
                - success: Boolean indicating success
                - message: Status message
                - task: Task object with:
                    - task_id: Task identifier
                    - model_id: Model ID
                    - status: Task status ("pending")
                    - assigned_instance: Instance ID (empty initially, assigned by background scheduler)
                    - submitted_at: ISO 8601 timestamp

        Raises:
            SchedulerValidationError: If task_id already exists or validation fails
            SchedulerNotFoundError: If no instances are available for the model_id
        """
        return await self._request(
            "POST",
            "/task/submit",
            json_data={"task_id": task_id, "model_id": model_id, "task_input": task_input, "metadata": metadata or {}},
        )

    async def list_tasks(
        self,
        status: Optional[str] = None,
        model_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List tasks with optional filtering and pagination.

        Args:
            status: Filter by task status ("pending", "running", "completed", "failed")
            model_id: Filter by model ID
            instance_id: Filter by instance ID
            limit: Maximum number of tasks to return (1-1000, default 100)
            offset: Number of tasks to skip (for pagination, default 0)

        Returns:
            Response dictionary with keys:
                - success: Boolean indicating success
                - count: Number of tasks in this page
                - total: Total number of tasks matching the filter
                - offset: Offset value used
                - limit: Limit value used
                - tasks: List of task summary objects
        """
        params = {"limit": min(limit, 1000), "offset": max(offset, 0)}
        if status:
            params["status"] = status
        if model_id:
            params["model_id"] = model_id
        if instance_id:
            params["instance_id"] = instance_id

        return await self._request("GET", "/task/list", params=params)

    async def get_task_info(self, task_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific task.

        Args:
            task_id: ID of the task

        Returns:
            Response dictionary with keys:
                - success: Boolean indicating success
                - task: Task object with full details including:
                    - task_id, model_id, status, assigned_instance
                    - task_input: Original input data
                    - metadata: Task metadata
                    - result: Task result (if completed)
                    - error: Error message (if failed)
                    - timestamps: submitted_at, started_at, completed_at
                    - execution_time_ms: Execution time in milliseconds

        Raises:
            SchedulerNotFoundError: If task does not exist
        """
        return await self._request("GET", "/task/info", params={"task_id": task_id})

    async def clear_tasks(self) -> Dict[str, Any]:
        """Clear all tasks from the scheduler and all instances.

        This calls /task/clear on all registered instances.

        Returns:
            Response dictionary with keys:
                - success: Boolean indicating success
                - message: Status message
                - cleared_count: Number of tasks cleared
        """
        return await self._request("POST", "/task/clear")

    # ==================== Strategy Management API ====================

    async def get_strategy(self) -> Dict[str, Any]:
        """Get the current scheduling strategy and parameters.

        Returns:
            Response dictionary with keys:
                - success: Boolean indicating success
                - strategy_info: Dictionary with:
                    - strategy_name: "round_robin", "min_time", or "probabilistic"
                    - parameters: Strategy-specific parameters (e.g., target_quantile for probabilistic)
        """
        return await self._request("GET", "/strategy/get")

    async def set_strategy(self, strategy_name: str, target_quantile: Optional[float] = None) -> Dict[str, Any]:
        """Switch to a different scheduling strategy.

        IMPORTANT: This clears all tasks and reinitializes all instances. Cannot be called
        while tasks are running (returns 400 error if any tasks are in progress).

        Args:
            strategy_name: Strategy to use ("round_robin", "min_time", "probabilistic")
            target_quantile: For "probabilistic" strategy, target quantile (0.0-1.0, default 0.9)

        Returns:
            Response dictionary with keys:
                - success: Boolean indicating success
                - message: Status message
                - cleared_tasks: Number of tasks cleared
                - reinitialized_instances: Number of instances reinitialized
                - strategy_info: New strategy configuration

        Raises:
            SchedulerValidationError: If tasks are currently running or invalid strategy name
            SchedulerServiceError: If strategy switch fails
        """
        json_data = {"strategy_name": strategy_name}
        if target_quantile is not None:
            json_data["target_quantile"] = target_quantile

        return await self._request("POST", "/strategy/set", json_data=json_data)

    # ==================== Health Check API ====================

    async def health_check(self) -> Dict[str, Any]:
        """Check the health status of the scheduler service.

        Returns:
            Response dictionary with keys:
                - success: Boolean indicating success
                - status: "healthy" or "unhealthy"
                - timestamp: ISO 8601 timestamp
                - version: Service version string
                - stats: Statistics with:
                    - total_instances, active_instances
                    - total_tasks, pending_tasks, running_tasks, completed_tasks, failed_tasks

        Raises:
            SchedulerServiceError: If service is unhealthy (503 status)
        """
        return await self._request("GET", "/health")

    # ==================== WebSocket API ====================

    @asynccontextmanager
    async def subscribe_to_tasks(
        self, task_ids: List[str], ws_timeout: float = 300.0
    ) -> AsyncIterator[WebSocketResultStream]:
        """Subscribe to task result notifications via WebSocket.

        This is a context manager that establishes a WebSocket connection, subscribes to
        the specified task IDs, and yields a stream for receiving results.

        Args:
            task_ids: List of task IDs to subscribe to
            ws_timeout: WebSocket connection timeout in seconds (default 300s)

        Yields:
            WebSocketResultStream for receiving task results

        Raises:
            SchedulerConnectionError: If WebSocket connection fails
            SchedulerWebSocketError: If subscription fails

        Example:
            ```python
            async with client.subscribe_to_tasks(["task-1", "task-2"]) as ws:
                async for result in ws.receive_results():
                    print(f"Task {result['task_id']} status: {result['status']}")
                    if result['status'] == 'completed':
                        print(f"Result: {result['result']}")
            ```
        """
        # Convert HTTP URL to WebSocket URL
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = urljoin(ws_url, "/task/get_result")

        try:
            async with websockets.connect(ws_url, open_timeout=ws_timeout) as websocket:
                # Send subscription message
                subscribe_msg = {"type": "subscribe", "task_ids": task_ids}
                await websocket.send(json.dumps(subscribe_msg))

                # Wait for acknowledgment
                ack_message = await websocket.recv()
                ack = json.loads(ack_message)
                if ack.get("type") != "ack":
                    raise SchedulerWebSocketError(f"Expected ack, got: {ack}")

                # Yield the stream
                stream = WebSocketResultStream(websocket)
                yield stream

        except SchedulerWebSocketError:
            # Re-raise our own WebSocket errors
            raise
        except websockets.exceptions.WebSocketException as e:
            raise SchedulerConnectionError(f"WebSocket connection failed: {e}", original_exception=e)
        except json.JSONDecodeError as e:
            raise SchedulerWebSocketError(f"Invalid JSON in WebSocket communication: {e}")
        except (OSError, Exception) as e:
            # Catch connection errors and other exceptions
            raise SchedulerConnectionError(f"WebSocket connection failed: {e}", original_exception=e)
