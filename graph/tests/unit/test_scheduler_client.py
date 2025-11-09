"""
Unit tests for SchedulerClient.

Tests cover all API methods, error handling, WebSocket functionality, and retry logic.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
import websockets

from src.clients.exceptions import (
    SchedulerClientError,
    SchedulerConnectionError,
    SchedulerNotFoundError,
    SchedulerServiceError,
    SchedulerTimeoutError,
    SchedulerValidationError,
    SchedulerWebSocketError,
)
from src.clients.scheduler_client import SchedulerClient, WebSocketResultStream


class TestSchedulerClientInit:
    """Test suite for SchedulerClient initialization."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        client = SchedulerClient(base_url="http://localhost:8000")

        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 30.0
        assert client.max_retries == 3
        assert client.retry_backoff_base == 0.5
        assert client._client is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        client = SchedulerClient(
            base_url="https://scheduler.example.com/",
            timeout=60.0,
            max_retries=5,
            retry_backoff_base=1.0,
            max_connections=200,
        )

        assert client.base_url == "https://scheduler.example.com"
        assert client.timeout == 60.0
        assert client.max_retries == 5
        assert client.retry_backoff_base == 1.0

    def test_init_strips_trailing_slash(self):
        """Test base_url trailing slash is stripped."""
        client = SchedulerClient(base_url="http://localhost:8000/")

        assert client.base_url == "http://localhost:8000"


class TestSchedulerClientContextManager:
    """Test suite for async context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager_entry(self):
        """Test entering async context manager initializes client."""
        client = SchedulerClient(base_url="http://localhost:8000")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value = mock_client

            async with client:
                assert client._client is not None

    @pytest.mark.asyncio
    async def test_context_manager_exit(self):
        """Test exiting async context manager closes client."""
        client = SchedulerClient(base_url="http://localhost:8000")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value = mock_client

            async with client:
                assert client._client is not None

            assert client._client is None

    @pytest.mark.asyncio
    async def test_ensure_client_without_context(self):
        """Test _ensure_client raises error without context manager."""
        client = SchedulerClient(base_url="http://localhost:8000")

        with pytest.raises(RuntimeError, match="must be used as an async context manager"):
            client._ensure_client()


class TestSchedulerClientRequestHandling:
    """Test suite for HTTP request handling and error management."""

    @pytest.mark.asyncio
    async def test_request_success(self):
        """Test successful HTTP request."""
        client = SchedulerClient(base_url="http://localhost:8000")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value = mock_client

            async with client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"success": True}

                with patch.object(client._client, "request", return_value=mock_response):
                    result = await client._request("GET", "/health")

                    assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_request_404_error(self):
        """Test request handling for 404 errors."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.text = "Not found"

            with patch.object(client._client, "request", return_value=mock_response):
                with pytest.raises(SchedulerNotFoundError, match="Resource not found"):
                    await client._request("GET", "/task/info", params={"task_id": "unknown"})

    @pytest.mark.asyncio
    async def test_request_400_error(self):
        """Test request handling for 400 validation errors."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Validation failed"

            with patch.object(client._client, "request", return_value=mock_response):
                with pytest.raises(SchedulerValidationError, match="Request validation failed"):
                    await client._request("POST", "/task/submit", json_data={})

    @pytest.mark.asyncio
    async def test_request_422_error(self):
        """Test request handling for 422 validation errors."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            mock_response = MagicMock()
            mock_response.status_code = 422
            mock_response.text = "Unprocessable entity"

            with patch.object(client._client, "request", return_value=mock_response):
                with pytest.raises(SchedulerValidationError):
                    await client._request("POST", "/instance/register", json_data={})

    @pytest.mark.asyncio
    async def test_request_500_error_with_retry(self):
        """Test request retries on 500 errors."""
        async with SchedulerClient(base_url="http://localhost:8000", max_retries=2) as client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal server error"

            with patch.object(client._client, "request", return_value=mock_response):
                with pytest.raises(SchedulerServiceError):
                    await client._request("GET", "/health")

    @pytest.mark.asyncio
    async def test_request_timeout_with_retry(self):
        """Test request retries on timeout."""
        async with SchedulerClient(base_url="http://localhost:8000", max_retries=2, retry_backoff_base=0.01) as client:
            with patch.object(client._client, "request", side_effect=httpx.TimeoutException("Timeout")):
                with pytest.raises(SchedulerTimeoutError):
                    await client._request("GET", "/health")

    @pytest.mark.asyncio
    async def test_request_connection_error_with_retry(self):
        """Test request retries on connection errors."""
        async with SchedulerClient(base_url="http://localhost:8000", max_retries=2, retry_backoff_base=0.01) as client:
            with patch.object(client._client, "request", side_effect=httpx.ConnectError("Connection failed")):
                with pytest.raises(SchedulerConnectionError):
                    await client._request("GET", "/health")

    @pytest.mark.asyncio
    async def test_request_other_http_error(self):
        """Test request handling for other HTTP errors."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            with patch.object(client._client, "request", side_effect=httpx.HTTPError("HTTP error")):
                with pytest.raises(SchedulerClientError, match="HTTP error"):
                    await client._request("GET", "/health")


class TestSchedulerClientInstanceManagement:
    """Test suite for instance management API methods.

    Note: Instance registration is handled automatically by instances when
    start_model() is called. Direct registration is not exposed in SchedulerClient.
    """

    @pytest.mark.asyncio
    async def test_remove_instance_success(self):
        """Test successful instance removal."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            with patch.object(client, "_request", return_value={"success": True}) as mock_request:
                result = await client.remove_instance("instance-1")

                assert result["success"] is True
                mock_request.assert_called_once_with("POST", "/instance/remove", json_data={"instance_id": "instance-1"})

    @pytest.mark.asyncio
    async def test_drain_instance_success(self):
        """Test successful instance draining."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            with patch.object(
                client,
                "_request",
                return_value={
                    "success": True,
                    "instance_id": "instance-1",
                    "status": "draining",
                    "pending_tasks": 5,
                    "running_tasks": 1,
                },
            ) as mock_request:
                result = await client.drain_instance("instance-1")

                assert result["status"] == "draining"
                assert result["pending_tasks"] == 5

    @pytest.mark.asyncio
    async def test_get_drain_status_success(self):
        """Test get drain status."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            with patch.object(
                client,
                "_request",
                return_value={
                    "success": True,
                    "instance_id": "instance-1",
                    "status": "draining",
                    "can_remove": False,
                    "pending_tasks": 3,
                },
            ) as mock_request:
                result = await client.get_drain_status("instance-1")

                assert result["status"] == "draining"
                assert result["can_remove"] is False
                mock_request.assert_called_once_with("GET", "/instance/drain/status", params={"instance_id": "instance-1"})

    @pytest.mark.asyncio
    async def test_list_instances_all(self):
        """Test listing all instances."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            with patch.object(
                client, "_request", return_value={"success": True, "count": 2, "instances": []}
            ) as mock_request:
                result = await client.list_instances()

                assert result["count"] == 2
                mock_request.assert_called_once_with("GET", "/instance/list", params=None)

    @pytest.mark.asyncio
    async def test_list_instances_by_model(self):
        """Test listing instances filtered by model ID."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            with patch.object(client, "_request", return_value={"success": True, "count": 1, "instances": []}) as mock_request:
                result = await client.list_instances(model_id="gpt-4")

                assert result["count"] == 1
                mock_request.assert_called_once_with("GET", "/instance/list", params={"model_id": "gpt-4"})

    @pytest.mark.asyncio
    async def test_get_instance_info_success(self):
        """Test getting instance information."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            with patch.object(
                client, "_request", return_value={"success": True, "instance": {}, "stats": {}}
            ) as mock_request:
                result = await client.get_instance_info("instance-1")

                assert result["success"] is True
                mock_request.assert_called_once_with("GET", "/instance/info", params={"instance_id": "instance-1"})


class TestSchedulerClientTaskManagement:
    """Test suite for task management API methods."""

    @pytest.mark.asyncio
    async def test_submit_task_success(self):
        """Test successful task submission."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            with patch.object(
                client, "_request", return_value={"success": True, "task": {"task_id": "task-1", "status": "pending"}}
            ) as mock_request:
                result = await client.submit_task(
                    task_id="task-1", model_id="gpt-4", task_input={"prompt": "Hello"}, metadata={"user_id": "123"}
                )

                assert result["task"]["task_id"] == "task-1"
                assert result["task"]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_submit_task_no_metadata(self):
        """Test task submission without metadata."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            with patch.object(client, "_request", return_value={"success": True, "task": {}}) as mock_request:
                await client.submit_task(task_id="task-1", model_id="gpt-4", task_input={"prompt": "Hello"})

                call_args = mock_request.call_args
                assert call_args.kwargs["json_data"]["metadata"] == {}

    @pytest.mark.asyncio
    async def test_list_tasks_all(self):
        """Test listing all tasks."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            with patch.object(
                client, "_request", return_value={"success": True, "count": 10, "total": 10, "tasks": []}
            ) as mock_request:
                result = await client.list_tasks()

                assert result["count"] == 10
                mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_tasks_with_filters(self):
        """Test listing tasks with filters."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            with patch.object(client, "_request", return_value={"success": True, "tasks": []}) as mock_request:
                await client.list_tasks(status="pending", model_id="gpt-4", instance_id="instance-1", limit=50, offset=10)

                call_args = mock_request.call_args
                params = call_args.kwargs["params"]
                assert params["status"] == "pending"
                assert params["model_id"] == "gpt-4"
                assert params["instance_id"] == "instance-1"
                assert params["limit"] == 50
                assert params["offset"] == 10

    @pytest.mark.asyncio
    async def test_list_tasks_limit_clamping(self):
        """Test list_tasks clamps limit to maximum 1000."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            with patch.object(client, "_request", return_value={"success": True, "tasks": []}) as mock_request:
                await client.list_tasks(limit=2000)

                call_args = mock_request.call_args
                assert call_args.kwargs["params"]["limit"] == 1000

    @pytest.mark.asyncio
    async def test_list_tasks_negative_offset(self):
        """Test list_tasks handles negative offset."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            with patch.object(client, "_request", return_value={"success": True, "tasks": []}) as mock_request:
                await client.list_tasks(offset=-10)

                call_args = mock_request.call_args
                assert call_args.kwargs["params"]["offset"] == 0

    @pytest.mark.asyncio
    async def test_get_task_info_success(self):
        """Test getting task information."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            with patch.object(client, "_request", return_value={"success": True, "task": {"task_id": "task-1"}}) as mock_request:
                result = await client.get_task_info("task-1")

                assert result["task"]["task_id"] == "task-1"
                mock_request.assert_called_once_with("GET", "/task/info", params={"task_id": "task-1"})

    @pytest.mark.asyncio
    async def test_clear_tasks_success(self):
        """Test clearing all tasks."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            with patch.object(
                client, "_request", return_value={"success": True, "cleared_count": 15}
            ) as mock_request:
                result = await client.clear_tasks()

                assert result["cleared_count"] == 15
                mock_request.assert_called_once_with("POST", "/task/clear")


class TestSchedulerClientStrategyManagement:
    """Test suite for strategy management API methods."""

    @pytest.mark.asyncio
    async def test_get_strategy_success(self):
        """Test getting current strategy."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            with patch.object(
                client,
                "_request",
                return_value={"success": True, "strategy_info": {"strategy_name": "min_time", "parameters": {}}},
            ) as mock_request:
                result = await client.get_strategy()

                assert result["strategy_info"]["strategy_name"] == "min_time"
                mock_request.assert_called_once_with("GET", "/strategy/get")

    @pytest.mark.asyncio
    async def test_set_strategy_round_robin(self):
        """Test setting round robin strategy."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            with patch.object(client, "_request", return_value={"success": True}) as mock_request:
                result = await client.set_strategy("round_robin")

                assert result["success"] is True
                call_args = mock_request.call_args
                assert call_args.kwargs["json_data"]["strategy_name"] == "round_robin"

    @pytest.mark.asyncio
    async def test_set_strategy_probabilistic_with_quantile(self):
        """Test setting probabilistic strategy with quantile."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            with patch.object(client, "_request", return_value={"success": True}) as mock_request:
                await client.set_strategy("probabilistic", target_quantile=0.95)

                call_args = mock_request.call_args
                assert call_args.kwargs["json_data"]["strategy_name"] == "probabilistic"
                assert call_args.kwargs["json_data"]["target_quantile"] == 0.95


class TestSchedulerClientHealthCheck:
    """Test suite for health check API."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check when service is healthy."""
        async with SchedulerClient(base_url="http://localhost:8000") as client:
            with patch.object(
                client,
                "_request",
                return_value={
                    "success": True,
                    "status": "healthy",
                    "version": "1.0.0",
                    "stats": {"total_instances": 5, "total_tasks": 100},
                },
            ) as mock_request:
                result = await client.health_check()

                assert result["status"] == "healthy"
                assert result["stats"]["total_instances"] == 5


class TestWebSocketResultStream:
    """Test suite for WebSocketResultStream."""

    @pytest.mark.asyncio
    async def test_receive_result_success(self):
        """Test receiving a result from WebSocket."""
        mock_ws = AsyncMock()
        mock_ws.recv.return_value = json.dumps({"type": "result", "task_id": "task-1", "status": "completed"})

        stream = WebSocketResultStream(mock_ws)
        result = await stream.receive_result()

        assert result["type"] == "result"
        assert result["task_id"] == "task-1"

    @pytest.mark.asyncio
    async def test_receive_result_timeout(self):
        """Test receive_result with timeout."""
        mock_ws = AsyncMock()
        mock_ws.recv.side_effect = asyncio.TimeoutError()

        stream = WebSocketResultStream(mock_ws)

        with pytest.raises(SchedulerTimeoutError):
            await stream.receive_result(timeout=5.0)

    @pytest.mark.asyncio
    async def test_receive_result_connection_closed(self):
        """Test receive_result when connection is closed."""
        mock_ws = AsyncMock()
        mock_ws.recv.side_effect = websockets.exceptions.ConnectionClosed(rcvd=None, sent=None)

        stream = WebSocketResultStream(mock_ws)

        with pytest.raises(SchedulerWebSocketError, match="WebSocket connection closed"):
            await stream.receive_result()

    @pytest.mark.asyncio
    async def test_receive_result_invalid_json(self):
        """Test receive_result with invalid JSON."""
        mock_ws = AsyncMock()
        mock_ws.recv.return_value = "invalid json"

        stream = WebSocketResultStream(mock_ws)

        with pytest.raises(SchedulerWebSocketError, match="Invalid JSON"):
            await stream.receive_result()

    @pytest.mark.asyncio
    async def test_subscribe_success(self):
        """Test subscribing to additional task IDs."""
        mock_ws = AsyncMock()
        stream = WebSocketResultStream(mock_ws)

        await stream.subscribe(["task-2", "task-3"])

        mock_ws.send.assert_called_once()
        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "subscribe"
        assert sent_message["task_ids"] == ["task-2", "task-3"]

    @pytest.mark.asyncio
    async def test_subscribe_connection_closed(self):
        """Test subscribe when connection is closed."""
        mock_ws = AsyncMock()
        mock_ws.send.side_effect = websockets.exceptions.ConnectionClosed(rcvd=None, sent=None)

        stream = WebSocketResultStream(mock_ws)

        with pytest.raises(SchedulerWebSocketError, match="Cannot subscribe"):
            await stream.subscribe(["task-2"])

    @pytest.mark.asyncio
    async def test_unsubscribe_success(self):
        """Test unsubscribing from task IDs."""
        mock_ws = AsyncMock()
        stream = WebSocketResultStream(mock_ws)

        await stream.unsubscribe(["task-1"])

        mock_ws.send.assert_called_once()
        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "unsubscribe"
        assert sent_message["task_ids"] == ["task-1"]

    @pytest.mark.asyncio
    async def test_unsubscribe_connection_closed(self):
        """Test unsubscribe when connection is closed."""
        mock_ws = AsyncMock()
        mock_ws.send.side_effect = websockets.exceptions.ConnectionClosed(rcvd=None, sent=None)

        stream = WebSocketResultStream(mock_ws)

        with pytest.raises(SchedulerWebSocketError, match="Cannot unsubscribe"):
            await stream.unsubscribe(["task-1"])

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing WebSocket stream."""
        mock_ws = AsyncMock()
        stream = WebSocketResultStream(mock_ws)

        await stream.close()

        mock_ws.close.assert_called_once()


class TestSchedulerClientWebSocket:
    """Test suite for WebSocket subscription functionality."""

    @pytest.mark.asyncio
    async def test_subscribe_to_tasks_success(self):
        """Test successful WebSocket subscription."""
        client = SchedulerClient(base_url="http://localhost:8000")

        mock_ws = AsyncMock()
        mock_ws.recv.return_value = json.dumps({"type": "ack", "message": "Subscribed"})
        mock_ws.__aenter__.return_value = mock_ws
        mock_ws.__aexit__.return_value = None

        with patch("websockets.connect", return_value=mock_ws):
            async with client.subscribe_to_tasks(["task-1", "task-2"]) as stream:
                assert isinstance(stream, WebSocketResultStream)
                mock_ws.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_subscribe_to_tasks_connection_error(self):
        """Test WebSocket subscription with connection error."""
        client = SchedulerClient(base_url="http://localhost:8000")

        def failing_connect(*args, **kwargs):
            raise OSError("Connection refused")

        with patch("websockets.connect", side_effect=failing_connect):
            with pytest.raises(SchedulerConnectionError, match="WebSocket connection failed"):
                async with client.subscribe_to_tasks(["task-1"]):
                    pass

    @pytest.mark.asyncio
    async def test_subscribe_to_tasks_ack_not_received(self):
        """Test WebSocket subscription when ack is not received."""
        client = SchedulerClient(base_url="http://localhost:8000")

        mock_ws = AsyncMock()
        mock_ws.recv.return_value = json.dumps({"type": "error", "message": "Subscription failed"})
        mock_ws.__aenter__.return_value = mock_ws
        mock_ws.__aexit__.return_value = None

        with patch("websockets.connect", return_value=mock_ws):
            with pytest.raises(SchedulerWebSocketError, match="Expected ack"):
                async with client.subscribe_to_tasks(["task-1"]):
                    pass

    @pytest.mark.asyncio
    async def test_subscribe_to_tasks_invalid_json_ack(self):
        """Test WebSocket subscription with invalid JSON in ack."""
        client = SchedulerClient(base_url="http://localhost:8000")

        mock_ws = AsyncMock()
        mock_ws.recv.return_value = "invalid json"
        mock_ws.__aenter__.return_value = mock_ws
        mock_ws.__aexit__.return_value = None

        with patch("websockets.connect", return_value=mock_ws):
            with pytest.raises(SchedulerWebSocketError, match="Invalid JSON"):
                async with client.subscribe_to_tasks(["task-1"]):
                    pass
