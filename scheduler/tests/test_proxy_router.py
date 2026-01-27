"""Unit tests for the catch-all proxy router.

TDD tests for the transparent proxy that forwards requests
to backend instances via WorkerQueueManager.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.algorithms.base import ScheduleResult
from src.clients.predictor_client import Prediction
from src.proxy.router import ProxyRouter
from src.services.task_result_callback import TaskResultCallback
from src.services.worker_queue_thread import TaskResult


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_scheduling_strategy():
    """Create a mock scheduling strategy."""
    strategy = MagicMock()
    strategy.schedule_task = AsyncMock(
        return_value=ScheduleResult(
            selected_instance_id="instance-1",
            selected_prediction=Prediction(
                instance_id="instance-1",
                predicted_time_ms=100.0,
                confidence=0.9,
                quantiles={0.5: 80.0, 0.9: 120.0},
            ),
        )
    )
    return strategy


@pytest.fixture
def mock_instance_registry():
    """Create a mock instance registry with one active instance."""
    registry = MagicMock()
    instance = MagicMock()
    instance.instance_id = "instance-1"
    instance.model_id = "test-model"
    instance.endpoint = "http://localhost:8001"
    registry.list_active = AsyncMock(return_value=[instance])
    return registry


@pytest.fixture
def mock_instance_registry_empty():
    """Create a mock instance registry with no instances."""
    registry = MagicMock()
    registry.list_active = AsyncMock(return_value=[])
    return registry


@pytest.fixture
def mock_task_registry():
    """Create a mock task registry."""
    registry = MagicMock()
    registry.create_task = AsyncMock()
    return registry


@pytest.fixture
def mock_callback():
    """Create a mock TaskResultCallback with Future support."""
    callback = MagicMock(spec=TaskResultCallback)
    # By default, register_future returns a future that will be resolved
    loop = asyncio.new_event_loop()
    future = loop.create_future()
    callback.register_future = MagicMock(return_value=future)
    callback.cleanup_future = MagicMock()
    callback.has_future = MagicMock(return_value=True)
    callback._future = future
    callback._loop = loop
    return callback


@pytest.fixture
def mock_queue_manager():
    """Create a mock WorkerQueueManager."""
    manager = MagicMock()
    manager.has_worker = MagicMock(return_value=True)
    manager.enqueue_task = MagicMock(return_value=1)
    return manager


@pytest.fixture
def proxy_app(
    mock_scheduling_strategy,
    mock_instance_registry,
    mock_task_registry,
    mock_callback,
    mock_queue_manager,
):
    """Create a FastAPI app with proxy router and a test scheduler route."""
    app = FastAPI()

    # Add a scheduler-internal route first (should take priority)
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    proxy_router = ProxyRouter(
        scheduling_strategy=mock_scheduling_strategy,
        instance_registry=mock_instance_registry,
        task_registry=mock_task_registry,
        task_result_callback=mock_callback,
        worker_queue_manager=mock_queue_manager,
        proxy_timeout=5.0,
    )
    app.include_router(proxy_router.router)

    return app


# ============================================================================
# Proxy Forwarding Tests
# ============================================================================


class TestProxyForwarding:
    """Tests for basic proxy request forwarding."""

    def test_post_v1_chat_completions_returns_response(
        self,
        mock_scheduling_strategy,
        mock_instance_registry,
        mock_task_registry,
        mock_callback,
        mock_queue_manager,
    ):
        """Test POST /v1/chat/completions returns synchronous response."""
        app = FastAPI()

        proxy_router = ProxyRouter(
            scheduling_strategy=mock_scheduling_strategy,
            instance_registry=mock_instance_registry,
            task_registry=mock_task_registry,
            task_result_callback=mock_callback,
            worker_queue_manager=mock_queue_manager,
            proxy_timeout=5.0,
        )
        app.include_router(proxy_router.router)

        # Set up the future to be resolved with a successful result
        result = TaskResult(
            task_id="proxy-test",
            worker_id="instance-1",
            status="completed",
            result={"choices": [{"message": {"content": "Hello!"}}]},
            execution_time_ms=100.0,
            http_status_code=200,
            response_headers={"content-type": "application/json"},
        )

        # Make register_future return a pre-resolved future
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        future.set_result(result)
        mock_callback.register_future = MagicMock(return_value=future)

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        loop.close()

    def test_get_v1_models_forwarded(
        self,
        mock_scheduling_strategy,
        mock_instance_registry,
        mock_task_registry,
        mock_callback,
        mock_queue_manager,
    ):
        """Test GET /v1/models is forwarded to instance."""
        app = FastAPI()

        proxy_router = ProxyRouter(
            scheduling_strategy=mock_scheduling_strategy,
            instance_registry=mock_instance_registry,
            task_registry=mock_task_registry,
            task_result_callback=mock_callback,
            worker_queue_manager=mock_queue_manager,
            proxy_timeout=5.0,
        )
        app.include_router(proxy_router.router)

        result = TaskResult(
            task_id="proxy-test",
            worker_id="instance-1",
            status="completed",
            result={"data": [{"id": "test-model"}]},
            execution_time_ms=50.0,
            http_status_code=200,
            response_headers={},
        )

        loop = asyncio.new_event_loop()
        future = loop.create_future()
        future.set_result(result)
        mock_callback.register_future = MagicMock(return_value=future)

        client = TestClient(app)
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["data"][0]["id"] == "test-model"
        loop.close()

    def test_unknown_path_forwarded(
        self,
        mock_scheduling_strategy,
        mock_instance_registry,
        mock_task_registry,
        mock_callback,
        mock_queue_manager,
    ):
        """Test unknown path is forwarded to instance."""
        app = FastAPI()

        proxy_router = ProxyRouter(
            scheduling_strategy=mock_scheduling_strategy,
            instance_registry=mock_instance_registry,
            task_registry=mock_task_registry,
            task_result_callback=mock_callback,
            worker_queue_manager=mock_queue_manager,
            proxy_timeout=5.0,
        )
        app.include_router(proxy_router.router)

        result = TaskResult(
            task_id="proxy-test",
            worker_id="instance-1",
            status="completed",
            result={"custom": "response"},
            execution_time_ms=50.0,
            http_status_code=200,
            response_headers={},
        )

        loop = asyncio.new_event_loop()
        future = loop.create_future()
        future.set_result(result)
        mock_callback.register_future = MagicMock(return_value=future)

        client = TestClient(app)
        response = client.post("/some/custom/path", json={"data": "test"})

        assert response.status_code == 200
        assert response.json() == {"custom": "response"}
        loop.close()


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestProxyErrors:
    """Tests for proxy error scenarios."""

    def test_no_instances_returns_503(
        self,
        mock_scheduling_strategy,
        mock_instance_registry_empty,
        mock_task_registry,
        mock_callback,
        mock_queue_manager,
    ):
        """Test 503 when no instances are available."""
        app = FastAPI()

        proxy_router = ProxyRouter(
            scheduling_strategy=mock_scheduling_strategy,
            instance_registry=mock_instance_registry_empty,
            task_registry=mock_task_registry,
            task_result_callback=mock_callback,
            worker_queue_manager=mock_queue_manager,
            proxy_timeout=5.0,
        )
        app.include_router(proxy_router.router)

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test"},
        )

        assert response.status_code == 503
        assert "No backend instances" in response.json()["error"]["message"]

    def test_timeout_returns_504(
        self,
        mock_scheduling_strategy,
        mock_instance_registry,
        mock_task_registry,
        mock_callback,
        mock_queue_manager,
    ):
        """Test 504 when request times out."""
        app = FastAPI()

        proxy_router = ProxyRouter(
            scheduling_strategy=mock_scheduling_strategy,
            instance_registry=mock_instance_registry,
            task_registry=mock_task_registry,
            task_result_callback=mock_callback,
            worker_queue_manager=mock_queue_manager,
            proxy_timeout=0.1,  # Very short timeout
        )
        app.include_router(proxy_router.router)

        # Return a future that never resolves, created on the running loop
        # We use a side_effect that creates the future on the current event loop
        def create_pending_future(task_id):
            loop = asyncio.get_running_loop()
            return loop.create_future()

        mock_callback.register_future = MagicMock(side_effect=create_pending_future)

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test"},
        )

        assert response.status_code == 504
        assert "timed out" in response.json()["error"]["message"]


# ============================================================================
# Route Priority Tests
# ============================================================================


class TestRoutePriority:
    """Tests for scheduler route priority over proxy."""

    def test_health_not_proxied(self, proxy_app):
        """Test /health is handled by scheduler, not proxy."""
        client = TestClient(proxy_app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_scheduler_enqueue_called_with_correct_task(
        self,
        mock_scheduling_strategy,
        mock_instance_registry,
        mock_task_registry,
        mock_callback,
        mock_queue_manager,
    ):
        """Test that enqueue is called with correct QueuedTask metadata."""
        app = FastAPI()

        proxy_router = ProxyRouter(
            scheduling_strategy=mock_scheduling_strategy,
            instance_registry=mock_instance_registry,
            task_registry=mock_task_registry,
            task_result_callback=mock_callback,
            worker_queue_manager=mock_queue_manager,
            proxy_timeout=5.0,
        )
        app.include_router(proxy_router.router)

        result = TaskResult(
            task_id="proxy-test",
            worker_id="instance-1",
            status="completed",
            result={"ok": True},
            execution_time_ms=50.0,
            http_status_code=200,
            response_headers={},
        )

        loop = asyncio.new_event_loop()
        future = loop.create_future()
        future.set_result(result)
        mock_callback.register_future = MagicMock(return_value=future)

        client = TestClient(app)
        client.post("/v1/chat/completions", json={"model": "test"})

        # Verify enqueue was called
        mock_queue_manager.enqueue_task.assert_called_once()
        call_args = mock_queue_manager.enqueue_task.call_args
        assert call_args[0][0] == "instance-1"  # worker_id
        queued_task = call_args[0][1]
        assert queued_task.metadata["path"] == "v1/chat/completions"
        assert queued_task.metadata["method"] == "POST"
        assert queued_task.metadata["proxy"] is True
        assert queued_task.task_input == {"model": "test"}
        loop.close()


# ============================================================================
# Backend Error Forwarding Tests
# ============================================================================


class TestBackendErrorForwarding:
    """Tests for forwarding backend errors transparently."""

    def test_backend_500_forwarded(
        self,
        mock_scheduling_strategy,
        mock_instance_registry,
        mock_task_registry,
        mock_callback,
        mock_queue_manager,
    ):
        """Test backend 500 error is forwarded to client."""
        app = FastAPI()

        proxy_router = ProxyRouter(
            scheduling_strategy=mock_scheduling_strategy,
            instance_registry=mock_instance_registry,
            task_registry=mock_task_registry,
            task_result_callback=mock_callback,
            worker_queue_manager=mock_queue_manager,
            proxy_timeout=5.0,
        )
        app.include_router(proxy_router.router)

        result = TaskResult(
            task_id="proxy-test",
            worker_id="instance-1",
            status="failed",
            result={"error": {"message": "Internal server error"}},
            error="HTTP 500",
            execution_time_ms=50.0,
            http_status_code=500,
            response_headers={},
        )

        loop = asyncio.new_event_loop()
        future = loop.create_future()
        future.set_result(result)
        mock_callback.register_future = MagicMock(return_value=future)

        client = TestClient(app)
        response = client.post("/v1/chat/completions", json={"model": "test"})

        assert response.status_code == 500
        assert response.json()["error"]["message"] == "Internal server error"
        loop.close()

    def test_connection_failure_returns_502(
        self,
        mock_scheduling_strategy,
        mock_instance_registry,
        mock_task_registry,
        mock_callback,
        mock_queue_manager,
    ):
        """Test connection failure to backend returns 502."""
        app = FastAPI()

        proxy_router = ProxyRouter(
            scheduling_strategy=mock_scheduling_strategy,
            instance_registry=mock_instance_registry,
            task_registry=mock_task_registry,
            task_result_callback=mock_callback,
            worker_queue_manager=mock_queue_manager,
            proxy_timeout=5.0,
        )
        app.include_router(proxy_router.router)

        result = TaskResult(
            task_id="proxy-test",
            worker_id="instance-1",
            status="failed",
            error="Connection refused",
            execution_time_ms=50.0,
            # No http_status_code = connection-level failure
        )

        loop = asyncio.new_event_loop()
        future = loop.create_future()
        future.set_result(result)
        mock_callback.register_future = MagicMock(return_value=future)

        client = TestClient(app)
        response = client.post("/v1/chat/completions", json={"model": "test"})

        assert response.status_code == 502
        error_msg = response.json()["error"]["message"]
        assert "Connection refused" in error_msg or "Backend request failed" in error_msg
        loop.close()
