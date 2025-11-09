"""
Unit tests for lazy initialization behavior.

Tests that instances start without automatically connecting to scheduler
or starting models, and that connections are only established when
/model/start is explicitly called.

NOTE: Most tests in this file are temporarily disabled because WebSocket
communication with scheduler has been disabled. All Instance-Scheduler
communication now uses HTTP API only.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from fastapi.testclient import TestClient

# Skip all WebSocket-related tests since WebSocket is temporarily disabled
pytestmark = pytest.mark.skip(reason="WebSocket communication temporarily disabled")


@pytest.fixture
def mock_docker_manager():
    """Mock docker manager for testing."""
    with patch("src.api.get_docker_manager") as mock:
        manager = AsyncMock()
        manager.is_model_running = AsyncMock(return_value=False)
        manager.start_model = AsyncMock(return_value=MagicMock(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        ))
        manager.get_current_model = AsyncMock(return_value=None)
        manager.check_model_health = AsyncMock(return_value=True)
        manager.stop_model = AsyncMock(return_value="test-model")
        manager.close = AsyncMock()
        mock.return_value = manager
        yield manager


@pytest.fixture
def mock_registry():
    """Mock model registry for testing."""
    with patch("src.api.get_registry") as mock:
        registry = MagicMock()
        registry.model_exists = MagicMock(return_value=True)
        registry.models = []
        mock.return_value = registry
        yield registry


@pytest.fixture
def mock_scheduler_client():
    """Mock scheduler client for testing."""
    with patch("src.api.get_scheduler_client") as mock:
        client = AsyncMock()
        client.is_enabled = True
        client.scheduler_url = "http://localhost:8000"
        client._registered = False
        client.register_instance = AsyncMock(return_value=True)
        client.deregister_instance = AsyncMock(return_value=True)
        mock.return_value = client
        yield client


@pytest.fixture
def mock_websocket_client():
    """Mock WebSocket client for testing."""
    with patch("src.api.get_websocket_client") as mock:
        client = AsyncMock()
        client.model_id = "unknown"
        client.platform_info = {"software_name": "test"}
        client.is_connected = MagicMock(return_value=False)
        client.start = AsyncMock()
        client.send_message = AsyncMock()
        client.stop = AsyncMock()
        mock.return_value = client
        yield client


@pytest.fixture
def mock_task_queue():
    """Mock task queue for testing."""
    with patch("src.api.get_task_queue") as mock:
        queue = AsyncMock()
        queue.stop_processing = AsyncMock()
        queue.get_queue_stats = AsyncMock(return_value={
            "total": 0,
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0
        })
        mock.return_value = queue
        yield queue


class TestLazyInitialization:
    """Test suite for lazy initialization behavior."""

    @pytest.mark.asyncio
    async def test_instance_starts_without_scheduler_connection(
        self,
        mock_docker_manager,
        mock_registry,
        mock_scheduler_client,
        mock_websocket_client
    ):
        """
        Test that instance service starts successfully without connecting to scheduler.

        The instance should start in "idle" state without:
        - Connecting to scheduler via HTTP
        - Starting WebSocket connection
        - Starting any models
        """
        # Import after mocks are set up
        from src.api import app

        # Create test client (this triggers lifespan startup)
        with TestClient(app) as client:
            # Verify instance is healthy
            response = client.get("/health")
            assert response.status_code == 200

            # Verify no scheduler registration was attempted during startup
            mock_scheduler_client.register_instance.assert_not_called()

            # Verify WebSocket was not started during startup
            mock_websocket_client.start.assert_not_called()

            # Verify no model was started
            mock_docker_manager.start_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_instance_info_shows_idle_state(
        self,
        mock_docker_manager,
        mock_registry,
        mock_scheduler_client,
        mock_task_queue
    ):
        """Test that /info endpoint shows instance in idle state after startup."""
        from src.api import app

        with TestClient(app) as client:
            response = client.get("/info")
            assert response.status_code == 200

            data = response.json()
            assert data["success"] is True
            assert data["instance"]["status"] == "idle"
            assert data["instance"]["current_model"] is None

    @pytest.mark.asyncio
    async def test_model_start_triggers_scheduler_registration(
        self,
        mock_docker_manager,
        mock_registry,
        mock_scheduler_client,
        mock_websocket_client
    ):
        """
        Test that calling /model/start triggers scheduler registration.

        When /model/start is called:
        1. Model should be started
        2. Scheduler registration should be triggered (HTTP)
        3. WebSocket connection should be established
        4. WebSocket registration message should be sent
        """
        from src.api import app

        with TestClient(app) as client:
            # Start a model
            response = client.post("/model/start", json={
                "model_id": "test-model",
                "parameters": {}
            })

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["model_id"] == "test-model"

            # Verify model was started
            mock_docker_manager.start_model.assert_called_once()

            # Verify scheduler registration was triggered
            mock_scheduler_client.register_instance.assert_called_once()
            call_args = mock_scheduler_client.register_instance.call_args
            assert call_args[1]["model_id"] == "test-model"

    @pytest.mark.asyncio
    async def test_model_start_establishes_websocket_connection(
        self,
        mock_docker_manager,
        mock_registry,
        mock_scheduler_client,
        mock_websocket_client
    ):
        """
        Test that /model/start establishes WebSocket connection.

        WebSocket should:
        1. Have model_id updated before connection
        2. Be started (connected) after model start
        3. Send REGISTER message to scheduler
        """
        from src.api import app

        with TestClient(app) as client:
            # Reset mock to clear any setup calls
            mock_websocket_client.start.reset_mock()
            mock_websocket_client.send_message.reset_mock()

            # Start a model
            response = client.post("/model/start", json={
                "model_id": "test-model",
                "parameters": {}
            })

            assert response.status_code == 200

            # Verify model_id was updated
            assert mock_websocket_client.model_id == "test-model"

            # Verify WebSocket was started
            mock_websocket_client.start.assert_called_once()

            # Verify REGISTER message was sent
            mock_websocket_client.send_message.assert_called_once()
            register_msg = mock_websocket_client.send_message.call_args[0][0]
            assert register_msg["type"] == "register"
            assert register_msg["model_id"] == "test-model"

    @pytest.mark.asyncio
    async def test_model_start_with_custom_scheduler_url(
        self,
        mock_docker_manager,
        mock_registry,
        mock_scheduler_client,
        mock_websocket_client
    ):
        """
        Test that /model/start accepts custom scheduler_url parameter.

        When scheduler_url is provided:
        1. Scheduler client should be updated with new URL
        2. Registration should use the new URL
        """
        from src.api import app

        custom_scheduler_url = "http://custom-scheduler:9000"

        with TestClient(app) as client:
            response = client.post("/model/start", json={
                "model_id": "test-model",
                "parameters": {},
                "scheduler_url": custom_scheduler_url
            })

            assert response.status_code == 200

            # Verify scheduler_url was updated
            assert mock_scheduler_client.scheduler_url == custom_scheduler_url

            # Verify registration was attempted
            mock_scheduler_client.register_instance.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_start_fails_without_model_in_registry(
        self,
        mock_docker_manager,
        mock_registry,
        mock_scheduler_client,
        mock_websocket_client
    ):
        """Test that /model/start fails when model doesn't exist in registry."""
        from src.api import app

        # Configure registry to report model not found
        mock_registry.model_exists.return_value = False

        with TestClient(app) as client:
            response = client.post("/model/start", json={
                "model_id": "nonexistent-model",
                "parameters": {}
            })

            assert response.status_code == 400
            assert "not found in registry" in response.json()["detail"]

            # Verify no model was started
            mock_docker_manager.start_model.assert_not_called()

            # Verify no scheduler registration
            mock_scheduler_client.register_instance.assert_not_called()

    @pytest.mark.asyncio
    async def test_model_start_continues_on_websocket_failure(
        self,
        mock_docker_manager,
        mock_registry,
        mock_scheduler_client,
        mock_websocket_client
    ):
        """
        Test that /model/start succeeds even if WebSocket connection fails.

        Instance should fall back to HTTP-only communication.
        """
        from src.api import app

        # Configure WebSocket to fail on start
        mock_websocket_client.start.side_effect = Exception("WebSocket connection failed")

        with TestClient(app) as client:
            response = client.post("/model/start", json={
                "model_id": "test-model",
                "parameters": {}
            })

            # Model start should still succeed
            assert response.status_code == 200
            assert response.json()["success"] is True

            # Verify model was started
            mock_docker_manager.start_model.assert_called_once()

            # Verify HTTP registration still happened
            mock_scheduler_client.register_instance.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_not_connected_if_no_scheduler_url(
        self,
        mock_docker_manager,
        mock_registry,
        mock_scheduler_client,
        mock_websocket_client
    ):
        """
        Test that WebSocket is not started if scheduler integration is disabled.
        """
        from src.api import app

        # Disable scheduler integration
        mock_scheduler_client.is_enabled = False
        mock_websocket_client_get = None

        with patch("src.api.get_websocket_client", return_value=None):
            with TestClient(app) as client:
                response = client.post("/model/start", json={
                    "model_id": "test-model",
                    "parameters": {}
                })

                assert response.status_code == 200

                # Verify model was started
                mock_docker_manager.start_model.assert_called_once()

                # Verify no scheduler registration
                mock_scheduler_client.register_instance.assert_not_called()

    @pytest.mark.asyncio
    async def test_model_start_reconnects_existing_websocket(
        self,
        mock_docker_manager,
        mock_registry,
        mock_scheduler_client,
        mock_websocket_client
    ):
        """
        Test that if WebSocket is already connected, it sends updated registration.
        """
        from src.api import app

        # Configure WebSocket as already connected
        mock_websocket_client.is_connected.return_value = True

        with TestClient(app) as client:
            # Start a model
            response = client.post("/model/start", json={
                "model_id": "test-model",
                "parameters": {}
            })

            assert response.status_code == 200

            # Verify WebSocket start was NOT called (already connected)
            mock_websocket_client.start.assert_not_called()

            # Verify updated REGISTER message was sent
            mock_websocket_client.send_message.assert_called_once()
            register_msg = mock_websocket_client.send_message.call_args[0][0]
            assert register_msg["type"] == "register"
            assert register_msg["model_id"] == "test-model"


class TestBackwardCompatibility:
    """Test backward compatibility with existing behavior."""

    @pytest.mark.asyncio
    async def test_scheduler_url_environment_variable_still_works(
        self,
        mock_docker_manager,
        mock_registry,
        mock_scheduler_client,
        mock_websocket_client
    ):
        """
        Test that SCHEDULER_URL environment variable still works.

        When SCHEDULER_URL is set, scheduler_client should be initialized
        with that URL, even if not provided in /model/start request.
        """
        from src.api import app

        # scheduler_client is already mocked with is_enabled=True
        # and scheduler_url="http://localhost:8000"

        with TestClient(app) as client:
            response = client.post("/model/start", json={
                "model_id": "test-model",
                "parameters": {}
            })

            assert response.status_code == 200

            # Verify scheduler registration was triggered
            mock_scheduler_client.register_instance.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_remains_functional(
        self,
        mock_docker_manager,
        mock_registry,
        mock_scheduler_client
    ):
        """Test that health check endpoint still works correctly."""
        from src.api import app

        with TestClient(app) as client:
            # Health check without model should succeed
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"

            # Start a model
            mock_docker_manager.is_model_running.return_value = True
            mock_docker_manager.check_model_health.return_value = True

            # Health check with healthy model should succeed
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"

            # Health check with unhealthy model should fail
            mock_docker_manager.check_model_health.return_value = False
            response = client.get("/health")
            assert response.status_code == 503
            assert response.json()["status"] == "unhealthy"
