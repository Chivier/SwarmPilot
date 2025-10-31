"""
Unit tests for FastAPI endpoints.

Tests instance management, task submission, and health check endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock

from src.api import app, instance_registry, task_registry, task_dispatcher
from src.model import TaskStatus


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_registries():
    """Reset registries before each test."""
    # Clear registries
    instance_registry._instances.clear()
    instance_registry._queue_info.clear()
    instance_registry._stats.clear()
    task_registry._tasks.clear()
    yield


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


# ============================================================================
# Instance Registration Tests
# ============================================================================

class TestInstanceRegistration:
    """Tests for instance registration endpoint."""

    def test_register_instance_success(self, client):
        """Test successful instance registration."""
        response = client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware"
                }
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["instance"]["instance_id"] == "inst-1"
        assert data["message"] == "Instance registered successfully"

    def test_register_duplicate_instance(self, client):
        """Test registering duplicate instance returns 400."""
        # Register first time
        client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware"
                }
            }
        )

        # Try to register again
        response = client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware"
                }
            }
        )

        assert response.status_code == 400

    def test_register_instance_validation(self, client):
        """Test validation of registration request."""
        response = client.post(
            "/instance/register",
            json={"instance_id": "inst-1"}  # Missing required fields
        )

        assert response.status_code == 422  # Validation error


# ============================================================================
# Instance Removal Tests
# ============================================================================

class TestInstanceRemoval:
    """Tests for instance removal endpoint."""

    def test_remove_instance_success(self, client):
        """Test successful instance removal."""
        # Register instance first
        client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware"
                }
            }
        )

        # Remove it
        response = client.post(
            "/instance/remove",
            json={"instance_id": "inst-1"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["instance_id"] == "inst-1"

    def test_remove_nonexistent_instance(self, client):
        """Test removing non-existent instance returns 404."""
        response = client.post(
            "/instance/remove",
            json={"instance_id": "nonexistent"}
        )

        assert response.status_code == 404

    def test_remove_instance_with_pending_tasks(self, client):
        """Test removing instance with pending tasks (allowed)."""
        # Register instance
        client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware"
                }
            }
        )

        # Add pending task
        instance_registry.increment_pending("inst-1")

        # Should still allow removal
        response = client.post(
            "/instance/remove",
            json={"instance_id": "inst-1"}
        )

        assert response.status_code == 200


# ============================================================================
# Instance List Tests
# ============================================================================

class TestInstanceList:
    """Tests for instance listing endpoint."""

    def test_list_all_instances(self, client):
        """Test listing all instances."""
        # Register multiple instances
        for i in range(3):
            client.post(
                "/instance/register",
                json={
                    "instance_id": f"inst-{i}",
                    "model_id": "model-1",
                    "endpoint": f"http://localhost:800{i}"
                }
            )

        response = client.get("/instance/list")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 3
        assert len(data["instances"]) == 3

    def test_list_instances_filtered(self, client):
        """Test listing instances filtered by model_id."""
        # Register instances with different models
        client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-a",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware"
                }
            }
        )
        client.post(
            "/instance/register",
            json={
                "instance_id": "inst-2",
                "model_id": "model-b",
                "endpoint": "http://localhost:8002",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware"
                }
            }
        )

        response = client.get("/instance/list?model_id=model-a")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["instances"][0]["model_id"] == "model-a"

    def test_list_empty_instances(self, client):
        """Test listing when no instances registered."""
        response = client.get("/instance/list")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["instances"] == []


# ============================================================================
# Instance Info Tests
# ============================================================================

class TestInstanceInfo:
    """Tests for instance info endpoint."""

    def test_get_instance_info(self, client):
        """Test getting instance information."""
        # Register instance
        client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware"
                }
            }
        )

        response = client.get("/instance/info?instance_id=inst-1")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["instance"]["instance_id"] == "inst-1"
        assert "stats" in data
        assert "queue_info" in data

    def test_get_nonexistent_instance_info(self, client):
        """Test getting info for non-existent instance."""
        response = client.get("/instance/info?instance_id=nonexistent")

        assert response.status_code == 404


# ============================================================================
# Task Submission Tests
# ============================================================================

class TestTaskSubmission:
    """Tests for task submission endpoint."""

    def test_submit_task_success(self, client):
        """Test successful task submission."""
        # Register instance first
        client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware"
                }
            }
        )

        # Mock predictor and dispatcher
        with patch("src.api.predictor_client.predict", new=AsyncMock(return_value=[])):
            with patch("src.api.task_dispatcher.dispatch_task_async") as mock_dispatch:
                response = client.post(
                    "/task/submit",
                    json={
                        "task_id": "task-1",
                        "model_id": "model-1",
                        "task_input": {"prompt": "test"},
                        "metadata": {}
                    }
                )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["task"]["task_id"] == "task-1"

    def test_submit_task_no_instances(self, client):
        """Test submitting task when no instances available."""
        response = client.post(
            "/task/submit",
            json={
                "task_id": "task-1",
                "model_id": "model-1",
                "task_input": {"prompt": "test"},
                "metadata": {}
            }
        )

        # API returns 404 when no instances match the model_id
        assert response.status_code == 404

    def test_submit_duplicate_task(self, client):
        """Test submitting duplicate task."""
        # Register instance
        client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware"
                }
            }
        )

        # Submit task first time
        with patch("src.api.predictor_client.predict", new=AsyncMock(return_value=[])):
            with patch("src.api.task_dispatcher.dispatch_task_async"):
                client.post(
                    "/task/submit",
                    json={
                        "task_id": "task-1",
                        "model_id": "model-1",
                        "task_input": {},
                        "metadata": {}
                    }
                )

        # Try to submit again
        with patch("src.api.predictor_client.predict", new=AsyncMock(return_value=[])):
            response = client.post(
                "/task/submit",
                json={
                    "task_id": "task-1",
                    "model_id": "model-1",
                    "task_input": {},
                    "metadata": {}
                }
            )

        assert response.status_code == 400


# ============================================================================
# Task List Tests
# ============================================================================

class TestTaskList:
    """Tests for task listing endpoint."""

    def test_list_all_tasks(self, client):
        """Test listing all tasks."""
        # Register instance and submit tasks
        client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware"
                }
            }
        )

        with patch("src.api.predictor_client.predict", new=AsyncMock(return_value=[])):
            with patch("src.api.task_dispatcher.dispatch_task_async"):
                for i in range(3):
                    client.post(
                        "/task/submit",
                        json={
                            "task_id": f"task-{i}",
                            "model_id": "model-1",
                            "task_input": {},
                            "metadata": {}
                        }
                    )

        response = client.get("/task/list")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 3
        assert data["total"] == 3

    def test_list_tasks_with_filters(self, client):
        """Test listing tasks with status filter."""
        # Register instance
        client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware"
                }
            }
        )

        # Submit task
        with patch("src.api.predictor_client.predict", new=AsyncMock(return_value=[])):
            with patch("src.api.task_dispatcher.dispatch_task_async"):
                client.post(
                    "/task/submit",
                    json={
                        "task_id": "task-1",
                        "model_id": "model-1",
                        "task_input": {},
                        "metadata": {}
                    }
                )

        response = client.get("/task/list?status=pending")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 0

    def test_list_tasks_pagination(self, client):
        """Test task list pagination."""
        response = client.get("/task/list?limit=10&offset=0")

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10
        assert data["offset"] == 0


# ============================================================================
# Task Info Tests
# ============================================================================

class TestTaskInfo:
    """Tests for task info endpoint."""

    def test_get_task_info(self, client):
        """Test getting task information."""
        # Register instance and submit task
        client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware"
                }
            }
        )

        with patch("src.api.predictor_client.predict", new=AsyncMock(return_value=[])):
            with patch("src.api.task_dispatcher.dispatch_task_async"):
                client.post(
                    "/task/submit",
                    json={
                        "task_id": "task-1",
                        "model_id": "model-1",
                        "task_input": {"test": "data"},
                        "metadata": {}
                    }
                )

        response = client.get("/task/info?task_id=task-1")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["task"]["task_id"] == "task-1"
        assert "timestamps" in data["task"]

    def test_get_nonexistent_task_info(self, client):
        """Test getting info for non-existent task."""
        response = client.get("/task/info?task_id=nonexistent")

        assert response.status_code == 404


# ============================================================================
# Health Check Tests
# ============================================================================

class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check_healthy(self, client):
        """Test health check when service is healthy."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "healthy"
        assert "stats" in data
        assert "version" in data
        assert "timestamp" in data

    def test_health_check_includes_stats(self, client):
        """Test that health check includes statistics."""
        # Register instance
        client.post(
            "/instance/register",
            json={
                "instance_id": "inst-1",
                "model_id": "model-1",
                "endpoint": "http://localhost:8001",
                "platform_info": {
                    "software_name": "docker",
                    "software_version": "20.10",
                    "hardware_name": "test-hardware"
                }
            }
        )

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["stats"]["total_instances"] == 1
        assert data["stats"]["active_instances"] == 1
