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
    from src.api import scheduling_strategy, predictor_client, config
    from src.scheduler import get_strategy
    import src.api as api_module

    # Clear registries
    instance_registry._instances.clear()
    instance_registry._queue_info.clear()
    instance_registry._stats.clear()
    task_registry._tasks.clear()

    # Reset instance registry queue type to probabilistic (default)
    instance_registry._queue_info_type = "probabilistic"

    # Reset scheduling strategy to default (probabilistic)
    api_module.scheduling_strategy = get_strategy(
        strategy_name="probabilistic",
        predictor_client=predictor_client,
        instance_registry=instance_registry,
        target_quantile=0.9,
    )

    # Reset config to default
    config.scheduling.default_strategy = "probabilistic"
    config.scheduling.probabilistic_quantile = 0.9

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
        """Test successful instance removal with drain-before-remove workflow."""
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

        # Drain the instance first (required for safe removal)
        drain_response = client.post(
            "/instance/drain",
            json={"instance_id": "inst-1"}
        )
        assert drain_response.status_code == 200

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

        assert response.status_code == 400


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
                    "endpoint": f"http://localhost:800{i}",
                    "platform_info": {
                        "software_name": "docker",
                        "software_version": "20.10",
                        "hardware_name": "test-hardware"
                    }
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


class TestTaskClear:
    """Tests for task clear endpoint."""

    def test_clear_empty_tasks(self, client):
        """Test clearing tasks when registry is empty."""
        response = client.post("/task/clear")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["cleared_count"] == 0
        assert "Successfully cleared 0 task(s)" in data["message"]

    def test_clear_multiple_tasks(self, client):
        """Test clearing multiple tasks."""
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

        # Submit multiple tasks
        with patch("src.api.predictor_client.predict", new=AsyncMock(return_value=[])):
            with patch("src.api.task_dispatcher.dispatch_task_async"):
                for i in range(3):
                    client.post(
                        "/task/submit",
                        json={
                            "task_id": f"task-{i}",
                            "model_id": "model-1",
                            "task_input": {"test": "data"},
                            "metadata": {}
                        }
                    )

        # Verify tasks exist
        response = client.get("/task/list")
        assert response.json()["count"] == 3

        # Clear all tasks
        response = client.post("/task/clear")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["cleared_count"] == 3
        assert "Successfully cleared 3 task(s)" in data["message"]

        # Verify tasks are cleared
        response = client.get("/task/list")
        assert response.json()["count"] == 0


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


# ============================================================================
# Strategy Management Tests
# ============================================================================

class TestStrategyManagement:
    """Tests for strategy management endpoints."""

    def test_get_strategy_default(self, client):
        """Test getting the default strategy (probabilistic)."""
        response = client.get("/strategy/get")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["strategy_info"]["strategy_name"] == "probabilistic"
        assert "target_quantile" in data["strategy_info"]["parameters"]
        assert data["strategy_info"]["parameters"]["target_quantile"] == 0.9

    def test_set_strategy_to_min_time(self, client):
        """Test switching to min_time strategy."""
        response = client.post(
            "/strategy/set",
            json={
                "strategy_name": "min_time"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["strategy_info"]["strategy_name"] == "min_time"
        assert data["cleared_tasks"] == 0
        assert data["reinitialized_instances"] == 0
        assert "Successfully switched" in data["message"]

    def test_set_strategy_to_round_robin(self, client):
        """Test switching to round_robin strategy."""
        response = client.post(
            "/strategy/set",
            json={
                "strategy_name": "round_robin"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["strategy_info"]["strategy_name"] == "round_robin"
        assert data["strategy_info"]["parameters"] == {}

    def test_set_strategy_to_probabilistic_with_custom_quantile(self, client):
        """Test switching to probabilistic strategy with custom quantile."""
        response = client.post(
            "/strategy/set",
            json={
                "strategy_name": "probabilistic",
                "target_quantile": 0.95
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["strategy_info"]["strategy_name"] == "probabilistic"
        assert data["strategy_info"]["parameters"]["target_quantile"] == 0.95

    def test_set_strategy_invalid_name(self, client):
        """Test setting strategy with invalid name."""
        response = client.post(
            "/strategy/set",
            json={
                "strategy_name": "invalid_strategy"
            }
        )

        assert response.status_code == 422  # Pydantic validation error

    def test_set_strategy_clears_tasks(self, client):
        """Test that setting strategy clears all tasks."""
        from src.predictor_client import Prediction

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

        # Submit a task (mock predictor)
        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0, 0.95: 120.0, 0.99: 130.0},
            error_margin_ms=10.0
        )

        with patch("src.api.predictor_client.predict", new=AsyncMock(return_value=[mock_prediction])):
            with patch("src.api.task_dispatcher.dispatch_task_async"):
                client.post(
                    "/task/submit",
                    json={
                        "task_id": "task-1",
                        "model_id": "model-1",
                        "task_input": {"prompt": "test"},
                        "metadata": {}
                    }
                )

        # Switch strategy
        response = client.post(
            "/strategy/set",
            json={
                "strategy_name": "min_time"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["cleared_tasks"] == 1

        # Verify tasks are cleared
        tasks_response = client.get("/task/list")
        assert tasks_response.json()["total"] == 0

    def test_set_strategy_reinitializes_instances(self, client):
        """Test that setting strategy reinitializes instance queues."""
        # Register two instances
        for i in range(1, 3):
            client.post(
                "/instance/register",
                json={
                    "instance_id": f"inst-{i}",
                    "model_id": "model-1",
                    "endpoint": f"http://localhost:800{i}",
                    "platform_info": {
                        "software_name": "docker",
                        "software_version": "20.10",
                        "hardware_name": "test-hardware"
                    }
                }
            )

        # Switch from probabilistic to min_time
        response = client.post(
            "/strategy/set",
            json={
                "strategy_name": "min_time"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["reinitialized_instances"] == 2

        # Verify instances still exist
        instances_response = client.get("/instance/list")
        assert instances_response.json()["count"] == 2

        # Verify queue info type changed by checking instance info
        instance_info = client.get("/instance/info?instance_id=inst-1")
        assert instance_info.status_code == 200
        queue_info = instance_info.json()["queue_info"]
        # For min_time, should have expected_time_ms and error_margin_ms
        assert "expected_time_ms" in queue_info
        assert "error_margin_ms" in queue_info

    def test_set_strategy_rejects_when_tasks_running(self, client):
        """Test that setting strategy is rejected when tasks are running."""
        from src.predictor_client import Prediction

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

        # Submit a task
        mock_prediction = Prediction(
            instance_id="inst-1",
            predicted_time_ms=100.0,
            quantiles={0.5: 90.0, 0.9: 110.0, 0.95: 120.0, 0.99: 130.0},
            error_margin_ms=10.0
        )

        with patch("src.api.predictor_client.predict", new=AsyncMock(return_value=[mock_prediction])):
            with patch("src.api.task_dispatcher.dispatch_task_async"):
                client.post(
                    "/task/submit",
                    json={
                        "task_id": "task-1",
                        "model_id": "model-1",
                        "task_input": {"prompt": "test"},
                        "metadata": {}
                    }
                )

        # Manually set task to RUNNING status
        task_registry.update_status("task-1", TaskStatus.RUNNING)

        # Try to switch strategy - should fail
        response = client.post(
            "/strategy/set",
            json={
                "strategy_name": "min_time"
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert "Cannot switch strategy while" in data["detail"]["error"]

    def test_strategy_switch_round_trip(self, client):
        """Test switching between all strategies."""
        # Start with probabilistic (default)
        response = client.get("/strategy/get")
        assert response.json()["strategy_info"]["strategy_name"] == "probabilistic"

        # Switch to min_time
        response = client.post("/strategy/set", json={"strategy_name": "min_time"})
        assert response.status_code == 200

        response = client.get("/strategy/get")
        assert response.json()["strategy_info"]["strategy_name"] == "min_time"

        # Switch to round_robin
        response = client.post("/strategy/set", json={"strategy_name": "round_robin"})
        assert response.status_code == 200

        response = client.get("/strategy/get")
        assert response.json()["strategy_info"]["strategy_name"] == "round_robin"

        # Switch back to probabilistic
        response = client.post("/strategy/set", json={"strategy_name": "probabilistic", "target_quantile": 0.8})
        assert response.status_code == 200

        response = client.get("/strategy/get")
        data = response.json()
        assert data["strategy_info"]["strategy_name"] == "probabilistic"
        assert data["strategy_info"]["parameters"]["target_quantile"] == 0.8

    def test_set_strategy_invalid_quantile(self, client):
        """Test setting strategy with invalid quantile value."""
        # Quantile > 1.0
        response = client.post(
            "/strategy/set",
            json={
                "strategy_name": "probabilistic",
                "target_quantile": 1.5
            }
        )
        assert response.status_code == 422  # Pydantic validation error

        # Quantile < 0.0
        response = client.post(
            "/strategy/set",
            json={
                "strategy_name": "probabilistic",
                "target_quantile": -0.1
            }
        )
        assert response.status_code == 422  # Pydantic validation error
