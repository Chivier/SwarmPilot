"""Tests for FastAPI endpoints."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from swarmpilot.planner.api import app


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """Test health check returns healthy status."""
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestInfoEndpoint:
    """Tests for /info endpoint."""

    def test_service_info(self, client):
        """Test service info returns correct metadata."""
        response = client.get("/v1/info")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "planner"
        assert "version" in data
        assert "simulated_annealing" in data["algorithms"]
        assert "integer_programming" in data["algorithms"]
        assert "relative_error" in data["objective_methods"]


class TestPlanEndpoint:
    """Tests for /plan endpoint."""

    def test_plan_with_simulated_annealing(self, client, sample_planner_input):
        """Test /plan with simulated annealing algorithm."""
        # Use a simple problem for faster test
        sample_planner_input["max_iterations"] = 10
        sample_planner_input["verbose"] = False

        with patch(
            "swarmpilot.planner.api.SimulatedAnnealingOptimizer"
        ) as mock_optimizer_class:
            # Mock optimizer instance and its methods
            mock_optimizer = MagicMock()
            mock_optimizer.optimize.return_value = (
                np.array([0, 1, 1, 2]),  # deployment
                0.0667,  # score
                {"algorithm": "simulated_annealing", "iterations": 10},  # stats
            )
            mock_optimizer.compute_service_capacity.return_value = np.array(
                [10.0, 16.0, 12.0]
            )
            mock_optimizer.compute_changes.return_value = 1
            mock_optimizer_class.return_value = mock_optimizer

            response = client.post("/v1/plan", json=sample_planner_input)

            assert response.status_code == 200
            data = response.json()
            assert data["deployment"] == [0, 1, 1, 2]
            assert data["score"] == 0.0667
            assert data["changes_count"] == 1
            assert data["service_capacity"] == [10.0, 16.0, 12.0]
            assert "stats" in data

            # Verify optimizer was called with correct parameters
            mock_optimizer_class.assert_called_once()
            mock_optimizer.optimize.assert_called_once()

    def test_plan_with_integer_programming(self, client, sample_planner_input):
        """Test /plan with integer programming algorithm."""
        sample_planner_input["algorithm"] = "integer_programming"
        sample_planner_input["verbose"] = False

        with patch(
            "swarmpilot.planner.api.IntegerProgrammingOptimizer"
        ) as mock_optimizer_class:
            mock_optimizer = MagicMock()
            mock_optimizer.optimize.return_value = (
                np.array([0, 1, 1, 2]),
                0.05,
                {
                    "algorithm": "integer_programming",
                    "solver_status": "optimal",
                },
            )
            mock_optimizer.compute_service_capacity.return_value = np.array(
                [10.0, 16.0, 12.0]
            )
            mock_optimizer.compute_changes.return_value = 1
            mock_optimizer_class.return_value = mock_optimizer

            response = client.post("/v1/plan", json=sample_planner_input)

            assert response.status_code == 200
            data = response.json()
            assert data["score"] == 0.05
            assert data["stats"]["algorithm"] == "integer_programming"

    def test_plan_invalid_input(self, client, sample_planner_input):
        """Test /plan rejects invalid input."""
        sample_planner_input["M"] = -1  # Invalid
        response = client.post("/v1/plan", json=sample_planner_input)
        assert response.status_code == 422  # Validation error

    def test_plan_unknown_algorithm(self, client, sample_planner_input):
        """Test /plan rejects unknown algorithm."""
        sample_planner_input["algorithm"] = "unknown_algorithm"
        response = client.post("/v1/plan", json=sample_planner_input)
        assert response.status_code == 422  # Validation error

    def test_plan_optimization_failure(self, client, sample_planner_input):
        """Test /plan handles optimization failures."""
        sample_planner_input["max_iterations"] = 10

        with patch(
            "swarmpilot.planner.api.SimulatedAnnealingOptimizer"
        ) as mock_optimizer_class:
            mock_optimizer_class.side_effect = ValueError("Optimization failed")

            response = client.post("/v1/plan", json=sample_planner_input)

            assert response.status_code == 400
            assert "Invalid input" in response.json()["detail"]


class TestInstanceRegisterEndpoint:
    """Tests for /instance/register endpoint."""

    def test_register_instance_success(self, client):
        """Test successful instance registration."""
        request_data = {
            "instance_id": "test-instance-1",
            "model_id": "model_0",
            "endpoint": "http://test:8080",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "nvidia-a100",
            },
        }

        response = client.post("/v1/instance/register", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "test-instance-1" in data["message"]
        assert "model_0" in data["message"]

    def test_register_instance_minimal_info(self, client):
        """Test instance registration with minimal platform_info."""
        request_data = {
            "instance_id": "test-instance-2",
            "model_id": "model_1",
            "endpoint": "http://test:8081",
            "platform_info": {},
        }

        response = client.post("/v1/instance/register", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_register_multiple_instances_same_model(self, client):
        """Test registering multiple instances for the same model."""
        # Register first instance
        request_data1 = {
            "instance_id": "instance-1",
            "model_id": "shared_model",
            "endpoint": "http://instance1:8080",
            "platform_info": {},
        }
        response1 = client.post("/v1/instance/register", json=request_data1)
        assert response1.status_code == 200

        # Register second instance for same model
        request_data2 = {
            "instance_id": "instance-2",
            "model_id": "shared_model",
            "endpoint": "http://instance2:8080",
            "platform_info": {},
        }
        response2 = client.post("/v1/instance/register", json=request_data2)
        assert response2.status_code == 200

        data = response2.json()
        assert data["success"] is True

    def test_register_instance_missing_required_fields(self, client):
        """Test registration fails with missing required fields."""
        # Missing model_id
        request_data = {
            "instance_id": "test-instance",
            "endpoint": "http://test:8080",
        }

        response = client.post("/v1/instance/register", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_register_instance_missing_endpoint(self, client):
        """Test registration fails without endpoint."""
        request_data = {"instance_id": "test-instance", "model_id": "model_0"}

        response = client.post("/v1/instance/register", json=request_data)

        assert response.status_code == 422  # Validation error


class TestDummyEndpoints:
    """Tests for dummy scheduler compatibility endpoints."""

    def test_instance_drain_success(self, client):
        """Test /instance/drain dummy endpoint returns success."""
        request_data = {"instance_id": "test-instance"}

        response = client.post("/v1/instance/drain", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["instance_id"] == "test-instance"
        assert data["status"] == "draining"
        assert data["pending_tasks"] == 0
        assert data["running_tasks"] == 0

    def test_instance_drain_status_success(self, client):
        """Test /instance/drain/status dummy endpoint returns can_remove=True."""
        response = client.get(
            "/v1/instance/drain/status", params={"instance_id": "test-instance"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["can_remove"] is True
        assert data["instance_id"] == "test-instance"

    def test_instance_remove_success(self, client):
        """Test /instance/remove dummy endpoint returns success."""
        request_data = {"instance_id": "test-instance"}

        response = client.post("/v1/instance/remove", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["instance_id"] == "test-instance"

    def test_task_resubmit_success(self, client):
        """Test /task/resubmit dummy endpoint returns success."""
        request_data = {
            "task_id": "task-123",
            "original_instance_id": "instance-1",
        }

        response = client.post("/v1/task/resubmit", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_timeline_get(self, client):
        """Test /timeline GET endpoint."""
        response = client.get("/v1/timeline")

        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
        assert isinstance(data["entries"], list)

    def test_timeline_clear(self, client):
        """Test /timeline/clear POST endpoint."""
        response = client.post("/v1/timeline/clear")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestPlanEndpointErrors:
    """Tests for /plan endpoint error paths."""

    def test_plan_invalid_a_value(self, client, sample_planner_input):
        """Test plan fails with a value out of range."""
        sample_planner_input["a"] = 1.5  # Out of 0-1 range

        response = client.post("/v1/plan", json=sample_planner_input)

        # Pydantic validation catches this
        assert response.status_code == 422

    def test_plan_negative_iterations(self, client, sample_planner_input):
        """Test plan handles negative max_iterations."""
        sample_planner_input["max_iterations"] = -10

        response = client.post("/v1/plan", json=sample_planner_input)

        # Pydantic validation catches this
        assert response.status_code == 422
