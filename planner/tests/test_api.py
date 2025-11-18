"""Tests for FastAPI endpoints."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import numpy as np

from src.api import app


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestInfoEndpoint:
    """Tests for /info endpoint."""

    def test_service_info(self, client):
        """Test service info returns correct metadata."""
        response = client.get("/info")
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

        with patch("src.api.SimulatedAnnealingOptimizer") as mock_optimizer_class:
            # Mock optimizer instance and its methods
            mock_optimizer = MagicMock()
            mock_optimizer.optimize.return_value = (
                np.array([0, 1, 1, 2]),  # deployment
                0.0667,  # score
                {"algorithm": "simulated_annealing", "iterations": 10}  # stats
            )
            mock_optimizer.compute_service_capacity.return_value = np.array([10.0, 16.0, 12.0])
            mock_optimizer.compute_changes.return_value = 1
            mock_optimizer_class.return_value = mock_optimizer

            response = client.post("/plan", json=sample_planner_input)

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

        with patch("src.api.IntegerProgrammingOptimizer") as mock_optimizer_class:
            mock_optimizer = MagicMock()
            mock_optimizer.optimize.return_value = (
                np.array([0, 1, 1, 2]),
                0.05,
                {"algorithm": "integer_programming", "solver_status": "optimal"}
            )
            mock_optimizer.compute_service_capacity.return_value = np.array([10.0, 16.0, 12.0])
            mock_optimizer.compute_changes.return_value = 1
            mock_optimizer_class.return_value = mock_optimizer

            response = client.post("/plan", json=sample_planner_input)

            assert response.status_code == 200
            data = response.json()
            assert data["score"] == 0.05
            assert data["stats"]["algorithm"] == "integer_programming"

    def test_plan_invalid_input(self, client, sample_planner_input):
        """Test /plan rejects invalid input."""
        sample_planner_input["M"] = -1  # Invalid
        response = client.post("/plan", json=sample_planner_input)
        assert response.status_code == 422  # Validation error

    def test_plan_unknown_algorithm(self, client, sample_planner_input):
        """Test /plan rejects unknown algorithm."""
        sample_planner_input["algorithm"] = "unknown_algorithm"
        response = client.post("/plan", json=sample_planner_input)
        assert response.status_code == 422  # Validation error

    def test_plan_optimization_failure(self, client, sample_planner_input):
        """Test /plan handles optimization failures."""
        sample_planner_input["max_iterations"] = 10

        with patch("src.api.SimulatedAnnealingOptimizer") as mock_optimizer_class:
            mock_optimizer_class.side_effect = ValueError("Optimization failed")

            response = client.post("/plan", json=sample_planner_input)

            assert response.status_code == 400
            assert "Invalid input" in response.json()["detail"]


class TestDeployEndpoint:
    """Tests for /deploy endpoint."""

    @pytest.mark.asyncio
    async def test_deploy_success(self, client, sample_deployment_input):
        """Test successful deployment with optimization."""
        sample_deployment_input["planner_input"]["max_iterations"] = 10
        sample_deployment_input["planner_input"]["verbose"] = False

        with patch("src.api.SimulatedAnnealingOptimizer") as mock_optimizer_class, \
             patch("src.api.InstanceDeployer") as mock_deployer_class:

            # Mock optimizer
            mock_optimizer = MagicMock()
            mock_optimizer.optimize.return_value = (
                np.array([0, 1, 1, 2]),
                0.0667,
                {"algorithm": "simulated_annealing", "iterations": 10}
            )
            mock_optimizer.compute_service_capacity.return_value = np.array([10.0, 16.0, 12.0])
            mock_optimizer.compute_changes.return_value = 1
            mock_optimizer_class.return_value = mock_optimizer

            # Mock deployer
            from src.models import DeploymentStatus
            mock_deployer = MagicMock()
            mock_deployer.deploy_to_instances = AsyncMock(return_value=[
                DeploymentStatus(
                    instance_index=i,
                    endpoint=f"http://instance-{i+1}:8080",
                    target_model=f"model_{[0,1,1,2][i]}",
                    previous_model=f"model_{[0,1,2,2][i]}",
                    success=True,
                    error_message=None,
                    deployment_time=0.0 if i in [0, 1] else 2.0
                )
                for i in range(4)
            ])
            mock_deployer_class.return_value = mock_deployer

            response = client.post("/deploy", json=sample_deployment_input)

            assert response.status_code == 200
            data = response.json()
            assert data["deployment"] == [0, 1, 1, 2]
            assert data["score"] == 0.0667
            assert data["success"] is True
            assert len(data["failed_instances"]) == 0
            assert len(data["deployment_status"]) == 4

            # Verify deployer was called
            mock_deployer.deploy_to_instances.assert_called_once()

    @pytest.mark.asyncio
    async def test_deploy_partial_failure(self, client, sample_deployment_input):
        """Test deployment with some instance failures."""
        sample_deployment_input["planner_input"]["max_iterations"] = 10
        sample_deployment_input["planner_input"]["verbose"] = False

        with patch("src.api.SimulatedAnnealingOptimizer") as mock_optimizer_class, \
             patch("src.api.InstanceDeployer") as mock_deployer_class:

            # Mock optimizer
            mock_optimizer = MagicMock()
            mock_optimizer.optimize.return_value = (
                np.array([0, 1, 1, 2]),
                0.0667,
                {"algorithm": "simulated_annealing"}
            )
            mock_optimizer.compute_service_capacity.return_value = np.array([10.0, 16.0, 12.0])
            mock_optimizer.compute_changes.return_value = 1
            mock_optimizer_class.return_value = mock_optimizer

            # Mock deployer with one failure
            from src.models import DeploymentStatus
            mock_deployer = MagicMock()
            statuses = [
                DeploymentStatus(
                    instance_index=i,
                    endpoint=f"http://instance-{i+1}:8080",
                    target_model=f"model_{[0,1,1,2][i]}",
                    previous_model=f"model_{[0,1,2,2][i]}",
                    success=(i != 2),  # Instance 2 fails
                    error_message="Connection timeout" if i == 2 else None,
                    deployment_time=0.0
                )
                for i in range(4)
            ]
            mock_deployer.deploy_to_instances = AsyncMock(return_value=statuses)
            mock_deployer_class.return_value = mock_deployer

            response = client.post("/deploy", json=sample_deployment_input)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert 2 in data["failed_instances"]
            assert data["deployment_status"][2]["success"] is False
            assert "Connection timeout" in data["deployment_status"][2]["error_message"]

    def test_deploy_invalid_instances_count(self, client, sample_deployment_input):
        """Test deploy rejects mismatched instance count."""
        sample_deployment_input["instances"] = sample_deployment_input["instances"][:2]
        # M is still 4 but only 2 instances provided

        response = client.post("/deploy", json=sample_deployment_input)
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_deploy_model_mapping(self, client, sample_deployment_input):
        """Test model name to ID mapping works correctly."""
        sample_deployment_input["planner_input"]["max_iterations"] = 10
        sample_deployment_input["planner_input"]["verbose"] = False

        # Set instances to have non-sequential model names
        sample_deployment_input["instances"][0]["current_model"] = "bert_large"
        sample_deployment_input["instances"][1]["current_model"] = "gpt_small"
        sample_deployment_input["instances"][2]["current_model"] = "bert_large"
        sample_deployment_input["instances"][3]["current_model"] = "t5_base"

        with patch("src.api.SimulatedAnnealingOptimizer") as mock_optimizer_class, \
             patch("src.api.InstanceDeployer") as mock_deployer_class:

            # Mock optimizer
            mock_optimizer = MagicMock()
            mock_optimizer.optimize.return_value = (
                np.array([0, 1, 0, 2]),  # IDs: bert_large=0, gpt_small=1, t5_base=2
                0.05,
                {"algorithm": "simulated_annealing"}
            )
            mock_optimizer.compute_service_capacity.return_value = np.array([10.0, 5.0, 8.0])
            mock_optimizer.compute_changes.return_value = 0
            mock_optimizer_class.return_value = mock_optimizer

            # Mock deployer
            from src.models import DeploymentStatus
            mock_deployer = MagicMock()
            mock_deployer.deploy_to_instances = AsyncMock(return_value=[
                DeploymentStatus(
                    instance_index=i,
                    endpoint=f"http://instance-{i+1}:8080",
                    target_model=["bert_large", "gpt_small", "bert_large", "t5_base"][i],
                    previous_model=["bert_large", "gpt_small", "bert_large", "t5_base"][i],
                    success=True,
                    error_message=None,
                    deployment_time=0.0
                )
                for i in range(4)
            ])
            mock_deployer_class.return_value = mock_deployer

            response = client.post("/deploy", json=sample_deployment_input)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

            # Verify optimizer was called with correct mapped initial state
            call_args = mock_optimizer_class.call_args
            initial = call_args[1]["initial"]
            # bert_large=0, gpt_small=1, t5_base=2
            assert list(initial) == [0, 1, 0, 2]



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
                "hardware_name": "nvidia-a100"
            }
        }

        response = client.post("/instance/register", json=request_data)

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
            "platform_info": {}
        }

        response = client.post("/instance/register", json=request_data)

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
            "platform_info": {}
        }
        response1 = client.post("/instance/register", json=request_data1)
        assert response1.status_code == 200

        # Register second instance for same model
        request_data2 = {
            "instance_id": "instance-2",
            "model_id": "shared_model",
            "endpoint": "http://instance2:8080",
            "platform_info": {}
        }
        response2 = client.post("/instance/register", json=request_data2)
        assert response2.status_code == 200

        data = response2.json()
        assert data["success"] is True

    def test_register_instance_missing_required_fields(self, client):
        """Test registration fails with missing required fields."""
        # Missing model_id
        request_data = {
            "instance_id": "test-instance",
            "endpoint": "http://test:8080"
        }

        response = client.post("/instance/register", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_register_instance_missing_endpoint(self, client):
        """Test registration fails without endpoint."""
        request_data = {
            "instance_id": "test-instance",
            "model_id": "model_0"
        }

        response = client.post("/instance/register", json=request_data)

        assert response.status_code == 422  # Validation error
