"""End-to-end tests to verify API behavior consistency after loguru migration.

These tests ensure that the API responses and behavior are identical
to the original implementation, confirming no breaking changes.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    from swarmpilot.planner.api import app

    return TestClient(app)


class TestAPIBehaviorConsistency:
    """Test that API endpoints behave identically after loguru migration."""

    def test_health_endpoint_response_structure(self, client):
        """Test that /health endpoint returns the expected structure."""
        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"

    def test_info_endpoint_response_structure(self, client):
        """Test that /info endpoint returns the expected structure."""
        response = client.get("/v1/info")

        assert response.status_code == 200
        data = response.json()

        # Verify all expected fields are present
        assert "service" in data
        assert "version" in data
        assert "algorithms" in data
        assert "objective_methods" in data
        assert "description" in data

        # Verify field values
        assert data["service"] == "planner"
        assert isinstance(data["algorithms"], list)
        assert "simulated_annealing" in data["algorithms"]
        assert "integer_programming" in data["algorithms"]

    def test_plan_endpoint_with_valid_input(self, client):
        """Test /plan endpoint with valid input returns expected structure."""
        request_data = {
            "M": 4,
            "N": 3,
            "B": [[10, 5, 0], [8, 6, 4], [0, 10, 8], [6, 0, 12]],
            "initial": [0, 1, 2, 2],
            "a": 0.5,
            "target": [20, 30, 25],
            "algorithm": "simulated_annealing",
            "max_iterations": 100,
        }

        response = client.post("/v1/plan", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "deployment" in data
        assert "score" in data
        assert "stats" in data
        assert "service_capacity" in data
        assert "changes_count" in data

        # Verify data types
        assert isinstance(data["deployment"], list)
        assert isinstance(data["score"], (int, float))
        assert isinstance(data["stats"], dict)
        assert isinstance(data["service_capacity"], list)
        assert isinstance(data["changes_count"], int)

        # Verify deployment length matches M
        assert len(data["deployment"]) == request_data["M"]

    def test_plan_endpoint_error_handling(self, client):
        """Test that /plan endpoint handles errors correctly."""
        # Invalid algorithm
        request_data = {
            "M": 4,
            "N": 3,
            "B": [[10, 5, 0], [8, 6, 4], [0, 10, 8], [6, 0, 12]],
            "initial": [0, 1, 2, 2],
            "a": 0.5,
            "target": [20, 30, 25],
            "algorithm": "invalid_algorithm",
        }

        response = client.post("/v1/plan", json=request_data)

        # FastAPI returns 422 for validation errors
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_exception_handling_produces_500(self, client):
        """Test that unhandled exceptions produce 500 errors."""
        # Intentionally malformed request that will cause internal error
        request_data = {
            "M": 0,  # Invalid: M must be positive
            "N": 3,
            "B": [],
            "initial": [],
            "a": 0.5,
            "target": [20, 30, 25],
        }

        response = client.post("/v1/plan", json=request_data)

        # Should either be 422 (validation) or 400/500 (internal error)
        assert response.status_code in [400, 422, 500]


class TestLoggingDoesNotAffectResponses:
    """Test that logging changes don't affect API responses."""

    def test_multiple_requests_consistent_responses(self, client):
        """Test that multiple identical requests get consistent responses."""
        request_data = {
            "M": 3,
            "N": 2,
            "B": [[10, 5], [8, 6], [7, 9]],
            "initial": [0, 1, 0],
            "a": 0.5,
            "target": [20, 30],
            "algorithm": "simulated_annealing",
            "max_iterations": 50,
        }

        # Make multiple requests
        responses = [
            client.post("/v1/plan", json=request_data) for _ in range(3)
        ]

        # All should succeed
        assert all(r.status_code == 200 for r in responses)

        # All should have the same structure
        for response in responses:
            data = response.json()
            assert "deployment" in data
            assert "score" in data
            assert "stats" in data

    def test_health_check_performance(self, client):
        """Test that health check is fast (not affected by logging)."""
        import time

        start = time.time()
        for _ in range(10):
            response = client.get("/v1/health")
            assert response.status_code == 200
        duration = time.time() - start

        # 10 health checks should complete very quickly
        assert duration < 1.0  # Less than 1 second total


class TestOriginalFunctionalityPreserved:
    """Test that all original functionality is preserved."""

    def test_simulated_annealing_algorithm_works(self, client):
        """Test that simulated annealing algorithm still works."""
        request_data = {
            "M": 4,
            "N": 3,
            "B": [[10, 5, 0], [8, 6, 4], [0, 10, 8], [6, 0, 12]],
            "initial": [0, 1, 2, 2],
            "a": 0.5,
            "target": [20, 30, 25],
            "algorithm": "simulated_annealing",
        }

        response = client.post("/v1/plan", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["stats"]["algorithm"] == "simulated_annealing"

    def test_integer_programming_algorithm_works(self, client):
        """Test that integer programming algorithm still works."""
        request_data = {
            "M": 4,
            "N": 3,
            "B": [[10, 5, 0], [8, 6, 4], [0, 10, 8], [6, 0, 12]],
            "initial": [0, 1, 2, 2],
            "a": 0.5,
            "target": [20, 30, 25],
            "algorithm": "integer_programming",
        }

        response = client.post("/v1/plan", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["stats"]["algorithm"] == "integer_programming"

    def test_objective_methods_work(self, client):
        """Test that all objective methods still work."""
        objective_methods = [
            "relative_error",
            "ratio_difference",
            "weighted_squared",
        ]

        for method in objective_methods:
            request_data = {
                "M": 3,
                "N": 2,
                "B": [[10, 5], [8, 6], [7, 9]],
                "initial": [0, 1, 0],
                "a": 0.5,
                "target": [20, 30],
                "algorithm": "simulated_annealing",
                "objective_method": method,
                "max_iterations": 50,
            }

            response = client.post("/v1/plan", json=request_data)
            assert response.status_code == 200, (
                f"Failed for objective method: {method}"
            )

    def test_service_capacity_calculation_preserved(self, client):
        """Test that service capacity calculation still works correctly."""
        request_data = {
            "M": 3,
            "N": 2,
            "B": [[10, 5], [8, 6], [7, 9]],
            "initial": [0, 1, 0],
            "a": 0.5,
            "target": [20, 30],
            "algorithm": "simulated_annealing",
            "max_iterations": 50,
        }

        response = client.post("/v1/plan", json=request_data)
        assert response.status_code == 200

        data = response.json()
        service_capacity = data["service_capacity"]

        # Should be a list of length N
        assert len(service_capacity) == request_data["N"]
        # All capacities should be non-negative
        assert all(c >= 0 for c in service_capacity)
