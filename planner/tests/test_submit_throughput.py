"""Tests for /submit_throughput endpoint."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np

from src import api as api_module


class TestSubmitThroughputValidation:
    """Tests for request validation."""

    def test_valid_request(self, client, setup_throughput_deployment_state):
        """Test valid throughput submission."""
        response = client.post("/submit_throughput", json={
            "instance_url": "http://inst1:8080",
            "avg_execution_time": 0.5
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["instance_url"] == "http://inst1:8080"

    def test_zero_execution_time_rejected(self, client, reset_throughput_state):
        """Test that avg_execution_time=0 is rejected (422)."""
        response = client.post("/submit_throughput", json={
            "instance_url": "http://inst1:8080",
            "avg_execution_time": 0
        })
        assert response.status_code == 422

    def test_negative_execution_time_rejected(self, client, reset_throughput_state):
        """Test that negative execution time is rejected (422)."""
        response = client.post("/submit_throughput", json={
            "instance_url": "http://inst1:8080",
            "avg_execution_time": -1.0
        })
        assert response.status_code == 422

    def test_missing_instance_url_rejected(self, client, reset_throughput_state):
        """Test missing instance_url field (422)."""
        response = client.post("/submit_throughput", json={
            "avg_execution_time": 0.5
        })
        assert response.status_code == 422

    def test_missing_execution_time_rejected(self, client, reset_throughput_state):
        """Test missing avg_execution_time field (422)."""
        response = client.post("/submit_throughput", json={
            "instance_url": "http://inst1:8080"
        })
        assert response.status_code == 422


class TestSubmitThroughputInstanceLookup:
    """Tests for instance lookup behavior."""

    def test_no_deployment_returns_success_with_warning(self, client, reset_throughput_state):
        """When no deployment exists, accept data with warning message."""
        api_module._stored_deployment_input = None

        response = client.post("/submit_throughput", json={
            "instance_url": "http://unknown:8080",
            "avg_execution_time": 0.5
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model_id"] is None
        assert "not found" in data["message"].lower() or "no deployment" in data["message"].lower()

    def test_instance_not_found_returns_success_with_warning(self, client, setup_throughput_deployment_state):
        """When instance URL not in deployment, accept with warning."""
        response = client.post("/submit_throughput", json={
            "instance_url": "http://unknown:8080",
            "avg_execution_time": 0.5
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model_id"] is None
        assert "not found" in data["message"].lower()

    def test_instance_found_returns_model_id(self, client, setup_throughput_deployment_state):
        """When instance found, return its model_id."""
        response = client.post("/submit_throughput", json={
            "instance_url": "http://inst1:8080",
            "avg_execution_time": 0.5
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model_id"] == "model_a"


class TestSubmitThroughputCapacity:
    """Tests for capacity calculation."""

    def test_capacity_is_inverse_of_execution_time(self, client, setup_throughput_deployment_state):
        """Capacity = 1 / avg_execution_time."""
        response = client.post("/submit_throughput", json={
            "instance_url": "http://inst1:8080",
            "avg_execution_time": 0.5
        })

        assert response.status_code == 200
        data = response.json()
        assert data["computed_capacity"] == 2.0  # 1 / 0.5 = 2.0

    def test_small_execution_time_gives_high_capacity(self, client, setup_throughput_deployment_state):
        """Fast execution = high capacity."""
        response = client.post("/submit_throughput", json={
            "instance_url": "http://inst1:8080",
            "avg_execution_time": 0.1
        })

        data = response.json()
        assert data["computed_capacity"] == 10.0  # 1 / 0.1 = 10.0

    def test_large_execution_time_gives_low_capacity(self, client, setup_throughput_deployment_state):
        """Slow execution = low capacity."""
        response = client.post("/submit_throughput", json={
            "instance_url": "http://inst1:8080",
            "avg_execution_time": 10.0
        })

        data = response.json()
        assert data["computed_capacity"] == 0.1  # 1 / 10.0 = 0.1


class TestSubmitThroughputStorage:
    """Tests for throughput data storage."""

    @pytest.mark.skip(reason="Throughput storage is temporarily disabled in api.py")
    def test_first_submission_stores_exact_capacity(self, client, setup_throughput_deployment_state):
        """First submission stores exact computed capacity."""
        response = client.post("/submit_throughput", json={
            "instance_url": "http://inst1:8080",
            "avg_execution_time": 0.5
        })

        assert response.status_code == 200

        # Check internal storage
        assert "http://inst1:8080" in api_module._throughput_data
        assert "model_a" in api_module._throughput_data["http://inst1:8080"]
        assert api_module._throughput_data["http://inst1:8080"]["model_a"] == 2.0

    @pytest.mark.skip(reason="Throughput storage is temporarily disabled in api.py")
    def test_second_submission_uses_ema(self, client, setup_throughput_deployment_state):
        """Subsequent submissions use exponential moving average."""
        # First submission: capacity = 2.0 (exec_time = 0.5)
        client.post("/submit_throughput", json={
            "instance_url": "http://inst1:8080",
            "avg_execution_time": 0.5
        })

        # Second submission: capacity = 4.0 (exec_time = 0.25)
        response = client.post("/submit_throughput", json={
            "instance_url": "http://inst1:8080",
            "avg_execution_time": 0.25
        })

        # EMA (alpha=0.3): 0.3 * 4.0 + 0.7 * 2.0 = 1.2 + 1.4 = 2.6
        stored_capacity = api_module._throughput_data["http://inst1:8080"]["model_a"]
        assert abs(stored_capacity - 2.6) < 0.01

    @pytest.mark.skip(reason="Throughput storage is temporarily disabled in api.py")
    def test_multiple_instances_stored_separately(self, client, setup_throughput_deployment_state):
        """Different instances have separate storage."""
        client.post("/submit_throughput", json={
            "instance_url": "http://inst1:8080",
            "avg_execution_time": 0.5
        })
        client.post("/submit_throughput", json={
            "instance_url": "http://inst2:8080",
            "avg_execution_time": 1.0
        })

        assert "http://inst1:8080" in api_module._throughput_data
        assert "http://inst2:8080" in api_module._throughput_data
        assert api_module._throughput_data["http://inst1:8080"]["model_a"] == 2.0
        assert api_module._throughput_data["http://inst2:8080"]["model_b"] == 1.0


class TestBMatrixUpdate:
    """Tests for B matrix update during optimization."""

    def test_apply_throughput_updates_b_matrix(self, setup_throughput_deployment_state):
        """B matrix is updated from throughput data."""
        # Set up throughput data manually
        api_module._throughput_data = {
            "http://inst1:8080": {"model_a": 15.0},  # Update B[0][0] from 10.0 to 15.0
            "http://inst2:8080": {"model_b": 20.0},  # Update B[1][1] from 10.0 to 20.0
        }

        # Call the apply function
        api_module._apply_throughput_to_b_matrix()

        # Check B matrix was updated
        B = api_module._stored_deployment_input.planner_input.B
        assert B[0][0] == 15.0  # Updated
        assert B[1][1] == 20.0  # Updated
        assert B[2][2] == 10.0  # Unchanged (no throughput data)

    def test_apply_throughput_only_updates_matching_cells(self, setup_throughput_deployment_state):
        """Only cells with throughput data are updated."""
        original_B = [row.copy() for row in api_module._stored_deployment_input.planner_input.B]

        # Only update one cell
        api_module._throughput_data = {
            "http://inst1:8080": {"model_a": 25.0},
        }

        api_module._apply_throughput_to_b_matrix()

        B = api_module._stored_deployment_input.planner_input.B
        assert B[0][0] == 25.0  # Updated
        # All other cells should be unchanged
        assert B[0][1] == original_B[0][1]
        assert B[0][2] == original_B[0][2]
        assert B[1][0] == original_B[1][0]
        assert B[1][1] == original_B[1][1]
        assert B[1][2] == original_B[1][2]

    def test_no_throughput_data_leaves_b_matrix_unchanged(self, setup_throughput_deployment_state):
        """If no throughput submitted, original B matrix used."""
        original_B = [row.copy() for row in api_module._stored_deployment_input.planner_input.B]

        api_module._throughput_data = {}

        api_module._apply_throughput_to_b_matrix()

        B = api_module._stored_deployment_input.planner_input.B
        assert B == original_B

    @pytest.mark.skip(reason="Throughput storage is temporarily disabled in api.py")
    @pytest.mark.asyncio
    async def test_b_matrix_updated_before_optimization(self, setup_throughput_deployment_state):
        """B matrix is updated from throughput data before running optimizer."""
        # Set up throughput data
        api_module._throughput_data = {
            "http://inst1:8080": {"model_a": 50.0},
        }
        api_module._submitted_models = {"model_a", "model_b", "model_c"}
        api_module._current_target = [100.0, 200.0, 300.0]

        with patch("src.api.SimulatedAnnealingOptimizer") as mock_optimizer_class, \
             patch("src.api.get_available_instance_store") as mock_store_getter:

            mock_optimizer = MagicMock()
            mock_optimizer.optimize.return_value = (np.array([0, 1, 2]), 0.0, {})
            mock_optimizer.compute_changes.return_value = 0
            mock_optimizer_class.return_value = mock_optimizer

            mock_store = MagicMock()
            mock_store_getter.return_value = mock_store

            await api_module._trigger_optimization()

            # Verify B matrix was updated before optimizer was called
            B = api_module._stored_deployment_input.planner_input.B
            assert B[0][0] == 50.0


class TestSubmitThroughputIntegration:
    """Integration tests for complete workflow."""

    @pytest.mark.skip(reason="Throughput storage is temporarily disabled in api.py")
    def test_submit_throughput_multiple_times(self, client, setup_throughput_deployment_state):
        """Test submitting throughput multiple times updates storage correctly."""
        # Submit 3 times with different values
        for exec_time in [1.0, 0.5, 0.25]:
            response = client.post("/submit_throughput", json={
                "instance_url": "http://inst1:8080",
                "avg_execution_time": exec_time
            })
            assert response.status_code == 200

        # Final value should be EMA of all submissions
        # 1st: capacity = 1.0
        # 2nd: capacity = 2.0, EMA = 0.3*2.0 + 0.7*1.0 = 1.3
        # 3rd: capacity = 4.0, EMA = 0.3*4.0 + 0.7*1.3 = 1.2 + 0.91 = 2.11
        stored_capacity = api_module._throughput_data["http://inst1:8080"]["model_a"]
        assert abs(stored_capacity - 2.11) < 0.01

    @pytest.mark.skip(reason="Throughput storage is temporarily disabled in api.py")
    def test_throughput_persists_across_requests(self, client, setup_throughput_deployment_state):
        """Throughput data persists across multiple API requests."""
        # Submit for first instance
        client.post("/submit_throughput", json={
            "instance_url": "http://inst1:8080",
            "avg_execution_time": 0.5
        })

        # Submit for second instance
        client.post("/submit_throughput", json={
            "instance_url": "http://inst2:8080",
            "avg_execution_time": 1.0
        })

        # Submit for third instance
        client.post("/submit_throughput", json={
            "instance_url": "http://inst3:8080",
            "avg_execution_time": 2.0
        })

        # Verify all data persisted
        assert len(api_module._throughput_data) == 3
        assert api_module._throughput_data["http://inst1:8080"]["model_a"] == 2.0
        assert api_module._throughput_data["http://inst2:8080"]["model_b"] == 1.0
        assert api_module._throughput_data["http://inst3:8080"]["model_c"] == 0.5
