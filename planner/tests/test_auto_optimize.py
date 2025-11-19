"""Tests for auto-optimization feature."""

import pytest
import asyncio
import time
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import numpy as np

from src.api import app
from src import api as api_module


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def reset_global_state():
    """Reset global state before and after each test."""
    # Save original state
    original_mapping = api_module._stored_model_mapping
    original_reverse = api_module._stored_reverse_mapping
    original_target = api_module._current_target
    original_submitted = api_module._submitted_models.copy() if api_module._submitted_models else set()
    original_last_update = api_module._last_target_update
    original_running = api_module._auto_optimize_running
    original_deployment_input = api_module._stored_deployment_input

    yield

    # Restore original state
    api_module._stored_model_mapping = original_mapping
    api_module._stored_reverse_mapping = original_reverse
    api_module._current_target = original_target
    api_module._submitted_models = original_submitted
    api_module._last_target_update = original_last_update
    api_module._auto_optimize_running = original_running
    api_module._stored_deployment_input = original_deployment_input


@pytest.fixture
def setup_model_mapping(reset_global_state):
    """Set up model mapping for testing submit_target."""
    api_module._stored_model_mapping = {"model_a": 0, "model_b": 1, "model_c": 2}
    api_module._stored_reverse_mapping = {0: "model_a", 1: "model_b", 2: "model_c"}
    api_module._current_target = [0.0, 0.0, 0.0]
    api_module._submitted_models = set()
    api_module._last_target_update = 0.0


class TestConfigAutoOptimize:
    """Tests for auto-optimization configuration."""

    def test_config_default_values(self):
        """Test default configuration values."""
        from src.config import PlannerConfig

        with patch.dict('os.environ', {}, clear=True):
            config = PlannerConfig()
            assert config.auto_optimize_enabled is False
            assert config.auto_optimize_interval == 60.0
            assert config.auto_optimize_cooldown == 5.0

    def test_config_enabled_true(self):
        """Test AUTO_OPTIMIZE_ENABLED=true."""
        from src.config import PlannerConfig

        with patch.dict('os.environ', {'AUTO_OPTIMIZE_ENABLED': 'true'}):
            config = PlannerConfig()
            assert config.auto_optimize_enabled is True

    def test_config_enabled_yes(self):
        """Test AUTO_OPTIMIZE_ENABLED=yes."""
        from src.config import PlannerConfig

        with patch.dict('os.environ', {'AUTO_OPTIMIZE_ENABLED': 'yes'}):
            config = PlannerConfig()
            assert config.auto_optimize_enabled is True

    def test_config_enabled_1(self):
        """Test AUTO_OPTIMIZE_ENABLED=1."""
        from src.config import PlannerConfig

        with patch.dict('os.environ', {'AUTO_OPTIMIZE_ENABLED': '1'}):
            config = PlannerConfig()
            assert config.auto_optimize_enabled is True

    def test_config_custom_interval(self):
        """Test custom AUTO_OPTIMIZE_INTERVAL."""
        from src.config import PlannerConfig

        with patch.dict('os.environ', {'AUTO_OPTIMIZE_INTERVAL': '30.0'}):
            config = PlannerConfig()
            assert config.auto_optimize_interval == 30.0

    def test_config_custom_cooldown(self):
        """Test custom AUTO_OPTIMIZE_COOLDOWN."""
        from src.config import PlannerConfig

        with patch.dict('os.environ', {'AUTO_OPTIMIZE_COOLDOWN': '10.0'}):
            config = PlannerConfig()
            assert config.auto_optimize_cooldown == 10.0

    def test_config_validation_invalid_interval(self):
        """Test validation rejects invalid interval."""
        from src.config import PlannerConfig

        with patch.dict('os.environ', {'AUTO_OPTIMIZE_INTERVAL': '0'}):
            config = PlannerConfig()
            with pytest.raises(ValueError, match="AUTO_OPTIMIZE_INTERVAL must be positive"):
                config.validate()

    def test_config_validation_negative_cooldown(self):
        """Test validation rejects negative cooldown."""
        from src.config import PlannerConfig

        with patch.dict('os.environ', {'AUTO_OPTIMIZE_COOLDOWN': '-1'}):
            config = PlannerConfig()
            with pytest.raises(ValueError, match="AUTO_OPTIMIZE_COOLDOWN must be non-negative"):
                config.validate()


class TestSubmitTargetEndpoint:
    """Tests for /submit_target endpoint with tracking."""

    def test_submit_target_no_mapping(self, client, reset_global_state):
        """Test submit_target when no mapping exists."""
        api_module._stored_model_mapping = None

        response = client.post("/submit_target", json={
            "model_id": "model_a",
            "value": 100.0
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "No active mapping" in data["message"]
        assert data["current_target"] is None

    def test_submit_target_unknown_model(self, client, setup_model_mapping):
        """Test submit_target with unknown model_id."""
        response = client.post("/submit_target", json={
            "model_id": "unknown_model",
            "value": 100.0
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "not in current mapping" in data["message"]

    def test_submit_target_success(self, client, setup_model_mapping):
        """Test successful target submission."""
        response = client.post("/submit_target", json={
            "model_id": "model_a",
            "value": 100.0
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["current_target"][0] == 100.0
        assert "1/3 submitted" in data["message"]

    def test_submit_target_tracks_models(self, client, setup_model_mapping):
        """Test that submit_target tracks submitted models."""
        # Submit first model
        client.post("/submit_target", json={"model_id": "model_a", "value": 100.0})
        assert "model_a" in api_module._submitted_models
        assert len(api_module._submitted_models) == 1

        # Submit second model
        client.post("/submit_target", json={"model_id": "model_b", "value": 200.0})
        assert "model_b" in api_module._submitted_models
        assert len(api_module._submitted_models) == 2

    def test_submit_target_updates_timestamp(self, client, setup_model_mapping):
        """Test that submit_target updates last_target_update timestamp."""
        old_time = api_module._last_target_update

        client.post("/submit_target", json={"model_id": "model_a", "value": 100.0})

        assert api_module._last_target_update > old_time
        assert api_module._last_target_update > 0

    def test_submit_target_all_models(self, client, setup_model_mapping):
        """Test submitting targets for all models."""
        # Submit all three models
        client.post("/submit_target", json={"model_id": "model_a", "value": 100.0})
        client.post("/submit_target", json={"model_id": "model_b", "value": 200.0})
        response = client.post("/submit_target", json={"model_id": "model_c", "value": 300.0})

        data = response.json()
        assert len(api_module._submitted_models) == 3
        assert data["current_target"] == [100.0, 200.0, 300.0]
        assert "3/3 submitted" in data["message"]

    def test_submit_target_updates_existing(self, client, setup_model_mapping):
        """Test updating an already submitted model's target."""
        # Submit first time
        client.post("/submit_target", json={"model_id": "model_a", "value": 100.0})

        # Update with new value
        response = client.post("/submit_target", json={"model_id": "model_a", "value": 150.0})

        data = response.json()
        assert data["current_target"][0] == 150.0
        # Model should still only be counted once
        assert len(api_module._submitted_models) == 1


class TestGetTargetEndpoint:
    """Tests for /target endpoint."""

    def test_get_target_no_mapping(self, client, reset_global_state):
        """Test get_target when no mapping exists."""
        api_module._stored_model_mapping = None
        api_module._current_target = None

        response = client.get("/target")

        assert response.status_code == 200
        data = response.json()
        assert data["target"] is None
        assert data["model_mapping"] is None

    def test_get_target_with_data(self, client, setup_model_mapping):
        """Test get_target returns current state."""
        api_module._current_target = [10.0, 20.0, 30.0]

        response = client.get("/target")

        assert response.status_code == 200
        data = response.json()
        assert data["target"] == [10.0, 20.0, 30.0]
        assert data["model_mapping"] == {"model_a": 0, "model_b": 1, "model_c": 2}
        assert data["reverse_mapping"] == {"0": "model_a", "1": "model_b", "2": "model_c"}


class TestAutoOptimizeLoop:
    """Tests for auto-optimization loop logic."""

    @pytest.mark.asyncio
    async def test_auto_optimize_loop_skips_when_disabled(self, reset_global_state):
        """Test loop skips when auto-optimize is disabled."""
        with patch.object(api_module.config, 'auto_optimize_enabled', False), \
             patch.object(api_module.config, 'auto_optimize_interval', 0.01):

            # Run loop briefly
            task = asyncio.create_task(api_module._auto_optimize_loop())
            await asyncio.sleep(0.05)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # Should not have triggered optimization
            assert api_module._auto_optimize_running is False

    @pytest.mark.asyncio
    async def test_auto_optimize_loop_skips_no_mapping(self, reset_global_state):
        """Test loop skips when no model mapping exists."""
        api_module._stored_model_mapping = None
        api_module._stored_deployment_input = None

        with patch.object(api_module.config, 'auto_optimize_enabled', True), \
             patch.object(api_module.config, 'auto_optimize_interval', 0.01):

            task = asyncio.create_task(api_module._auto_optimize_loop())
            await asyncio.sleep(0.05)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            assert api_module._auto_optimize_running is False

    @pytest.mark.asyncio
    async def test_auto_optimize_loop_skips_not_all_submitted(self, setup_model_mapping):
        """Test loop skips when not all models have submitted."""
        # Only submit 2 of 3 models
        api_module._submitted_models = {"model_a", "model_b"}
        api_module._last_target_update = time.time()

        # Mock deployment input
        mock_input = MagicMock()
        api_module._stored_deployment_input = mock_input

        with patch.object(api_module.config, 'auto_optimize_enabled', True), \
             patch.object(api_module.config, 'auto_optimize_interval', 0.01), \
             patch.object(api_module.config, 'auto_optimize_cooldown', 0):

            task = asyncio.create_task(api_module._auto_optimize_loop())
            await asyncio.sleep(0.05)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # Optimization should not have been triggered
            assert api_module._auto_optimize_running is False

    @pytest.mark.asyncio
    async def test_auto_optimize_loop_skips_during_cooldown(self, setup_model_mapping):
        """Test loop skips during cooldown period."""
        api_module._submitted_models = {"model_a", "model_b", "model_c"}
        api_module._last_target_update = time.time()  # Just updated

        mock_input = MagicMock()
        api_module._stored_deployment_input = mock_input

        with patch.object(api_module.config, 'auto_optimize_enabled', True), \
             patch.object(api_module.config, 'auto_optimize_interval', 0.01), \
             patch.object(api_module.config, 'auto_optimize_cooldown', 100):  # Long cooldown

            task = asyncio.create_task(api_module._auto_optimize_loop())
            await asyncio.sleep(0.05)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            assert api_module._auto_optimize_running is False

    @pytest.mark.asyncio
    async def test_auto_optimize_loop_skips_when_running(self, setup_model_mapping):
        """Test loop skips when optimization is already running."""
        api_module._submitted_models = {"model_a", "model_b", "model_c"}
        api_module._last_target_update = time.time() - 10
        api_module._auto_optimize_running = True  # Already running

        mock_input = MagicMock()
        api_module._stored_deployment_input = mock_input

        with patch.object(api_module.config, 'auto_optimize_enabled', True), \
             patch.object(api_module.config, 'auto_optimize_interval', 0.01), \
             patch.object(api_module.config, 'auto_optimize_cooldown', 0), \
             patch.object(api_module, '_trigger_optimization') as mock_trigger:

            task = asyncio.create_task(api_module._auto_optimize_loop())
            await asyncio.sleep(0.05)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # Should not call trigger when already running
            mock_trigger.assert_not_called()


class TestTriggerOptimization:
    """Tests for _trigger_optimization function."""

    @pytest.mark.asyncio
    async def test_trigger_optimization_sets_running_flag(self, setup_model_mapping):
        """Test that trigger_optimization sets and clears running flag."""
        # Set up deployment input
        from src.models import DeploymentInput, PlannerInput, InstanceInfo

        mock_input = DeploymentInput(
            instances=[
                InstanceInfo(endpoint="http://inst1:8080", current_model="model_a"),
                InstanceInfo(endpoint="http://inst2:8080", current_model="model_b"),
                InstanceInfo(endpoint="http://inst3:8080", current_model="model_c"),
            ],
            planner_input=PlannerInput(
                M=3, N=3,
                B=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                a=0.5,
                target=[100.0, 200.0, 300.0],
                algorithm="simulated_annealing",
                max_iterations=10,
                verbose=False
            ),
            scheduler_mapping={"model_a": "http://scheduler:8100"}
        )
        api_module._stored_deployment_input = mock_input
        api_module._current_target = [100.0, 200.0, 300.0]
        api_module._submitted_models = {"model_a", "model_b", "model_c"}

        with patch("src.api.SimulatedAnnealingOptimizer") as mock_optimizer_class, \
             patch("src.api.get_available_instance_store") as mock_store_getter:

            # Mock optimizer
            mock_optimizer = MagicMock()
            mock_optimizer.optimize.return_value = (
                np.array([0, 1, 2]),
                0.05,
                {"algorithm": "simulated_annealing"}
            )
            mock_optimizer.compute_changes.return_value = 0
            mock_optimizer_class.return_value = mock_optimizer

            # Mock instance store
            mock_store = MagicMock()
            mock_store.fetch_one_available_instance = AsyncMock(return_value=None)
            mock_store_getter.return_value = mock_store

            # Run trigger
            assert api_module._auto_optimize_running is False
            await api_module._trigger_optimization()

            # Should have cleared running flag
            assert api_module._auto_optimize_running is False
            # Should have cleared submitted models
            assert len(api_module._submitted_models) == 0

    @pytest.mark.asyncio
    async def test_trigger_optimization_clears_submitted_on_success(self, setup_model_mapping):
        """Test that successful optimization clears submitted models."""
        from src.models import DeploymentInput, PlannerInput, InstanceInfo

        mock_input = DeploymentInput(
            instances=[
                InstanceInfo(endpoint="http://inst1:8080", current_model="model_a"),
            ],
            planner_input=PlannerInput(
                M=1, N=3,
                B=[[10.0, 10.0, 10.0]],
                a=1.0,
                target=[100.0, 200.0, 300.0],
                algorithm="simulated_annealing",
                max_iterations=10,
                verbose=False
            ),
            scheduler_mapping={}
        )
        api_module._stored_deployment_input = mock_input
        api_module._current_target = [100.0, 200.0, 300.0]
        api_module._submitted_models = {"model_a", "model_b", "model_c"}

        with patch("src.api.SimulatedAnnealingOptimizer") as mock_optimizer_class, \
             patch("src.api.get_available_instance_store") as mock_store_getter:

            mock_optimizer = MagicMock()
            mock_optimizer.optimize.return_value = (np.array([0]), 0.0, {})
            mock_optimizer.compute_changes.return_value = 0
            mock_optimizer_class.return_value = mock_optimizer

            mock_store = MagicMock()
            mock_store_getter.return_value = mock_store

            await api_module._trigger_optimization()

            # Should clear submitted models after success
            assert len(api_module._submitted_models) == 0

    @pytest.mark.asyncio
    async def test_trigger_optimization_keeps_submitted_on_failure(self, setup_model_mapping):
        """Test that failed optimization keeps submitted models for retry."""
        from src.models import DeploymentInput, PlannerInput, InstanceInfo

        mock_input = DeploymentInput(
            instances=[
                InstanceInfo(endpoint="http://inst1:8080", current_model="model_a"),
            ],
            planner_input=PlannerInput(
                M=1, N=3,
                B=[[10.0, 10.0, 10.0]],
                a=1.0,
                target=[100.0, 200.0, 300.0],
                algorithm="simulated_annealing",
                max_iterations=10,
                verbose=False
            ),
            scheduler_mapping={}
        )
        api_module._stored_deployment_input = mock_input
        api_module._current_target = [100.0, 200.0, 300.0]
        api_module._submitted_models = {"model_a", "model_b", "model_c"}

        with patch("src.api.SimulatedAnnealingOptimizer") as mock_optimizer_class:
            # Make optimization fail
            mock_optimizer_class.side_effect = Exception("Optimization failed")

            await api_module._trigger_optimization()

            # Should keep submitted models for retry
            assert len(api_module._submitted_models) == 3
            # But running flag should be cleared
            assert api_module._auto_optimize_running is False


class TestDeployStoresState:
    """Tests for /deploy storing state for auto-optimization."""

    @pytest.mark.asyncio
    async def test_deploy_stores_deployment_input(self, client, sample_deployment_input, reset_global_state):
        """Test that /deploy stores deployment input for auto-optimization."""
        sample_deployment_input["planner_input"]["max_iterations"] = 10
        sample_deployment_input["planner_input"]["verbose"] = False
        # Remove fields that are not valid or calculated by the endpoint
        sample_deployment_input.pop("scheduler_url", None)
        sample_deployment_input["planner_input"].pop("initial", None)  # Calculated from instances

        with patch("src.api.SimulatedAnnealingOptimizer") as mock_optimizer_class, \
             patch("src.api.InstanceDeployer") as mock_deployer_class:

            mock_optimizer = MagicMock()
            mock_optimizer.optimize.return_value = (
                np.array([0, 1, 1, 2]),
                0.0667,
                {}
            )
            mock_optimizer.compute_service_capacity.return_value = np.array([10.0, 16.0, 12.0])
            mock_optimizer.compute_changes.return_value = 1
            mock_optimizer_class.return_value = mock_optimizer

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
                    deployment_time=0.0
                )
                for i in range(4)
            ])
            mock_deployer_class.return_value = mock_deployer

            response = client.post("/deploy", json=sample_deployment_input)

            assert response.status_code == 200
            # Check that state was stored
            assert api_module._stored_deployment_input is not None
            assert api_module._stored_model_mapping is not None
            assert api_module._current_target is not None
            assert len(api_module._submitted_models) == 0  # Reset after deploy

    @pytest.mark.asyncio
    async def test_deploy_resets_submitted_models(self, client, sample_deployment_input, reset_global_state):
        """Test that /deploy resets submitted models."""
        # Pre-populate submitted models
        api_module._submitted_models = {"old_model_1", "old_model_2"}

        sample_deployment_input["planner_input"]["max_iterations"] = 10
        sample_deployment_input["planner_input"]["verbose"] = False
        # Remove fields that are not valid or calculated by the endpoint
        sample_deployment_input.pop("scheduler_url", None)
        sample_deployment_input["planner_input"].pop("initial", None)  # Calculated from instances

        with patch("src.api.SimulatedAnnealingOptimizer") as mock_optimizer_class, \
             patch("src.api.InstanceDeployer") as mock_deployer_class:

            mock_optimizer = MagicMock()
            mock_optimizer.optimize.return_value = (np.array([0, 1, 1, 2]), 0.05, {})
            mock_optimizer.compute_service_capacity.return_value = np.array([10.0, 16.0, 12.0])
            mock_optimizer.compute_changes.return_value = 0
            mock_optimizer_class.return_value = mock_optimizer

            from src.models import DeploymentStatus
            mock_deployer = MagicMock()
            mock_deployer.deploy_to_instances = AsyncMock(return_value=[
                DeploymentStatus(
                    instance_index=i,
                    endpoint=f"http://instance-{i+1}:8080",
                    target_model="model_0",
                    previous_model="model_0",
                    success=True,
                    error_message=None,
                    deployment_time=0.0
                )
                for i in range(4)
            ])
            mock_deployer_class.return_value = mock_deployer

            client.post("/deploy", json=sample_deployment_input)

            # Should have cleared old submitted models
            assert len(api_module._submitted_models) == 0


class TestIntegrationAutoOptimize:
    """Integration tests for complete auto-optimization workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, client, sample_deployment_input, reset_global_state):
        """Test complete workflow: deploy -> submit_target x N -> auto-optimize."""
        sample_deployment_input["planner_input"]["max_iterations"] = 10
        sample_deployment_input["planner_input"]["verbose"] = False
        # Remove fields that are not valid or calculated by the endpoint
        sample_deployment_input.pop("scheduler_url", None)
        sample_deployment_input["planner_input"].pop("initial", None)  # Calculated from instances

        # Step 1: Deploy to set up mapping
        with patch("src.api.SimulatedAnnealingOptimizer") as mock_optimizer_class, \
             patch("src.api.InstanceDeployer") as mock_deployer_class:

            mock_optimizer = MagicMock()
            mock_optimizer.optimize.return_value = (np.array([0, 1, 1, 2]), 0.05, {})
            mock_optimizer.compute_service_capacity.return_value = np.array([10.0, 16.0, 12.0])
            mock_optimizer.compute_changes.return_value = 0
            mock_optimizer_class.return_value = mock_optimizer

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
                    deployment_time=0.0
                )
                for i in range(4)
            ])
            mock_deployer_class.return_value = mock_deployer

            response = client.post("/deploy", json=sample_deployment_input)
            assert response.status_code == 200

        # Step 2: Submit targets for all models
        models = list(api_module._stored_model_mapping.keys())
        for i, model_id in enumerate(models):
            response = client.post("/submit_target", json={
                "model_id": model_id,
                "value": float((i + 1) * 100)
            })
            assert response.status_code == 200

        # Verify all models submitted
        assert len(api_module._submitted_models) == len(models)

        # Step 3: Check target endpoint shows correct state
        response = client.get("/target")
        data = response.json()
        assert data["target"] is not None
        assert len(data["target"]) == len(models)
