"""Unit tests for PyLet API endpoints.

Tests for:
- PyLetDeploymentService
- PyLet API endpoints (/status, /deploy, /scale, /migrate, /optimize)
- PyLet models
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

pylet = pytest.importorskip("pylet")

from swarmpilot.planner.models.pylet import (  # noqa: E402
    PyLetDeploymentInput,
    PyLetInstanceStatus,
    PyLetOptimizeInput,
    PyLetScaleInput,
)
from swarmpilot.planner.pylet.deployment_service import (  # noqa: E402
    DeploymentServiceResult,
    PyLetDeploymentService,
)
from swarmpilot.planner.pylet.instance_manager import (  # noqa: E402
    ManagedInstance,
    ManagedInstanceStatus,
)


class TestPyLetModels:
    """Tests for PyLet Pydantic models."""

    def test_pylet_instance_status(self):
        """Test PyLetInstanceStatus model."""
        status = PyLetInstanceStatus(
            pylet_id="pylet-123",
            instance_id="inst-123",
            model_id="Qwen/Qwen3-0.6B",
            endpoint="http://localhost:8001",
            status="active",
            error=None,
        )

        assert status.pylet_id == "pylet-123"
        assert status.model_id == "Qwen/Qwen3-0.6B"
        assert status.status == "active"

    def test_pylet_deployment_input(self):
        """Test PyLetDeploymentInput model."""
        input_data = PyLetDeploymentInput(
            target_state={"model-a": 2, "model-b": 1},
            wait_for_ready=True,
            register_with_scheduler=True,
        )

        assert input_data.target_state["model-a"] == 2
        assert input_data.wait_for_ready is True

    def test_pylet_scale_input(self):
        """Test PyLetScaleInput model."""
        input_data = PyLetScaleInput(
            model_id="model-a",
            target_count=3,
            wait_for_ready=False,
        )

        assert input_data.model_id == "model-a"
        assert input_data.target_count == 3

    def test_pylet_scale_input_validation(self):
        """Test that target_count must be non-negative."""
        with pytest.raises(ValueError):
            PyLetScaleInput(
                model_id="model-a",
                target_count=-1,
            )

    def test_pylet_optimize_input(self):
        """Test PyLetOptimizeInput model."""
        input_data = PyLetOptimizeInput(
            target=[0.5, 0.3, 0.2],
            model_ids=["model-a", "model-b", "model-c"],
            B=[[1.0, 0.8, 0.6], [0.9, 1.0, 0.7]],
            a=0.3,
            objective_method="ratio_difference",
            algorithm="simulated_annealing",
        )

        assert len(input_data.target) == 3
        assert len(input_data.model_ids) == 3
        assert input_data.a == 0.3


class TestDeploymentServiceResult:
    """Tests for DeploymentServiceResult."""

    def test_total_added(self):
        """Test total_added property."""
        from swarmpilot.planner.pylet.deployment_executor import ExecutionResult

        mock_result = ExecutionResult(
            success=True,
            added_instances=[MagicMock(), MagicMock()],
            removed_instances=[],
            failed_adds=[],
            failed_removes=[],
        )

        result = DeploymentServiceResult(
            success=True,
            deployment_result=mock_result,
        )

        assert result.total_added == 2

    def test_total_removed(self):
        """Test total_removed property."""
        from swarmpilot.planner.pylet.deployment_executor import ExecutionResult

        mock_result = ExecutionResult(
            success=True,
            added_instances=[],
            removed_instances=["id-1", "id-2", "id-3"],
            failed_adds=[],
            failed_removes=[],
        )

        result = DeploymentServiceResult(
            success=True,
            deployment_result=mock_result,
        )

        assert result.total_removed == 3

    def test_total_migrated(self):
        """Test total_migrated property."""
        from swarmpilot.planner.pylet.migration_executor import MigrationResult

        mock_result = MigrationResult(
            success=True,
            completed=[MagicMock(), MagicMock()],
            failed=[],
            total_duration=10.0,
        )

        result = DeploymentServiceResult(
            success=True,
            migration_result=mock_result,
        )

        assert result.total_migrated == 2


class TestPyLetDeploymentService:
    """Tests for PyLetDeploymentService."""

    def test_init_not_initialized(self):
        """Test service starts not initialized."""
        service = PyLetDeploymentService(
            pylet_head_url="http://localhost:8000",
            scheduler_url="http://localhost:8001",
        )
        assert not service.initialized

    @patch("swarmpilot.planner.pylet.deployment_service.InstanceManager")
    def test_get_current_state(self, mock_manager_class):
        """Test get_current_state method."""
        mock_manager = MagicMock()
        mock_manager.health_check = MagicMock(return_value=True)
        mock_manager_class.return_value = mock_manager

        # Create mock instances
        mock_instances = [
            ManagedInstance(
                pylet_id="p1",
                instance_id="i1",
                model_id="model-a",
                status=ManagedInstanceStatus.ACTIVE,
            ),
            ManagedInstance(
                pylet_id="p2",
                instance_id="i2",
                model_id="model-a",
                status=ManagedInstanceStatus.ACTIVE,
            ),
            ManagedInstance(
                pylet_id="p3",
                instance_id="i3",
                model_id="model-b",
                status=ManagedInstanceStatus.ACTIVE,
            ),
        ]
        mock_manager.get_active_instances.return_value = mock_instances

        service = PyLetDeploymentService(
            pylet_head_url="http://localhost:8000",
            scheduler_url="http://localhost:8001",
        )
        service._initialized = True
        service._instance_manager = mock_manager

        state = service.get_current_state()

        assert state["model-a"] == 2
        assert state["model-b"] == 1

    @patch("swarmpilot.planner.pylet.deployment_service.InstanceManager")
    def test_get_active_instances(self, mock_manager_class):
        """Test get_active_instances method."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager

        mock_instances = [
            ManagedInstance(
                pylet_id="p1",
                instance_id="i1",
                model_id="model-a",
                status=ManagedInstanceStatus.ACTIVE,
            ),
        ]
        mock_manager.get_active_instances.return_value = mock_instances

        service = PyLetDeploymentService(
            pylet_head_url="http://localhost:8000",
            scheduler_url="http://localhost:8001",
        )
        service._initialized = True
        service._instance_manager = mock_manager

        instances = service.get_active_instances()

        assert len(instances) == 1
        assert instances[0].model_id == "model-a"


class TestPyLetAPIEndpoints:
    """Tests for PyLet API endpoints."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config with PyLet disabled."""
        with patch("swarmpilot.planner.pylet_api.config") as mock:
            mock.pylet_enabled = False
            yield mock

    @pytest.fixture
    def mock_config_enabled(self):
        """Create mock config with PyLet enabled."""
        with patch("swarmpilot.planner.pylet_api.config") as mock:
            mock.pylet_enabled = True
            yield mock

    def test_pylet_status_disabled(self, mock_config):
        """Test /status when PyLet is disabled."""
        with patch(
            "swarmpilot.planner.pylet_api.get_pylet_service_optional"
        ) as mock_get:
            mock_get.return_value = None

            from fastapi import FastAPI

            from swarmpilot.planner.pylet_api import router

            app = FastAPI()
            app.include_router(router, prefix="/v1/pylet")
            client = TestClient(app)

            response = client.get("/v1/pylet/status")

            assert response.status_code == 200
            data = response.json()
            assert data["enabled"] is False
            assert data["initialized"] is False

    def test_pylet_status_enabled(self, mock_config_enabled):
        """Test /status when PyLet is enabled."""
        mock_service = MagicMock()
        mock_service.initialized = True
        mock_service.get_current_state.return_value = {"model-a": 2}
        mock_service.get_active_instances.return_value = [
            ManagedInstance(
                pylet_id="p1",
                instance_id="i1",
                model_id="model-a",
                status=ManagedInstanceStatus.ACTIVE,
                endpoint="http://localhost:8001",
            ),
        ]

        with patch(
            "swarmpilot.planner.pylet_api.get_pylet_service_optional"
        ) as mock_get:
            mock_get.return_value = mock_service

            from fastapi import FastAPI

            from swarmpilot.planner.pylet_api import router

            app = FastAPI()
            app.include_router(router, prefix="/v1/pylet")
            client = TestClient(app)

            response = client.get("/v1/pylet/status")

            assert response.status_code == 200
            data = response.json()
            assert data["enabled"] is True
            assert data["initialized"] is True
            assert data["current_state"]["model-a"] == 2
            assert len(data["active_instances"]) == 1

    def test_pylet_deploy_manually_disabled(self, mock_config):
        """Test /deploy_manually returns 503 when PyLet is disabled."""
        with patch(
            "swarmpilot.planner.pylet_api.get_pylet_service_optional"
        ) as mock_get:
            mock_get.return_value = None

            from fastapi import FastAPI

            from swarmpilot.planner.pylet_api import router

            app = FastAPI()
            app.include_router(router, prefix="/v1/pylet")
            client = TestClient(app)

            response = client.post(
                "/v1/pylet/deploy_manually",
                json={"target_state": {"model-a": 2}},
            )

            assert response.status_code == 503

    def test_pylet_scale_disabled(self, mock_config):
        """Test /scale returns 503 when PyLet is disabled."""
        with patch(
            "swarmpilot.planner.pylet_api.get_pylet_service_optional"
        ) as mock_get:
            mock_get.return_value = None

            from fastapi import FastAPI

            from swarmpilot.planner.pylet_api import router

            app = FastAPI()
            app.include_router(router, prefix="/v1/pylet")
            client = TestClient(app)

            response = client.post(
                "/v1/pylet/scale",
                json={"model_id": "model-a", "target_count": 3},
            )

            assert response.status_code == 503


class TestPyLetConfigValidation:
    """Tests for PyLet configuration validation."""

    def test_pylet_enabled_without_url_activates_local_mode(self):
        """Test that enabling PyLet without URL auto-activates local mode."""
        import os
        from unittest.mock import patch

        with patch.dict(
            os.environ,
            {
                "PYLET_ENABLED": "true",
                "PYLET_HEAD_URL": "",
                "PYLET_LOCAL_MODE": "false",
            },
            clear=False,
        ):
            from swarmpilot.planner.config import PlannerConfig

            config = PlannerConfig()
            assert config.pylet_local_mode is True
            assert config.pylet_head_url == (
                f"http://localhost:{config.pylet_local_port}"
            )
            # Validation should pass — head_url was auto-derived
            config.validate()

    def test_pylet_invalid_backend(self):
        """Test that invalid backend raises error."""
        from swarmpilot.planner.config import PlannerConfig

        config = PlannerConfig()
        config.pylet_enabled = True
        config.pylet_head_url = "http://localhost:8000"
        config.pylet_backend = "invalid"

        with pytest.raises(ValueError, match="PYLET_BACKEND must be"):
            config.validate()

    def test_pylet_invalid_gpu_count(self):
        """Test that invalid GPU count raises error."""
        from swarmpilot.planner.config import PlannerConfig

        config = PlannerConfig()
        config.pylet_enabled = True
        config.pylet_head_url = "http://localhost:8000"
        config.pylet_backend = "vllm"
        config.pylet_gpu_count = -1  # Negative GPU count is invalid

        with pytest.raises(
            ValueError, match="PYLET_GPU_COUNT must be non-negative"
        ):
            config.validate()

    def test_pylet_invalid_cpu_count(self):
        """Test that invalid CPU count raises error."""
        from swarmpilot.planner.config import PlannerConfig

        config = PlannerConfig()
        config.pylet_enabled = True
        config.pylet_head_url = "http://localhost:8000"
        config.pylet_backend = "vllm"
        config.pylet_cpu_count = 0  # CPU count must be positive

        with pytest.raises(
            ValueError, match="PYLET_CPU_COUNT must be positive"
        ):
            config.validate()

    def test_pylet_valid_config(self):
        """Test that valid PyLet config passes validation."""
        from swarmpilot.planner.config import PlannerConfig

        config = PlannerConfig()
        config.pylet_enabled = True
        config.pylet_head_url = "http://localhost:8000"
        config.pylet_backend = "vllm"
        config.pylet_gpu_count = 1
        config.pylet_deploy_timeout = 300.0

        # Should not raise
        config.validate()
