"""Tests for SDK deployment management endpoints.

Tests the REST endpoints in swarmpilot.planner.routes.sdk_api
using mocked PyLet deployment services.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from swarmpilot.planner.api import app


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


def _make_managed_instance(
    pylet_id: str = "pylet-001",
    instance_id: str = "inst-001",
    model_id: str = "Qwen/Qwen3-0.6B",
    endpoint: str = "http://worker:8080",
    status_value: str = "active",
    gpu_count: int = 1,
    error: str | None = None,
) -> MagicMock:
    """Build a MagicMock resembling a ManagedInstance.

    Args:
        pylet_id: PyLet instance UUID.
        instance_id: Scheduler instance ID.
        model_id: Model identifier.
        endpoint: HTTP endpoint.
        status_value: Status string value.
        gpu_count: GPU count.
        error: Error message.

    Returns:
        MagicMock configured as a ManagedInstance.
    """
    inst = MagicMock()
    inst.pylet_id = pylet_id
    inst.instance_id = instance_id
    inst.model_id = model_id
    inst.endpoint = endpoint
    inst.status.value = status_value
    inst.gpu_count = gpu_count
    inst.error = error
    return inst


def _make_deployment_result(
    success: bool = True,
    active_instances: list | None = None,
    error: str | None = None,
) -> MagicMock:
    """Build a MagicMock resembling a DeploymentServiceResult.

    Args:
        success: Whether the operation succeeded.
        active_instances: List of mock instances.
        error: Error message.

    Returns:
        MagicMock configured as a DeploymentServiceResult.
    """
    result = MagicMock()
    result.success = success
    result.active_instances = active_instances or []
    result.error = error
    result.total_added = len(active_instances) if active_instances else 0
    result.total_removed = 0
    return result


# ------------------------------------------------------------------ #
# POST /v1/serve
# ------------------------------------------------------------------ #


class TestServeEndpoint:
    """Tests for POST /v1/serve."""

    @patch(
        "swarmpilot.planner.routes.sdk_api"
        ".get_pylet_service_optional"
    )
    @patch("swarmpilot.planner.routes.sdk_api.config")
    def test_serve_auto_command_generation(
        self, mock_config, mock_get_svc, client
    ):
        """Auto-generate vllm serve command for model names."""
        mock_config.pylet_enabled = True
        inst = _make_managed_instance()
        svc = MagicMock()
        svc.initialized = True
        svc.apply_deployment.return_value = (
            _make_deployment_result(active_instances=[inst])
        )
        mock_get_svc.return_value = svc

        response = client.post(
            "/v1/serve",
            json={
                "model_or_command": "Qwen/Qwen3-0.6B",
                "replicas": 2,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["name"] == "Qwen-Qwen3-0.6B"
        assert data["model"] == "Qwen/Qwen3-0.6B"
        assert data["replicas"] == 2
        assert len(data["instances"]) == 1

        # Verify deployment was called with model as key
        svc.apply_deployment.assert_called_once_with(
            target_state={"Qwen/Qwen3-0.6B": 2},
            wait_for_ready=True,
        )

    @patch(
        "swarmpilot.planner.routes.sdk_api"
        ".get_pylet_service_optional"
    )
    @patch("swarmpilot.planner.routes.sdk_api.config")
    def test_serve_scheduler_auto_resolution(
        self, mock_config, mock_get_svc, client
    ):
        """Scheduler 'auto' resolves from SchedulerRegistry."""
        mock_config.pylet_enabled = True
        svc = MagicMock()
        svc.initialized = True
        svc.apply_deployment.return_value = (
            _make_deployment_result()
        )
        mock_get_svc.return_value = svc

        # Register a scheduler for the model
        from swarmpilot.planner.scheduler_registry import (
            get_scheduler_registry,
        )

        registry = get_scheduler_registry()
        registry.register(
            "Qwen/Qwen3-0.6B", "http://scheduler:8000"
        )

        try:
            response = client.post(
                "/v1/serve",
                json={
                    "model_or_command": "Qwen/Qwen3-0.6B",
                    "scheduler": "auto",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert (
                data["scheduler_url"]
                == "http://scheduler:8000"
            )
        finally:
            registry.deregister("Qwen/Qwen3-0.6B")

    @patch(
        "swarmpilot.planner.routes.sdk_api"
        ".get_pylet_service_optional"
    )
    @patch("swarmpilot.planner.routes.sdk_api.config")
    def test_serve_scheduler_none_skips(
        self, mock_config, mock_get_svc, client
    ):
        """Scheduler None means no scheduler resolution."""
        mock_config.pylet_enabled = True
        svc = MagicMock()
        svc.initialized = True
        svc.apply_deployment.return_value = (
            _make_deployment_result()
        )
        mock_get_svc.return_value = svc

        response = client.post(
            "/v1/serve",
            json={
                "model_or_command": "Qwen/Qwen3-0.6B",
                "scheduler": None,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["scheduler_url"] is None

    def test_serve_pylet_disabled(self, client):
        """Returns 503 when PyLet is disabled."""
        response = client.post(
            "/v1/serve",
            json={"model_or_command": "some-model"},
        )
        assert response.status_code == 503


# ------------------------------------------------------------------ #
# POST /v1/run
# ------------------------------------------------------------------ #


class TestRunEndpoint:
    """Tests for POST /v1/run."""

    @patch(
        "swarmpilot.planner.routes.sdk_api"
        ".get_pylet_service_optional"
    )
    @patch("swarmpilot.planner.routes.sdk_api.config")
    def test_run_custom_command(
        self, mock_config, mock_get_svc, client
    ):
        """Deploy with a custom command."""
        mock_config.pylet_enabled = True
        inst = _make_managed_instance(
            model_id="python train.py"
        )
        svc = MagicMock()
        svc.initialized = True
        svc.apply_deployment.return_value = (
            _make_deployment_result(active_instances=[inst])
        )
        mock_get_svc.return_value = svc

        response = client.post(
            "/v1/run",
            json={
                "command": "python train.py",
                "name": "training-job",
                "replicas": 1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["name"] == "training-job"
        assert data["command"] == "python train.py"
        assert data["replicas"] == 1

    def test_run_pylet_disabled(self, client):
        """Returns 503 when PyLet is disabled."""
        response = client.post(
            "/v1/run",
            json={"command": "echo hello"},
        )
        assert response.status_code == 503


# ------------------------------------------------------------------ #
# POST /v1/register + GET /v1/registered
# ------------------------------------------------------------------ #


class TestRegisterEndpoints:
    """Tests for POST /v1/register and GET /v1/registered."""

    def test_register_and_list(self, client):
        """Register a model and verify it appears in listing."""
        # Clear previous state
        from swarmpilot.planner.routes.sdk_api import (
            _registered_models,
            _registered_models_lock,
        )

        with _registered_models_lock:
            _registered_models.clear()

        # Register a model
        response = client.post(
            "/v1/register",
            json={
                "model": "meta-llama/Llama-3-8B",
                "replicas": 3,
                "gpu_count": 2,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "registered"
        assert data["model"] == "meta-llama/Llama-3-8B"

        # List registered models
        response = client.get("/v1/registered")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert "meta-llama/Llama-3-8B" in data["models"]
        assert data["models"]["meta-llama/Llama-3-8B"]["replicas"] == 3
        assert (
            data["models"]["meta-llama/Llama-3-8B"]["gpu_count"]
            == 2
        )

    def test_register_overwrites(self, client):
        """Re-registering the same model overwrites."""
        from swarmpilot.planner.routes.sdk_api import (
            _registered_models,
            _registered_models_lock,
        )

        with _registered_models_lock:
            _registered_models.clear()

        client.post(
            "/v1/register",
            json={"model": "m1", "replicas": 1},
        )
        client.post(
            "/v1/register",
            json={"model": "m1", "replicas": 5},
        )

        response = client.get("/v1/registered")
        data = response.json()
        assert data["total"] == 1
        assert data["models"]["m1"]["replicas"] == 5


# ------------------------------------------------------------------ #
# POST /v1/terminate
# ------------------------------------------------------------------ #


class TestTerminateEndpoint:
    """Tests for POST /v1/terminate."""

    @patch(
        "swarmpilot.planner.routes.sdk_api"
        ".get_pylet_service_optional"
    )
    @patch("swarmpilot.planner.routes.sdk_api.config")
    def test_terminate_all(
        self, mock_config, mock_get_svc, client
    ):
        """Terminate all instances."""
        mock_config.pylet_enabled = True
        svc = MagicMock()
        svc.initialized = True
        svc.terminate_all.return_value = {
            "id-1": True,
            "id-2": True,
        }
        mock_get_svc.return_value = svc

        response = client.post(
            "/v1/terminate", json={"all": True}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["terminated_count"] == 2

    @patch(
        "swarmpilot.planner.routes.sdk_api"
        ".get_pylet_service_optional"
    )
    @patch("swarmpilot.planner.routes.sdk_api.config")
    def test_terminate_by_model(
        self, mock_config, mock_get_svc, client
    ):
        """Terminate instances by model."""
        mock_config.pylet_enabled = True
        inst = _make_managed_instance(
            pylet_id="p1", model_id="model-a"
        )
        svc = MagicMock()
        svc.initialized = True
        svc.get_instances_by_model.return_value = [inst]
        svc.instance_manager.terminate_instances.return_value = {
            "p1": True
        }
        mock_get_svc.return_value = svc

        response = client.post(
            "/v1/terminate", json={"model": "model-a"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["terminated_count"] == 1

    @patch(
        "swarmpilot.planner.routes.sdk_api"
        ".get_pylet_service_optional"
    )
    @patch("swarmpilot.planner.routes.sdk_api.config")
    def test_terminate_no_criteria(
        self, mock_config, mock_get_svc, client
    ):
        """Returns failure message when no criteria given."""
        mock_config.pylet_enabled = True
        svc = MagicMock()
        svc.initialized = True
        mock_get_svc.return_value = svc

        response = client.post("/v1/terminate", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "No termination criteria" in data["message"]

    def test_terminate_pylet_disabled(self, client):
        """Returns 503 when PyLet is disabled."""
        response = client.post(
            "/v1/terminate", json={"all": True}
        )
        assert response.status_code == 503


# ------------------------------------------------------------------ #
# GET /v1/instances
# ------------------------------------------------------------------ #


class TestInstancesEndpoint:
    """Tests for GET /v1/instances."""

    @patch(
        "swarmpilot.planner.routes.sdk_api"
        ".get_pylet_service_optional"
    )
    @patch("swarmpilot.planner.routes.sdk_api.config")
    def test_list_instances(
        self, mock_config, mock_get_svc, client
    ):
        """List active instances."""
        mock_config.pylet_enabled = True
        inst1 = _make_managed_instance(
            pylet_id="p1", model_id="model-a"
        )
        inst2 = _make_managed_instance(
            pylet_id="p2", model_id="model-b"
        )
        svc = MagicMock()
        svc.initialized = True
        svc.get_active_instances.return_value = [inst1, inst2]
        mock_get_svc.return_value = svc

        response = client.get("/v1/instances")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["pylet_id"] == "p1"
        assert data[1]["pylet_id"] == "p2"

    @patch(
        "swarmpilot.planner.routes.sdk_api"
        ".get_pylet_service_optional"
    )
    @patch("swarmpilot.planner.routes.sdk_api.config")
    def test_get_instance_by_name(
        self, mock_config, mock_get_svc, client
    ):
        """Get single instance by pylet_id."""
        mock_config.pylet_enabled = True
        inst = _make_managed_instance(pylet_id="target-id")
        svc = MagicMock()
        svc.initialized = True
        svc.get_active_instances.return_value = [inst]
        mock_get_svc.return_value = svc

        response = client.get("/v1/instances/target-id")

        assert response.status_code == 200
        data = response.json()
        assert data["pylet_id"] == "target-id"

    @patch(
        "swarmpilot.planner.routes.sdk_api"
        ".get_pylet_service_optional"
    )
    @patch("swarmpilot.planner.routes.sdk_api.config")
    def test_get_instance_not_found(
        self, mock_config, mock_get_svc, client
    ):
        """404 when instance not found."""
        mock_config.pylet_enabled = True
        svc = MagicMock()
        svc.initialized = True
        svc.get_active_instances.return_value = []
        mock_get_svc.return_value = svc

        response = client.get("/v1/instances/nonexistent")

        assert response.status_code == 404

    def test_list_instances_pylet_disabled(self, client):
        """Returns 503 when PyLet is disabled."""
        response = client.get("/v1/instances")
        assert response.status_code == 503


# ------------------------------------------------------------------ #
# GET /v1/schedulers
# ------------------------------------------------------------------ #


class TestSchedulersEndpoint:
    """Tests for GET /v1/schedulers."""

    def test_get_schedulers_empty(self, client):
        """Returns empty map when no schedulers registered."""
        response = client.get("/v1/schedulers")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 0
        assert isinstance(data["schedulers"], dict)

    def test_get_schedulers_with_entries(self, client):
        """Returns registered scheduler mapping."""
        from swarmpilot.planner.scheduler_registry import (
            get_scheduler_registry,
        )

        registry = get_scheduler_registry()
        registry.register("model-x", "http://sched-x:8000")

        try:
            response = client.get("/v1/schedulers")

            assert response.status_code == 200
            data = response.json()
            assert "model-x" in data["schedulers"]
            assert (
                data["schedulers"]["model-x"]
                == "http://sched-x:8000"
            )
        finally:
            registry.deregister("model-x")


# ------------------------------------------------------------------ #
# POST /v1/scale
# ------------------------------------------------------------------ #


class TestScaleEndpoint:
    """Tests for POST /v1/scale."""

    @patch(
        "swarmpilot.planner.routes.sdk_api"
        ".get_pylet_service_optional"
    )
    @patch("swarmpilot.planner.routes.sdk_api.config")
    def test_scale_model(
        self, mock_config, mock_get_svc, client
    ):
        """Scale a model to target count."""
        mock_config.pylet_enabled = True
        inst = _make_managed_instance()
        svc = MagicMock()
        svc.initialized = True
        svc.get_instances_by_model.return_value = [inst]
        svc.scale_model.return_value = (
            _make_deployment_result(
                active_instances=[inst, _make_managed_instance()]
            )
        )
        mock_get_svc.return_value = svc

        response = client.post(
            "/v1/scale",
            json={"model": "Qwen/Qwen3-0.6B", "replicas": 2},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model"] == "Qwen/Qwen3-0.6B"
        assert data["previous_count"] == 1
        assert data["current_count"] == 2

    def test_scale_pylet_disabled(self, client):
        """Returns 503 when PyLet is disabled."""
        response = client.post(
            "/v1/scale",
            json={"model": "m", "replicas": 1},
        )
        assert response.status_code == 503
