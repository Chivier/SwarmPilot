"""End-to-end integration tests for SDK -> Planner -> (mocked) PyLet.

These tests exercise real FastAPI routing through the Planner app
using ``httpx.AsyncClient`` with ASGI transport while mocking the
PyLet deployment service layer.  No real PyLet cluster is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from swarmpilot.planner.api import app as planner_app

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


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
    result.total_added = (
        len(active_instances) if active_instances else 0
    )
    result.total_removed = 0
    return result


def _mock_pylet_service(
    active_instances: list[MagicMock] | None = None,
) -> MagicMock:
    """Build a fully wired mock PyLet service.

    Args:
        active_instances: Instances the mock service tracks.

    Returns:
        MagicMock configured as a PyLetDeploymentService.
    """
    instances = active_instances or []
    svc = MagicMock()
    svc.initialized = True
    svc.get_active_instances.return_value = instances
    svc.get_instances_by_model.return_value = instances
    svc.apply_deployment.return_value = (
        _make_deployment_result(active_instances=instances)
    )
    svc.scale_model.return_value = (
        _make_deployment_result(active_instances=instances)
    )
    svc.terminate_all.return_value = {
        i.pylet_id: True for i in instances
    }
    svc.instance_manager.terminate_instances.return_value = {
        i.pylet_id: True for i in instances
    }
    return svc


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture()
async def planner_client():
    """Async httpx client backed by the real Planner ASGI app.

    Yields:
        httpx.AsyncClient wired to Planner via ASGITransport.
    """
    transport = httpx.ASGITransport(app=planner_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        yield client


@pytest.fixture(autouse=True)
def _clear_registered_models():
    """Reset the SDK route module-level model registry.

    Prevents test pollution from leftover registrations.
    """
    from swarmpilot.planner.routes.sdk_api import (
        _registered_models,
        _registered_models_lock,
    )

    with _registered_models_lock:
        _registered_models.clear()

    yield

    with _registered_models_lock:
        _registered_models.clear()


@pytest.fixture(autouse=True)
def _clear_scheduler_registry():
    """Reset the global scheduler registry between tests."""
    from swarmpilot.planner.scheduler_registry import (
        get_scheduler_registry,
    )

    registry = get_scheduler_registry()
    for model in list(registry.get_registered_models()):
        registry.deregister(model)

    yield

    for model in list(registry.get_registered_models()):
        registry.deregister(model)


# ------------------------------------------------------------------ #
# 1. Manual serve flow
# ------------------------------------------------------------------ #


class TestServeFlow:
    """POST /v1/serve with mocked PyLet -> verify response."""

    @patch(
        "swarmpilot.planner.routes.sdk_api"
        ".get_pylet_service_optional"
    )
    @patch("swarmpilot.planner.routes.sdk_api.config")
    async def test_serve_returns_correct_structure(
        self,
        mock_config: MagicMock,
        mock_get_svc: MagicMock,
        planner_client: httpx.AsyncClient,
    ) -> None:
        """Serve a model and verify the response shape."""
        mock_config.pylet_enabled = True
        inst = _make_managed_instance()
        mock_get_svc.return_value = _mock_pylet_service(
            active_instances=[inst]
        )

        resp = await planner_client.post(
            "/v1/serve",
            json={
                "model_or_command": "Qwen/Qwen3-0.6B",
                "replicas": 2,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["name"] == "Qwen-Qwen3-0.6B"
        assert data["model"] == "Qwen/Qwen3-0.6B"
        assert data["replicas"] == 2
        assert isinstance(data["instances"], list)
        assert len(data["instances"]) == 1


# ------------------------------------------------------------------ #
# 2. Register + deploy flow
# ------------------------------------------------------------------ #


class TestRegisterDeployFlow:
    """Register models -> POST /v1/deploy -> verify plan."""

    @patch(
        "swarmpilot.planner.routes.sdk_api"
        ".get_pylet_service_optional"
    )
    @patch("swarmpilot.planner.routes.sdk_api.config")
    async def test_register_then_deploy(
        self,
        mock_config: MagicMock,
        mock_get_svc: MagicMock,
        planner_client: httpx.AsyncClient,
    ) -> None:
        """Register two models, deploy, and verify response."""
        mock_config.pylet_enabled = True
        inst_a = _make_managed_instance(
            pylet_id="p-a", model_id="model-a"
        )
        inst_b = _make_managed_instance(
            pylet_id="p-b", model_id="model-b"
        )
        mock_get_svc.return_value = _mock_pylet_service(
            active_instances=[inst_a, inst_b]
        )

        # Step 1: Register models
        r1 = await planner_client.post(
            "/v1/register",
            json={
                "model": "model-a",
                "replicas": 1,
                "gpu_count": 1,
            },
        )
        assert r1.status_code == 200
        assert r1.json()["status"] == "registered"

        r2 = await planner_client.post(
            "/v1/register",
            json={
                "model": "model-b",
                "replicas": 2,
                "gpu_count": 2,
            },
        )
        assert r2.status_code == 200

        # Step 2: Deploy
        resp = await planner_client.post("/v1/deploy")

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert set(data["deployed_models"]) == {
            "model-a",
            "model-b",
        }
        assert data["total_instances"] == 2


# ------------------------------------------------------------------ #
# 3. Custom run flow
# ------------------------------------------------------------------ #


class TestRunFlow:
    """POST /v1/run -> verify no scheduler association."""

    @patch(
        "swarmpilot.planner.routes.sdk_api"
        ".get_pylet_service_optional"
    )
    @patch("swarmpilot.planner.routes.sdk_api.config")
    async def test_run_custom_command(
        self,
        mock_config: MagicMock,
        mock_get_svc: MagicMock,
        planner_client: httpx.AsyncClient,
    ) -> None:
        """Run a custom command and verify response."""
        mock_config.pylet_enabled = True
        inst = _make_managed_instance(
            model_id="python train.py"
        )
        mock_get_svc.return_value = _mock_pylet_service(
            active_instances=[inst]
        )

        resp = await planner_client.post(
            "/v1/run",
            json={
                "command": "python train.py",
                "name": "training-job",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["name"] == "training-job"
        assert data["command"] == "python train.py"
        # Run responses do not carry a scheduler_url field
        assert "scheduler_url" not in data


# ------------------------------------------------------------------ #
# 4. List instances
# ------------------------------------------------------------------ #


class TestListInstancesFlow:
    """Deploy -> GET /v1/instances -> verify listing."""

    @patch(
        "swarmpilot.planner.routes.sdk_api"
        ".get_pylet_service_optional"
    )
    @patch("swarmpilot.planner.routes.sdk_api.config")
    async def test_list_instances_after_deploy(
        self,
        mock_config: MagicMock,
        mock_get_svc: MagicMock,
        planner_client: httpx.AsyncClient,
    ) -> None:
        """List instances and verify the returned details."""
        mock_config.pylet_enabled = True
        inst1 = _make_managed_instance(
            pylet_id="p1", model_id="model-a"
        )
        inst2 = _make_managed_instance(
            pylet_id="p2", model_id="model-b"
        )
        mock_get_svc.return_value = _mock_pylet_service(
            active_instances=[inst1, inst2]
        )

        resp = await planner_client.get("/v1/instances")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        ids = {item["pylet_id"] for item in data}
        assert ids == {"p1", "p2"}

    @patch(
        "swarmpilot.planner.routes.sdk_api"
        ".get_pylet_service_optional"
    )
    @patch("swarmpilot.planner.routes.sdk_api.config")
    async def test_get_single_instance(
        self,
        mock_config: MagicMock,
        mock_get_svc: MagicMock,
        planner_client: httpx.AsyncClient,
    ) -> None:
        """Get a single instance by pylet_id."""
        mock_config.pylet_enabled = True
        inst = _make_managed_instance(pylet_id="target-id")
        mock_get_svc.return_value = _mock_pylet_service(
            active_instances=[inst]
        )

        resp = await planner_client.get(
            "/v1/instances/target-id"
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["pylet_id"] == "target-id"
        assert data["status"] == "active"


# ------------------------------------------------------------------ #
# 5. Scale flow
# ------------------------------------------------------------------ #


class TestScaleFlow:
    """Deploy -> POST /v1/scale -> verify count changes."""

    @patch(
        "swarmpilot.planner.routes.sdk_api"
        ".get_pylet_service_optional"
    )
    @patch("swarmpilot.planner.routes.sdk_api.config")
    async def test_scale_model_replica_count(
        self,
        mock_config: MagicMock,
        mock_get_svc: MagicMock,
        planner_client: httpx.AsyncClient,
    ) -> None:
        """Scale from 1 to 3 replicas and verify counts."""
        mock_config.pylet_enabled = True
        original = _make_managed_instance(pylet_id="p1")
        scaled = [
            _make_managed_instance(pylet_id=f"p{i}")
            for i in range(3)
        ]

        svc = _mock_pylet_service(active_instances=[original])
        # get_instances_by_model returns 1 (previous count)
        svc.get_instances_by_model.return_value = [original]
        # scale_model returns 3 (new count)
        svc.scale_model.return_value = (
            _make_deployment_result(active_instances=scaled)
        )
        mock_get_svc.return_value = svc

        resp = await planner_client.post(
            "/v1/scale",
            json={
                "model": "Qwen/Qwen3-0.6B",
                "replicas": 3,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["model"] == "Qwen/Qwen3-0.6B"
        assert data["previous_count"] == 1
        assert data["current_count"] == 3


# ------------------------------------------------------------------ #
# 6. Terminate flow
# ------------------------------------------------------------------ #


class TestTerminateFlow:
    """Deploy -> POST /v1/terminate -> verify termination."""

    @patch(
        "swarmpilot.planner.routes.sdk_api"
        ".get_pylet_service_optional"
    )
    @patch("swarmpilot.planner.routes.sdk_api.config")
    async def test_terminate_all_instances(
        self,
        mock_config: MagicMock,
        mock_get_svc: MagicMock,
        planner_client: httpx.AsyncClient,
    ) -> None:
        """Terminate all instances and verify count."""
        mock_config.pylet_enabled = True
        inst1 = _make_managed_instance(pylet_id="p1")
        inst2 = _make_managed_instance(pylet_id="p2")
        mock_get_svc.return_value = _mock_pylet_service(
            active_instances=[inst1, inst2]
        )

        resp = await planner_client.post(
            "/v1/terminate", json={"all": True}
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["terminated_count"] == 2

    @patch(
        "swarmpilot.planner.routes.sdk_api"
        ".get_pylet_service_optional"
    )
    @patch("swarmpilot.planner.routes.sdk_api.config")
    async def test_terminate_by_model(
        self,
        mock_config: MagicMock,
        mock_get_svc: MagicMock,
        planner_client: httpx.AsyncClient,
    ) -> None:
        """Terminate instances by model identifier."""
        mock_config.pylet_enabled = True
        inst = _make_managed_instance(
            pylet_id="p1", model_id="model-a"
        )
        svc = _mock_pylet_service(active_instances=[inst])
        mock_get_svc.return_value = svc

        resp = await planner_client.post(
            "/v1/terminate", json={"model": "model-a"}
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["terminated_count"] == 1
        assert "model-a" in data["message"]


# ------------------------------------------------------------------ #
# 7. Scheduler mapping
# ------------------------------------------------------------------ #


class TestSchedulerMapping:
    """Register scheduler -> GET /v1/schedulers -> verify."""

    async def test_scheduler_mapping_roundtrip(
        self,
        planner_client: httpx.AsyncClient,
    ) -> None:
        """Register a scheduler and retrieve the mapping."""
        # Register via the scheduler/register endpoint
        reg_resp = await planner_client.post(
            "/v1/scheduler/register",
            json={
                "model_id": "llama-7b",
                "scheduler_url": "http://sched:8000",
            },
        )
        assert reg_resp.status_code == 200

        # Retrieve via the SDK schedulers endpoint
        resp = await planner_client.get("/v1/schedulers")

        assert resp.status_code == 200
        data = resp.json()
        assert "llama-7b" in data["schedulers"]
        assert (
            data["schedulers"]["llama-7b"]
            == "http://sched:8000"
        )
        assert data["total"] >= 1

    async def test_empty_scheduler_map(
        self,
        planner_client: httpx.AsyncClient,
    ) -> None:
        """Empty cluster returns empty scheduler map."""
        resp = await planner_client.get("/v1/schedulers")

        assert resp.status_code == 200
        data = resp.json()
        assert data["schedulers"] == {}
        assert data["total"] == 0


# ------------------------------------------------------------------ #
# 8. Predictor train (via scheduler)
# ------------------------------------------------------------------ #


class TestPredictorTrainFlow:
    """POST /v1/predictor/train with mocked training client."""

    @patch(
        "swarmpilot.scheduler.routes.predictor._get_clients"
    )
    async def test_predictor_train_success(
        self,
        mock_get_clients: MagicMock,
    ) -> None:
        """Train the predictor and verify the response.

        Uses a separate ASGI transport for the scheduler app
        because the predictor endpoints live on the scheduler.
        """
        from swarmpilot.scheduler.api import app as sched_app

        # Build mock training client
        mock_training = MagicMock()
        mock_training.flush = MagicMock(return_value=True)
        # Make flush awaitable
        import asyncio

        future: asyncio.Future[bool] = asyncio.Future()
        future.set_result(True)
        mock_training.flush.return_value = future
        mock_training.get_buffer_size.return_value = 5

        # Build mock predictor client
        mock_predictor = MagicMock()

        mock_get_clients.return_value = (
            mock_predictor,
            mock_training,
        )

        transport = httpx.ASGITransport(app=sched_app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/v1/predictor/train",
                json={"model_id": "llama-7b"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["model_id"] == "llama-7b"
        assert data["samples_trained"] == 5
        assert "Training completed" in data["message"]

    @patch(
        "swarmpilot.scheduler.routes.predictor._get_clients"
    )
    async def test_predictor_train_no_client_returns_503(
        self,
        mock_get_clients: MagicMock,
    ) -> None:
        """Returns 503 when training client is not configured."""
        from swarmpilot.scheduler.api import app as sched_app

        mock_predictor = MagicMock()
        mock_get_clients.return_value = (mock_predictor, None)

        transport = httpx.ASGITransport(app=sched_app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/v1/predictor/train",
                json={"model_id": "llama-7b"},
            )

        assert resp.status_code == 503
