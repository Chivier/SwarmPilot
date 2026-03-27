"""Tests for the SwarmPilot SDK client."""

from __future__ import annotations

import httpx
import pytest
import respx

from swarmpilot.errors import ModelNotDeployed, SchedulerNotFound
from swarmpilot.sdk.client import SwarmPilotClient
from swarmpilot.sdk.models import (
    ClusterState,
    DeploymentResult,
    InstanceGroup,
    ModelStatus,
    PredictResult,
    Process,
    TrainResult,
)

PLANNER = "http://planner:8002"
SCHEDULER = "http://scheduler:8000"


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def client() -> SwarmPilotClient:
    """Create a client pointed at test URLs."""
    return SwarmPilotClient(
        planner_url=PLANNER,
        scheduler_url=SCHEDULER,
        timeout=5.0,
    )


# ------------------------------------------------------------------
# serve()
# ------------------------------------------------------------------


class TestServe:
    """serve() sends POST /v1/serve and returns InstanceGroup."""

    @respx.mock
    async def test_serve_returns_instance_group(
        self, client: SwarmPilotClient
    ) -> None:
        """Successful serve returns an InstanceGroup."""
        respx.post(f"{PLANNER}/v1/serve").mock(
            return_value=httpx.Response(
                200,
                json={
                    "name": "llama-group",
                    "model": "llama-7b",
                    "command": "vllm serve llama-7b",
                    "replicas": [
                        {
                            "name": "llama-group-0",
                            "endpoint": "http://w:8080",
                            "status": "running",
                        }
                    ],
                    "scheduler": "http://sched:8000",
                    "status": "deployed",
                },
            )
        )

        group = await client.serve("llama-7b", gpu=1, replicas=1)

        assert isinstance(group, InstanceGroup)
        assert group.name == "llama-group"
        assert group.model == "llama-7b"
        assert len(group.instances) == 1
        assert group.instances[0].name == "llama-group-0"
        assert group.scheduler == "http://sched:8000"

    @respx.mock
    async def test_serve_sends_correct_payload(
        self, client: SwarmPilotClient
    ) -> None:
        """serve() includes all parameters in the POST body."""
        route = respx.post(f"{PLANNER}/v1/serve").mock(
            return_value=httpx.Response(
                200,
                json={
                    "name": "my-svc",
                    "model": "llama-7b",
                    "command": "vllm serve llama-7b",
                    "replicas": [],
                    "scheduler": None,
                    "status": "pending",
                },
            )
        )

        await client.serve(
            "llama-7b",
            name="my-svc",
            replicas=2,
            gpu=4,
            scheduler=None,
        )

        assert route.called
        body = route.calls[0].request.content
        import json

        payload = json.loads(body)
        assert payload["model_or_command"] == "llama-7b"
        assert payload["name"] == "my-svc"
        assert payload["replicas"] == 2
        assert payload["gpu_count"] == 4
        assert payload["scheduler"] is None


# ------------------------------------------------------------------
# run()
# ------------------------------------------------------------------


class TestRun:
    """run() sends POST /v1/run and returns Process."""

    @respx.mock
    async def test_run_returns_process(self, client: SwarmPilotClient) -> None:
        """Successful run returns a Process."""
        respx.post(f"{PLANNER}/v1/run").mock(
            return_value=httpx.Response(
                200,
                json={
                    "name": "etl-job",
                    "command": "python etl.py",
                    "endpoint": None,
                    "scheduler": None,
                    "status": "running",
                },
            )
        )

        proc = await client.run("python etl.py", name="etl-job", gpu=0)

        assert isinstance(proc, Process)
        assert proc.name == "etl-job"
        assert proc.command == "python etl.py"
        assert proc.status == "running"
        assert proc.scheduler is None

    @respx.mock
    async def test_run_sends_correct_payload(
        self, client: SwarmPilotClient
    ) -> None:
        """run() includes all parameters in the POST body."""
        route = respx.post(f"{PLANNER}/v1/run").mock(
            return_value=httpx.Response(
                200,
                json={
                    "name": "job-1",
                    "command": "python job.py",
                    "endpoint": None,
                    "scheduler": None,
                    "status": "pending",
                },
            )
        )

        await client.run(
            "python job.py",
            name="job-1",
            gpu=2,
        )

        assert route.called
        import json

        payload = json.loads(route.calls[0].request.content)
        assert payload["command"] == "python job.py"
        assert payload["name"] == "job-1"
        assert payload["gpu_count"] == 2


# ------------------------------------------------------------------
# register()
# ------------------------------------------------------------------


class TestRegister:
    """register() sends POST /v1/register."""

    @respx.mock
    async def test_register_sends_correct_payload(
        self, client: SwarmPilotClient
    ) -> None:
        """register() sends model, gpu_count, and replicas."""
        route = respx.post(f"{PLANNER}/v1/register").mock(
            return_value=httpx.Response(
                200,
                json={"status": "registered"},
            )
        )

        await client.register("llama-7b", gpu=1, replicas=3)

        assert route.called
        import json

        payload = json.loads(route.calls[0].request.content)
        assert payload["model"] == "llama-7b"
        assert payload["gpu_count"] == 1
        assert payload["replicas"] == 3


# ------------------------------------------------------------------
# deploy()
# ------------------------------------------------------------------


class TestDeploy:
    """deploy() sends POST /v1/deploy and returns DeploymentResult."""

    @respx.mock
    async def test_deploy_returns_deployment_result(
        self, client: SwarmPilotClient
    ) -> None:
        """Successful deploy returns a DeploymentResult."""
        respx.post(f"{PLANNER}/v1/deploy").mock(
            return_value=httpx.Response(
                200,
                json={
                    "plan": {"llama-7b": 2},
                    "groups": {
                        "llama-7b": {
                            "name": "llama-group",
                            "model": "llama-7b",
                            "command": "vllm serve llama-7b",
                            "replicas": [
                                {
                                    "name": "w-0",
                                    "endpoint": "http://a:8080",
                                    "status": "running",
                                },
                                {
                                    "name": "w-1",
                                    "endpoint": "http://b:8080",
                                    "status": "running",
                                },
                            ],
                            "scheduler": None,
                            "status": "deployed",
                        }
                    },
                    "status": "deployed",
                },
            )
        )

        result = await client.deploy()

        assert isinstance(result, DeploymentResult)
        assert result.status == "deployed"
        assert result.plan == {"llama-7b": 2}
        assert "llama-7b" in result.groups
        group = result["llama-7b"]
        assert isinstance(group, InstanceGroup)
        assert len(group.instances) == 2


# ------------------------------------------------------------------
# instances()
# ------------------------------------------------------------------


class TestInstances:
    """instances() sends GET /v1/instances and returns ClusterState."""

    @respx.mock
    async def test_instances_returns_cluster_state(
        self, client: SwarmPilotClient
    ) -> None:
        """Successful listing returns a ClusterState."""
        respx.get(f"{PLANNER}/v1/instances").mock(
            return_value=httpx.Response(
                200,
                json={
                    "instances": [
                        {
                            "name": "w-0",
                            "model": "llama-7b",
                            "command": "vllm serve",
                            "endpoint": "http://a:8080",
                            "scheduler": None,
                            "status": "running",
                            "gpu": 1,
                        }
                    ],
                    "processes": [
                        {
                            "name": "etl",
                            "command": "python etl.py",
                            "endpoint": None,
                            "status": "running",
                            "gpu": 0,
                        }
                    ],
                    "groups": [],
                },
            )
        )

        state = await client.instances()

        assert isinstance(state, ClusterState)
        assert len(state.instances) == 1
        assert state.instances[0].name == "w-0"
        assert len(state.processes) == 1
        assert state.processes[0].name == "etl"


# ------------------------------------------------------------------
# scale()
# ------------------------------------------------------------------


class TestScale:
    """scale() sends POST /v1/scale and returns InstanceGroup."""

    @respx.mock
    async def test_scale_returns_instance_group(
        self, client: SwarmPilotClient
    ) -> None:
        """Successful scale returns an InstanceGroup."""
        respx.post(f"{PLANNER}/v1/scale").mock(
            return_value=httpx.Response(
                200,
                json={
                    "name": "llama-group",
                    "model": "llama-7b",
                    "command": "vllm serve llama-7b",
                    "replicas": [
                        {
                            "name": "w-0",
                            "endpoint": "http://a:8080",
                            "status": "running",
                        },
                        {
                            "name": "w-1",
                            "endpoint": "http://b:8080",
                            "status": "running",
                        },
                    ],
                    "scheduler": None,
                    "status": "scaled",
                },
            )
        )

        group = await client.scale("llama-7b", replicas=2)

        assert isinstance(group, InstanceGroup)
        assert len(group.instances) == 2


# ------------------------------------------------------------------
# terminate()
# ------------------------------------------------------------------


class TestTerminate:
    """terminate() sends POST /v1/terminate."""

    @respx.mock
    async def test_terminate_by_name(self, client: SwarmPilotClient) -> None:
        """terminate(name=...) sends the correct payload."""
        route = respx.post(f"{PLANNER}/v1/terminate").mock(
            return_value=httpx.Response(
                200,
                json={"status": "terminated"},
            )
        )

        await client.terminate(name="w-0")

        assert route.called
        import json

        payload = json.loads(route.calls[0].request.content)
        assert payload["name"] == "w-0"

    @respx.mock
    async def test_terminate_all(self, client: SwarmPilotClient) -> None:
        """terminate(all=True) sends all=True."""
        route = respx.post(f"{PLANNER}/v1/terminate").mock(
            return_value=httpx.Response(
                200,
                json={"status": "terminated"},
            )
        )

        await client.terminate(all=True)

        import json

        payload = json.loads(route.calls[0].request.content)
        assert payload["all"] is True


# ------------------------------------------------------------------
# schedulers()
# ------------------------------------------------------------------


class TestSchedulers:
    """schedulers() sends GET /v1/schedulers."""

    @respx.mock
    async def test_schedulers_returns_dict(
        self, client: SwarmPilotClient
    ) -> None:
        """Successful call returns a model-to-scheduler mapping."""
        respx.get(f"{PLANNER}/v1/schedulers").mock(
            return_value=httpx.Response(
                200,
                json={
                    "schedulers": {
                        "llama-7b": "http://sched:8000",
                        "gpt-4": "http://sched:8001",
                    }
                },
            )
        )

        result = await client.schedulers()

        assert result == {
            "llama-7b": "http://sched:8000",
            "gpt-4": "http://sched:8001",
        }


# ------------------------------------------------------------------
# train()
# ------------------------------------------------------------------


class TestTrain:
    """train() sends POST to scheduler /v1/predictor/train."""

    @respx.mock
    async def test_train_returns_train_result(
        self, client: SwarmPilotClient
    ) -> None:
        """Successful train returns a TrainResult."""
        respx.post(f"{SCHEDULER}/v1/predictor/train").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "llama-7b",
                    "samples_trained": 500,
                    "metrics": {"mae": 12.0},
                    "strategy": "expect_error",
                },
            )
        )

        result = await client.train("llama-7b")

        assert isinstance(result, TrainResult)
        assert result.model == "llama-7b"
        assert result.samples_trained == 500
        assert result.metrics == {"mae": 12.0}
        assert result.strategy == "expect_error"

    @respx.mock
    async def test_train_sends_to_scheduler_url(
        self, client: SwarmPilotClient
    ) -> None:
        """train() sends to the scheduler, not the planner."""
        route = respx.post(f"{SCHEDULER}/v1/predictor/train").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "llama-7b",
                    "samples_trained": 0,
                    "metrics": {},
                    "strategy": "quantile",
                },
            )
        )

        await client.train("llama-7b", prediction_type="quantile")

        assert route.called
        import json

        payload = json.loads(route.calls[0].request.content)
        assert payload["model"] == "llama-7b"
        assert payload["prediction_type"] == "quantile"


# ------------------------------------------------------------------
# predict()
# ------------------------------------------------------------------


class TestPredict:
    """predict() sends POST to scheduler /v1/predictor/predict."""

    @respx.mock
    async def test_predict_returns_predict_result(
        self, client: SwarmPilotClient
    ) -> None:
        """Successful predict returns a PredictResult."""
        respx.post(f"{SCHEDULER}/v1/predictor/predict").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "llama-7b",
                    "expected_runtime_ms": 150.0,
                    "error_margin_ms": 20.0,
                    "quantiles": None,
                },
            )
        )

        result = await client.predict(
            "llama-7b",
            features={"input_len": 128},
        )

        assert isinstance(result, PredictResult)
        assert result.model == "llama-7b"
        assert result.expected_runtime_ms == 150.0
        assert result.error_margin_ms == 20.0
        assert result.quantiles is None

    @respx.mock
    async def test_predict_sends_to_scheduler_url(
        self, client: SwarmPilotClient
    ) -> None:
        """predict() sends to the scheduler, not the planner."""
        route = respx.post(f"{SCHEDULER}/v1/predictor/predict").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "llama-7b",
                    "expected_runtime_ms": 100.0,
                    "error_margin_ms": 10.0,
                },
            )
        )

        await client.predict(
            "llama-7b",
            features={"input_len": 64},
            prediction_type="expect_error",
        )

        assert route.called
        import json

        payload = json.loads(route.calls[0].request.content)
        assert payload["model"] == "llama-7b"
        assert payload["features"] == {"input_len": 64}
        assert payload["prediction_type"] == "expect_error"


# ------------------------------------------------------------------
# predictor_status()
# ------------------------------------------------------------------


class TestPredictorStatus:
    """predictor_status() sends GET to scheduler."""

    @respx.mock
    async def test_returns_model_status(self, client: SwarmPilotClient) -> None:
        """Successful call returns a ModelStatus."""
        respx.get(f"{SCHEDULER}/v1/predictor/status/llama-7b").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "llama-7b",
                    "samples_collected": 1000,
                    "last_trained": "2026-01-15T10:00:00Z",
                    "prediction_types": [
                        "expect_error",
                        "quantile",
                    ],
                    "metrics": {"mae": 10.0},
                    "strategy": "expect_error",
                    "preprocessors": [
                        {
                            "name": "tok",
                            "feature": "input_ids",
                            "path": "/p/tok.json",
                        }
                    ],
                },
            )
        )

        status = await client.predictor_status("llama-7b")

        assert isinstance(status, ModelStatus)
        assert status.model == "llama-7b"
        assert status.samples_collected == 1000
        assert len(status.preprocessors) == 1
        assert status.preprocessors[0].name == "tok"


# ------------------------------------------------------------------
# Error mapping
# ------------------------------------------------------------------


class TestErrorMapping:
    """HTTP error codes map to SwarmPilot exceptions."""

    @respx.mock
    async def test_404_from_scheduler_raises_scheduler_not_found(
        self, client: SwarmPilotClient
    ) -> None:
        """404 on scheduler URL raises SchedulerNotFound."""
        respx.post(f"{SCHEDULER}/v1/predictor/train").mock(
            return_value=httpx.Response(
                404,
                json={"detail": "not found"},
            )
        )

        with pytest.raises(SchedulerNotFound):
            await client.train("nonexistent-model")

    @respx.mock
    async def test_404_from_planner_raises_model_not_deployed(
        self, client: SwarmPilotClient
    ) -> None:
        """404 on planner URL raises ModelNotDeployed."""
        respx.get(f"{PLANNER}/v1/instances").mock(
            return_value=httpx.Response(
                404,
                json={"detail": "not found"},
            )
        )

        with pytest.raises(ModelNotDeployed):
            await client.instances()

    @respx.mock
    async def test_400_raises_value_error(
        self, client: SwarmPilotClient
    ) -> None:
        """400 Bad Request maps to ValueError."""
        respx.post(f"{PLANNER}/v1/serve").mock(
            return_value=httpx.Response(
                400,
                json={"detail": "invalid gpu value"},
            )
        )

        with pytest.raises(ValueError, match="Bad request"):
            await client.serve("bad-model", gpu=-1)

    @respx.mock
    async def test_500_raises_http_status_error(
        self, client: SwarmPilotClient
    ) -> None:
        """500 raises httpx.HTTPStatusError."""
        respx.post(f"{PLANNER}/v1/serve").mock(
            return_value=httpx.Response(
                500,
                json={"detail": "internal error"},
            )
        )

        with pytest.raises(httpx.HTTPStatusError):
            await client.serve("model", gpu=1)


# ------------------------------------------------------------------
# Context manager and close()
# ------------------------------------------------------------------


class TestLifecycle:
    """Context manager and close() work correctly."""

    @respx.mock
    async def test_async_context_manager(self) -> None:
        """Async context manager creates and closes the client."""
        respx.get(f"{PLANNER}/v1/schedulers").mock(
            return_value=httpx.Response(
                200,
                json={"schedulers": {}},
            )
        )

        async with SwarmPilotClient(
            planner_url=PLANNER,
            scheduler_url=SCHEDULER,
        ) as sp:
            result = await sp.schedulers()
            assert result == {}

        # After exiting, the internal client should be closed
        assert sp._client.is_closed

    @respx.mock
    async def test_close_closes_http_client(self) -> None:
        """Explicit close() closes the httpx client."""
        sp = SwarmPilotClient(
            planner_url=PLANNER,
            scheduler_url=SCHEDULER,
        )
        assert not sp._client.is_closed

        await sp.close()

        assert sp._client.is_closed


# ------------------------------------------------------------------
# Scheduler URL validation
# ------------------------------------------------------------------


class TestSchedulerUrlValidation:
    """Predictor methods require a scheduler_url."""

    async def test_train_without_scheduler_url_raises(
        self,
    ) -> None:
        """train() raises ValueError when no scheduler_url."""
        sp = SwarmPilotClient(
            planner_url=PLANNER,
            scheduler_url=None,
        )
        try:
            with pytest.raises(
                ValueError,
                match="scheduler_url is required",
            ):
                await sp.train("model")
        finally:
            await sp.close()

    async def test_predict_without_scheduler_url_raises(
        self,
    ) -> None:
        """predict() raises ValueError when no scheduler_url."""
        sp = SwarmPilotClient(
            planner_url=PLANNER,
            scheduler_url=None,
        )
        try:
            with pytest.raises(
                ValueError,
                match="scheduler_url is required",
            ):
                await sp.predict("model", features={"x": 1})
        finally:
            await sp.close()

    async def test_predictor_status_without_scheduler_url_raises(
        self,
    ) -> None:
        """predictor_status() raises ValueError with no URL."""
        sp = SwarmPilotClient(
            planner_url=PLANNER,
            scheduler_url=None,
        )
        try:
            with pytest.raises(
                ValueError,
                match="scheduler_url is required",
            ):
                await sp.predictor_status("model")
        finally:
            await sp.close()
