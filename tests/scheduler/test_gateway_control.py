"""Unit tests for the gateway control channel (phase R3)."""

import asyncio
import json

import grpc
import pytest

from swarmpilot.gateway.v1 import control_pb2 as pb, control_pb2_grpc as pb_grpc
from swarmpilot.scheduler import gateway_control as gc
from swarmpilot.scheduler.models.status import InstanceStatus

GPU = {"software_name": "vllm", "software_version": "1", "hardware_name": "a"}
EXP = {
    "software_name": "exp",
    "software_version": "exp",
    "hardware_name": "exp",
}


class Fake:
    """State sources with what the tests change, and the reports seen."""

    def __init__(self):
        self.instances = [
            gc.InstanceView(
                "a", "http://a", "m", GPU, InstanceStatus.ACTIVE, 8, 2
            ),
            gc.InstanceView(
                "b", "http://b", "m", EXP, InstanceStatus.DRAINING, 0, 0
            ),
        ]
        self.strategy = "min_time"
        self.completions: list[pb.Completion] = []
        self.arrivals: list[tuple[int, dict]] = []
        self.unreachable: list[tuple[str, str]] = []

    def sources(self) -> gc.Sources:
        async def instances():
            return list(self.instances)

        async def on_completion(c):
            self.completions.append(c)

        async def on_unreachable(i, e):
            self.unreachable.append((i, e))

        def hardware(info):
            if info is EXP:
                raise ValueError("no GPU")
            return {"gpu_mem": 80.0}

        return gc.Sources(
            instances=instances,
            strategy=lambda: self.strategy,
            model_id=lambda: "fallback",
            proxy=lambda: pb.ProxySettings(timeout_s=5, max_retries=3),
            manifest=lambda sw: {"name": sw or "passthrough"},
            storage_dir=lambda: "/models",
            sketch_quantiles=lambda: [0.5, 0.9],
            model_key=lambda m, info, t: f"{m}-{info['hardware_name']}-{t}",
            hardware_features=hardware,
            preprocessor_rules=lambda m: False,
            style_paths=lambda: {"v1/messages": "anthropic_messages"},
            on_completion=on_completion,
            on_arrivals=lambda n, c: self.arrivals.append((n, c)),
            on_unreachable=on_unreachable,
        )


async def test_build_state_describes_instances_and_platforms():
    fake = Fake()
    state = await gc.build_state(fake.sources(), 7)
    assert state.version == 7
    assert state.model_id == "m"
    assert state.strategy == "min_time"
    a, b = state.instances
    assert a.status == pb.Instance.ACTIVE and b.status == pb.Instance.DRAINING
    assert (a.max_in_flight, a.servers) == (8, 2)
    # Limits are at least 1, as the gateway's queues need.
    assert (b.max_in_flight, b.servers) == (1, 1)
    assert a.platform_key == json.dumps(GPU, sort_keys=True)
    assert set(state.manifests) == {"", "vllm", "exp"}
    assert json.loads(state.manifests[""]) == {"name": "passthrough"}
    platforms = {p.platform_key: p for p in state.predictor.platforms}
    gpu = platforms[a.platform_key]
    assert gpu.quantile_model_key == "m-a-quantile"
    assert gpu.expect_error_model_key == "m-a-expect_error"
    assert dict(gpu.hardware_features) == {"gpu_mem": 80.0}
    assert not gpu.experiment
    exp = platforms[b.platform_key]
    assert exp.experiment and not exp.hardware_features
    assert len(state.predictor.experiment_normals) == 2000
    assert list(state.predictor.sketch_quantiles) == [0.5, 0.9]
    assert dict(state.api_style_paths) == {"v1/messages": "anthropic_messages"}


def test_scheduler_sends_style_by_route_path():
    """The gateway looks styles up by the path a client called."""
    from swarmpilot.scheduler import api

    paths = api._gateway_sources().style_paths()
    assert paths["v1/messages"] == "anthropic_messages"
    assert paths["v1/chat/completions"] == "openai_chat"
    assert paths["v1/responses"] == "openai_responses"


async def test_build_state_without_instances_uses_configured_model():
    fake = Fake()
    fake.instances = []
    state = await gc.build_state(fake.sources(), 1)
    assert state.model_id == "fallback"
    assert not state.instances
    assert set(state.manifests) == {""}


async def test_unknown_status_is_unhealthy():
    fake = Fake()
    fake.instances[0].status = InstanceStatus.UNHEALTHY
    state = await gc.build_state(fake.sources(), 1)
    assert state.instances[0].status == pb.Instance.UNHEALTHY


@pytest.fixture
async def channel(tmp_path):
    fake = Fake()
    path = tmp_path / "ctl.sock"
    server = await gc.start_server(fake.sources(), str(path))
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    async with grpc.aio.insecure_channel(f"unix:{path}") as ch:
        yield fake, pb_grpc.GatewayControlStub(ch)
    await server.stop(0)


async def test_watch_sends_only_changes(channel):
    fake, stub = channel
    call = stub.WatchState(pb.WatchStateRequest(gateway_id="t"))
    first = await call.read()
    assert first.version == 1 and first.strategy == "min_time"
    fake.strategy = "random"
    second = await asyncio.wait_for(call.read(), 5)
    assert second.version == 2 and second.strategy == "random"
    call.cancel()


async def test_reports_reach_the_sinks(channel):
    fake, stub = channel

    async def batches():
        yield pb.CompletionBatch(
            completions=[pb.Completion(task_id="t1", succeeded=True)],
            arrivals=3,
            calls_from={"parent": 1},
        )
        yield pb.CompletionBatch(completions=[pb.Completion(task_id="t2")])

    await stub.ReportCompletions(batches())
    assert [c.task_id for c in fake.completions] == ["t1", "t2"]
    assert fake.arrivals == [(3, {"parent": 1})]
    await stub.ReportUnreachable(
        pb.UnreachableReport(instance_id="a", error="refused")
    )
    assert fake.unreachable == [("a", "refused")]
