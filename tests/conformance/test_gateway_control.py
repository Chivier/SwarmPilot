"""End to end: the Scheduler and ``swarmpilot-gateway`` over the control channel.

The dispatch conformance suite checks what clients see. These tests check
what the Scheduler sees when the gateway serves the traffic: requests it
finished are counted on their instances and become training samples, an
instance it cannot reach is marked UNHEALTHY in the Scheduler, and changes
made through the Scheduler's API (registrations, strategy switches,
removals) reach the gateway. They are skipped when the gateway binary is
not built.
"""

import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "perf"))
from _harness import (
    fake_engine_cmd,
    free_port,
    gateway_binary,
    gateway_cmd,
    scheduler_cmd,
    spawn,
    wait_gateway_instances,
)

PLATFORM = {
    "software_name": "vllm",
    "software_version": "0.0.0",
    "hardware_name": "cpu",
}


class Stack:
    """Scheduler, gateway and one fake engine, started for the module."""

    def __init__(self, tmp: Path, scheduler: str, gateway: str, engine: str):
        self.tmp = tmp
        self.scheduler = scheduler
        self.gateway = gateway
        self.engine = engine

    def register(self, instance_id: str, endpoint: str) -> None:
        r = httpx.post(
            f"{self.scheduler}/v1/instance/register",
            json={
                "instance_id": instance_id,
                "model_id": "fake-model",
                "endpoint": endpoint,
                "platform_info": PLATFORM,
            },
        )
        r.raise_for_status()

    def remove(self, instance_id: str) -> None:
        httpx.post(
            f"{self.scheduler}/v1/instance/remove",
            json={"instance_id": instance_id},
        ).raise_for_status()

    def info(self, instance_id: str) -> dict:
        r = httpx.get(
            f"{self.scheduler}/v1/instance/info",
            params={"instance_id": instance_id},
        )
        r.raise_for_status()
        return r.json()

    def gateway_health(self) -> dict:
        return httpx.get(f"{self.gateway}/_gateway/health").json()


def _wait(check: Callable[[], bool], what: str, timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture(scope="module")
def stack():
    binary = gateway_binary()
    if binary is None:
        pytest.skip("swarmpilot-gateway is not built")
    eport, sport, gport = free_port(), free_port(), free_port()
    tmp = Path(tempfile.mkdtemp(prefix="sp-gw-"))
    control = tmp / "gateway.sock"
    env = {
        "SCHEDULING_STRATEGY": "round_robin",
        "SCHEDULER_LOGURU_LEVEL": "WARNING",
        "SCHEDULER_LOG_DIR": str(tmp / "logs"),
        "PREDICTOR_STORAGE_DIR": str(tmp / "models"),
        "SCHEDULER_MODEL_ID": "fake-model",
        "SCHEDULER_GATEWAY_SOCKET": str(control),
        "TRAINING_ENABLE_AUTO": "true",
        "TRAINING_BATCH_SIZE": "20",
        "TRAINING_MIN_SAMPLES": "10",
        "TRAINING_HOLDOUT": "0",
        "PROXY_TIMEOUT": "5",
    }
    s = Stack(
        tmp,
        f"http://127.0.0.1:{sport}",
        f"http://127.0.0.1:{gport}",
        f"http://127.0.0.1:{eport}",
    )
    with ExitStack() as es:
        es.enter_context(
            spawn(
                fake_engine_cmd(eport, latency_ms=0),
                f"{s.engine}/health",
                log_path=tmp / "engine.log",
            )
        )
        es.enter_context(
            spawn(
                scheduler_cmd(sport),
                f"{s.scheduler}/v1/health",
                env=env,
                log_path=tmp / "scheduler.log",
            )
        )
        es.enter_context(
            spawn(
                gateway_cmd(binary, gport, control),
                f"{s.gateway}/_gateway/health",
                log_path=tmp / "gateway.log",
            )
        )
        yield s


def test_no_instances_gives_503(stack):
    wait_gateway_instances(stack.gateway, 0)
    r = httpx.post(
        f"{stack.gateway}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 503
    assert "error" in r.json()


def test_registration_reaches_gateway(stack):
    stack.register("fake-0", stack.engine)
    wait_gateway_instances(stack.gateway, 1)
    health = stack.gateway_health()
    assert health["model_id"] == "fake-model"
    assert health["strategy"] == "round_robin"


def test_completions_are_counted_on_the_instance(stack):
    before = stack.info("fake-0")["stats"]
    for status in (200, 200, 500):
        httpx.post(
            f"{stack.gateway}/v1/completions",
            json={"prompt": "x", "fake_status": status},
        )

    def counted() -> bool:
        stats = stack.info("fake-0")["stats"]
        return (
            stats["completed_tasks"] == before["completed_tasks"] + 2
            and stats["failed_tasks"] == before["failed_tasks"] + 1
        )

    _wait(counted, "completion reports")


def test_completions_train_the_predictor(stack):
    """Samples the gateway reports train models the gateway then loads."""
    for i in range(30):
        r = httpx.post(
            f"{stack.gateway}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "x" * (10 + i)}],
                "max_tokens": 8 + i,
                "fake_latency_ms": 5 + i,
            },
        )
        assert r.status_code == 200
    models = stack.tmp / "models"
    _wait(
        lambda: models.exists() and any(models.glob("*.json")),
        "a trained model",
        timeout=60,
    )
    log = stack.tmp / "gateway.log"
    _wait(
        lambda: "loaded predictor model" in log.read_text(),
        "the gateway to load the model",
        timeout=10,
    )


def test_strategy_switch_reaches_gateway(stack):
    r = httpx.post(
        f"{stack.scheduler}/v1/strategy/set",
        json={"strategy_name": "random"},
    )
    r.raise_for_status()
    _wait(
        lambda: stack.gateway_health()["strategy"] == "random",
        "the strategy switch",
    )
    httpx.post(
        f"{stack.scheduler}/v1/strategy/set",
        json={"strategy_name": "round_robin"},
    ).raise_for_status()


def test_unreachable_instance_is_reported_and_avoided(stack):
    """A refused connection marks the instance UNHEALTHY in the Scheduler.

    Every request still succeeds on the other instance.
    """
    stack.register("dead-0", f"http://127.0.0.1:{free_port()}")
    wait_gateway_instances(stack.gateway, 2)
    for _ in range(6):
        r = httpx.post(f"{stack.gateway}/v1/completions", json={"prompt": "x"})
        assert r.status_code == 200
    _wait(
        lambda: stack.info("dead-0")["instance"]["status"] == "unhealthy",
        "the Scheduler to mark the instance unhealthy",
    )
    stack.remove("dead-0")
    wait_gateway_instances(stack.gateway, 1)
