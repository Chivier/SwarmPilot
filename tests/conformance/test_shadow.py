"""Shadow mode: the gateway routes every request the Python proxy routes.

Three fake engines sit behind three instances on the experiment platform,
so predictions come from each request's ``exp_*`` fields. The Scheduler
serves the traffic and mirrors each decision, with the draws its strategy
made, to a gateway started with ``--shadow``. For every strategy, random
ones included, the gateway must make the same decision every time.
Skipped when the gateway binary is not built.
"""

import asyncio
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import aiohttp
import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "perf"))
from _harness import (
    fake_engine_cmd,
    free_port,
    gateway_binary,
    scheduler_cmd,
    spawn,
    wait_gateway_instances,
)

EXP = {
    "software_name": "exp",
    "software_version": "exp",
    "hardware_name": "exp",
}
N_ENGINES = 3
GAP_S = 0.03
# Shorter than the sequence: charges are released while it is sent.
LATENCY_MS = 300
STRATEGIES = [
    "min_time",
    "probabilistic",
    "po2",
    "random",
    "round_robin",
    "adaptive_bootstrap",
]

# Varied runtimes and spreads, as in test_routing_ab.
REQUESTS = [
    {"exp_runtime": 100 + 37 * (i % 7) ** 2, "exp_cv": 0.05 * (1 + i % 5)}
    | ({"exp_skewness": 1.5} if i % 4 == 1 else {})
    | (
        {"exp_modes": [{"mean": 80, "weight": 0.6}, {"mean": 400}]}
        if i % 6 == 3
        else {}
    )
    for i in range(24)
]


async def _send(url: str) -> None:
    async with aiohttp.ClientSession() as s:

        async def one(body: dict) -> None:
            async with s.post(
                f"{url}/v1/completions",
                json={"prompt": "x", "fake_latency_ms": LATENCY_MS, **body},
            ) as r:
                assert r.status == 200, await r.text()

        tasks = []
        for body in REQUESTS:
            tasks.append(asyncio.create_task(one(body)))
            await asyncio.sleep(GAP_S)
        await asyncio.gather(*tasks)


def _wait(check, what: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture(scope="module")
def shadow_stack():
    binary = gateway_binary()
    if binary is None:
        pytest.skip("swarmpilot-gateway is not built")
    tmp = Path(tempfile.mkdtemp(prefix="sp-shadow-"))
    sport, gport = free_port(), free_port()
    scheduler = f"http://127.0.0.1:{sport}"
    gateway = f"http://127.0.0.1:{gport}"
    control = tmp / "gateway.sock"
    env = {
        "SCHEDULING_STRATEGY": "min_time",
        "SCHEDULER_LOGURU_LEVEL": "WARNING",
        "SCHEDULER_LOG_DIR": str(tmp / "logs"),
        "PREDICTOR_STORAGE_DIR": str(tmp / "models"),
        "SCHEDULER_MODEL_ID": "fake-model",
        "TRAINING_ENABLE_AUTO": "false",
        "PROXY_TIMEOUT": "30",
        "SCHEDULER_GATEWAY_SOCKET": str(control),
        "SCHEDULER_GATEWAY_SHADOW_URL": gateway,
    }
    with ExitStack() as es:
        engines = []
        for i in range(N_ENGINES):
            port = free_port()
            engines.append(f"http://127.0.0.1:{port}")
            es.enter_context(
                spawn(
                    fake_engine_cmd(port, latency_ms=0),
                    f"{engines[-1]}/health",
                    env={"FAKE_ENGINE_MODEL": f"engine-{i}"},
                )
            )
        es.enter_context(
            spawn(
                scheduler_cmd(sport),
                f"{scheduler}/v1/health",
                env=env,
                log_path=tmp / "scheduler.log",
            )
        )
        for i, endpoint in enumerate(engines):
            httpx.post(
                f"{scheduler}/v1/instance/register",
                json={
                    "instance_id": f"i{i}",
                    "model_id": "fake-model",
                    "endpoint": endpoint,
                    "platform_info": EXP,
                },
            ).raise_for_status()
        es.enter_context(
            spawn(
                [
                    str(binary),
                    "--listen",
                    f"127.0.0.1:{gport}",
                    "--control",
                    str(control),
                    "--shadow",
                ],
                f"{gateway}/_gateway/health",
                log_path=tmp / "gateway.log",
            )
        )
        wait_gateway_instances(gateway, N_ENGINES)
        yield scheduler, gateway


def _shadow(scheduler: str) -> dict:
    return httpx.get(f"{scheduler}/v1/gateway/shadow").json()


@pytest.mark.slow
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_gateway_decides_like_python(shadow_stack, strategy):
    scheduler, gateway = shadow_stack
    httpx.post(
        f"{scheduler}/v1/strategy/set", json={"strategy_name": strategy}
    ).raise_for_status()
    _wait(
        lambda: httpx.get(f"{gateway}/_gateway/health").json()["strategy"]
        == strategy,
        "the gateway to follow the strategy",
    )
    before = _shadow(scheduler)
    asyncio.run(_send(scheduler))
    n = len(REQUESTS)
    _wait(
        lambda: _shadow(scheduler)["decisions"] == before["decisions"] + n,
        "the shadow decisions",
    )
    after = _shadow(scheduler)
    assert after["errors"] == before["errors"], after["last_error"]
    assert after["dropped"] == 0
    assert after["draws_diverged"] == before["draws_diverged"]
    mismatched = after["mismatched"] - before["mismatched"]
    assert mismatched == 0, after["recent_mismatches"]


def test_shadow_gateway_serves_no_traffic(shadow_stack):
    _, gateway = shadow_stack
    r = httpx.post(f"{gateway}/v1/completions", json={"prompt": "x"})
    assert r.status_code == 503
    assert "shadow" in r.json()["error"]["message"]


def test_shadow_is_off_without_the_setting():
    from fastapi.testclient import TestClient

    from swarmpilot.scheduler.api import app

    with TestClient(app) as client:
        assert client.get("/v1/gateway/shadow").json() == {"enabled": False}
