"""Routing A/B: the Python proxy and the gateway make the same decisions.

Three fake engines, each answering with its own model name, sit behind
three instances on the experiment platform, so every prediction comes
from the request's ``exp_*`` fields and is deterministic. Requests are
sent 40 ms apart and each runs for a second, so the queues only grow
while the sequence is sent and every decision depends on the charges of
the ones before it. With a strategy that draws no random numbers, the
sequence of chosen instances must be identical on both data planes.
Skipped when the gateway binary is not built.
"""

import asyncio
import sys
import tempfile
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
    gateway_cmd,
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
GAP_S = 0.04
LATENCY_MS = 1000

# Varied runtimes and spreads; a few modes and skewed ones too.
REQUESTS = [
    {"exp_runtime": 100 + 37 * (i % 7) ** 2, "exp_cv": 0.05 * (1 + i % 5)}
    | ({"exp_skewness": 1.5} if i % 4 == 1 else {})
    | (
        {"exp_modes": [{"mean": 80, "weight": 0.6}, {"mean": 400}]}
        if i % 6 == 3
        else {}
    )
    for i in range(18)
]


async def _send_sequence(url: str) -> list[str]:
    async with aiohttp.ClientSession() as s:

        async def one(body: dict) -> str:
            async with s.post(
                f"{url}/v1/completions",
                json={"prompt": "x", "fake_latency_ms": LATENCY_MS, **body},
            ) as r:
                assert r.status == 200, await r.text()
                return (await r.json())["model"]

        tasks = []
        for body in REQUESTS:
            tasks.append(asyncio.create_task(one(body)))
            await asyncio.sleep(GAP_S)
        return list(await asyncio.gather(*tasks))


def _run(strategy: str, gateway: bool) -> list[str]:
    binary = gateway_binary() if gateway else None
    tmp = Path(tempfile.mkdtemp(prefix="sp-ab-"))
    sport = free_port()
    scheduler = f"http://127.0.0.1:{sport}"
    control = tmp / "gateway.sock"
    env = {
        "SCHEDULING_STRATEGY": strategy,
        "SCHEDULER_LOGURU_LEVEL": "WARNING",
        "SCHEDULER_LOG_DIR": str(tmp / "logs"),
        "PREDICTOR_STORAGE_DIR": str(tmp / "models"),
        "SCHEDULER_MODEL_ID": "fake-model",
        "TRAINING_ENABLE_AUTO": "false",
        "PROXY_TIMEOUT": "30",
    }
    if gateway:
        env["SCHEDULER_GATEWAY_SOCKET"] = str(control)
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
            spawn(scheduler_cmd(sport), f"{scheduler}/v1/health", env=env)
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
        url = scheduler
        if gateway:
            gport = free_port()
            url = f"http://127.0.0.1:{gport}"
            es.enter_context(
                spawn(
                    gateway_cmd(binary, gport, control),
                    f"{url}/_gateway/health",
                    log_path=tmp / "gateway.log",
                )
            )
            wait_gateway_instances(url, N_ENGINES)
        return asyncio.run(_send_sequence(url))


@pytest.mark.slow
@pytest.mark.parametrize("strategy", ["min_time", "serverless", "round_robin"])
def test_same_decisions_as_python(strategy):
    if gateway_binary() is None:
        pytest.skip("swarmpilot-gateway is not built")
    python = _run(strategy, gateway=False)
    rust = _run(strategy, gateway=True)
    # The sequence is not trivial: every engine gets work.
    assert set(python) == {f"engine-{i}" for i in range(N_ENGINES)}
    assert rust == python
