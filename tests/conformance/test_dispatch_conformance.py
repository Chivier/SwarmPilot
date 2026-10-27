"""Black-box dispatch semantics every SwarmPilot data plane must keep.

These tests only speak HTTP: to a data plane and to the fake engine behind
it, whose ``/_stats`` shows what actually reached it. By default they run
twice, once against the Python Scheduler's proxy and once against
``swarmpilot-gateway`` (phase R3) in front of the same Scheduler; the
gateway run is skipped when the binary is not built (``cargo build
--release --manifest-path rust/Cargo.toml``). To check another data plane,
start it in front of a fake
engine with one instance registered as ``vllm`` and a per-instance limit of
4 requests in flight and a 2 s request timeout, then set::

    SWARMPILOT_CONFORMANCE_PROXY=http://host:port
    SWARMPILOT_CONFORMANCE_ENGINE=http://host:port
"""

import asyncio
import json
import os
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
    gateway_cmd,
    scheduler_cmd,
    spawn,
    wait_gateway_instances,
)

MAX_IN_FLIGHT = 4
TIMEOUT_S = 2


@pytest.fixture(scope="module", params=["python", "gateway"])
def urls(request):
    proxy = os.getenv("SWARMPILOT_CONFORMANCE_PROXY")
    engine = os.getenv("SWARMPILOT_CONFORMANCE_ENGINE")
    if proxy and engine:
        if request.param != "python":
            pytest.skip("an external data plane is checked once")
        yield proxy.rstrip("/"), engine.rstrip("/")
        return
    binary = None
    if request.param == "gateway":
        binary = gateway_binary()
        if binary is None:
            pytest.skip("swarmpilot-gateway is not built")

    eport, sport = free_port(), free_port()
    engine = f"http://127.0.0.1:{eport}"
    proxy = f"http://127.0.0.1:{sport}"
    tmp = Path(tempfile.mkdtemp(prefix="sp-conf-"))
    env = {
        "SCHEDULING_STRATEGY": "round_robin",
        "SCHEDULER_LOGURU_LEVEL": "WARNING",
        "SCHEDULER_LOG_DIR": str(tmp / "logs"),
        "PREDICTOR_STORAGE_DIR": str(tmp / "models"),
        "SCHEDULER_MODEL_ID": "fake-model",
        "TRAINING_ENABLE_AUTO": "false",
        "PROXY_MAX_IN_FLIGHT": str(MAX_IN_FLIGHT),
        "PROXY_TIMEOUT": str(TIMEOUT_S),
    }
    control = tmp / "gateway.sock"
    if binary is not None:
        env["SCHEDULER_GATEWAY_SOCKET"] = str(control)
    with ExitStack() as stack:
        stack.enter_context(
            spawn(
                fake_engine_cmd(eport, latency_ms=0),
                f"{engine}/health",
                log_path=tmp / "engine.log",
            )
        )
        stack.enter_context(
            spawn(
                scheduler_cmd(sport),
                f"{proxy}/v1/health",
                env=env,
                log_path=tmp / "scheduler.log",
            )
        )
        r = httpx.post(
            f"{proxy}/v1/instance/register",
            json={
                "instance_id": "fake-0",
                "model_id": "fake-model",
                "endpoint": engine,
                "platform_info": {
                    "software_name": "vllm",
                    "software_version": "0.0.0",
                    "hardware_name": "cpu",
                },
            },
        )
        r.raise_for_status()
        if binary is None:
            yield proxy, engine
            return
        gport = free_port()
        gateway = f"http://127.0.0.1:{gport}"
        stack.enter_context(
            spawn(
                gateway_cmd(binary, gport, control),
                f"{gateway}/_gateway/health",
                log_path=tmp / "gateway.log",
            )
        )
        wait_gateway_instances(gateway, 1)
        yield gateway, engine


@pytest.fixture
def engine_stats(urls):
    _, engine = urls
    httpx.post(f"{engine}/_reset").raise_for_status()
    return lambda: httpx.get(f"{engine}/_stats").json()


def _chat(**extra) -> dict:
    return {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        **extra,
    }


def test_json_response_is_passed_through(urls, engine_stats):
    proxy, _ = urls
    r = httpx.post(f"{proxy}/v1/chat/completions", json=_chat(), timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["usage"]["completion_tokens"] == 16
    assert engine_stats()["requests"] == {"/v1/chat/completions": 1}


def test_engine_error_status_and_body_are_passed_through(urls):
    proxy, _ = urls
    r = httpx.post(
        f"{proxy}/v1/chat/completions", json=_chat(fake_status=400), timeout=10
    )
    assert r.status_code == 400
    assert r.json()["error"]["message"] == "fake error"


def test_stream_arrives_incrementally(urls):
    """The first SSE chunk is relayed long before the stream ends."""
    proxy, _ = urls
    body = _chat(stream=True, fake_latency_ms=1000, fake_tokens=10)
    start = time.monotonic()
    first = None
    chunks = []
    with httpx.stream(
        "POST", f"{proxy}/v1/chat/completions", json=body, timeout=10
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if not line:
                continue
            if first is None:
                first = time.monotonic() - start
            chunks.append(line)
    total = time.monotonic() - start
    assert chunks[-1] == "data: [DONE]"
    assert len(chunks) == 11
    assert json.loads(chunks[0][6:])["choices"][0]["delta"] == {"content": "x"}
    assert first < 0.5 < total


async def test_in_flight_limit_per_instance(urls, engine_stats):
    """No more than the instance's limit reaches the engine at once.

    The limit is also used: requests do not go one at a time.
    """
    proxy, _ = urls

    async def run() -> float:
        async with aiohttp.ClientSession() as s:

            async def one():
                async with s.post(
                    f"{proxy}/v1/completions",
                    json={"prompt": "x", "fake_latency_ms": 200},
                ) as r:
                    assert r.status == 200
                    await r.read()

            start = time.monotonic()
            await asyncio.gather(*(one() for _ in range(4 * MAX_IN_FLIGHT)))
            return time.monotonic() - start

    elapsed = await run()
    stats = engine_stats()
    assert stats["max_in_flight"] == MAX_IN_FLIGHT
    assert stats["requests"]["/v1/completions"] == 4 * MAX_IN_FLIGHT
    # 4 waves of 0.2 s, far from 16 sequential requests (3.2 s).
    assert elapsed < 2.0


async def test_lightweight_route_skips_busy_queue(urls):
    """GET /v1/models answers while inference requests fill the queue."""
    proxy, _ = urls

    async def run() -> float:
        async with aiohttp.ClientSession() as s:

            async def slow():
                async with s.post(
                    f"{proxy}/v1/completions",
                    json={"prompt": "x", "fake_latency_ms": 600},
                ) as r:
                    await r.read()

            busy = [
                asyncio.create_task(slow()) for _ in range(3 * MAX_IN_FLIGHT)
            ]
            await asyncio.sleep(0.1)
            start = time.monotonic()
            async with s.get(f"{proxy}/v1/models") as r:
                assert r.status == 200
                assert (await r.json())["data"][0]["id"] == "fake-model"
            took = time.monotonic() - start
            await asyncio.gather(*busy)
            return took

    assert await run() < 0.4


def test_slow_engine_times_out_with_504(urls):
    proxy, _ = urls
    r = httpx.post(
        f"{proxy}/v1/completions",
        json={"prompt": "x", "fake_latency_ms": (TIMEOUT_S + 1) * 1000},
        timeout=TIMEOUT_S + 5,
    )
    assert r.status_code == 504
