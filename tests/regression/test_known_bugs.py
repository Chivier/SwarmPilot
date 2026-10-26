"""Regression tests for the five known scheduling bugs (plan item 0.4).

Each test describes the behaviour the paper promises and is marked
``xfail(strict=True)`` because it fails at the baseline commit. When the
phase that fixes a bug lands, its test starts passing, strict xfail turns
that into a failure, and the fixing PR must remove the marker.

| Test                         | Gap ids | Fixed in plan item |
|------------------------------|---------|--------------------|
| test_drain_counts_and_reroutes_queued_tasks | S5, S4 | 2.1 |
| test_queue_sketch_returns_to_empty          | R2, A1 | 2.2 |
| test_random_choices_are_random              | R4, R7 | 2.3 |
| test_retraining_does_not_block_event_loop   | A3     | 2.4 |
| test_planner_accepts_scheduler_reports      | S3     | 3.2 |
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import Counter
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.stats import chisquare

from swarmpilot.shared.loop_lag import EventLoopLagMonitor

EXP_PLATFORM = {
    "software_name": "exp",
    "software_version": "exp",
    "hardware_name": "exp",
}


def _fake_engine(delay_s: float) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("content-length", 0)))
            time.sleep(delay_s)
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def slow_engine() -> Iterator[str]:
    """A fake model server that takes 0.5 s per request."""
    srv = _fake_engine(0.5)
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@pytest.fixture
def fast_engine() -> Iterator[str]:
    """A fake model server that answers immediately."""
    srv = _fake_engine(0.0)
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _run(coro) -> None:
    # asyncio.run() would unset the thread's event loop, which other test
    # modules still read with asyncio.get_event_loop().
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def scheduler() -> Iterator[TestClient]:
    """The real Scheduler app with empty registries."""
    from swarmpilot.scheduler import api

    _run(api.task_registry.clear_all())
    _run(api.instance_registry.clear_all())
    with TestClient(api.app) as client:
        yield client
    _run(api.task_registry.clear_all())
    _run(api.instance_registry.clear_all())


def _set_strategy(client: TestClient, name: str) -> None:
    r = client.post("/v1/strategy/set", json={"strategy_name": name})
    assert r.status_code == 200, r.text


def _register(
    client: TestClient, iid: str, endpoint: str, platform: dict | None = None
) -> None:
    r = client.post(
        "/v1/instance/register",
        json={
            "instance_id": iid,
            "model_id": "m",
            "endpoint": endpoint,
            "platform_info": platform
            or {
                "software_name": "vllm",
                "software_version": "x",
                "hardware_name": "h",
            },
        },
    )
    assert r.status_code == 200, r.text


def _submit(client: TestClient, task_id: str, metadata: dict) -> None:
    r = client.post(
        "/v1/task/submit",
        json={
            "task_id": task_id,
            "model_id": "m",
            "task_input": {"prompt": "x"},
            "metadata": metadata,
        },
    )
    assert r.status_code == 200, r.text


def _wait_all_done(
    client: TestClient, task_ids: list[str], timeout: float
) -> dict[str, str]:
    deadline = time.monotonic() + timeout
    statuses: dict[str, str] = {}
    while time.monotonic() < deadline:
        statuses = {
            t: client.get("/v1/task/info", params={"task_id": t}).json()[
                "task"
            ]["status"]
            for t in task_ids
        }
        if all(s in ("completed", "failed") for s in statuses.values()):
            break
        time.sleep(0.1)
    return statuses


@pytest.mark.xfail(
    strict=True,
    reason="gap S5/S4: drain ignores queued tasks and remove drops them "
    "(fixed by plan item 2.1)",
)
def test_drain_counts_and_reroutes_queued_tasks(
    scheduler: TestClient, slow_engine: str
) -> None:
    """Drain must report queued work, and remove must not lose it."""
    _set_strategy(scheduler, "round_robin")
    _register(scheduler, "A", slow_engine)
    _register(scheduler, "B", slow_engine)
    task_ids = [f"t{i}" for i in range(6)]
    for t in task_ids:
        _submit(scheduler, t, {})
    time.sleep(0.2)

    scheduler.post("/v1/instance/drain", json={"instance_id": "A"})
    st = scheduler.get(
        "/v1/instance/drain/status", params={"instance_id": "A"}
    ).json()
    assert st["pending_tasks"] >= 2, st
    assert st["can_remove"] is False, st

    scheduler.post("/v1/instance/remove", json={"instance_id": "A"})
    statuses = _wait_all_done(scheduler, task_ids, timeout=10)
    assert statuses == dict.fromkeys(task_ids, "completed"), statuses


@pytest.mark.xfail(
    strict=True,
    reason="gap R2/A1: queue sketches are never decremented on completion "
    "(fixed by plan item 2.2)",
)
def test_queue_sketch_returns_to_empty(
    scheduler: TestClient, fast_engine: str
) -> None:
    """Every queue sketch must be empty once all tasks have finished."""
    _set_strategy(scheduler, "probabilistic")
    for iid in ("A", "B"):
        _register(scheduler, iid, fast_engine, EXP_PLATFORM)
    task_ids = [f"s{i}" for i in range(20)]
    for t in task_ids:
        _submit(scheduler, t, {"exp_runtime": 100.0, "exp_cv": 0.3})
    statuses = _wait_all_done(scheduler, task_ids, timeout=10)
    assert set(statuses.values()) == {"completed"}, statuses
    time.sleep(0.2)

    for iid in ("A", "B"):
        info = scheduler.get("/v1/instance/info", params={"instance_id": iid})
        queue = info.json()["queue_info"]
        values = queue.get("values") or [queue.get("expected_time_ms", 0.0)]
        assert max(values) == pytest.approx(0.0, abs=1e-6), (iid, queue)


@pytest.mark.xfail(
    strict=True,
    reason="gap R4/R7: the predictor reseeds random and np.random to 42 on "
    "every call (fixed by plan item 2.3)",
)
async def test_random_choices_are_random(tmp_path) -> None:
    """Random must be uniform over replicas, and PO2 must use all of them."""
    from loguru import logger

    from swarmpilot.scheduler.algorithms.factory import get_strategy
    from swarmpilot.scheduler.clients.predictor_library_client import (
        PredictorClient,
    )
    from swarmpilot.scheduler.models import Instance
    from swarmpilot.scheduler.registry.instance_registry import (
        InstanceRegistry,
    )

    logger.disable("swarmpilot")
    try:
        insts = [
            Instance(
                instance_id=f"i{k}",
                model_id="m",
                endpoint=f"http://x{k}",
                platform_info=EXP_PLATFORM,
            )
            for k in range(8)
        ]
        predictor = PredictorClient(storage_dir=str(tmp_path))
        md = {"exp_runtime": 100.0, "exp_cv": 0.3}
        picks: dict[str, Counter] = {}
        for name, n in (("random", 10_000), ("po2", 2_000)):
            qtype = "expect_error" if name == "po2" else "probabilistic"
            reg = InstanceRegistry(queue_info_type=qtype)
            for inst in insts:
                await reg.register(inst)
            strategy = get_strategy(name, predictor, reg)
            counts: Counter = Counter()
            for _ in range(n):
                res = await strategy.schedule_task("m", md, insts)
                counts[res.selected_instance_id] += 1
            picks[name] = counts
    finally:
        logger.enable("swarmpilot")

    observed = [picks["random"][f"i{k}"] for k in range(8)]
    assert chisquare(observed).pvalue > 0.01, observed
    assert len(picks["po2"]) == 8, picks["po2"]


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="gap A3: auto-training fits the model on the event loop "
    "(fixed by plan item 2.4)",
)
async def test_retraining_does_not_block_event_loop(tmp_path) -> None:
    """A retrain must keep event-loop lag P99 under 50 ms."""
    from loguru import logger

    from swarmpilot.scheduler.clients.predictor_library_client import (
        PredictorClient,
    )
    from swarmpilot.scheduler.clients.training_library_client import (
        TrainingClient,
    )

    predictor = PredictorClient(storage_dir=str(tmp_path))
    trainer = TrainingClient(
        storage=predictor._low_level._storage,
        cache=predictor._low_level._cache,
        batch_size=100,
        prediction_types=["quantile"],
    )
    rng = np.random.default_rng(0)
    plat = {
        "software_name": "vllm",
        "software_version": "x",
        "hardware_name": "h",
    }
    for _ in range(100):
        plen, maxt = int(rng.integers(50, 4000)), int(rng.choice([256, 1024]))
        trainer.add_sample(
            model_id="m",
            platform_info=plat,
            features={"prompt_length": plen, "max_tokens": maxt},
            actual_runtime_ms=50 + 0.05 * plen + 0.8 * maxt,
        )

    monitor = EventLoopLagMonitor(interval_s=0.01, warn_ms=1e9)
    monitor.start()
    logger.disable("swarmpilot")
    try:
        # Same call TaskResultCallback makes after every completed task.
        await asyncio.sleep(0.1)
        assert await trainer.flush_if_ready()
        await asyncio.sleep(0.1)
    finally:
        logger.enable("swarmpilot")
        await monitor.stop()

    snap = monitor.snapshot()
    assert snap["p99_ms"] <= 50, json.dumps(snap)


@pytest.mark.xfail(
    strict=True,
    reason="gap S3: the Planner has no /v1/submit_target or "
    "/v1/submit_throughput (fixed by plan item 3.2)",
)
def test_planner_accepts_scheduler_reports() -> None:
    """The Planner must accept the reports the Scheduler already sends."""
    from swarmpilot.planner.api import app

    with TestClient(app) as client:
        r = client.post(
            "/v1/submit_target", json={"model_id": "m", "value": 3.0}
        )
        assert r.is_success, (r.status_code, r.text)
        r = client.post(
            "/v1/submit_throughput",
            json={"instance_url": "http://x:8000", "avg_execution_time": 1.5},
        )
        assert r.is_success, (r.status_code, r.text)
