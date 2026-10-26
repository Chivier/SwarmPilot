"""Drain counts queued work and remove re-routes it (plan item 2.1)."""

import asyncio
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

# An engine name with no adapter: one request in flight per instance, so
# extra tasks wait in the Scheduler's queue.
ONE_AT_A_TIME = {
    "software_name": "test-serial",
    "software_version": "1",
    "hardware_name": "cpu",
}


class _Engine:
    def __init__(self, delay_s: float):
        self.seen: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                n = int(self.headers.get("content-length", 0))
                outer.seen.append(self.rfile.read(n).decode())
                time.sleep(delay_s)
                body = b'{"ok": true}'
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}"


@pytest.fixture
def engines() -> Iterator[tuple[_Engine, _Engine]]:
    a, b = _Engine(0.3), _Engine(0.0)
    yield a, b
    a.srv.shutdown()
    b.srv.shutdown()


def _run(coro) -> None:
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def client() -> Iterator[TestClient]:
    from swarmpilot.scheduler import api

    _run(api.task_registry.clear_all())
    _run(api.instance_registry.clear_all())
    with TestClient(api.app) as c:
        r = c.post("/v1/strategy/set", json={"strategy_name": "round_robin"})
        assert r.status_code == 200
        yield c
    _run(api.task_registry.clear_all())
    _run(api.instance_registry.clear_all())


def _register(c: TestClient, iid: str, url: str) -> None:
    r = c.post(
        "/v1/instance/register",
        json={
            "instance_id": iid,
            "model_id": "m",
            "endpoint": url,
            "platform_info": ONE_AT_A_TIME,
        },
    )
    assert r.status_code == 200, r.text


def _submit(c: TestClient, tid: str) -> None:
    r = c.post(
        "/v1/task/submit",
        json={
            "task_id": tid,
            "model_id": "m",
            "task_input": {"prompt": tid},
            "metadata": {},
        },
    )
    assert r.status_code == 200, r.text


def _statuses(c: TestClient, ids: list[str], timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        st = {
            t: c.get("/v1/task/info", params={"task_id": t}).json()["task"]
            for t in ids
        }
        done = all(v["status"] in ("completed", "failed") for v in st.values())
        if done or time.monotonic() > deadline:
            return st
        time.sleep(0.05)


def test_drain_status_counts_queue_and_in_flight(client, engines):
    a, _ = engines
    _register(client, "A", a.url)
    for i in range(3):
        _submit(client, f"t{i}")
    time.sleep(0.1)
    client.post("/v1/instance/drain", json={"instance_id": "A"})
    st = client.get(
        "/v1/instance/drain/status", params={"instance_id": "A"}
    ).json()
    assert st["pending_tasks"] == 3
    assert st["running_tasks"] == 1
    assert st["can_remove"] is False

    _statuses(client, ["t0", "t1", "t2"], timeout=5)
    time.sleep(0.05)
    st = client.get(
        "/v1/instance/drain/status", params={"instance_id": "A"}
    ).json()
    assert st == {**st, "pending_tasks": 0, "can_remove": True}


def test_remove_reroutes_queued_tasks(client, engines):
    a, b = engines
    _register(client, "A", a.url)
    ids = [f"t{i}" for i in range(4)]
    for t in ids:
        _submit(client, t)
    time.sleep(0.1)
    _register(client, "B", b.url)
    time.sleep(0.2)  # let B finish registering (work stealing) first
    client.post("/v1/instance/drain", json={"instance_id": "A"})
    client.post("/v1/instance/remove", json={"instance_id": "A"})

    st = _statuses(client, ids, timeout=5)
    assert {v["status"] for v in st.values()} == {"completed"}, st
    # The task in flight on A finished there; the rest ran on B.
    assert len(a.seen) + len(b.seen) == 4
    assert len(b.seen) >= 1
    assert all(st[t]["assigned_instance"] in ("A", "B") for t in ids)


def test_remove_without_other_instance_fails_queued_tasks(client, engines):
    a, _ = engines
    _register(client, "A", a.url)
    ids = [f"t{i}" for i in range(3)]
    for t in ids:
        _submit(client, t)
    time.sleep(0.1)
    client.post("/v1/instance/remove", json={"instance_id": "A"})

    st = _statuses(client, ids, timeout=5)
    statuses = sorted(v["status"] for v in st.values())
    assert statuses == ["completed", "failed", "failed"], st
    failed = [v for v in st.values() if v["status"] == "failed"]
    assert all("could not be re-routed" in v["error"] for v in failed)
