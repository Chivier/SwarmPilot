"""An unreachable instance stops getting traffic at once.

A crashed engine refuses connections. Its requests move to other
instances instead of failing after seconds of retries, and the instance
comes back when it answers again or is removed after a while.
"""

import asyncio
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

# No adapter: requests pass through, and any answer counts as healthy.
PLATFORM = {
    "software_name": "test-plain",
    "software_version": "1",
    "hardware_name": "cpu",
}


class _Engine:
    def __init__(self, port: int = 0, status: int = 200, drop: bool = False):
        self.seen = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                n = int(self.headers.get("content-length", 0))
                self.rfile.read(n)
                outer.seen += 1
                if drop:
                    # Dies mid-request: the connection closes unanswered.
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                self._reply(status)

            def do_GET(self) -> None:
                self._reply()

            def _reply(self, code: int = 200) -> None:
                body = b'{"ok": true}'
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        ThreadingHTTPServer.allow_reuse_address = True
        self.srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}"

    def stop(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()


def _dead_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(coro) -> None:
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def client(monkeypatch) -> Iterator[TestClient]:
    from swarmpilot.scheduler import api

    monkeypatch.setattr(api.config.health, "check_interval", 0.1)
    monkeypatch.setattr(api.config.health, "remove_after", 0)
    _run(api.task_registry.clear_all())
    _run(api.instance_registry.clear_all())
    with TestClient(api.app) as c:
        r = c.post("/v1/strategy/set", json={"strategy_name": "round_robin"})
        assert r.status_code == 200
        yield c
    _run(api.task_registry.clear_all())
    _run(api.instance_registry.clear_all())


@pytest.fixture
def engine() -> Iterator[_Engine]:
    e = _Engine()
    yield e
    e.stop()


def _register(c: TestClient, iid: str, url: str) -> None:
    r = c.post(
        "/v1/instance/register",
        json={
            "instance_id": iid,
            "model_id": "m",
            "endpoint": url,
            "platform_info": PLATFORM,
        },
    )
    assert r.status_code == 200, r.text


def _status(c: TestClient, iid: str) -> str | None:
    for inst in c.get("/v1/instance/list").json()["instances"]:
        if inst["instance_id"] == iid:
            return inst["status"]
    return None


def _until(pred, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def _active(c: TestClient, *ids: str) -> None:
    _until(lambda: all(_status(c, i) == "active" for i in ids))


def test_requests_move_off_a_dead_instance(client, engine):
    dead = _dead_port()
    _register(client, "dead", f"http://127.0.0.1:{dead}")
    _register(client, "live", engine.url)
    _active(client, "dead", "live")

    for _ in range(6):
        start = time.monotonic()
        r = client.post("/infer", json={"x": 1})
        assert r.status_code == 200, r.text
        # No retry backoff on the dead instance (it was 1 s + 2 s).
        assert time.monotonic() - start < 0.9
    assert engine.seen == 6
    assert _status(client, "dead") == "unhealthy"


def test_instance_comes_back_when_it_answers(client, engine):
    port = _dead_port()
    _register(client, "flaky", f"http://127.0.0.1:{port}")
    _register(client, "live", engine.url)
    _active(client, "flaky", "live")
    for _ in range(2):
        assert client.post("/infer", json={}).status_code == 200
    assert _status(client, "flaky") == "unhealthy"

    revived = _Engine(port)
    try:
        _until(lambda: _status(client, "flaky") == "active")
        for _ in range(4):
            assert client.post("/infer", json={}).status_code == 200
        assert revived.seen >= 1
    finally:
        revived.stop()


def test_long_dead_instance_is_removed(client, engine, monkeypatch):
    from swarmpilot.scheduler import api

    monkeypatch.setattr(api.config.health, "remove_after", 0.3)
    _register(client, "dead", f"http://127.0.0.1:{_dead_port()}")
    _register(client, "live", engine.url)
    _active(client, "dead", "live")
    for _ in range(2):
        assert client.post("/infer", json={}).status_code == 200
    _until(lambda: _status(client, "dead") is None)
    assert _status(client, "live") == "active"


def test_only_instance_dead_fails_fast(client):
    _register(client, "dead", f"http://127.0.0.1:{_dead_port()}")
    _active(client, "dead")
    start = time.monotonic()
    r = client.post("/infer", json={})
    assert r.status_code == 502
    assert "unreachable" in r.text
    assert time.monotonic() - start < 0.9


def test_reroute_survives_a_strategy_error(client, engine, monkeypatch):
    """Re-routing falls back like the proxy when scheduling raises."""
    from swarmpilot.scheduler import api

    _register(client, "dead", f"http://127.0.0.1:{_dead_port()}")
    _register(client, "live", engine.url)
    _active(client, "dead", "live")

    async def broken(**_):
        raise RuntimeError("Model not found")

    # Both the proxy and re-routing fail to schedule; the proxy then falls
    # back to the first instance, the dead one. (The proxy keeps the
    # strategy it started with, so it is patched separately.)
    monkeypatch.setattr(api.scheduling_strategy, "schedule_task", broken)
    proxy_strategy = api.proxy_router_instance._scheduling_strategy
    monkeypatch.setattr(proxy_strategy, "schedule_task", broken)
    r = client.post("/infer", json={})
    assert r.status_code == 200, r.text
    assert engine.seen == 1
    assert _status(client, "dead") == "unhealthy"


@pytest.mark.parametrize(
    "broken", [{"status": 503}, {"drop": True}], ids=["503", "dropped"]
)
def test_request_lost_by_a_dying_engine_runs_elsewhere(client, engine, broken):
    """A dying engine's open requests run again on another instance.

    An engine that stops with requests open (it crashed, or was removed
    after a drain timeout) answers 503 or drops the connection.
    """
    dying = _Engine(**broken)
    try:
        _register(client, "dying", dying.url)
        _register(client, "live", engine.url)
        _active(client, "dying", "live")
        for _ in range(4):
            r = client.post("/infer", json={"x": 1})
            assert r.status_code == 200, r.text
        assert dying.seen >= 1
        assert engine.seen == 4
    finally:
        dying.stop()


def test_lost_request_fails_when_nowhere_else_to_go(client):
    dying = _Engine(status=503)
    try:
        _register(client, "dying", dying.url)
        _active(client, "dying")
        r = client.post("/infer", json={})
        assert r.status_code >= 500
        assert dying.seen == 1
    finally:
        dying.stop()
