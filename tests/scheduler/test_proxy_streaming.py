"""End-to-end proxy tests with async dispatch (plan item 1.3)."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

DELAY_S = 0.2


class _Handler(BaseHTTPRequestHandler):
    """OpenAI-ish engine: JSON, or SSE when the body asks to stream."""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        time.sleep(DELAY_S if body.get("slow") else 0)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("connection", "close")
            self.end_headers()
            for i in range(3):
                self.wfile.write(f"data: {i}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.close_connection = True
            return
        out = json.dumps({"echo": body}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def engine_url():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@pytest.fixture
def client(engine_url):
    from swarmpilot.scheduler import api

    with TestClient(api.app) as c:
        assert api.worker_queue_manager.async_dispatch
        r = c.post(
            "/v1/instance/register",
            json={
                "instance_id": "e0",
                "model_id": "m",
                "endpoint": engine_url,
                "platform_info": {
                    "software_name": "vllm",
                    "software_version": "0.11",
                    "hardware_name": "cpu",
                },
            },
        )
        assert r.status_code == 200, r.text
        yield c
        c.post("/v1/instance/remove", json={"instance_id": "e0"})


def test_stream_is_passed_through(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    assert text == "data: 0\n\ndata: 1\n\ndata: 2\n\ndata: [DONE]\n\n"


def test_json_still_works(client):
    r = client.post("/v1/completions", json={"prompt": "x", "max_tokens": 2})
    assert r.status_code == 200
    assert r.json() == {"echo": {"prompt": "x", "max_tokens": 2}}


def test_requests_to_one_instance_run_concurrently(client):
    def call(i: int) -> int:
        return client.post(
            "/v1/completions", json={"prompt": str(i), "slow": True}
        ).status_code

    t0 = time.monotonic()
    with ThreadPoolExecutor(8) as pool:
        codes = list(pool.map(call, range(8)))
    elapsed = time.monotonic() - t0
    assert codes == [200] * 8
    # One at a time would take 8 * 0.2 s.
    assert elapsed < 4 * DELAY_S, elapsed


def test_registration_sets_sketch_capacity(client):
    from swarmpilot.scheduler import api

    worker = api.worker_queue_manager.get_worker("e0")
    assert api.instance_registry.get_capacity("e0") == worker.max_in_flight
    assert worker.max_in_flight > 1
