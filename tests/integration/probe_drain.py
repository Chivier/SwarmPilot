"""E2E probe: submit tasks, drain, remove, and see what happens to queued work.

Runs the real scheduler FastAPI app (round_robin strategy so no trained model
is needed) against a fake model server that takes 1 s per request.
The pytest version of this check is
tests/regression/test_known_bugs.py::test_drain_counts_and_reroutes_queued_tasks.

Usage:
    uv run python tests/integration/probe_drain.py
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["SCHEDULING_STRATEGY"] = "round_robin"
os.environ["SCHEDULER_MODEL_ID"] = "m"


class Slow(BaseHTTPRequestHandler):
    """Fake model server that takes one second per request."""

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        self.rfile.read(n)
        time.sleep(1.0)
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


srv = ThreadingHTTPServer(("127.0.0.1", 0), Slow)
ENGINE = f"http://127.0.0.1:{srv.server_port}"
threading.Thread(target=srv.serve_forever, daemon=True).start()

from fastapi.testclient import TestClient  # noqa: E402
from loguru import logger  # noqa: E402

logger.remove()
from swarmpilot.scheduler.api import app  # noqa: E402

out = {}
with TestClient(app) as c:
    plat = {
        "software_name": "vllm",
        "software_version": "x",
        "hardware_name": "h",
    }
    r = c.post(
        "/v1/instance/register",
        json={
            "instance_id": "A",
            "model_id": "m",
            "endpoint": ENGINE,
            "platform_info": plat,
        },
    )
    out["register"] = r.status_code
    for i in range(6):
        c.post(
            "/v1/task/submit",
            json={
                "task_id": f"t{i}",
                "model_id": "m",
                "task_input": {"prompt": "x"},
                "metadata": {},
            },
        )
    time.sleep(0.3)
    info = c.get("/v1/task/info", params={"task_id": "t0"}).json()
    out["t0_assigned_instance_after_submit"] = info["task"]["assigned_instance"]
    out["drain"] = c.post(
        "/v1/instance/drain", json={"instance_id": "A"}
    ).json()
    st = c.get("/v1/instance/drain/status", params={"instance_id": "A"}).json()
    out["drain_status_right_after"] = {
        k: st.get(k) for k in ("pending_tasks", "can_remove")
    }
    out["remove"] = c.post(
        "/v1/instance/remove", json={"instance_id": "A"}
    ).status_code
    time.sleep(4)
    statuses = {}
    for i in range(6):
        t = c.get("/v1/task/info", params={"task_id": f"t{i}"}).json()["task"]
        statuses[f"t{i}"] = t["status"]
    out["task_status_4s_after_remove"] = statuses
srv.shutdown()
print(json.dumps(out, indent=1))
