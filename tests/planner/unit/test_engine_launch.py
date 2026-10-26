"""Tests for adapter-based launch commands and live platform info (1.2)."""

import shlex
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock

import pytest

from swarmpilot.engines import get_adapter
from swarmpilot.engines.live import probe_platform_info
from swarmpilot.planner.pylet.client import build_launch_command
from swarmpilot.planner.pylet.instance_manager import (
    InstanceManager,
    ManagedInstance,
)
from swarmpilot.planner.pylet.scheduler_client import default_platform_info


class TestBuildLaunchCommand:
    """Tests for build_launch_command."""

    def test_vllm(self):
        cmd = build_launch_command(
            "Qwen/Qwen3-8B",
            "vllm",
            gpu_count=2,
            extra_args="--max-model-len 8192",
        )
        assert cmd == (
            'vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port "$PORT" '
            "--tensor-parallel-size 2 --max-model-len 8192"
        )

    def test_model_id_cannot_inject(self):
        cmd = build_launch_command("x; touch /tmp/pwned", "vllm")
        argv = shlex.split(cmd.replace('"$PORT"', "1"))
        assert argv[2] == "x; touch /tmp/pwned"

    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="Unsupported backend"):
            build_launch_command("m", "nope")


class _Engine(BaseHTTPRequestHandler):
    """Answers /health and /version like vLLM."""

    def do_GET(self) -> None:
        body = b'{"version": "0.11.2"}' if self.path == "/version" else b"{}"
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def engine_addr():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Engine)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"127.0.0.1:{srv.server_port}"
    srv.shutdown()


class TestPlatformInfo:
    """Tests for live platform info."""

    def test_probe_reads_live_version(self, engine_addr):
        info = probe_platform_info(get_adapter("vllm"), engine_addr, "H100")
        assert info == {
            "software_name": "vllm",
            "software_version": "0.11",
            "hardware_name": "H100",
        }

    def test_probe_unhealthy_is_unknown(self):
        info = probe_platform_info(
            get_adapter("sglang"), "127.0.0.1:9", ready_timeout_s=0
        )
        assert info["software_version"] == "unknown"

    def test_default_is_unknown_not_hardcoded(self):
        assert default_platform_info("vllm")["software_version"] == "unknown"
        assert default_platform_info("custom") == {
            "software_name": "custom",
            "software_version": "unknown",
            "hardware_name": "gpu",
        }

    def test_instance_manager_registers_live_version(self, engine_addr):
        mgr = InstanceManager(
            "http://pylet:1", "http://sched:1", engine_ready_timeout=5
        )
        mgr._initialized = True
        mgr.pylet_client = MagicMock()
        mgr.pylet_client.wait_instance_running.return_value = MagicMock(
            endpoint=engine_addr
        )
        scheduler = MagicMock()
        scheduler.register_instance.return_value = MagicMock(success=True)
        mgr._get_scheduler_client = MagicMock(return_value=scheduler)
        mgr._instances["p1"] = ManagedInstance(
            pylet_id="p1", instance_id="i1", model_id="m", backend="vllm"
        )
        mgr.wait_instance_ready("p1")
        kwargs = scheduler.register_instance.call_args.kwargs
        assert kwargs["platform_info"]["software_version"] == "0.11"
