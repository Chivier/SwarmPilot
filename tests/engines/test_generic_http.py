"""Generic YAML engines and the out-of-tree plugin example (plan item 1.5)."""

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from pydantic import ValidationError

from swarmpilot.engines import adapter_names, get_adapter
from swarmpilot.engines.contract import check_adapter
from swarmpilot.engines.extractors import ExtractorError, compile_extractor
from swarmpilot.engines.generic_http import (
    config_files,
    load_config_file,
    make_adapter_class,
)
from swarmpilot.engines.registry import reset_registry

EXAMPLE = Path(__file__).parents[2] / "examples" / "9.engine_plugin"
PLUGIN_SRC = EXAMPLE / "toy_ranker_plugin"
YAML = EXAMPLE / "toy-ranker-http.yaml"

CHAT = {
    "messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ],
    "max_tokens": 64,
    "width": 640,
    "height": 480,
    "n": True,
}


class TestExtractors:
    """Declarative feature expressions."""

    @pytest.mark.parametrize(
        ("expr", "want"),
        [
            ("field(max_tokens)", 64),
            ("max_tokens", 64),
            ("len(messages)", 2),
            ("count(messages[*].role)", 2),
            ("chars(messages[*].content)", 13),
            ("chars(messages[1].content)", 5),
            ("product(width, height)", 640 * 480),
            ("sum(max_tokens, 1, len(messages))", 67),
            ("max(width, height)", 640),
            ("product(width, 0.5)", 320),
            ("7", 7),
        ],
    )
    def test_values(self, expr, want):
        assert compile_extractor(expr)(CHAT) == want

    @pytest.mark.parametrize(
        "expr",
        ["field(missing)", "field(n)", "len(max_tokens)", "product(width, x)"],
    )
    def test_missing_or_non_numeric_gives_default(self, expr):
        assert compile_extractor(expr, default=-1)(CHAT) == -1

    @pytest.mark.parametrize(
        "expr",
        ["", "field(", "nope(x)", "field(a) b", "sum()", "len(1)", "a..b"],
    )
    def test_bad_expressions_rejected(self, expr):
        with pytest.raises(ExtractorError):
            compile_extractor(expr)

    def test_no_code_execution(self):
        with pytest.raises(ExtractorError):
            compile_extractor("__import__('os').system('true')")


def _config(**over):
    cfg = {
        "name": "toy",
        "launch": {"argv": ["srv", "--model", "{model}", "--port", "{port}"]},
        "routes": [{"path": "/v1/run"}],
        "features": [{"name": "n", "expr": "len(items)"}],
    }
    cfg.update(over)
    return cfg


class TestGenericAdapter:
    """Adapters built from YAML descriptions."""

    def test_example_yaml_passes_contract(self):
        cls = load_config_file(YAML)
        assert cls.name == "toy-ranker-http"
        assert check_adapter(cls()) == []

    def test_minimal_config_passes_contract(self):
        assert check_adapter(make_adapter_class(_config())()) == []

    def test_launch_fills_placeholders(self):
        cfg = _config(
            launch={
                "argv": ["srv", "--model={model}", "--port", "{port}"],
                "gpu_args": ["--tp", "{gpus}"],
                "env": {"A": "1"},
            }
        )
        adapter = make_adapter_class(cfg)()
        spec = adapter.launch_spec("org/m; rm -rf /", gpus=2)
        assert spec.render_argv(7000) == [
            "srv",
            "--model=org/m; rm -rf /",
            "--port",
            "7000",
            "--tp",
            "2",
        ]
        assert spec.env == {"A": "1"}
        assert adapter.launch_spec("m").argv[-1] == "{port}"

    def test_routes_and_features(self):
        adapter = make_adapter_class(_config())()
        assert [r.path for r in adapter.routes()] == ["v1/run"]
        assert adapter.features("v1/run", {"items": [1, 2, 3]}) == {"n": 3}
        assert adapter.features("v1/run", {}) == {"n": 0}

    @pytest.mark.parametrize(
        "bad",
        [
            {"launch": {"argv": ["srv"]}},
            {"routes": []},
            {"features": []},
            {"name": "bad name"},
            {"surprise": 1},
            {"capacity": {"max_in_flight": 0}},
        ],
    )
    def test_invalid_configs_rejected(self, bad):
        with pytest.raises(ValidationError):
            make_adapter_class(_config(**bad))

    def test_bad_feature_expression_rejected(self):
        with pytest.raises(ExtractorError):
            make_adapter_class(
                _config(features=[{"name": "n", "expr": "eval(x)"}])
            )

    def test_config_files_expands_directories(self, tmp_path):
        (tmp_path / "a.yaml").write_text("x: 1")
        (tmp_path / "b.yml").write_text("x: 1")
        (tmp_path / "c.txt").write_text("x: 1")
        other = tmp_path / "solo.yaml"
        found = config_files(f"{tmp_path}:{other}")
        assert [p.name for p in found] == ["a.yaml", "b.yml", "solo.yaml"]


@pytest.fixture
def registry_with_yaml(monkeypatch, tmp_path):
    broken = tmp_path / "broken.yaml"
    broken.write_text("name: broken\n")
    monkeypatch.setenv("SWARMPILOT_ENGINE_CONFIGS", f"{YAML}:{broken}")
    reset_registry()
    yield
    monkeypatch.delenv("SWARMPILOT_ENGINE_CONFIGS")
    reset_registry()


def test_yaml_engines_are_discovered(registry_with_yaml):
    names = adapter_names()
    assert "toy-ranker-http" in names
    assert "broken" not in names
    assert {"vllm", "sglang"} <= set(names)


@pytest.fixture
def plugin_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(PLUGIN_SRC))
    from toy_ranker.adapter import ToyRankerAdapter

    return ToyRankerAdapter


def test_plugin_adapter_passes_contract(plugin_on_path):
    assert check_adapter(plugin_on_path()) == []


def test_plugin_declares_entry_point():
    text = (PLUGIN_SRC / "pyproject.toml").read_text()
    assert '[project.entry-points."swarmpilot.engines"]' in text
    assert 'toy-ranker = "toy_ranker.adapter:ToyRankerAdapter"' in text


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def toy_server(registry_with_yaml):
    """Launch the toy server exactly as the YAML engine says to."""
    port = _free_port()
    spec = get_adapter("toy-ranker-http").launch_spec("toy-small")
    argv = spec.render_argv(port)
    argv[0] = sys.executable
    proc = subprocess.Popen(
        argv,
        cwd=PLUGIN_SRC,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    while True:
        try:
            with urllib.request.urlopen(f"{url}{spec.health_path}", timeout=1):
                break
        except OSError:
            if proc.poll() is not None or time.monotonic() > deadline:
                proc.kill()
                pytest.fail("toy server did not start")
            time.sleep(0.05)
    yield url
    proc.terminate()
    proc.wait(timeout=5)


def test_yaml_engine_served_and_routed(toy_server):
    """A private server is launched, versioned and routed from YAML.

    Nothing under swarmpilot/ knows about the toy server.
    """
    from fastapi.testclient import TestClient

    from swarmpilot.engines.live import probe_platform_info
    from swarmpilot.scheduler import api

    adapter = get_adapter("toy-ranker-http")
    pinfo = probe_platform_info(adapter, toy_server, "cpu")
    assert pinfo["software_version"] == "1.4"
    assert pinfo["workload"] == "generic_job"

    seen: list[dict] = []
    real = api.scheduling_strategy.schedule_task

    async def spy(model_id, metadata, available_instances, **kw):
        seen.append(dict(metadata))
        return await real(model_id, metadata, available_instances, **kw)

    with TestClient(api.app) as c:
        r = c.post(
            "/v1/instance/register",
            json={
                "instance_id": "toy0",
                "model_id": "toy-small",
                "endpoint": toy_server,
                "platform_info": pinfo,
            },
        )
        assert r.status_code == 200, r.text
        assert api.worker_queue_manager._workers["toy0"].max_in_flight == 4
        api.scheduling_strategy.schedule_task = spy
        try:
            r = c.post(
                "/v1/rank",
                json={"query": "red fox", "documents": ["a red fox", "b"]},
            )
        finally:
            api.scheduling_strategy.schedule_task = real
            c.post("/v1/instance/remove", json={"instance_id": "toy0"})
    assert r.status_code == 200, r.text
    assert json.loads(r.content)["results"][0] == {"index": 0, "score": 2}
    assert seen[0]["n_docs"] == 2
    assert seen[0]["query_chars"] == 7


def test_planner_accepts_yaml_engine(registry_with_yaml):
    """The Planner builds a shell-safe launch command for scaling it."""
    from swarmpilot.planner.pylet.client import build_launch_command

    cmd = build_launch_command("toy-small", "toy-ranker-http", gpu_count=1)
    assert cmd == (
        'python -m toy_ranker.server --model toy-small --port "$PORT"'
    )
