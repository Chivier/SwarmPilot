"""Unit tests for the engine plugin layer and the built-in adapters."""

import json
import shlex
from typing import Any, ClassVar

import aiohttp
import httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from swarmpilot.engines import (
    PORT,
    EngineAdapter,
    EngineRequest,
    LaunchSpec,
    RouteSpec,
    WorkloadSpec,
    WorkloadType,
    get_adapter,
    major_minor,
    register_adapter,
    unregister_adapter,
)
from swarmpilot.engines.contract import check_adapter
from swarmpilot.engines.openai_compat import llm_features, parse_prometheus


class TestLaunchSpec:
    """Tests for Launch Spec."""

    def test_render_argv_fills_port(self):
        spec = LaunchSpec(argv=("srv", "--port", PORT, f"--url=:{PORT}/x"))
        assert spec.render_argv(9000) == [
            "srv",
            "--port",
            "9000",
            "--url=:9000/x",
        ]

    def test_render_shell_quotes_everything_but_port(self):
        spec = LaunchSpec(argv=("vllm", "serve", "a; rm -rf /", "--port", PORT))
        line = spec.render_shell()
        assert line == "vllm serve 'a; rm -rf /' --port \"$PORT\""
        assert shlex.split(line)[2] == "a; rm -rf /"

    def test_render_shell_embedded_placeholder(self):
        spec = LaunchSpec(argv=(f"--bind=0.0.0.0:{PORT}",))
        assert spec.render_shell() == '--bind=0.0.0.0:"$PORT"'


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("0.11.0", "0.11"),
        ("v0.4.3.post2", "0.4"),
        ("0.32.0rc1", "0.32"),
        ("2", "2"),
        ("", "unknown"),
        ("dev", "dev"),
    ],
)
def test_major_minor(raw, want):
    assert major_minor(raw) == want


class TestLLMFeatures:
    """Tests for L L M Features."""

    def test_chat_counts_text_characters(self):
        body = {
            "messages": [
                {"role": "system", "content": "abc"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "image_url", "image_url": {"url": "x"}},
                    ],
                },
            ],
            "max_tokens": 64,
        }
        assert llm_features(body, 1024) == {
            "prompt_length": 8.0,
            "max_tokens": 64.0,
        }

    def test_completion_and_defaults(self):
        assert llm_features({"prompt": "abcd"}, 1024) == {
            "prompt_length": 4.0,
            "max_tokens": 1024.0,
        }
        feats = llm_features({"prompt": "x", "max_completion_tokens": 7}, 1)
        assert feats["max_tokens"] == 7.0

    def test_garbage_does_not_raise(self):
        feats = llm_features({"messages": "nope", "max_tokens": "big"}, 5)
        assert feats == {"prompt_length": 0.0, "max_tokens": 5.0}


def test_parse_prometheus_sums_labels():
    text = """
# HELP vllm:num_requests_running Number of requests running.
vllm:num_requests_running{model_name="a"} 3.0
vllm:num_requests_running{model_name="b"} 2.0
vllm:kv_cache_usage_perc{model_name="a"} 0.25
"""
    assert parse_prometheus(text) == {
        "vllm:num_requests_running": 5.0,
        "vllm:kv_cache_usage_perc": 0.25,
    }


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestBuiltinAdapters:
    """Tests for Builtin Adapters."""

    def test_vllm_launch(self):
        spec = get_adapter("vllm").launch_spec("Qwen/Qwen3-8B", gpus=2)
        assert spec.render_argv(8100) == [
            "vllm",
            "serve",
            "Qwen/Qwen3-8B",
            "--host",
            "0.0.0.0",
            "--port",
            "8100",
            "--tensor-parallel-size",
            "2",
        ]

    def test_sglang_launch(self):
        spec = get_adapter("sglang").launch_spec("Qwen/Qwen3-8B")
        argv = spec.render_argv(8100)
        assert argv[:5] == [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            "Qwen/Qwen3-8B",
        ]
        assert "--tp-size" not in argv

    async def test_vllm_version_and_metrics(self):
        def handler(req: httpx.Request) -> httpx.Response:
            if req.url.path == "/version":
                return httpx.Response(200, json={"version": "0.11.0"})
            if req.url.path == "/metrics":
                return httpx.Response(
                    200,
                    text="vllm:num_requests_running 4\n"
                    "vllm:num_requests_waiting 1\n"
                    "vllm:gpu_cache_usage_perc 0.5\n",
                )
            return httpx.Response(404)

        adapter = get_adapter("vllm")
        async with _client(handler) as c:
            version = await adapter.version(c, "http://e:1")
            m = await adapter.runtime_metrics(c, "http://e:1")
        assert version == "0.11.0"
        assert adapter.platform_info(version, "H100") == {
            "software_name": "vllm",
            "software_version": "0.11",
            "hardware_name": "H100",
        }
        assert (m.running_requests, m.waiting_requests, m.kv_cache_usage) == (
            4,
            1,
            0.5,
        )

    async def test_sglang_version_from_server_info(self):
        def handler(req: httpx.Request) -> httpx.Response:
            if req.url.path == "/get_server_info":
                return httpx.Response(200, json={"version": "0.4.6.post1"})
            return httpx.Response(404)

        async with _client(handler) as c:
            v = await get_adapter("sglang").version(c, "http://e:1")
        assert v == "0.4.6.post1"

    async def test_version_unknown_when_unreachable(self):
        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        async with _client(handler) as c:
            assert (
                await get_adapter("vllm").version(c, "http://e:1") == "unknown"
            )
            assert await get_adapter("vllm").health(c, "http://e:1") is False


SSE = b"data: 1\n\ndata: 2\n\ndata: [DONE]\n\n"


async def _engine_handler(req: web.Request) -> web.StreamResponse:
    body = json.loads(await req.read() or b"{}")
    if req.path == "/v1/bad":
        return web.json_response({"error": "bad"}, status=400)
    if body.get("stream"):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)
        for part in SSE.split(b"\n\n")[:-1]:
            await resp.write(part + b"\n\n")
        await resp.write_eof()
        return resp
    return web.json_response({"path": req.path, "body": body})


@pytest.fixture
async def engine_url():
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", _engine_handler)
    server = TestServer(app)
    await server.start_server()
    yield str(server.make_url("")).rstrip("/")
    await server.close()


class TestDefaultDispatch:
    """Tests for the default aiohttp dispatch."""

    async def test_json(self, engine_url):
        async with aiohttp.ClientSession() as s:
            r = await get_adapter("vllm").dispatch(
                s,
                engine_url + "/",
                EngineRequest(
                    path="v1/chat/completions", body={"max_tokens": 3}
                ),
            )
        assert r.status_code == 200 and r.chunks is None
        assert r.body == {
            "path": "/v1/chat/completions",
            "body": {"max_tokens": 3},
        }

    async def test_stream(self, engine_url):
        async with aiohttp.ClientSession() as s:
            r = await get_adapter("vllm").dispatch(
                s,
                engine_url,
                EngineRequest(
                    path="v1/completions", body={"stream": True}, stream=True
                ),
            )
            got = b"".join([chunk async for chunk in r.chunks])
            await r.close()
        assert r.body is None
        assert got == SSE

    async def test_stream_error_is_json(self, engine_url):
        async with aiohttp.ClientSession() as s:
            r = await get_adapter("vllm").dispatch(
                s,
                engine_url,
                EngineRequest(
                    path="v1/bad", body={"stream": True}, stream=True
                ),
            )
        assert (r.status_code, r.body, r.chunks) == (
            400,
            {"error": "bad"},
            None,
        )


class _ToyAdapter(EngineAdapter):
    name: ClassVar[str] = "toy"
    workload: ClassVar[WorkloadSpec] = WorkloadSpec(
        type=WorkloadType.GENERIC_JOB, feature_names=("size",)
    )

    def launch_spec(self, model_id, *, gpus=1, extra_args=()):
        return LaunchSpec(
            argv=("toy", "--model", model_id, "--port", PORT, *extra_args),
            gpus=gpus,
        )

    def routes(self):
        return (RouteSpec("run"),)

    def features(self, path: str, body: dict[str, Any]) -> dict[str, float]:
        try:
            return {"size": float(body.get("size", 0))}
        except (TypeError, ValueError):
            return {"size": 0.0}


class TestRegistry:
    """Tests for Registry."""

    def test_register_and_lookup(self):
        register_adapter(_ToyAdapter)
        try:
            adapter = get_adapter("toy")
            assert isinstance(adapter, _ToyAdapter)
            assert check_adapter(adapter) == []
        finally:
            unregister_adapter("toy")
        with pytest.raises(KeyError):
            get_adapter("toy")

    def test_contract_catches_shell_strings(self):
        class Bad(_ToyAdapter):
            name: ClassVar[str] = "bad"

            def launch_spec(self, model_id, *, gpus=1, extra_args=()):
                return LaunchSpec(
                    argv=(f"toy --model {model_id} --port $PORT",)
                )

        problems = check_adapter(Bad())
        assert any("one argv element" in p for p in problems)
        assert any("PORT" in p for p in problems)
