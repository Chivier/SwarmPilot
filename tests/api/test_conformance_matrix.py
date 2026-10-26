"""Phase T conformance matrix: every client style against every engine style.

Each cell runs the official SDK of the client style against a real
Scheduler process whose only instance is a fake engine speaking the engine
style (see ``fake_engines.py``). The Scheduler either passes the body
through (same style) or translates it both ways. Cells cover plain text and
tool calls, streamed and not, plus a tool-result round trip.
"""

from __future__ import annotations

import json
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

import anthropic
import httpx
import openai
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "perf"))
from _harness import free_port, scheduler_cmd, spawn

STYLES = ("openai_chat", "anthropic_messages", "openai_responses")
ROUTES = {
    "openai_chat": "v1/chat/completions",
    "anthropic_messages": "v1/messages",
    "openai_responses": "v1/responses",
}
FEATURES = {
    "openai_chat": (
        "sum(chars(messages[*].content), chars(messages[*].content[*].text))",
        "coalesce(max_tokens, max_completion_tokens, 4096)",
    ),
    "anthropic_messages": (
        "sum(chars(system), chars(messages[*].content), "
        "chars(messages[*].content[*].text))",
        "coalesce(max_tokens, 4096)",
    ),
    "openai_responses": (
        "sum(chars(instructions), chars(input), chars(input[*].content))",
        "coalesce(max_output_tokens, 4096)",
    ),
}
SYSTEM, USER = "be brief", "hello"
EXPECTED = f"[{SYSTEM}] {USER}"
SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
TOOL_ARGS = {"city": "Paris"}


def _engine_config(style: str, path: Path) -> None:
    prompt, max_tokens = FEATURES[style]
    path.write_text(
        yaml.safe_dump(
            {
                "name": f"fake-{style}",
                "workload": "llm_chat",
                "launch": {"argv": ["fake", "{port}"]},
                "routes": [{"path": ROUTES[style], "streaming": True}],
                "features": [
                    {"name": "prompt_length", "expr": prompt},
                    {"name": "max_tokens", "expr": max_tokens},
                ],
                "capacity": {"max_in_flight": 8},
            }
        )
    )


@pytest.fixture(scope="module")
def gateways():
    """Start one fake engine and one Scheduler per engine style."""
    tmp = Path(tempfile.mkdtemp(prefix="sp-api-"))
    configs = tmp / "engines"
    configs.mkdir()
    for style in STYLES:
        _engine_config(style, configs / f"{style}.yaml")
    urls: dict[str, tuple[str, str]] = {}
    with ExitStack() as stack:
        for style in STYLES:
            eport, sport = free_port(), free_port()
            engine = f"http://127.0.0.1:{eport}"
            proxy = f"http://127.0.0.1:{sport}"
            stack.enter_context(
                spawn(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("serve_fake.py")),
                        style,
                        str(eport),
                    ],
                    f"{engine}/health",
                    log_path=tmp / f"engine-{style}.log",
                )
            )
            stack.enter_context(
                spawn(
                    scheduler_cmd(sport),
                    f"{proxy}/v1/health",
                    env={
                        "SCHEDULING_STRATEGY": "round_robin",
                        "SCHEDULER_LOGURU_LEVEL": "WARNING",
                        "SCHEDULER_LOG_DIR": str(tmp / f"logs-{style}"),
                        "PREDICTOR_STORAGE_DIR": str(tmp / f"models-{style}"),
                        "SCHEDULER_MODEL_ID": "fake-model",
                        "TRAINING_ENABLE_AUTO": "false",
                        "SWARMPILOT_ENGINE_CONFIGS": str(configs),
                        "PROXY_TIMEOUT": "10",
                    },
                    log_path=tmp / f"scheduler-{style}.log",
                )
            )
            r = httpx.post(
                f"{proxy}/v1/instance/register",
                json={
                    "instance_id": f"fake-{style}",
                    "model_id": "fake-model",
                    "endpoint": engine,
                    "platform_info": {
                        "software_name": f"fake-{style}",
                        "software_version": "0.0.0",
                        "hardware_name": "cpu",
                    },
                },
            )
            r.raise_for_status()
            urls[style] = (proxy, engine)
        yield urls


async def _openai_chat(proxy: str, tool: bool, stream: bool):
    client = openai.AsyncOpenAI(base_url=f"{proxy}/v1", api_key="k")
    kwargs = {
        "model": "fake-model",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        "max_tokens": 16,
    }
    if tool:
        kwargs["tools"] = [
            {
                "type": "function",
                "function": {"name": "get_weather", "parameters": SCHEMA},
            }
        ]
    if not stream:
        r = await client.chat.completions.create(**kwargs)
        msg = r.choices[0].message
        calls = [
            (c.function.name, json.loads(c.function.arguments))
            for c in msg.tool_calls or []
        ]
        return msg.content or "", calls, r.usage.completion_tokens
    text, args, name, usage = "", "", None, None
    out = await client.chat.completions.create(
        **kwargs, stream=True, stream_options={"include_usage": True}
    )
    async for chunk in out:
        if chunk.usage:
            usage = chunk.usage.completion_tokens
        for ch in chunk.choices:
            text += ch.delta.content or ""
            for tc in ch.delta.tool_calls or []:
                name = tc.function.name or name
                args += tc.function.arguments or ""
    calls = [(name, json.loads(args))] if name else []
    return text, calls, usage


async def _anthropic(proxy: str, tool: bool, stream: bool):
    client = anthropic.AsyncAnthropic(base_url=proxy, api_key="k")
    kwargs = {
        "model": "fake-model",
        "max_tokens": 16,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": USER}],
    }
    if tool:
        kwargs["tools"] = [{"name": "get_weather", "input_schema": SCHEMA}]
    if stream:
        async with client.messages.stream(**kwargs) as s:
            m = await s.get_final_message()
    else:
        m = await client.messages.create(**kwargs)
    text = "".join(b.text for b in m.content if b.type == "text")
    calls = [(b.name, b.input) for b in m.content if b.type == "tool_use"]
    if tool:
        assert m.stop_reason == "tool_use"
    return text, calls, m.usage.output_tokens


async def _responses(proxy: str, tool: bool, stream: bool):
    client = openai.AsyncOpenAI(base_url=f"{proxy}/v1", api_key="k")
    kwargs = {
        "model": "fake-model",
        "instructions": SYSTEM,
        "input": USER,
        "max_output_tokens": 16,
    }
    if tool:
        kwargs["tools"] = [
            {"type": "function", "name": "get_weather", "parameters": SCHEMA}
        ]
    if stream:
        async with client.responses.stream(**kwargs) as s:
            r = await s.get_final_response()
    else:
        r = await client.responses.create(**kwargs)
    calls = [
        (o.name, json.loads(o.arguments))
        for o in r.output
        if o.type == "function_call"
    ]
    return r.output_text, calls, r.usage.output_tokens


CLIENTS = {
    "openai_chat": _openai_chat,
    "anthropic_messages": _anthropic,
    "openai_responses": _responses,
}


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
@pytest.mark.parametrize("tool", [False, True], ids=["text", "tool"])
@pytest.mark.parametrize("engine", STYLES)
@pytest.mark.parametrize("client", STYLES)
async def test_matrix(gateways, client, engine, tool, stream):
    proxy, _ = gateways[engine]
    text, calls, output_tokens = await CLIENTS[client](proxy, tool, stream)
    if tool:
        assert calls == [("get_weather", TOOL_ARGS)]
    else:
        assert text == EXPECTED
        assert calls == []
    assert output_tokens == 3


def _round_trip_body(client: str) -> dict:
    call_id, args = "call_1", json.dumps(TOOL_ARGS)
    if client == "openai_chat":
        return {
            "model": "fake-model",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": USER},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": args,
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": call_id, "content": "sunny"},
            ],
        }
    if client == "anthropic_messages":
        return {
            "model": "fake-model",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": USER},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": call_id,
                            "name": "get_weather",
                            "input": TOOL_ARGS,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": "sunny",
                        }
                    ],
                },
            ],
        }
    return {
        "model": "fake-model",
        "max_output_tokens": 16,
        "input": [
            {"role": "user", "content": USER},
            {
                "type": "function_call",
                "call_id": call_id,
                "name": "get_weather",
                "arguments": args,
            },
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": "sunny",
            },
        ],
    }


def _answer_text(client: str, body: dict) -> str:
    if client == "openai_chat":
        return body["choices"][0]["message"]["content"]
    if client == "anthropic_messages":
        return "".join(
            b["text"] for b in body["content"] if b["type"] == "text"
        )
    return "".join(
        p["text"]
        for o in body["output"]
        if o["type"] == "message"
        for p in o["content"]
    )


@pytest.mark.parametrize("engine", STYLES)
@pytest.mark.parametrize("client", STYLES)
def test_tool_result_round_trip(gateways, client, engine):
    proxy, _ = gateways[engine]
    r = httpx.post(
        f"{proxy}/{ROUTES[client]}", json=_round_trip_body(client), timeout=10
    )
    assert r.status_code == 200, r.text
    assert _answer_text(client, r.json()) == "result: sunny"


def test_same_style_passes_body_through(gateways):
    proxy, engine = gateways["openai_chat"]
    body = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": USER}],
        "max_tokens": 16,
        "logit_bias": {"42": 1},
    }
    r = httpx.post(f"{proxy}/v1/chat/completions", json=body, timeout=10)
    assert r.status_code == 200
    assert "x-swarmpilot-dropped" not in r.headers
    assert httpx.get(f"{engine}/_seen").json()[-1] == body


def test_answer_changing_field_is_refused_in_client_style(gateways):
    proxy, _ = gateways["anthropic_messages"]
    r = httpx.post(
        f"{proxy}/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": USER}],
            "n": 2,
        },
        timeout=10,
    )
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "n"


def test_ignorable_field_is_dropped_and_reported(gateways):
    proxy, engine = gateways["anthropic_messages"]
    r = httpx.post(
        f"{proxy}/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": USER}],
            "presence_penalty": 0.5,
        },
        timeout=10,
    )
    assert r.status_code == 200
    assert r.headers["x-swarmpilot-dropped"] == "presence_penalty"
    seen = httpx.get(f"{engine}/_seen").json()[-1]
    assert "presence_penalty" not in seen
    assert seen["max_tokens"] == 4096


def test_translation_error_uses_anthropic_shape(gateways):
    proxy, _ = gateways["openai_chat"]
    r = httpx.post(
        f"{proxy}/v1/messages",
        json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
        timeout=10,
    )
    assert r.status_code == 400
    assert r.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "'max_tokens' is required",
        },
    }


def test_stateful_responses_field_is_refused(gateways):
    """SwarmPilot keeps no response state, so it says so up front."""
    proxy, _ = gateways["anthropic_messages"]
    r = httpx.post(
        f"{proxy}/v1/responses",
        json={"model": "m", "input": "x", "previous_response_id": "resp_1"},
        timeout=10,
    )
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "previous_response_id"
