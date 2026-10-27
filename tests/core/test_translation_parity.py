"""Differential tests: the Rust API translation against ``swarmpilot.api``.

The Rust gateway translates between API styles itself (R3b), so its
translation must give the same bytes as the Python one for any input:
valid requests, requests with fields of the wrong type, engine answers and
SSE streams that are malformed, cut short or split anywhere. Each case is
run through both and compared as JSON text (key order, int versus float
and bool included) and, for streams, byte for byte.

Ids and the clock are replaced on both sides: ids count up from 1 in call
order, so a difference in when an id is taken shows up too. Seeds are
fixed, so a failure repeats.
"""

from __future__ import annotations

import asyncio
import copy
import itertools
import json
import os
import random
from typing import Any

import pytest

from swarmpilot.api import TranslationError, plan_request
from swarmpilot.api.styles import (
    anthropic_messages,
    openai_chat,
    openai_responses,
)

NOW = 1_700_000_000
# Cases per test are multiplied by this; raise it for a longer hunt.
SCALE = int(os.getenv("SWARMPILOT_PARITY_SCALE", "1"))
STYLES = ("openai_chat", "anthropic_messages", "openai_responses")
PATHS = {
    "openai_chat": "v1/chat/completions",
    "anthropic_messages": "v1/messages",
    "openai_responses": "v1/responses",
}
NATIVES = (
    ["openai_chat"],
    ["anthropic_messages"],
    ["openai_responses"],
    ["openai_chat"],
    ["anthropic_messages"],
    ["openai_responses"],
    ["openai_responses", "anthropic_messages"],
    ["custom_style"],
)

# --- Python side ------------------------------------------------------------


def _patch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = itertools.count(1)

    def new_id(prefix: str) -> str:
        return f"{prefix}{next(counter):024x}"

    for mod in (openai_chat, anthropic_messages, openai_responses):
        monkeypatch.setattr(mod, "new_id", new_id, raising=False)
        monkeypatch.setattr(mod, "now", lambda: NOW, raising=False)


async def _drain(agen: Any) -> tuple[bytes, bool]:
    out = b""
    try:
        async for b in agen:
            out += b
    except Exception:
        return out, True
    return out, False


def py_translate(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    body: Any,
    natives: list[str],
    strict: bool,
    dmt: int,
    status: int = 200,
    response: Any = None,
    chunks: list[bytes] | None = None,
    fail: bool = False,
) -> dict[str, Any]:
    """Run one case through the Python translation layer."""
    with monkeypatch.context() as m:
        _patch_env(m)
        try:
            tr = plan_request(
                path, body, natives, strict=strict, default_max_tokens=dmt
            )
        except TranslationError as e:
            return {
                "plan": "error",
                "status": e.status,
                "message": e.message,
                "param": e.param,
                "type": e.error_type,
            }
        except Exception:
            return {"plan": "internal"}
        if tr is None:
            return {"plan": "none"}
        out: dict[str, Any] = {
            "plan": "ok",
            "passthrough": tr.passthrough,
            "stream": tr.stream,
            "path": tr.path,
            "body": tr.body,
            "dropped": list(tr.ctx.dropped),
        }
        if tr.passthrough:
            return out
        answer: Any = None
        if chunks is not None:

            async def src():
                for c in chunks:
                    yield c
                if fail:
                    raise ConnectionError("engine went away")

            sse, broken = asyncio.run(_drain(tr.stream_response(src())))
            answer = {"sse": sse.decode(), "broken": broken}
        elif response is not None:
            if tr.stream and status < 400 and isinstance(response, dict):
                try:
                    agen = tr.stream_from_body(response)
                except Exception:
                    answer = {"internal": True}
                else:
                    sse, broken = asyncio.run(_drain(agen))
                    answer = {"sse": sse.decode(), "broken": broken}
            else:
                try:
                    s, b = tr.response(status, response)
                    answer = {"status": s, "body": b}
                except Exception:
                    answer = {"internal": True}
        out["answer"] = answer
        return out


def rs_translate(
    rc: Any,
    path: str,
    body: Any,
    natives: list[str],
    strict: bool,
    dmt: int,
    **kw: Any,
) -> dict[str, Any]:
    """Run one case through the Rust translation layer."""
    if "response" in kw:
        kw["response"] = json.dumps(kw["response"])
    return json.loads(
        rc.api_translate(
            path, json.dumps(body), natives, strict, dmt, now=NOW, **kw
        )
    )


def _canon(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)


# --- Random inputs -----------------------------------------------------------

WORDS = ("hi", "Paris", "", " ", "日本", "a\nb", "x\u2028y", '{"k"', "0")


def _scalar(rng: random.Random) -> Any:
    return rng.choice(
        [
            None,
            True,
            False,
            0,
            1,
            -3,
            2**40,
            0.0,
            1.5,
            -0.25,
            1e16,
            1e-7,
            3.0,
            rng.choice(WORDS),
            rng.randint(-5, 5),
            round(rng.uniform(-3, 3), rng.choice([0, 2, 9])),
        ]
    )


def _value(rng: random.Random, depth: int = 0) -> Any:
    r = rng.random()
    if depth > 2 or r < 0.6:
        return _scalar(rng)
    if r < 0.8:
        return [_value(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    return {
        rng.choice(("text", "type", "a", "content", "id")): _value(
            rng, depth + 1
        )
        for _ in range(rng.randint(0, 3))
    }


def _nodes(x: Any, out: list) -> list:
    if isinstance(x, dict):
        for k in list(x):
            out.append((x, k))
            _nodes(x[k], out)
    elif isinstance(x, list):
        for i in range(len(x)):
            out.append((x, i))
            _nodes(x[i], out)
    return out


def mutate(rng: random.Random, x: Any, times: int) -> Any:
    """Replace, drop or add a few random nodes of ``x``."""
    x = copy.deepcopy(x)
    for _ in range(times):
        nodes = _nodes(x, [])
        if not nodes:
            return x
        parent, key = rng.choice(nodes)
        r = rng.random()
        if r < 0.6:
            parent[key] = _value(rng)
        elif r < 0.8:
            del parent[key]
        elif isinstance(parent, dict):
            parent[rng.choice(("extra", "cache_control", "n", "index"))] = (
                _value(rng)
            )
    return x


def _maybe(rng: random.Random, p: float = 0.5) -> bool:
    return rng.random() < p


def _text(rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(0, 3)))


def _schema(rng: random.Random) -> Any:
    return rng.choice(
        [
            {"type": "object", "properties": {"city": {"type": "string"}}},
            {},
            None,
        ]
    )


def chat_request(rng: random.Random) -> dict[str, Any]:
    msgs: list[Any] = []
    if _maybe(rng):
        msgs.append(
            {"role": rng.choice(["system", "developer"]), "content": _text(rng)}
        )
    for _ in range(rng.randint(1, 4)):
        kind = rng.choice(["user", "user_parts", "assistant", "tool"])
        if kind == "user":
            msgs.append({"role": "user", "content": _text(rng)})
        elif kind == "user_parts":
            parts: list[Any] = [{"type": "text", "text": _text(rng)}]
            if _maybe(rng):
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": rng.choice(
                            [
                                {"url": "https://x/y.png", "detail": "low"},
                                {"url": "data:image/png;base64,AAAA"},
                                "https://x/z.png",
                            ]
                        ),
                    }
                )
            msgs.append({"role": "user", "content": parts})
        elif kind == "assistant":
            m: dict[str, Any] = {
                "role": "assistant",
                "content": rng.choice([_text(rng), None]),
            }
            if _maybe(rng):
                m["tool_calls"] = [
                    {
                        "id": rng.choice(["call_1", "", None]),
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": rng.choice(
                                ['{"city":"Paris"}', "{", "", "[1]"]
                            ),
                        },
                    }
                ]
            if _maybe(rng, 0.2):
                m["reasoning_content"] = _text(rng)
            msgs.append(m)
        else:
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": _text(rng),
                }
            )
    body: dict[str, Any] = {
        "model": rng.choice(["m", "", None]),
        "messages": msgs,
    }
    for key, gen in (
        ("max_tokens", lambda: rng.choice([16, 0, 5000])),
        ("max_completion_tokens", lambda: rng.choice([32, 1])),
        ("temperature", lambda: rng.choice([0, 0.7, 1, 1.5, 2])),
        ("top_p", lambda: rng.choice([0.9, 1])),
        ("top_k", lambda: rng.choice([5, 40])),
        ("stop", lambda: rng.choice(["\n", ["a", "b"], []])),
        ("seed", lambda: 42),
        ("parallel_tool_calls", lambda: rng.choice([True, False])),
        ("reasoning_effort", lambda: rng.choice(["low", "high", "minimal"])),
        ("user", lambda: "u1"),
        ("n", lambda: rng.choice([1, 2])),
        ("logprobs", lambda: rng.choice([False, True])),
        (
            "stream_options",
            lambda: {"include_usage": rng.choice([True, False])},
        ),
        ("frequency_penalty", lambda: 0.5),
    ):
        if _maybe(rng, 0.3):
            body[key] = gen()
    if _maybe(rng, 0.4):
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": rng.choice(["Weather", None]),
                    "parameters": _schema(rng),
                    **({"strict": True} if _maybe(rng, 0.3) else {}),
                },
            }
        ]
        if _maybe(rng):
            body["tool_choice"] = rng.choice(
                [
                    "auto",
                    "none",
                    "required",
                    {"type": "function", "function": {"name": "get_weather"}},
                ]
            )
    if _maybe(rng, 0.3):
        body["response_format"] = rng.choice(
            [
                {"type": "text"},
                {"type": "json_object"},
                {
                    "type": "json_schema",
                    "json_schema": {"name": "r", "schema": {}, "strict": True},
                },
            ]
        )
    body["stream"] = _maybe(rng)
    return body


def _image_block(rng: random.Random) -> dict[str, Any]:
    return {
        "type": "image",
        "source": rng.choice(
            [
                {"type": "base64", "media_type": "image/png", "data": "AAAA"},
                {"type": "url", "url": "https://x/y.png"},
            ]
        ),
    }


def messages_request(rng: random.Random) -> dict[str, Any]:
    msgs: list[Any] = []
    for i in range(rng.randint(1, 4)):
        role = "user" if i % 2 == 0 else "assistant"
        if _maybe(rng, 0.3):
            msgs.append({"role": role, "content": _text(rng)})
            continue
        blocks: list[Any] = []
        for _ in range(rng.randint(0, 3)):
            if role == "user":
                kind = rng.choice(["text", "image", "tool_result"])
            else:
                kind = rng.choice(
                    ["text", "tool_use", "thinking", "redacted_thinking"]
                )
            if kind == "text":
                b: dict[str, Any] = {"type": "text", "text": _text(rng)}
                if _maybe(rng, 0.2):
                    b["cache_control"] = {"type": "ephemeral"}
                blocks.append(b)
            elif kind == "image":
                blocks.append(_image_block(rng))
            elif kind == "tool_result":
                b = {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": rng.choice(
                        [
                            _text(rng),
                            [{"type": "text", "text": _text(rng)}],
                            [_image_block(rng)],
                            None,
                        ]
                    ),
                }
                if _maybe(rng, 0.3):
                    b["is_error"] = True
                blocks.append(b)
            elif kind == "tool_use":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": rng.choice([{"city": "Paris"}, {}, None]),
                    }
                )
            elif kind == "thinking":
                blocks.append(
                    {
                        "type": "thinking",
                        "thinking": _text(rng),
                        "signature": rng.choice(["sig", "", None]),
                    }
                )
            else:
                blocks.append({"type": "redacted_thinking", "data": "x"})
        msgs.append({"role": role, "content": blocks})
    body: dict[str, Any] = {
        "model": rng.choice(["m", None]),
        "max_tokens": rng.choice([16, 1024, 3000, 0]),
        "messages": msgs,
    }
    if _maybe(rng, 0.4):
        body["system"] = rng.choice(
            [
                _text(rng),
                [
                    {
                        "type": "text",
                        "text": _text(rng),
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            ]
        )
    for key, gen in (
        ("temperature", lambda: rng.choice([0, 0.5, 1])),
        ("top_p", lambda: 0.9),
        ("top_k", lambda: 20),
        ("stop_sequences", lambda: ["END"]),
        ("metadata", lambda: rng.choice([{"user_id": "u1"}, {"user_id": 3}])),
        (
            "thinking",
            lambda: {
                "type": rng.choice(["enabled", "disabled"]),
                "budget_tokens": rng.choice([1024, 500, 4000]),
            },
        ),
        ("service_tier", lambda: "auto"),
    ):
        if _maybe(rng, 0.3):
            body[key] = gen()
    if _maybe(rng, 0.4):
        t: dict[str, Any] = {
            "name": "get_weather",
            "description": rng.choice(["Weather", None]),
            "input_schema": _schema(rng),
        }
        if _maybe(rng, 0.2):
            t["type"] = rng.choice(["custom", "web_search_20250305"])
        if _maybe(rng, 0.2):
            t["cache_control"] = {"type": "ephemeral"}
        body["tools"] = [t]
        if _maybe(rng):
            tc: dict[str, Any] = {
                "type": rng.choice(["auto", "any", "tool", "none"])
            }
            if tc["type"] == "tool":
                tc["name"] = "get_weather"
            if _maybe(rng, 0.3):
                tc["disable_parallel_tool_use"] = rng.choice([True, False])
            body["tool_choice"] = tc
    body["stream"] = _maybe(rng)
    return body


def responses_request(rng: random.Random) -> dict[str, Any]:
    body: dict[str, Any] = {"model": rng.choice(["m", None])}
    if _maybe(rng, 0.3):
        body["input"] = _text(rng)
    else:
        items: list[Any] = []
        for _ in range(rng.randint(1, 4)):
            kind = rng.choice(
                [
                    "message",
                    "message",
                    "function_call",
                    "function_call_output",
                    "reasoning",
                ]
            )
            if kind == "message":
                role = rng.choice(["user", "assistant", "system", "developer"])
                content: Any = rng.choice(
                    [
                        _text(rng),
                        [
                            {
                                "type": rng.choice(
                                    ["input_text", "output_text", "text"]
                                ),
                                "text": _text(rng),
                            }
                        ],
                        [
                            {"type": "input_image", "image_url": "https://x"},
                            {"type": "refusal", "refusal": "no"},
                        ],
                    ]
                )
                item: dict[str, Any] = {"role": role, "content": content}
                if _maybe(rng):
                    item["type"] = "message"
                items.append(item)
            elif kind == "function_call":
                items.append(
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": rng.choice(['{"city":"Paris"}', ""]),
                    }
                )
            elif kind == "function_call_output":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": rng.choice(
                            [
                                _text(rng),
                                [{"type": "input_text", "text": "x"}],
                                [{"type": "input_image", "image_url": "u"}],
                            ]
                        ),
                    }
                )
            else:
                items.append(
                    {
                        "type": "reasoning",
                        "summary": [
                            {"type": "summary_text", "text": _text(rng)}
                        ],
                    }
                )
        body["input"] = items
    for key, gen in (
        ("instructions", lambda: _text(rng)),
        ("max_output_tokens", lambda: rng.choice([16, 2048])),
        ("temperature", lambda: rng.choice([0.2, 1.7])),
        ("top_p", lambda: 1),
        ("parallel_tool_calls", lambda: rng.choice([True, False])),
        ("reasoning", lambda: {"effort": rng.choice(["low", "medium", 3])}),
        ("user", lambda: "u1"),
        ("store", lambda: rng.choice([True, False])),
        ("metadata", lambda: {"a": "b"}),
        ("include", lambda: ["reasoning.encrypted_content"]),
        ("previous_response_id", lambda: rng.choice(["resp_1", ""])),
        (
            "text",
            lambda: {
                "format": rng.choice(
                    [
                        {"type": "text"},
                        {"type": "json_object"},
                        {"type": "json_schema", "name": "r", "schema": {}},
                    ]
                )
            },
        ),
    ):
        if _maybe(rng, 0.25):
            body[key] = gen()
    if _maybe(rng, 0.4):
        body["tools"] = [
            rng.choice(
                [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Weather",
                        "parameters": _schema(rng),
                        "strict": rng.choice([True, None]),
                    },
                    {"type": "web_search"},
                ]
            )
        ]
        if _maybe(rng):
            body["tool_choice"] = rng.choice(
                [
                    "auto",
                    "required",
                    {"type": "function", "name": "get_weather"},
                    {"type": "file_search"},
                ]
            )
    body["stream"] = _maybe(rng)
    return body


REQUESTS = {
    "openai_chat": chat_request,
    "anthropic_messages": messages_request,
    "openai_responses": responses_request,
}

# Engine answers.


def _usage(rng: random.Random, style: str) -> Any:
    if _maybe(rng, 0.2):
        return None
    if style == "openai_chat":
        u: dict[str, Any] = {"prompt_tokens": 7, "completion_tokens": 3}
        if _maybe(rng):
            u["prompt_tokens_details"] = {"cached_tokens": 2}
            u["completion_tokens_details"] = {"reasoning_tokens": 1}
        return u
    if style == "anthropic_messages":
        u = {"input_tokens": 7, "output_tokens": 3}
        if _maybe(rng):
            u["cache_read_input_tokens"] = 2
        return u
    u = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    if _maybe(rng):
        u["input_tokens_details"] = {"cached_tokens": 2}
        u["output_tokens_details"] = {"reasoning_tokens": 1}
    return u


def engine_body(rng: random.Random, style: str) -> dict[str, Any]:
    text = _text(rng)
    tool = _maybe(rng, 0.3)
    if style == "openai_chat":
        msg: dict[str, Any] = {"role": "assistant", "content": text or None}
        if tool:
            msg["tool_calls"] = [
                {
                    "id": rng.choice(["call_1", None]),
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ]
        if _maybe(rng, 0.2):
            msg["reasoning_content"] = _text(rng)
        finish = rng.choice(["stop", "length", "tool_calls", "content_filter"])
        choice: dict[str, Any] = {
            "index": 0,
            "message": msg,
            "finish_reason": finish,
        }
        if _maybe(rng, 0.2):
            choice["stop_reason"] = "END"
        return {
            "id": rng.choice(["chatcmpl-1", "abc", None]),
            "object": "chat.completion",
            "created": rng.choice([1, 0]),
            "model": rng.choice(["m", None]),
            "choices": [choice],
            "usage": _usage(rng, style),
        }
    if style == "anthropic_messages":
        content: list[Any] = []
        if _maybe(rng, 0.2):
            content.append(
                {"type": "thinking", "thinking": _text(rng), "signature": "s"}
            )
        if text:
            content.append({"type": "text", "text": text})
        if tool:
            content.append(
                {
                    "type": "tool_use",
                    "id": rng.choice(["toolu_1", None]),
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                }
            )
        return {
            "id": rng.choice(["msg_1", "x", None]),
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": content,
            "stop_reason": rng.choice(
                [
                    "end_turn",
                    "max_tokens",
                    "tool_use",
                    "stop_sequence",
                    "refusal",
                ]
            ),
            "stop_sequence": rng.choice([None, "END"]),
            "usage": _usage(rng, style),
        }
    output: list[Any] = []
    if _maybe(rng, 0.2):
        output.append(
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": rng.choice(
                    [[{"type": "summary_text", "text": _text(rng)}], []]
                ),
                "content": [{"type": "reasoning_text", "text": "r"}],
            }
        )
    output.append(
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "content": [
                rng.choice(
                    [
                        {"type": "output_text", "text": text},
                        {"type": "refusal", "refusal": "no"},
                    ]
                )
            ],
        }
    )
    if tool:
        output.append(
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": rng.choice(["call_1", None]),
                "name": "get_weather",
                "arguments": '{"city":"Paris"}',
            }
        )
    body: dict[str, Any] = {
        "id": rng.choice(["resp_1", "r", None]),
        "object": "response",
        "created_at": rng.choice([1, 0]),
        "model": "m",
        "status": rng.choice(["completed", "incomplete"]),
        "output": output,
        "usage": _usage(rng, style),
    }
    if body["status"] == "incomplete":
        body["incomplete_details"] = {
            "reason": rng.choice(["max_output_tokens", "content_filter"])
        }
    return body


def _sse(obj: Any, event: str | None = None) -> bytes:
    head = f"event: {event}\n" if event else ""
    data = obj if isinstance(obj, str) else json.dumps(obj)
    return f"{head}data: {data}\n\n".encode()


def engine_stream(rng: random.Random, style: str) -> list[Any]:
    """Frames (dicts or raw strings) of an engine stream."""
    pieces = [_text(rng) for _ in range(rng.randint(0, 3))]
    tool = _maybe(rng, 0.3)
    frames: list[Any] = []
    if style == "openai_chat":

        def chunk(delta: Any, finish: Any = None, **extra: Any) -> Any:
            return {
                "id": "chatcmpl-1",
                "created": 1,
                "model": "m",
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish}
                ],
                **extra,
            }

        frames.append(chunk({"role": "assistant", "content": ""}))
        if _maybe(rng, 0.2):
            frames.append(chunk({"reasoning_content": _text(rng)}))
        frames += [chunk({"content": p}) for p in pieces]
        if tool:
            frames.append(
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "",
                                },
                            }
                        ]
                    }
                )
            )
            frames += [
                chunk(
                    {"tool_calls": [{"index": 0, "function": {"arguments": a}}]}
                )
                for a in ('{"city": ', '"Paris"}')
            ]
        frames.append(chunk({}, rng.choice(["stop", "length", "tool_calls"])))
        if _maybe(rng):
            frames.append(
                {"id": "chatcmpl-1", "choices": [], "usage": _usage(rng, style)}
            )
        frames.append("[DONE]")
        return frames
    if style == "anthropic_messages":
        base = rng.choice([0, 1])
        frames.append(
            (
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "model": "m",
                        "usage": {"input_tokens": 7, "output_tokens": 0},
                    },
                },
                "message_start",
            )
        )
        i = base
        if _maybe(rng, 0.2):
            frames.append(({"type": "ping"}, "ping"))
        if _maybe(rng, 0.2):
            frames += [
                (
                    {
                        "type": "content_block_start",
                        "index": i,
                        "content_block": {"type": "thinking", "thinking": ""},
                    },
                    None,
                ),
                (
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "thinking_delta", "thinking": "t"},
                    },
                    None,
                ),
                (
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "signature_delta", "signature": "s"},
                    },
                    None,
                ),
                ({"type": "content_block_stop", "index": i}, None),
            ]
            i += 1
        frames.append(
            (
                {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {"type": "text", "text": ""},
                },
                None,
            )
        )
        frames += [
            (
                {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {"type": "text_delta", "text": p},
                },
                None,
            )
            for p in pieces
        ]
        frames.append(({"type": "content_block_stop", "index": i}, None))
        if tool:
            i += 1
            frames.append(
                (
                    {
                        "type": "content_block_start",
                        "index": i,
                        "content_block": {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {},
                        },
                    },
                    None,
                )
            )
            frames += [
                (
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": a,
                        },
                    },
                    None,
                )
                for a in ('{"city": ', '"Paris"}')
            ]
            frames.append(({"type": "content_block_stop", "index": i}, None))
        frames.append(
            (
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": rng.choice(
                            ["end_turn", "tool_use", "max_tokens"]
                        ),
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": 3},
                },
                "message_delta",
            )
        )
        frames.append(({"type": "message_stop"}, "message_stop"))
        return frames
    frames.append(
        {
            "type": "response.created",
            "response": {"id": "resp_1", "model": "m", "created_at": 1},
        }
    )
    if _maybe(rng, 0.2):
        frames += [
            {
                "type": "response.output_item.added",
                "item": {"type": "reasoning"},
            },
            {"type": "response.reasoning_summary_text.delta", "delta": "r"},
            {"type": "response.output_item.done", "item": {}},
        ]
    frames += [
        {"type": "response.output_item.added", "item": {"type": "message"}},
        {"type": "response.content_part.added", "part": {}},
    ]
    frames += [
        {"type": "response.output_text.delta", "delta": p} for p in pieces
    ]
    frames += [
        {"type": "response.content_part.done"},
        {"type": "response.output_item.done", "item": {}},
    ]
    if tool:
        frames.append(
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                },
            }
        )
        frames += [
            {"type": "response.function_call_arguments.delta", "delta": a}
            for a in ('{"city": ', '"Paris"}')
        ]
        frames.append({"type": "response.output_item.done", "item": {}})
    status = rng.choice(["completed", "incomplete"])
    final: dict[str, Any] = {"status": status, "usage": _usage(rng, style)}
    if status == "incomplete":
        final["incomplete_details"] = {"reason": "max_output_tokens"}
    frames.append({"type": f"response.{status}", "response": final})
    return frames


def _encode_frames(rng: random.Random, frames: list[Any]) -> bytes:
    out = b""
    for f in frames:
        if isinstance(f, tuple):
            obj, event = f
        else:
            obj, event = f, None
        if _maybe(rng, 0.05):
            out += b": keep-alive\n\n"
        if _maybe(rng, 0.03):
            out += b"data: {not json\n\n"
        out += _sse(obj, event)
    if _maybe(rng, 0.2):
        # Cut the stream anywhere, even inside a UTF-8 character.
        out = out[: rng.randint(0, len(out))]
    if _maybe(rng, 0.2):
        out = out.replace(b"\n", b"\r\n")
    return out


def _split(rng: random.Random, data: bytes) -> list[bytes]:
    cuts = sorted(
        rng.sample(range(len(data) + 1), min(len(data), rng.randint(0, 6)))
    )
    pieces, last = [], 0
    for c in cuts:
        pieces.append(data[last:c])
        last = c
    pieces.append(data[last:])
    return pieces


def _mutate_frames(rng: random.Random, frames: list[Any]) -> list[Any]:
    out = []
    for f in frames:
        if _maybe(rng, 0.1):
            if isinstance(f, tuple):
                f = (mutate(rng, f[0], 1), f[1])
            elif isinstance(f, dict):
                f = mutate(rng, f, 1)
        out.append(f)
    return out


# --- Tests -------------------------------------------------------------------


def _check(rc, monkeypatch, path, body, natives, strict, dmt, **kw):
    py = py_translate(monkeypatch, path, body, natives, strict, dmt, **kw)
    rs = rs_translate(rc, path, body, natives, strict, dmt, **kw)
    assert _canon(rs) == _canon(py), (path, body, natives, strict, kw)


@pytest.mark.parametrize("client", STYLES)
def test_requests_translate_the_same(rc, monkeypatch, client):
    rng = random.Random(f"req-{client}-{SCALE}")
    for _ in range(1500 * SCALE):
        body = REQUESTS[client](rng)
        if _maybe(rng, 0.5):
            body = mutate(rng, body, rng.randint(1, 3))
        natives = list(rng.choice(NATIVES))
        _check(
            rc,
            monkeypatch,
            PATHS[client],
            body,
            natives,
            rng.choice([True, False]),
            rng.choice([4096, 1000]),
        )


def test_odd_bodies_and_paths(rc, monkeypatch):
    for path in ("v1/messages", "/v1/messages", "v1/other", "v1/responses"):
        for body in ({}, [], "x", 3, None, {"messages": "hi"}):
            for natives in ([], ["openai_chat"], ["anthropic_messages"]):
                _check(rc, monkeypatch, path, body, natives, True, 4096)


@pytest.mark.parametrize("client", STYLES)
@pytest.mark.parametrize("engine", STYLES)
def test_answers_translate_the_same(rc, monkeypatch, client, engine):
    if client == engine:
        pytest.skip("passthrough")
    rng = random.Random(f"answer-{client}-{engine}-{SCALE}")
    done = 0
    while done < 400 * SCALE:
        body = REQUESTS[client](rng)
        status = rng.choice([200, 200, 200, 400, 429, 500])
        if status == 200:
            answer: Any = engine_body(rng, engine)
            if _maybe(rng, 0.4):
                answer = mutate(rng, answer, rng.randint(1, 3))
            if _maybe(rng, 0.05):
                answer = rng.choice([[], "text", 3])
        else:
            answer = rng.choice(
                [
                    {"error": {"message": "bad", "type": "x"}},
                    {"error": "boom"},
                    {"message": "m"},
                    {"detail": [1]},
                    {"type": "error", "error": {"message": 5}},
                    "plain",
                ]
            )
        py = py_translate(
            monkeypatch,
            PATHS[client],
            body,
            [engine],
            False,
            4096,
            status=status,
            response=answer,
        )
        if py["plan"] != "ok":
            continue
        done += 1
        rs = rs_translate(
            rc,
            PATHS[client],
            body,
            [engine],
            False,
            4096,
            status=status,
            response=answer,
        )
        assert _canon(rs) == _canon(py), (body, status, answer)


@pytest.mark.parametrize("client", STYLES)
@pytest.mark.parametrize("engine", STYLES)
def test_streams_translate_the_same(rc, monkeypatch, client, engine):
    if client == engine:
        pytest.skip("passthrough")
    rng = random.Random(f"stream-{client}-{engine}-{SCALE}")
    done = 0
    while done < 400 * SCALE:
        body = REQUESTS[client](rng)
        body["stream"] = True
        if client == "openai_chat" and _maybe(rng):
            body["stream_options"] = {"include_usage": True}
        frames = engine_stream(rng, engine)
        if _maybe(rng, 0.4):
            frames = _mutate_frames(rng, frames)
        chunks = _split(rng, _encode_frames(rng, frames))
        fail = _maybe(rng, 0.15)
        py = py_translate(
            monkeypatch,
            PATHS[client],
            body,
            [engine],
            False,
            4096,
            chunks=chunks,
            fail=fail,
        )
        if py["plan"] != "ok":
            continue
        done += 1
        rs = rs_translate(
            rc,
            PATHS[client],
            body,
            [engine],
            False,
            4096,
            chunks=chunks,
            fail=fail,
        )
        assert _canon(rs) == _canon(py), (body, frames, chunks, fail)
