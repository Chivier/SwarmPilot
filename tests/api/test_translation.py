"""Unit tests for the canonical model, the styles and the translator."""

from __future__ import annotations

import json

import pytest

from swarmpilot.api import canonical as c, plan_request
from swarmpilot.api.base import Context, TranslationError
from swarmpilot.api.registry import (
    get_style,
    register_style,
    style_for_path,
    style_names,
    unregister_style,
)
from swarmpilot.api.sse import iter_sse
from swarmpilot.api.styles._common import collect
from swarmpilot.api.styles.openai_chat import OpenAIChatStyle
from swarmpilot.engines.extractors import compile_extractor
from swarmpilot.engines.registry import get_adapter

CHAT = ("openai_chat",)

# One conversation, spelled in each style.
CONVERSATION = {
    "openai_chat": {
        "model": "m",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
            {
                "role": "user",
                "content": [{"type": "text", "text": "And of Spain?"}],
            },
        ],
    },
    "anthropic_messages": {
        "model": "m",
        "max_tokens": 64,
        "system": "You are terse.",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Paris."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "And of Spain?"}],
            },
        ],
    },
    "openai_responses": {
        "model": "m",
        "max_output_tokens": 64,
        "instructions": "You are terse.",
        "input": [
            {"role": "user", "content": "What is the capital of France?"},
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Paris."}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "And of Spain?"}],
            },
        ],
    },
}


def test_builtin_styles_are_registered():
    assert style_names() == [
        "anthropic_messages",
        "openai_chat",
        "openai_responses",
    ]
    assert style_for_path("/v1/messages").name == "anthropic_messages"
    assert style_for_path("v1/embeddings") is None


def test_native_styles_follow_adapter_routes():
    assert get_adapter("vllm").native_styles() == CHAT
    assert get_adapter("sglang").native_styles() == CHAT
    assert get_adapter("diffusers").native_styles() == ()


def test_canonical_schema_file_is_current():
    """Regenerate with python -m swarmpilot.api.canonical."""
    assert json.loads(c.SCHEMA_PATH.read_text()) == c.json_schema()


@pytest.mark.parametrize("style", sorted(CONVERSATION))
def test_canonical_round_trips_through_dicts(style):
    req = get_style(style).decode_request(CONVERSATION[style], Context())
    assert c.request_from_dict(c.to_dict(req)) == req


@pytest.mark.parametrize("style", sorted(CONVERSATION))
def test_every_style_decodes_to_the_same_conversation(style):
    req = get_style(style).decode_request(CONVERSATION[style], Context())
    assert req.system_text() == "You are terse."
    assert [m.role for m in req.messages] == ["user", "assistant", "user"]
    assert req.messages[-1].content == [c.TextPart("And of Spain?")]
    assert req.max_output_tokens == 64


@pytest.mark.parametrize("style", sorted(CONVERSATION))
def test_features_match_across_styles(style):
    """The predictor sees the same features whatever style the client used."""
    adapter = get_adapter("vllm")
    body = CONVERSATION[style]
    expected = adapter.features(
        "v1/chat/completions", CONVERSATION["openai_chat"]
    )
    # Declarative form, for the gateway: expressions over the client body.
    pairs = get_style(style).feature_extractors(adapter.default_max_tokens)
    declared = {n: compile_extractor(e)(body) for n, e in pairs}
    assert declared == expected
    # Python path: the adapter's features over the translated body.
    tr = plan_request(ROUTE[style], body, CHAT)
    assert adapter.features(tr.path, tr.body) == expected


ROUTE = {
    "openai_chat": "v1/chat/completions",
    "anthropic_messages": "v1/messages",
    "openai_responses": "v1/responses",
}


def test_manifest_lists_native_styles_and_style_features():
    m = get_adapter("vllm").manifest()
    assert m["native_styles"] == ["openai_chat"]
    assert set(m["style_features"]) == {
        "anthropic_messages",
        "openai_responses",
    }


def test_same_style_is_passthrough():
    body = {"model": "m", "messages": [], "logit_bias": {"1": 2}}
    tr = plan_request("v1/chat/completions", body, CHAT)
    assert tr.passthrough and tr.body == body and tr.headers() == {}


def test_engine_without_styles_is_not_translated():
    assert plan_request("v1/messages", {}, ()) is None


def test_no_shared_style_is_404():
    with pytest.raises(TranslationError) as e:
        plan_request("v1/messages", {}, ("other_family_style",))
    assert e.value.status == 404


def test_strict_refuses_and_lenient_drops_semantic_fields():
    body = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    body["n"] = 3
    with pytest.raises(TranslationError) as e:
        plan_request("v1/chat/completions", body, ("anthropic_messages",))
    assert e.value.param == "n"
    tr = plan_request(
        "v1/chat/completions", body, ("anthropic_messages",), strict=False
    )
    assert tr.headers() == {"x-swarmpilot-dropped": "n"}
    assert "n" not in tr.body


def test_lenient_env(monkeypatch):
    monkeypatch.setenv("SWARMPILOT_API_TRANSLATION", "lenient")
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "x"}],
        "logprobs": True,
    }
    tr = plan_request("v1/chat/completions", body, ("anthropic_messages",))
    assert tr.ctx.dropped == ["logprobs"]


def test_hosted_tools_are_refused():
    with pytest.raises(TranslationError) as e:
        plan_request(
            "v1/responses",
            {"input": "x", "tools": [{"type": "web_search"}]},
            CHAT,
        )
    assert e.value.param == "tools[0].type"


def test_anthropic_tool_results_become_tool_messages_and_back():
    body = {
        "max_tokens": 10,
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "w",
                        "input": {"c": 1},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "ok",
                    },
                    {"type": "text", "text": "thanks"},
                ],
            },
        ],
    }
    tr = plan_request("v1/messages", body, CHAT)
    assert [m["role"] for m in tr.body["messages"]] == [
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert tr.body["messages"][1]["tool_calls"][0]["function"] == {
        "name": "w",
        "arguments": '{"c":1}',
    }
    # Back to Anthropic: tool result and text merge into one user turn.
    req = get_style("anthropic_messages").decode_request(body, Context())
    _, out = get_style("anthropic_messages").encode_request(req, Context())
    assert [m["role"] for m in out["messages"]] == ["user", "assistant", "user"]
    assert [b["type"] for b in out["messages"][2]["content"]] == [
        "tool_result",
        "text",
    ]


def test_images_convert_between_data_urls_and_base64_blocks():
    url = "data:image/png;base64,iVBORw0KGgo="
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": url}}],
            }
        ],
    }
    tr = plan_request("v1/chat/completions", body, ("anthropic_messages",))
    block = tr.body["messages"][0]["content"][0]
    assert block["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "iVBORw0KGgo=",
    }
    back = plan_request("v1/messages", tr.body, CHAT)
    assert back.body["messages"][0]["content"][0]["image_url"]["url"] == url


def test_engine_error_is_rewritten_in_client_style():
    tr = plan_request(
        "v1/messages",
        {"max_tokens": 5, "messages": [{"role": "user", "content": "x"}]},
        CHAT,
    )
    status, body = tr.response(
        422, {"error": {"message": "bad temperature", "type": "x"}}
    )
    assert status == 422
    assert body["type"] == "error"
    assert body["error"]["message"] == "bad temperature"


async def test_stream_from_one_body():
    """An engine that ignores stream=true still gives the client a stream."""
    tr = plan_request(
        "v1/messages",
        {
            "max_tokens": 5,
            "stream": True,
            "messages": [{"role": "user", "content": "x"}],
        },
        CHAT,
    )
    body = {
        "id": "c1",
        "choices": [{"message": {"content": "hi"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 5},
    }
    frames = b"".join(await collect(tr.stream_from_body(body)))
    events = [e async for e, _ in iter_sse(_one(frames))]
    assert events == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert b'"stop_reason":"max_tokens"' in frames


async def _one(data: bytes):
    yield data


async def test_sse_parser_handles_any_split():
    raw = (
        'event: a\ndata: {"x": "\xe9"}\n\n: comment\n\ndata: 1\ndata: 2\n\n'
    ).encode()

    async def chunks():
        for i in range(len(raw)):
            yield raw[i : i + 1]

    assert [x async for x in iter_sse(chunks())] == [
        ("a", '{"x": "\xe9"}'),
        (None, "1\n2"),
    ]


async def test_engine_stream_failure_becomes_error_event():
    tr = plan_request(
        "v1/messages",
        {
            "max_tokens": 5,
            "stream": True,
            "messages": [{"role": "user", "content": "x"}],
        },
        CHAT,
    )

    async def broken():
        yield b'data: {"id":"c","choices":[{"delta":{"content":"a"}}]}\n\n'
        raise ConnectionResetError("gone")

    frames = b"".join(await collect(tr.stream_response(broken())))
    assert b"event: error" in frames
    assert b"message_stop" not in frames


async def test_chat_stream_state_machine_opens_and_closes_parts():
    chunks = [
        {"id": "c", "choices": [{"delta": {"reasoning_content": "hmm"}}]},
        {"id": "c", "choices": [{"delta": {"content": "Let me check."}}]},
        {
            "id": "c",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "a",
                                "function": {"name": "f", "arguments": "{"},
                            }
                        ]
                    }
                }
            ],
        },
        {
            "id": "c",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": "}"}}
                        ]
                    }
                }
            ],
        },
        {
            "id": "c",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "b",
                                "function": {"name": "g", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]

    async def gen():
        for ch in chunks:
            yield f"data: {json.dumps(ch)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    events = await collect(OpenAIChatStyle().decode_stream(gen()))
    kinds = [
        (type(e).__name__, getattr(e, "index", None), getattr(e, "kind", None))
        for e in events
    ]
    assert kinds == [
        ("ResponseStart", None, None),
        ("PartStart", 0, "reasoning"),
        ("ReasoningDelta", 0, None),
        ("PartStop", 0, None),
        ("PartStart", 1, "text"),
        ("TextDelta", 1, None),
        ("PartStop", 1, None),
        ("PartStart", 2, "tool_call"),
        ("ToolArgsDelta", 2, None),
        ("ToolArgsDelta", 2, None),
        ("PartStop", 2, None),
        ("PartStart", 3, "tool_call"),
        ("ToolArgsDelta", 3, None),
        ("PartStop", 3, None),
        ("ResponseStop", None, None),
    ]
    assert events[-1].stop_reason == c.StopReason.TOOL_USE


def test_plugin_styles_can_be_registered():
    class Echo(OpenAIChatStyle):
        name = "echo"

        def routes(self):
            from swarmpilot.api.base import StyleRoute

            return (StyleRoute("v1/echo"),)

    register_style(Echo())
    try:
        assert style_for_path("v1/echo").name == "echo"
        tr = plan_request(
            "v1/echo", {"messages": [{"role": "user", "content": "x"}]}, CHAT
        )
        assert tr.path == "v1/chat/completions"
    finally:
        unregister_style("echo")
    assert style_for_path("v1/echo") is None


@pytest.mark.parametrize("style", sorted(CONVERSATION))
def test_prompt_text_is_style_independent(style):
    req = get_style(style).decode_request(CONVERSATION[style], Context())
    assert c.prompt_text(req) == (
        "You are terse.\n\nWhat is the capital of France?\n\nParis."
        "\n\nAnd of Spain?"
    )


def test_non_object_body_passes_through_or_is_refused():
    tr = plan_request("v1/chat/completions", [1, 2], CHAT)
    assert tr.passthrough and tr.body == [1, 2] and not tr.stream
    with pytest.raises(TranslationError) as e:
        plan_request("v1/messages", [1, 2], CHAT)
    assert e.value.status == 400


def _to_anthropic(**fields):
    body = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    body.update(fields)
    return plan_request("v1/chat/completions", body, ("anthropic_messages",))


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
def test_effort_fits_anthropic_thinking_limits(effort):
    """budget_tokens must be >= 1024 and below max_tokens."""
    out = _to_anthropic(reasoning_effort=effort).body
    budget = out["thinking"]["budget_tokens"]
    assert 1024 <= budget < out["max_tokens"]
    assert out["max_tokens"] - budget >= 4096


def test_thinking_budget_shrinks_to_the_client_limit():
    out = _to_anthropic(reasoning_effort="high", max_tokens=2000).body
    assert out["max_tokens"] == 2000
    assert out["thinking"]["budget_tokens"] == 1999


def test_thinking_is_dropped_when_it_cannot_fit():
    tr = _to_anthropic(reasoning_effort="high", max_tokens=500)
    assert "thinking" not in tr.body
    assert tr.ctx.dropped == ["reasoning"]


def test_thinking_drops_custom_temperature():
    tr = _to_anthropic(reasoning_effort="low", temperature=0.2)
    assert "temperature" not in tr.body
    assert "temperature" in tr.ctx.dropped


def test_temperature_above_one_is_clamped_for_anthropic():
    tr = _to_anthropic(temperature=1.7)
    assert tr.body["temperature"] == 1.0
    assert tr.ctx.dropped == ["temperature above 1"]


def test_stream_options_is_honoured_not_reported_dropped():
    tr = _to_anthropic(stream=True, stream_options={"include_usage": True})
    assert tr.headers() == {}


def test_empty_turns_are_not_sent_to_anthropic():
    tr = plan_request(
        "v1/chat/completions",
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": "b"},
            ],
        },
        ("anthropic_messages",),
    )
    assert tr.body["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ],
        }
    ]


def test_anthropic_tool_choice_tool_needs_a_name():
    with pytest.raises(TranslationError):
        plan_request(
            "v1/messages",
            {
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "x"}],
                "tool_choice": {"type": "tool"},
            },
            CHAT,
        )


def test_tool_result_images_are_not_silently_lost_on_responses_engines():
    body = {
        "max_tokens": 5,
        "messages": [
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t", "name": "f", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": "https://x/y.png",
                                },
                            }
                        ],
                    }
                ],
            },
        ],
    }
    with pytest.raises(TranslationError):
        plan_request("v1/messages", body, ("openai_responses",))
