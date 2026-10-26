"""Fake engines that each speak one API style, written by hand.

They deliberately share no code with ``swarmpilot.api``: each parses its
own wire format and answers with literal wire-format bodies, so the
conformance tests check SwarmPilot's translation against an independent
reading of every style.

Every engine answers the same way:

* if the request offers tools and the conversation does not end in a tool
  result, it calls ``get_weather`` with ``{"city": "Paris"}``;
* if the conversation ends in a tool result, it says
  ``result: <tool output>``;
* otherwise it says ``[<system prompt>] <last user text>``.

Streams split the text and the tool arguments into two deltas each.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

TOOL_ARGS = '{"city": "Paris"}'
ARG_PIECES = ('{"city": ', '"Paris"}')
USAGE_IN, USAGE_OUT = 7, 3


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        p.get("text", "")
        for p in content or []
        if p.get("type") in ("text", "input_text", "output_text")
    )


def _halves(s: str) -> tuple[str, str]:
    mid = len(s) // 2
    return s[:mid], s[mid:]


def _sse(obj: Any, event: str | None = None) -> bytes:
    head = f"event: {event}\n" if event else ""
    data = obj if isinstance(obj, str) else json.dumps(obj)
    return f"{head}data: {data}\n\n".encode()


def _stream(frames: list[bytes]) -> StreamingResponse:
    async def gen():
        for f in frames:
            yield f

    return StreamingResponse(gen(), media_type="text/event-stream")


def _base() -> FastAPI:
    app = FastAPI()
    app.state.seen = []

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/_seen")
    async def seen():
        return app.state.seen

    return app


# --- OpenAI Chat Completions -------------------------------------------


def chat_decide(body: dict[str, Any]) -> tuple[str, str]:
    msgs = body["messages"]
    system = " ".join(
        _text(m["content"])
        for m in msgs
        if m["role"] in ("system", "developer")
    )
    last = msgs[-1]
    if last["role"] == "tool":
        return "text", f"result: {_text(last['content'])}"
    if body.get("tools"):
        return "tool", ""
    user = [m for m in msgs if m["role"] == "user"][-1]
    return "text", f"[{system}] {_text(user['content'])}"


def chat_app() -> FastAPI:
    app = _base()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        app.state.seen.append(body)
        kind, text = chat_decide(body)
        model = body.get("model", "m")
        usage = {
            "prompt_tokens": USAGE_IN,
            "completion_tokens": USAGE_OUT,
            "total_tokens": USAGE_IN + USAGE_OUT,
        }
        call = {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": ""},
        }
        if not body.get("stream"):
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if kind == "tool":
                msg["tool_calls"] = [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": TOOL_ARGS,
                        },
                    }
                ]
            return {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": msg,
                        "finish_reason": "tool_calls"
                        if kind == "tool"
                        else "stop",
                    }
                ],
                "usage": usage,
            }

        def chunk(delta, finish=None, **extra):
            return _sse(
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": delta, "finish_reason": finish}
                    ],
                    **extra,
                }
            )

        frames = [chunk({"role": "assistant", "content": ""})]
        if kind == "tool":
            frames.append(chunk({"tool_calls": [call]}))
            for piece in ARG_PIECES:
                frames.append(
                    chunk(
                        {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": piece}}
                            ]
                        }
                    )
                )
            frames.append(chunk({}, "tool_calls"))
        else:
            for piece in _halves(text):
                frames.append(chunk({"content": piece}))
            frames.append(chunk({}, "stop"))
        if (body.get("stream_options") or {}).get("include_usage"):
            frames.append(
                _sse(
                    {
                        "id": "chatcmpl-1",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": model,
                        "choices": [],
                        "usage": usage,
                    }
                )
            )
        frames.append(_sse("[DONE]"))
        return _stream(frames)

    return app


# --- Anthropic Messages --------------------------------------------------


def messages_decide(body: dict[str, Any]) -> tuple[str, str]:
    system = body.get("system") or ""
    system = system if isinstance(system, str) else _text(system)
    last = body["messages"][-1]
    blocks = last["content"] if isinstance(last["content"], list) else []
    results = [b for b in blocks if b.get("type") == "tool_result"]
    if last["role"] == "user" and results:
        out = results[-1].get("content")
        return "text", f"result: {_text(out)}"
    if body.get("tools"):
        return "tool", ""
    user = [m for m in body["messages"] if m["role"] == "user"][-1]
    return "text", f"[{system}] {_text(user['content'])}"


def messages_app() -> FastAPI:
    app = _base()

    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        app.state.seen.append(body)
        if "max_tokens" not in body:
            return JSONResponse(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "max_tokens: Field required",
                    },
                },
                status_code=400,
            )
        kind, text = messages_decide(body)
        model = body.get("model", "m")
        if not body.get("stream"):
            content = (
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": json.loads(TOOL_ARGS),
                    }
                ]
                if kind == "tool"
                else [{"type": "text", "text": text}]
            )
            return {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": content,
                "stop_reason": "tool_use" if kind == "tool" else "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": USAGE_IN, "output_tokens": USAGE_OUT},
            }
        frames = [
            _sse(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": USAGE_IN, "output_tokens": 1},
                    },
                },
                "message_start",
            ),
            _sse({"type": "ping"}, "ping"),
        ]
        if kind == "tool":
            frames.append(
                _sse(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {},
                        },
                    },
                    "content_block_start",
                )
            )
            deltas = [
                {"type": "input_json_delta", "partial_json": p}
                for p in ARG_PIECES
            ]
        else:
            frames.append(
                _sse(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    "content_block_start",
                )
            )
            deltas = [{"type": "text_delta", "text": p} for p in _halves(text)]
        for d in deltas:
            frames.append(
                _sse(
                    {"type": "content_block_delta", "index": 0, "delta": d},
                    "content_block_delta",
                )
            )
        frames += [
            _sse(
                {"type": "content_block_stop", "index": 0}, "content_block_stop"
            ),
            _sse(
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "tool_use"
                        if kind == "tool"
                        else "end_turn",
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": USAGE_OUT},
                },
                "message_delta",
            ),
            _sse({"type": "message_stop"}, "message_stop"),
        ]
        return _stream(frames)

    return app


# --- OpenAI Responses ----------------------------------------------------


def responses_decide(body: dict[str, Any]) -> tuple[str, str]:
    system = body.get("instructions") or ""
    items = body["input"]
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    last = items[-1]
    if last.get("type") == "function_call_output":
        return "text", f"result: {_text(last['output'])}"
    if body.get("tools"):
        return "tool", ""
    user = [i for i in items if i.get("role") == "user"][-1]
    return "text", f"[{system}] {_text(user['content'])}"


def _resp(model: str, output: list, status: str = "completed") -> dict:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "status": status,
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": USAGE_IN,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": USAGE_OUT,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": USAGE_IN + USAGE_OUT,
        },
    }


def responses_app() -> FastAPI:
    app = _base()

    @app.post("/v1/responses")
    async def responses(request: Request):
        body = await request.json()
        app.state.seen.append(body)
        kind, text = responses_decide(body)
        model = body.get("model", "m")
        call_item = {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": TOOL_ARGS,
            "status": "completed",
        }
        part = {"type": "output_text", "text": text, "annotations": []}
        msg_item = {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [part],
        }
        final = _resp(model, [call_item if kind == "tool" else msg_item])
        if not body.get("stream"):
            return final
        n = iter(range(100))

        def ev(kind_: str, **data):
            return _sse(
                {"type": kind_, "sequence_number": next(n), **data}, kind_
            )

        start = _resp(model, [], "in_progress")
        frames = [
            ev("response.created", response=start),
            ev("response.in_progress", response=start),
        ]
        if kind == "tool":
            frames.append(
                ev(
                    "response.output_item.added",
                    output_index=0,
                    item={
                        **call_item,
                        "arguments": "",
                        "status": "in_progress",
                    },
                )
            )
            for p in ARG_PIECES:
                frames.append(
                    ev(
                        "response.function_call_arguments.delta",
                        item_id="fc_1",
                        output_index=0,
                        delta=p,
                    )
                )
            frames += [
                ev(
                    "response.function_call_arguments.done",
                    item_id="fc_1",
                    output_index=0,
                    arguments=TOOL_ARGS,
                ),
                ev("response.output_item.done", output_index=0, item=call_item),
            ]
        else:
            frames += [
                ev(
                    "response.output_item.added",
                    output_index=0,
                    item={**msg_item, "status": "in_progress", "content": []},
                ),
                ev(
                    "response.content_part.added",
                    item_id="msg_1",
                    output_index=0,
                    content_index=0,
                    part={**part, "text": ""},
                ),
            ]
            for p in _halves(text):
                frames.append(
                    ev(
                        "response.output_text.delta",
                        item_id="msg_1",
                        output_index=0,
                        content_index=0,
                        delta=p,
                        logprobs=[],
                    )
                )
            frames += [
                ev(
                    "response.output_text.done",
                    item_id="msg_1",
                    output_index=0,
                    content_index=0,
                    text=text,
                    logprobs=[],
                ),
                ev(
                    "response.content_part.done",
                    item_id="msg_1",
                    output_index=0,
                    content_index=0,
                    part=part,
                ),
                ev("response.output_item.done", output_index=0, item=msg_item),
            ]
        frames.append(ev("response.completed", response=final))
        return _stream(frames)

    return app


ENGINES = {
    "openai_chat": chat_app,
    "anthropic_messages": messages_app,
    "openai_responses": responses_app,
}
