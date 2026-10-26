"""OpenAI Chat Completions (``POST /v1/chat/completions``)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any, ClassVar

from swarmpilot.api import canonical as c
from swarmpilot.api.base import (
    ApiStyle,
    Context,
    StyleRoute,
    TranslationError,
    unsupported,
)
from swarmpilot.api.sse import frame, iter_sse
from swarmpilot.api.styles._common import (
    as_list,
    as_mapping,
    as_str,
    new_id,
    now,
    opt_float,
    opt_int,
    stop_list,
)

# Fields the canonical model holds; everything else is an extension.
_KNOWN = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stop",
        "seed",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "reasoning_effort",
        "stream",
        "user",
    }
)

_FINISH_TO_STOP = {
    "stop": c.StopReason.END_TURN,
    "length": c.StopReason.MAX_TOKENS,
    "tool_calls": c.StopReason.TOOL_USE,
    "function_call": c.StopReason.TOOL_USE,
    "content_filter": c.StopReason.CONTENT_FILTER,
}
_STOP_TO_FINISH = {
    c.StopReason.END_TURN: "stop",
    c.StopReason.STOP_SEQUENCE: "stop",
    c.StopReason.MAX_TOKENS: "length",
    c.StopReason.TOOL_USE: "tool_calls",
    c.StopReason.CONTENT_FILTER: "content_filter",
    c.StopReason.ERROR: "stop",
}
_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_exceeded",
    503: "service_unavailable",
    504: "gateway_timeout",
}


def _user_parts(content: Any, where: str) -> list[c.Part]:
    if isinstance(content, str):
        return [c.TextPart(content)]
    parts: list[c.Part] = []
    for i, p in enumerate(as_list(content, where)):
        p = as_mapping(p, f"{where}[{i}]")
        kind = p.get("type")
        if kind == "text":
            parts.append(
                c.TextPart(as_str(p.get("text"), f"{where}[{i}].text"))
            )
        elif kind == "image_url":
            img = p.get("image_url")
            url = img.get("url") if isinstance(img, Mapping) else img
            detail = img.get("detail") if isinstance(img, Mapping) else None
            parts.append(c.ImagePart(as_str(url, f"{where}[{i}].image_url")))
            if detail:
                parts[-1].detail = detail
        elif kind == "refusal":
            parts.append(c.TextPart(str(p.get("refusal") or "")))
        else:
            raise unsupported(f"{where}[{i}].type", f"content type {kind!r}")
    return parts


def _tool_results(m: Mapping[str, Any], where: str) -> c.Message:
    call_id = as_str(m.get("tool_call_id"), f"{where}.tool_call_id")
    content = [
        p
        for p in _user_parts(m.get("content") or "", f"{where}.content")
        if isinstance(p, c.TextPart | c.ImagePart)
    ]
    return c.Message("tool", [c.ToolResultPart(call_id, content)])


def _assistant(m: Mapping[str, Any], where: str) -> c.Message:
    parts: list[c.Part] = []
    reasoning = m.get("reasoning_content") or m.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        parts.append(c.ReasoningPart(reasoning))
    content = m.get("content")
    if content:
        parts.extend(_user_parts(content, f"{where}.content"))
    for j, call in enumerate(m.get("tool_calls") or []):
        call = as_mapping(call, f"{where}.tool_calls[{j}]")
        fn = as_mapping(call.get("function"), f"{where}.tool_calls[{j}]")
        args = fn.get("arguments")
        parts.append(
            c.ToolCallPart(
                id=str(call.get("id") or new_id("call_")),
                name=as_str(fn.get("name"), f"{where}.tool_calls[{j}].name"),
                arguments=args if isinstance(args, str) else "{}",
            )
        )
    return c.Message("assistant", parts)


def _tool_choice(v: Any) -> c.ToolChoice | None:
    if v is None:
        return None
    if v in ("auto", "none", "required"):
        return c.ToolChoice(v)
    if isinstance(v, Mapping) and isinstance(v.get("function"), Mapping):
        return c.ToolChoice(
            "tool", as_str(v["function"].get("name"), "tool_choice")
        )
    raise TranslationError("invalid 'tool_choice'", param="tool_choice")


def _response_format(v: Any) -> c.ResponseFormat | None:
    if v is None:
        return None
    v = as_mapping(v, "response_format")
    kind = v.get("type")
    if kind in ("text", "json_object"):
        return c.ResponseFormat(kind)
    if kind == "json_schema":
        js = as_mapping(v.get("json_schema"), "response_format.json_schema")
        return c.ResponseFormat(
            "json_schema",
            name=js.get("name"),
            schema=js.get("schema"),
            strict=js.get("strict"),
        )
    raise TranslationError("invalid 'response_format'", param="response_format")


class OpenAIChatStyle(ApiStyle):
    """OpenAI Chat Completions, as served by OpenAI, vLLM and SGLang."""

    name: ClassVar[str] = "openai_chat"
    semantic_extensions: ClassVar[frozenset[str]] = frozenset(
        {"n", "logprobs", "top_logprobs", "logit_bias", "audio", "modalities"}
    )

    def routes(self) -> tuple[StyleRoute, ...]:
        """Return the chat completions route."""
        return (StyleRoute("v1/chat/completions"),)

    def feature_extractors(
        self, default_max_tokens: int
    ) -> tuple[tuple[str, str], ...]:
        """Message text characters and the output limit."""
        return (
            (
                "prompt_length",
                "sum(chars(messages[*].content), "
                "chars(messages[*].content[*].text))",
            ),
            (
                "max_tokens",
                "coalesce(max_tokens, max_completion_tokens, "
                f"{default_max_tokens})",
            ),
        )

    # Client side.

    def decode_request(
        self, body: Mapping[str, Any], ctx: Context
    ) -> c.Request:
        """Parse a chat completions request."""
        req = c.Request(model=body.get("model"))
        for i, m in enumerate(as_list(body.get("messages"), "messages")):
            where = f"messages[{i}]"
            m = as_mapping(m, where)
            role = m.get("role")
            if role in ("system", "developer"):
                text = "".join(
                    p.text
                    for p in _user_parts(m.get("content") or "", where)
                    if isinstance(p, c.TextPart)
                )
                req.system.append(c.TextPart(text))
            elif role == "user":
                req.messages.append(
                    c.Message("user", _user_parts(m.get("content"), where))
                )
            elif role == "assistant":
                req.messages.append(_assistant(m, where))
            elif role == "tool":
                req.messages.append(_tool_results(m, where))
            else:
                raise unsupported(f"{where}.role", f"role {role!r}")
        req.max_output_tokens = opt_int(body, "max_completion_tokens")
        if req.max_output_tokens is None:
            req.max_output_tokens = opt_int(body, "max_tokens")
        req.temperature = opt_float(body, "temperature")
        req.top_p = opt_float(body, "top_p")
        req.top_k = opt_int(body, "top_k")
        req.stop = stop_list(body.get("stop"), "stop")
        req.seed = opt_int(body, "seed")
        for j, t in enumerate(body.get("tools") or []):
            t = as_mapping(t, f"tools[{j}]")
            if t.get("type") != "function":
                raise unsupported(
                    f"tools[{j}].type", f"tool type {t.get('type')!r}"
                )
            fn = as_mapping(t.get("function"), f"tools[{j}].function")
            req.tools.append(
                c.Tool(
                    name=as_str(fn.get("name"), f"tools[{j}].function.name"),
                    description=fn.get("description"),
                    parameters=fn.get("parameters")
                    or {"type": "object", "properties": {}},
                    strict=fn.get("strict"),
                )
            )
        req.tool_choice = _tool_choice(body.get("tool_choice"))
        req.parallel_tool_calls = body.get("parallel_tool_calls")
        req.response_format = _response_format(body.get("response_format"))
        req.reasoning_effort = body.get("reasoning_effort")
        req.stream = body.get("stream") is True
        req.user = body.get("user")
        ext = {k: v for k, v in body.items() if k not in _KNOWN}
        if ext.get("n") in (None, 1):
            ext.pop("n", None)
        if ext.get("logprobs") in (None, False):
            ext.pop("logprobs", None)
        if ext:
            req.extensions[self.name] = ext
        return req

    def encode_response(
        self, resp: c.Response, req: c.Request
    ) -> dict[str, Any]:
        """Write a chat completion object."""
        text = "".join(p.text for p in resp.output if isinstance(p, c.TextPart))
        reasoning = "".join(
            p.text for p in resp.output if isinstance(p, c.ReasoningPart)
        )
        calls = [
            {
                "id": p.id,
                "type": "function",
                "function": {"name": p.name, "arguments": p.arguments},
            }
            for p in resp.output
            if isinstance(p, c.ToolCallPart)
        ]
        message: dict[str, Any] = {
            "role": "assistant",
            "content": text if text or not calls else None,
        }
        if calls:
            message["tool_calls"] = calls
        if reasoning:
            message["reasoning_content"] = reasoning
        return {
            "id": resp.id
            if resp.id.startswith("chatcmpl")
            else f"chatcmpl-{resp.id}",
            "object": "chat.completion",
            "created": resp.created or now(),
            "model": resp.model or req.model or "",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "logprobs": None,
                    "finish_reason": _STOP_TO_FINISH[resp.stop_reason],
                }
            ],
            "usage": _usage_out(resp.usage),
        }

    async def encode_stream(
        self, events: AsyncIterator[Any], req: c.Request
    ) -> AsyncIterator[bytes]:
        """Write ``chat.completion.chunk`` events and ``[DONE]``."""
        opts = req.extensions.get(self.name, {}).get("stream_options")
        include_usage = isinstance(opts, Mapping) and bool(
            opts.get("include_usage")
        )
        cid, model, created = new_id("chatcmpl-"), req.model or "", now()
        tool_index: dict[int, int] = {}

        def chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
            return frame(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": delta, "finish_reason": finish}
                    ],
                }
            )

        async for ev in events:
            if isinstance(ev, c.ResponseStart):
                model = ev.model or model
                created = ev.created or created
                yield chunk({"role": "assistant", "content": ""})
            elif isinstance(ev, c.TextDelta):
                if ev.text:
                    yield chunk({"content": ev.text})
            elif isinstance(ev, c.ReasoningDelta):
                if ev.text:
                    yield chunk({"reasoning_content": ev.text})
            elif isinstance(ev, c.PartStart) and ev.kind == "tool_call":
                k = tool_index[ev.index] = len(tool_index)
                yield chunk(
                    {
                        "tool_calls": [
                            {
                                "index": k,
                                "id": ev.id,
                                "type": "function",
                                "function": {"name": ev.name, "arguments": ""},
                            }
                        ]
                    }
                )
            elif isinstance(ev, c.ToolArgsDelta):
                yield chunk(
                    {
                        "tool_calls": [
                            {
                                "index": tool_index.get(ev.index, 0),
                                "function": {"arguments": ev.delta},
                            }
                        ]
                    }
                )
            elif isinstance(ev, c.ResponseStop):
                yield chunk({}, _STOP_TO_FINISH[ev.stop_reason])
                if include_usage:
                    yield frame(
                        {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [],
                            "usage": _usage_out(ev.usage),
                        }
                    )
                yield b"data: [DONE]\n\n"
                return
            elif isinstance(ev, c.ErrorEvent):
                yield frame(self.encode_error(500, ev.message, ev.error_type))
                yield b"data: [DONE]\n\n"
                return

    def encode_error(
        self,
        status: int,
        message: str,
        error_type: str | None = None,
        param: str | None = None,
    ) -> dict[str, Any]:
        """Return an OpenAI error object."""
        return {
            "error": {
                "message": message,
                "type": error_type or _ERROR_TYPES.get(status, "api_error"),
                "param": param,
                "code": None,
            }
        }

    # Engine side.

    def encode_request(
        self, req: c.Request, ctx: Context
    ) -> tuple[str, dict[str, Any]]:
        """Build a chat completions body."""
        messages: list[dict[str, Any]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system_text()})
        for m in req.messages:
            messages.extend(_encode_message(m, ctx))
        body: dict[str, Any] = {"messages": messages}
        if req.model is not None:
            body["model"] = req.model
        if req.max_output_tokens is not None:
            body["max_tokens"] = req.max_output_tokens
        for key in ("temperature", "top_p", "top_k", "seed", "user"):
            v = getattr(req, key)
            if v is not None:
                body[key] = v
        if req.stop:
            body["stop"] = req.stop
        if req.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        k: v
                        for k, v in (
                            ("name", t.name),
                            ("description", t.description),
                            ("parameters", t.parameters),
                            ("strict", t.strict),
                        )
                        if v is not None
                    },
                }
                for t in req.tools
            ]
        if req.tool_choice is not None:
            tc = req.tool_choice
            body["tool_choice"] = (
                {"type": "function", "function": {"name": tc.name}}
                if tc.mode == "tool"
                else tc.mode
            )
        if req.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = req.parallel_tool_calls
        rf = req.response_format
        if rf is not None and rf.type != "text":
            if rf.type == "json_object":
                body["response_format"] = {"type": "json_object"}
            else:
                js: dict[str, Any] = {
                    "name": rf.name or "response",
                    "schema": rf.schema or {},
                }
                if rf.strict is not None:
                    js["strict"] = rf.strict
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": js,
                }
        if req.reasoning_effort is not None:
            body["reasoning_effort"] = req.reasoning_effort
        elif req.reasoning_budget is not None:
            ctx.drop("reasoning budget")
        if req.stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return self.generate_path(), body

    def decode_response(self, body: Mapping[str, Any]) -> c.Response:
        """Parse a chat completion object (the first choice)."""
        choices = body.get("choices") or [{}]
        choice = choices[0] if isinstance(choices[0], Mapping) else {}
        msg = choice.get("message") or {}
        output: list[c.Part] = []
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            output.append(c.ReasoningPart(reasoning))
        if isinstance(msg.get("content"), str) and msg["content"]:
            output.append(c.TextPart(msg["content"]))
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            output.append(
                c.ToolCallPart(
                    id=str(call.get("id") or new_id("call_")),
                    name=str(fn.get("name") or ""),
                    arguments=fn.get("arguments") or "{}",
                )
            )
        stop, seq = _stop_reason(choice)
        return c.Response(
            id=str(body.get("id") or new_id("chatcmpl-")),
            model=body.get("model"),
            output=output,
            stop_reason=stop,
            stop_sequence=seq,
            usage=_usage_in(body.get("usage")),
            created=int(body.get("created") or 0),
        )

    async def decode_stream(
        self, chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[Any]:
        """Turn chat completion chunks into canonical events.

        Chat streams have no part boundaries, so a part opens when a new
        kind of delta (content, reasoning, or a tool call with a new
        index) arrives and closes when the next one opens.
        """
        started = False
        index = -1
        open_key: tuple[str, int] | None = None
        stop, seq = c.StopReason.END_TURN, None
        usage = c.Usage()
        async for _event, data in iter_sse(chunks):
            if data.strip() == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if "error" in obj and "choices" not in obj:
                yield c.ErrorEvent(self.decode_error(obj))
                return
            if not started:
                started = True
                yield c.ResponseStart(
                    id=str(obj.get("id") or new_id("chatcmpl-")),
                    model=obj.get("model"),
                    created=int(obj.get("created") or 0),
                )
            if obj.get("usage"):
                usage = _usage_in(obj["usage"])
            for choice in obj.get("choices") or []:
                if choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta") or {}
                pieces: list[tuple[tuple[str, int], Any]] = []
                r = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(r, str) and r:
                    pieces.append((("reasoning", 0), r))
                if isinstance(delta.get("content"), str) and delta["content"]:
                    pieces.append((("text", 0), delta["content"]))
                for call in delta.get("tool_calls") or []:
                    pieces.append((("tool_call", call.get("index", 0)), call))
                for key, value in pieces:
                    if key != open_key:
                        if open_key is not None:
                            yield c.PartStop(index)
                        index += 1
                        open_key = key
                        if key[0] == "tool_call":
                            fn = value.get("function") or {}
                            yield c.PartStart(
                                index,
                                "tool_call",
                                id=value.get("id") or new_id("call_"),
                                name=fn.get("name") or "",
                            )
                        else:
                            yield c.PartStart(index, key[0])
                    if key[0] == "text":
                        yield c.TextDelta(index, value)
                    elif key[0] == "reasoning":
                        yield c.ReasoningDelta(index, value)
                    else:
                        args = (value.get("function") or {}).get("arguments")
                        if args:
                            yield c.ToolArgsDelta(index, args)
                if choice.get("finish_reason"):
                    stop, seq = _stop_reason(choice)
        if not started:
            yield c.ErrorEvent("Engine stream ended without a response")
            return
        if open_key is not None:
            yield c.PartStop(index)
        yield c.ResponseStop(stop, seq, usage)


def _encode_message(m: c.Message, ctx: Context) -> list[dict[str, Any]]:
    if m.role == "tool":
        out = []
        for p in m.content:
            if isinstance(p, c.ToolResultPart):
                text = "".join(
                    x.text for x in p.content if isinstance(x, c.TextPart)
                )
                if any(isinstance(x, c.ImagePart) for x in p.content):
                    ctx.drop("image in tool result", semantic=True)
                out.append(
                    {"role": "tool", "tool_call_id": p.call_id, "content": text}
                )
        return out
    content: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for p in m.content:
        if isinstance(p, c.TextPart):
            content.append({"type": "text", "text": p.text})
        elif isinstance(p, c.ImagePart):
            img: dict[str, Any] = {"url": p.url}
            if p.detail:
                img["detail"] = p.detail
            content.append({"type": "image_url", "image_url": img})
        elif isinstance(p, c.ToolCallPart):
            calls.append(
                {
                    "id": p.id,
                    "type": "function",
                    "function": {"name": p.name, "arguments": p.arguments},
                }
            )
        elif isinstance(p, c.ReasoningPart):
            # Chat engines do not take earlier reasoning back.
            continue
    out_msg: dict[str, Any] = {"role": m.role}
    if all(x["type"] == "text" for x in content):
        out_msg["content"] = "".join(x["text"] for x in content) or (
            None if calls else ""
        )
    else:
        out_msg["content"] = content
    if calls:
        out_msg["tool_calls"] = calls
    return [out_msg]


def _stop_reason(choice: Mapping[str, Any]) -> tuple[c.StopReason, str | None]:
    finish = choice.get("finish_reason")
    matched = choice.get("stop_reason")
    if finish == "stop" and isinstance(matched, str):
        return c.StopReason.STOP_SEQUENCE, matched
    return _FINISH_TO_STOP.get(finish, c.StopReason.END_TURN), None


def _usage_in(u: Any) -> c.Usage:
    if not isinstance(u, Mapping):
        return c.Usage()
    pd = u.get("prompt_tokens_details") or {}
    cd = u.get("completion_tokens_details") or {}
    return c.Usage(
        input_tokens=u.get("prompt_tokens"),
        output_tokens=u.get("completion_tokens"),
        cached_input_tokens=pd.get("cached_tokens") if pd else None,
        reasoning_tokens=cd.get("reasoning_tokens") if cd else None,
    )


def _usage_out(u: c.Usage) -> dict[str, Any]:
    out: dict[str, Any] = {
        "prompt_tokens": u.input_tokens or 0,
        "completion_tokens": u.output_tokens or 0,
        "total_tokens": (u.input_tokens or 0) + (u.output_tokens or 0),
    }
    if u.cached_input_tokens is not None:
        out["prompt_tokens_details"] = {"cached_tokens": u.cached_input_tokens}
    if u.reasoning_tokens is not None:
        out["completion_tokens_details"] = {
            "reasoning_tokens": u.reasoning_tokens
        }
    return out
