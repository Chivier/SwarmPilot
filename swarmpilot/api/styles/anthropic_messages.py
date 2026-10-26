"""Anthropic Messages (``POST /v1/messages``)."""

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
    dump_arguments,
    new_id,
    opt_float,
    opt_int,
    parse_arguments,
    split_data_url,
    stop_list,
)

_KNOWN = frozenset(
    {
        "model",
        "messages",
        "system",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "tools",
        "tool_choice",
        "thinking",
        "stream",
        "metadata",
    }
)

_STOP_IN = {
    "end_turn": c.StopReason.END_TURN,
    "max_tokens": c.StopReason.MAX_TOKENS,
    "stop_sequence": c.StopReason.STOP_SEQUENCE,
    "tool_use": c.StopReason.TOOL_USE,
    "pause_turn": c.StopReason.END_TURN,
    "refusal": c.StopReason.CONTENT_FILTER,
}
_STOP_OUT = {
    c.StopReason.END_TURN: "end_turn",
    c.StopReason.MAX_TOKENS: "max_tokens",
    c.StopReason.STOP_SEQUENCE: "stop_sequence",
    c.StopReason.TOOL_USE: "tool_use",
    c.StopReason.CONTENT_FILTER: "refusal",
    c.StopReason.ERROR: "end_turn",
}
_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    503: "overloaded_error",
    529: "overloaded_error",
}
# Reasoning effort to an Anthropic thinking budget, when an effort-style
# client reaches an Anthropic-style engine.
_MIN_THINKING_BUDGET = 1024
_EFFORT_BUDGET = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 24576}


def _text_blocks(value: Any, where: str, ctx: Context) -> list[c.TextPart]:
    if isinstance(value, str):
        return [c.TextPart(value)]
    out = []
    for i, b in enumerate(as_list(value, where)):
        b = as_mapping(b, f"{where}[{i}]")
        if b.get("type") != "text":
            raise unsupported(f"{where}[{i}].type", f"block {b.get('type')!r}")
        if "cache_control" in b:
            ctx.drop("cache_control")
        out.append(c.TextPart(as_str(b.get("text"), f"{where}[{i}].text")))
    return out


def _image(b: Mapping[str, Any], where: str) -> c.ImagePart:
    src = as_mapping(b.get("source"), f"{where}.source")
    if src.get("type") == "base64":
        media = as_str(src.get("media_type"), f"{where}.source.media_type")
        data = as_str(src.get("data"), f"{where}.source.data")
        return c.ImagePart(f"data:{media};base64,{data}")
    if src.get("type") == "url":
        return c.ImagePart(as_str(src.get("url"), f"{where}.source.url"))
    raise unsupported(f"{where}.source.type", f"source {src.get('type')!r}")


def _blocks(
    content: Any, role: str, where: str, ctx: Context
) -> tuple[list[c.Part], list[c.ToolResultPart]]:
    """Return a message's parts and, separately, its tool results."""
    if isinstance(content, str):
        return [c.TextPart(content)], []
    parts: list[c.Part] = []
    results: list[c.ToolResultPart] = []
    for i, b in enumerate(as_list(content, where)):
        w = f"{where}[{i}]"
        b = as_mapping(b, w)
        if "cache_control" in b:
            ctx.drop("cache_control")
        kind = b.get("type")
        if kind == "text":
            parts.append(c.TextPart(as_str(b.get("text"), f"{w}.text")))
        elif kind == "image":
            parts.append(_image(b, w))
        elif kind == "tool_use" and role == "assistant":
            parts.append(
                c.ToolCallPart(
                    id=as_str(b.get("id"), f"{w}.id"),
                    name=as_str(b.get("name"), f"{w}.name"),
                    arguments=dump_arguments(b.get("input") or {}),
                )
            )
        elif kind == "tool_result" and role == "user":
            raw = b.get("content")
            inner: list[c.TextPart | c.ImagePart] = []
            if isinstance(raw, str):
                inner.append(c.TextPart(raw))
            elif raw is not None:
                for j, x in enumerate(as_list(raw, f"{w}.content")):
                    x = as_mapping(x, f"{w}.content[{j}]")
                    if x.get("type") == "image":
                        inner.append(_image(x, f"{w}.content[{j}]"))
                    else:
                        inner.append(c.TextPart(str(x.get("text") or "")))
            results.append(
                c.ToolResultPart(
                    call_id=as_str(b.get("tool_use_id"), f"{w}.tool_use_id"),
                    content=inner,
                    is_error=bool(b.get("is_error")),
                )
            )
        elif kind == "thinking" and role == "assistant":
            parts.append(
                c.ReasoningPart(
                    str(b.get("thinking") or ""), b.get("signature")
                )
            )
        elif kind == "redacted_thinking":
            ctx.drop("redacted_thinking")
        else:
            raise unsupported(f"{w}.type", f"content block {kind!r}")
    return parts, results


class AnthropicMessagesStyle(ApiStyle):
    """Anthropic Messages, as served by Anthropic and compatible servers."""

    name: ClassVar[str] = "anthropic_messages"

    def routes(self) -> tuple[StyleRoute, ...]:
        """Return the messages route."""
        return (StyleRoute("v1/messages"),)

    def feature_extractors(
        self, default_max_tokens: int
    ) -> tuple[tuple[str, str], ...]:
        """System and message text characters and the output limit."""
        return (
            (
                "prompt_length",
                "sum(chars(system), chars(system[*].text), "
                "chars(messages[*].content), "
                "chars(messages[*].content[*].text), "
                "chars(messages[*].content[*].content), "
                "chars(messages[*].content[*].content[*].text))",
            ),
            ("max_tokens", f"coalesce(max_tokens, {default_max_tokens})"),
        )

    # Client side.

    def decode_request(
        self, body: Mapping[str, Any], ctx: Context
    ) -> c.Request:
        """Parse a Messages request."""
        req = c.Request(model=body.get("model"))
        if body.get("system") is not None:
            req.system = _text_blocks(body["system"], "system", ctx)
        for i, m in enumerate(as_list(body.get("messages"), "messages")):
            where = f"messages[{i}]"
            m = as_mapping(m, where)
            role = m.get("role")
            if role not in ("user", "assistant"):
                raise unsupported(f"{where}.role", f"role {role!r}")
            parts, results = _blocks(m.get("content"), role, where, ctx)
            # Tool results lead the user turn that carries them.
            for r in results:
                req.messages.append(c.Message("tool", [r]))
            if parts or not results:
                req.messages.append(c.Message(role, parts))
        req.max_output_tokens = opt_int(body, "max_tokens")
        if req.max_output_tokens is None:
            raise TranslationError(
                "'max_tokens' is required", param="max_tokens"
            )
        req.temperature = opt_float(body, "temperature")
        req.top_p = opt_float(body, "top_p")
        req.top_k = opt_int(body, "top_k")
        req.stop = stop_list(body.get("stop_sequences"), "stop_sequences")
        for j, t in enumerate(body.get("tools") or []):
            t = as_mapping(t, f"tools[{j}]")
            if t.get("type") not in (None, "custom"):
                raise unsupported(
                    f"tools[{j}].type", f"server tool {t.get('type')!r}"
                )
            if "cache_control" in t:
                ctx.drop("cache_control")
            req.tools.append(
                c.Tool(
                    name=as_str(t.get("name"), f"tools[{j}].name"),
                    description=t.get("description"),
                    parameters=t.get("input_schema")
                    or {"type": "object", "properties": {}},
                )
            )
        tc = body.get("tool_choice")
        if tc is not None:
            tc = as_mapping(tc, "tool_choice")
            mode = {
                "auto": "auto",
                "any": "required",
                "tool": "tool",
                "none": "none",
            }
            if tc.get("type") not in mode:
                raise TranslationError(
                    "invalid 'tool_choice'", param="tool_choice"
                )
            if tc["type"] == "tool":
                as_str(tc.get("name"), "tool_choice.name")
            req.tool_choice = c.ToolChoice(mode[tc["type"]], tc.get("name"))
            if tc.get("disable_parallel_tool_use") is not None:
                req.parallel_tool_calls = not tc["disable_parallel_tool_use"]
        thinking = body.get("thinking")
        if isinstance(thinking, Mapping) and thinking.get("type") == "enabled":
            req.reasoning_budget = opt_int(thinking, "budget_tokens")
        meta = body.get("metadata")
        if isinstance(meta, Mapping) and isinstance(meta.get("user_id"), str):
            req.user = meta["user_id"]
        req.stream = body.get("stream") is True
        ext = {k: v for k, v in body.items() if k not in _KNOWN}
        if ext:
            req.extensions[self.name] = ext
        return req

    def encode_response(
        self, resp: c.Response, req: c.Request
    ) -> dict[str, Any]:
        """Write a Messages response."""
        content = [_block_out(p) for p in resp.output]
        return {
            "id": resp.id if resp.id.startswith("msg_") else f"msg_{resp.id}",
            "type": "message",
            "role": "assistant",
            "model": resp.model or req.model or "",
            "content": [b for b in content if b is not None],
            "stop_reason": _STOP_OUT[resp.stop_reason],
            "stop_sequence": resp.stop_sequence,
            "usage": _usage_out(resp.usage),
        }

    async def encode_stream(
        self, events: AsyncIterator[Any], req: c.Request
    ) -> AsyncIterator[bytes]:
        """Write the Messages SSE event sequence."""
        kinds: dict[int, str] = {}
        async for ev in events:
            if isinstance(ev, c.ResponseStart):
                rid = ev.id if ev.id.startswith("msg_") else f"msg_{ev.id}"
                yield frame(
                    {
                        "type": "message_start",
                        "message": {
                            "id": rid,
                            "type": "message",
                            "role": "assistant",
                            "model": ev.model or req.model or "",
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            # The OpenAI engines report usage only at the end.
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    },
                    "message_start",
                )
            elif isinstance(ev, c.PartStart):
                kinds[ev.index] = ev.kind
                if ev.kind == "text":
                    block: dict[str, Any] = {"type": "text", "text": ""}
                elif ev.kind == "reasoning":
                    block = {
                        "type": "thinking",
                        "thinking": "",
                        "signature": "",
                    }
                else:
                    block = {
                        "type": "tool_use",
                        "id": ev.id,
                        "name": ev.name,
                        "input": {},
                    }
                yield frame(
                    {
                        "type": "content_block_start",
                        "index": ev.index,
                        "content_block": block,
                    },
                    "content_block_start",
                )
            elif isinstance(ev, c.TextDelta):
                yield _delta(ev.index, {"type": "text_delta", "text": ev.text})
            elif isinstance(ev, c.ToolArgsDelta):
                yield _delta(
                    ev.index,
                    {"type": "input_json_delta", "partial_json": ev.delta},
                )
            elif isinstance(ev, c.ReasoningDelta):
                if ev.text:
                    yield _delta(
                        ev.index,
                        {"type": "thinking_delta", "thinking": ev.text},
                    )
                if ev.signature:
                    yield _delta(
                        ev.index,
                        {"type": "signature_delta", "signature": ev.signature},
                    )
            elif isinstance(ev, c.PartStop):
                yield frame(
                    {"type": "content_block_stop", "index": ev.index},
                    "content_block_stop",
                )
            elif isinstance(ev, c.ResponseStop):
                yield frame(
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": _STOP_OUT[ev.stop_reason],
                            "stop_sequence": ev.stop_sequence,
                        },
                        "usage": _usage_out(ev.usage),
                    },
                    "message_delta",
                )
                yield frame({"type": "message_stop"}, "message_stop")
                return
            elif isinstance(ev, c.ErrorEvent):
                yield frame(self.encode_error(500, ev.message), "error")
                return

    def encode_error(
        self,
        status: int,
        message: str,
        error_type: str | None = None,
        param: str | None = None,
    ) -> dict[str, Any]:
        """Return an Anthropic error body."""
        kind = _ERROR_TYPES.get(status, "api_error")
        if status == 400 and error_type == "invalid_request_error":
            kind = error_type
        return {"type": "error", "error": {"type": kind, "message": message}}

    # Engine side.

    def encode_request(
        self, req: c.Request, ctx: Context
    ) -> tuple[str, dict[str, Any]]:
        """Build a Messages body."""
        messages: list[dict[str, Any]] = []
        for m in req.messages:
            role = "user" if m.role == "tool" else m.role
            blocks = [b for b in (_block_in(p, ctx) for p in m.content) if b]
            if not blocks:
                # Anthropic refuses empty turns; neighbours of the same
                # role merge below.
                continue
            if messages and messages[-1]["role"] == role:
                # Messages alternate; merge tool results and the user turn.
                messages[-1]["content"].extend(blocks)
            else:
                messages.append({"role": role, "content": blocks})
        body: dict[str, Any] = {
            "messages": messages,
            "max_tokens": req.max_output_tokens or ctx.default_max_tokens,
        }
        if req.model is not None:
            body["model"] = req.model
        if req.system:
            body["system"] = req.system_text()
        for key in ("temperature", "top_p", "top_k"):
            v = getattr(req, key)
            if v is not None:
                body[key] = v
        if req.temperature is not None and req.temperature > 1:
            # OpenAI styles allow 0-2, Anthropic 0-1.
            body["temperature"] = 1.0
            ctx.drop("temperature above 1")
        if req.stop:
            body["stop_sequences"] = req.stop
        if req.seed is not None:
            ctx.drop("seed")
        if req.tools:
            body["tools"] = [
                {
                    k: v
                    for k, v in (
                        ("name", t.name),
                        ("description", t.description),
                        ("input_schema", t.parameters),
                    )
                    if v is not None
                }
                for t in req.tools
            ]
        tc = req.tool_choice
        if tc is not None or req.parallel_tool_calls is False:
            mode = tc.mode if tc else "auto"
            out: dict[str, Any] = {
                "type": {"required": "any"}.get(mode, mode),
            }
            if mode == "tool":
                out["name"] = tc.name if tc else None
            if req.parallel_tool_calls is False and mode != "none":
                out["disable_parallel_tool_use"] = True
            body["tool_choice"] = out
        if (
            req.response_format is not None
            and req.response_format.type != "text"
        ):
            ctx.drop("response_format", semantic=True)
        budget = req.reasoning_budget
        if budget is None and req.reasoning_effort in _EFFORT_BUDGET:
            budget = _EFFORT_BUDGET[req.reasoning_effort]
        if budget is not None:
            self._add_thinking(body, budget, req.max_output_tokens, ctx)
        if req.user:
            body["metadata"] = {"user_id": req.user}
        if req.stream:
            body["stream"] = True
        return self.generate_path(), body

    @staticmethod
    def _add_thinking(
        body: dict[str, Any],
        budget: int,
        client_max: int | None,
        ctx: Context,
    ) -> None:
        """Turn on thinking within Anthropic's limits.

        The budget must be at least 1024 and below ``max_tokens``. When
        the client set no output limit, the limit grows to leave the
        default room for the answer; when it did, the budget shrinks to
        fit, and thinking is left off if even 1024 does not fit.
        Thinking also rules out a custom temperature or top_k.
        """
        budget = max(budget, _MIN_THINKING_BUDGET)
        if client_max is None:
            body["max_tokens"] = max(
                body["max_tokens"], budget + ctx.default_max_tokens
            )
        elif budget >= client_max:
            budget = client_max - 1
        if budget < _MIN_THINKING_BUDGET:
            ctx.drop("reasoning")
            return
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}
        for key in ("temperature", "top_k"):
            if key in body and not (key == "temperature" and body[key] == 1):
                del body[key]
                ctx.drop(key)

    def decode_response(self, body: Mapping[str, Any]) -> c.Response:
        """Parse a Messages response."""
        output: list[c.Part] = []
        for b in body.get("content") or []:
            kind = b.get("type")
            if kind == "text":
                output.append(c.TextPart(str(b.get("text") or "")))
            elif kind == "thinking":
                output.append(
                    c.ReasoningPart(
                        str(b.get("thinking") or ""), b.get("signature")
                    )
                )
            elif kind == "tool_use":
                output.append(
                    c.ToolCallPart(
                        id=str(b.get("id") or new_id("toolu_")),
                        name=str(b.get("name") or ""),
                        arguments=dump_arguments(b.get("input") or {}),
                    )
                )
        return c.Response(
            id=str(body.get("id") or new_id("msg_")),
            model=body.get("model"),
            output=output,
            stop_reason=_STOP_IN.get(
                body.get("stop_reason"), c.StopReason.END_TURN
            ),
            stop_sequence=body.get("stop_sequence"),
            usage=_usage_in(body.get("usage")),
        )

    async def decode_stream(
        self, chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[Any]:
        """Turn Messages SSE events into canonical events."""
        usage = c.Usage()
        stop, seq = c.StopReason.END_TURN, None
        kinds: dict[int, str] = {}
        offset: int | None = None
        started = False
        async for _event, data in iter_sse(chunks):
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            kind = obj.get("type")
            if kind == "message_start":
                msg = obj.get("message") or {}
                usage = _merge_usage(usage, msg.get("usage"))
                started = True
                yield c.ResponseStart(
                    id=str(msg.get("id") or new_id("msg_")),
                    model=msg.get("model"),
                )
            elif kind == "content_block_start":
                b = obj.get("content_block") or {}
                raw = int(obj.get("index", 0))
                if offset is None:
                    offset = raw
                idx = raw - offset
                t = b.get("type")
                if t == "tool_use":
                    kinds[idx] = "tool_call"
                    yield c.PartStart(
                        idx, "tool_call", id=b.get("id"), name=b.get("name")
                    )
                elif t == "thinking":
                    kinds[idx] = "reasoning"
                    yield c.PartStart(idx, "reasoning")
                elif t == "text":
                    kinds[idx] = "text"
                    yield c.PartStart(idx, "text")
                    if b.get("text"):
                        yield c.TextDelta(idx, b["text"])
            elif kind == "content_block_delta":
                idx = int(obj.get("index", 0)) - (offset or 0)
                d = obj.get("delta") or {}
                dt = d.get("type")
                if idx not in kinds:
                    continue
                if dt == "text_delta":
                    yield c.TextDelta(idx, str(d.get("text") or ""))
                elif dt == "input_json_delta":
                    yield c.ToolArgsDelta(idx, str(d.get("partial_json") or ""))
                elif dt == "thinking_delta":
                    yield c.ReasoningDelta(idx, str(d.get("thinking") or ""))
                elif dt == "signature_delta":
                    yield c.ReasoningDelta(idx, "", d.get("signature"))
            elif kind == "content_block_stop":
                idx = int(obj.get("index", 0)) - (offset or 0)
                if idx in kinds:
                    yield c.PartStop(idx)
            elif kind == "message_delta":
                d = obj.get("delta") or {}
                if d.get("stop_reason"):
                    stop = _STOP_IN.get(d["stop_reason"], c.StopReason.END_TURN)
                    seq = d.get("stop_sequence")
                usage = _merge_usage(usage, obj.get("usage"))
            elif kind == "message_stop":
                break
            elif kind == "error":
                err = obj.get("error") or {}
                yield c.ErrorEvent(str(err.get("message") or "Engine error"))
                return
        if not started:
            yield c.ErrorEvent("Engine stream ended without a response")
            return
        yield c.ResponseStop(stop, seq, usage)


def _delta(index: int, delta: dict[str, Any]) -> bytes:
    return frame(
        {"type": "content_block_delta", "index": index, "delta": delta},
        "content_block_delta",
    )


def _block_out(p: c.Part) -> dict[str, Any] | None:
    if isinstance(p, c.TextPart):
        return {"type": "text", "text": p.text}
    if isinstance(p, c.ReasoningPart):
        return {
            "type": "thinking",
            "thinking": p.text,
            "signature": p.signature or "",
        }
    if isinstance(p, c.ToolCallPart):
        return {
            "type": "tool_use",
            "id": p.id,
            "name": p.name,
            "input": parse_arguments(p.arguments),
        }
    return None


def _image_block(p: c.ImagePart) -> dict[str, Any]:
    split = split_data_url(p.url)
    if split:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": split[0],
                "data": split[1],
            },
        }
    return {"type": "image", "source": {"type": "url", "url": p.url}}


def _block_in(p: c.Part, ctx: Context) -> dict[str, Any] | None:
    if isinstance(p, c.TextPart):
        return {"type": "text", "text": p.text} if p.text else None
    if isinstance(p, c.ImagePart):
        return _image_block(p)
    if isinstance(p, c.ToolCallPart):
        return {
            "type": "tool_use",
            "id": p.id,
            "name": p.name,
            "input": parse_arguments(p.arguments),
        }
    if isinstance(p, c.ToolResultPart):
        inner = [
            {"type": "text", "text": x.text}
            if isinstance(x, c.TextPart)
            else _image_block(x)
            for x in p.content
        ]
        out: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": p.call_id,
            "content": inner,
        }
        if p.is_error:
            out["is_error"] = True
        return out
    if isinstance(p, c.ReasoningPart):
        if p.signature:
            return {
                "type": "thinking",
                "thinking": p.text,
                "signature": p.signature,
            }
        # Without its signature an Anthropic engine refuses the block.
        ctx.drop("reasoning without signature")
        return None
    return None


def _usage_in(u: Any) -> c.Usage:
    return _merge_usage(c.Usage(), u)


def _merge_usage(base: c.Usage, u: Any) -> c.Usage:
    if not isinstance(u, Mapping):
        return base
    return c.Usage(
        input_tokens=u.get("input_tokens", base.input_tokens),
        output_tokens=u.get("output_tokens", base.output_tokens),
        cached_input_tokens=u.get(
            "cache_read_input_tokens", base.cached_input_tokens
        ),
        reasoning_tokens=base.reasoning_tokens,
    )


def _usage_out(u: c.Usage) -> dict[str, Any]:
    out: dict[str, Any] = {
        "input_tokens": u.input_tokens or 0,
        "output_tokens": u.output_tokens or 0,
    }
    if u.cached_input_tokens is not None:
        out["cache_read_input_tokens"] = u.cached_input_tokens
    return out
