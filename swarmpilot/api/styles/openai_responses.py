"""OpenAI Responses (``POST /v1/responses``), stateless subset.

SwarmPilot keeps no conversation state (plan item 5.2), so
``previous_response_id``, ``conversation`` and ``background`` are refused,
``store`` is ignored and reported as dropped, and hosted tools (web search,
file search, code interpreter, ...) are refused. Function tools work.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any, ClassVar

from swarmpilot.api import canonical as c
from swarmpilot.api.base import (
    ApiStyle,
    Context,
    StyleRoute,
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
    text_of,
)

_KNOWN = frozenset(
    {
        "model",
        "input",
        "instructions",
        "max_output_tokens",
        "temperature",
        "top_p",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "text",
        "reasoning",
        "stream",
        "user",
        "store",
        "metadata",
    }
)
_STATEFUL = ("previous_response_id", "conversation", "background")


def _content(content: Any, where: str) -> list[c.Part]:
    if isinstance(content, str):
        return [c.TextPart(content)]
    parts: list[c.Part] = []
    for i, p in enumerate(as_list(content, where)):
        w = f"{where}[{i}]"
        p = as_mapping(p, w)
        kind = p.get("type")
        if kind in ("input_text", "output_text", "text"):
            parts.append(c.TextPart(as_str(p.get("text"), f"{w}.text")))
        elif kind == "refusal":
            parts.append(c.TextPart(str(p.get("refusal") or "")))
        elif kind == "input_image":
            url = p.get("image_url")
            if not isinstance(url, str):
                raise unsupported(f"{w}.file_id", "only image_url images")
            parts.append(c.ImagePart(url, p.get("detail")))
        else:
            raise unsupported(f"{w}.type", f"content type {kind!r}")
    return parts


def _last_assistant(req: c.Request) -> c.Message:
    if not req.messages or req.messages[-1].role != "assistant":
        req.messages.append(c.Message("assistant", []))
    return req.messages[-1]


class OpenAIResponsesStyle(ApiStyle):
    """OpenAI Responses without server-side state."""

    name: ClassVar[str] = "openai_responses"
    semantic_extensions: ClassVar[frozenset[str]] = frozenset(
        {"top_logprobs", "include"}
    )

    def routes(self) -> tuple[StyleRoute, ...]:
        """Return the responses route."""
        return (StyleRoute("v1/responses"),)

    def feature_extractors(
        self, default_max_tokens: int
    ) -> tuple[tuple[str, str], ...]:
        """Instruction and input text characters and the output limit."""
        return (
            (
                "prompt_length",
                "sum(chars(instructions), chars(input), "
                "chars(input[*].content), chars(input[*].content[*].text), "
                "chars(input[*].output))",
            ),
            (
                "max_tokens",
                f"coalesce(max_output_tokens, {default_max_tokens})",
            ),
        )

    # Client side.

    def decode_request(
        self, body: Mapping[str, Any], ctx: Context
    ) -> c.Request:
        """Parse a Responses request."""
        for key in _STATEFUL:
            if body.get(key):
                raise unsupported(key, "SwarmPilot keeps no response state")
        req = c.Request(model=body.get("model"))
        if body.get("instructions") is not None:
            req.system.append(
                c.TextPart(as_str(body["instructions"], "instructions"))
            )
        items = body.get("input")
        if isinstance(items, str):
            req.messages.append(c.Message("user", [c.TextPart(items)]))
        else:
            for i, it in enumerate(as_list(items, "input")):
                self._decode_item(
                    req, as_mapping(it, f"input[{i}]"), f"input[{i}]"
                )
        req.max_output_tokens = opt_int(body, "max_output_tokens")
        req.temperature = opt_float(body, "temperature")
        req.top_p = opt_float(body, "top_p")
        for j, t in enumerate(body.get("tools") or []):
            t = as_mapping(t, f"tools[{j}]")
            if t.get("type") != "function":
                raise unsupported(
                    f"tools[{j}].type", f"hosted tool {t.get('type')!r}"
                )
            req.tools.append(
                c.Tool(
                    name=as_str(t.get("name"), f"tools[{j}].name"),
                    description=t.get("description"),
                    parameters=t.get("parameters")
                    or {"type": "object", "properties": {}},
                    strict=t.get("strict"),
                )
            )
        tc = body.get("tool_choice")
        if tc in ("auto", "none", "required"):
            req.tool_choice = c.ToolChoice(tc)
        elif isinstance(tc, Mapping) and tc.get("type") == "function":
            req.tool_choice = c.ToolChoice(
                "tool", as_str(tc.get("name"), "tool_choice.name")
            )
        elif tc is not None:
            raise unsupported("tool_choice", f"{tc!r}")
        req.parallel_tool_calls = body.get("parallel_tool_calls")
        text = body.get("text")
        fmt = text.get("format") if isinstance(text, Mapping) else None
        if isinstance(fmt, Mapping) and fmt.get("type") in (
            "text",
            "json_object",
            "json_schema",
        ):
            req.response_format = c.ResponseFormat(
                fmt["type"],
                name=fmt.get("name"),
                schema=fmt.get("schema"),
                strict=fmt.get("strict"),
            )
        reasoning = body.get("reasoning")
        if isinstance(reasoning, Mapping) and reasoning.get("effort"):
            req.reasoning_effort = str(reasoning["effort"])
        req.stream = body.get("stream") is True
        req.user = body.get("user")
        if body.get("store"):
            ctx.drop("store")
        ext = {k: v for k, v in body.items() if k not in _KNOWN}
        if ext:
            req.extensions[self.name] = ext
        return req

    def _decode_item(
        self, req: c.Request, it: Mapping[str, Any], w: str
    ) -> None:
        kind = it.get("type", "message")
        if kind == "message":
            role = it.get("role")
            if role in ("system", "developer"):
                req.system.append(c.TextPart(text_of(it.get("content"))))
            elif role in ("user", "assistant"):
                parts = _content(it.get("content"), f"{w}.content")
                if role == "assistant":
                    _last_assistant(req).content.extend(parts)
                else:
                    req.messages.append(c.Message("user", parts))
            else:
                raise unsupported(f"{w}.role", f"role {role!r}")
        elif kind == "function_call":
            _last_assistant(req).content.append(
                c.ToolCallPart(
                    id=as_str(it.get("call_id"), f"{w}.call_id"),
                    name=as_str(it.get("name"), f"{w}.name"),
                    arguments=str(it.get("arguments") or "{}"),
                )
            )
        elif kind == "function_call_output":
            out = it.get("output")
            parts = (
                [c.TextPart(out)]
                if isinstance(out, str)
                else [
                    p
                    for p in _content(out or [], f"{w}.output")
                    if isinstance(p, c.TextPart | c.ImagePart)
                ]
            )
            req.messages.append(
                c.Message(
                    "tool",
                    [
                        c.ToolResultPart(
                            as_str(it.get("call_id"), f"{w}.call_id"), parts
                        )
                    ],
                )
            )
        elif kind == "reasoning":
            text = "".join(
                s.get("text", "")
                for s in it.get("summary") or []
                if isinstance(s, Mapping)
            )
            _last_assistant(req).content.append(c.ReasoningPart(text))
        else:
            raise unsupported(f"{w}.type", f"input item {kind!r}")

    def encode_response(
        self, resp: c.Response, req: c.Request
    ) -> dict[str, Any]:
        """Write a Responses object."""
        items = [_item(p) for p in resp.output]
        merged: list[dict[str, Any]] = []
        for item in items:
            if (
                item["type"] == "message"
                and merged
                and merged[-1]["type"] == "message"
            ):
                merged[-1]["content"].extend(item["content"])
            else:
                merged.append(item)
        return _response_obj(
            resp.id, resp.model or req.model, resp.created, merged, req
        ) | _status(resp.stop_reason, resp.usage)

    async def encode_stream(
        self, events: AsyncIterator[Any], req: c.Request
    ) -> AsyncIterator[bytes]:
        """Write the ``response.*`` SSE event sequence."""
        seq = 0
        rid, model, created = new_id("resp_"), req.model, now()
        output: list[dict[str, Any]] = []
        items: dict[int, dict[str, Any]] = {}
        texts: dict[int, list[str]] = {}

        def ev(kind: str, **data: Any) -> bytes:
            nonlocal seq
            seq += 1
            return frame(
                {"type": kind, "sequence_number": seq - 1, **data}, kind
            )

        async for e in events:
            if isinstance(e, c.ResponseStart):
                model = e.model or model
                base = _response_obj(rid, model, created, [], req) | {
                    "status": "in_progress",
                    "usage": None,
                }
                yield ev("response.created", response=base)
                yield ev("response.in_progress", response=base)
            elif isinstance(e, c.PartStart):
                oi = len(output)
                if e.kind == "text":
                    item = {
                        "type": "message",
                        "id": new_id("msg_"),
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    }
                elif e.kind == "tool_call":
                    item = {
                        "type": "function_call",
                        "id": new_id("fc_"),
                        "call_id": e.id,
                        "name": e.name,
                        "arguments": "",
                        "status": "in_progress",
                    }
                else:
                    item = {
                        "type": "reasoning",
                        "id": new_id("rs_"),
                        "summary": [],
                    }
                item["_oi"] = oi
                items[e.index] = item
                texts[e.index] = []
                output.append(item)
                yield ev(
                    "response.output_item.added",
                    output_index=oi,
                    item=_pub(item),
                )
                if e.kind == "text":
                    yield ev(
                        "response.content_part.added",
                        item_id=item["id"],
                        output_index=oi,
                        content_index=0,
                        part={
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                        },
                    )
                elif e.kind == "reasoning":
                    yield ev(
                        "response.reasoning_summary_part.added",
                        item_id=item["id"],
                        output_index=oi,
                        summary_index=0,
                        part={"type": "summary_text", "text": ""},
                    )
            elif isinstance(e, c.TextDelta) and e.index in items:
                item = items[e.index]
                texts[e.index].append(e.text)
                yield ev(
                    "response.output_text.delta",
                    item_id=item["id"],
                    output_index=item["_oi"],
                    content_index=0,
                    delta=e.text,
                    logprobs=[],
                )
            elif isinstance(e, c.ToolArgsDelta) and e.index in items:
                item = items[e.index]
                texts[e.index].append(e.delta)
                yield ev(
                    "response.function_call_arguments.delta",
                    item_id=item["id"],
                    output_index=item["_oi"],
                    delta=e.delta,
                )
            elif isinstance(e, c.ReasoningDelta) and e.index in items:
                if e.text:
                    item = items[e.index]
                    texts[e.index].append(e.text)
                    yield ev(
                        "response.reasoning_summary_text.delta",
                        item_id=item["id"],
                        output_index=item["_oi"],
                        summary_index=0,
                        delta=e.text,
                    )
            elif isinstance(e, c.PartStop) and e.index in items:
                item = items[e.index]
                oi, full = item["_oi"], "".join(texts[e.index])
                if item["type"] == "message":
                    part = {
                        "type": "output_text",
                        "text": full,
                        "annotations": [],
                    }
                    yield ev(
                        "response.output_text.done",
                        item_id=item["id"],
                        output_index=oi,
                        content_index=0,
                        text=full,
                        logprobs=[],
                    )
                    yield ev(
                        "response.content_part.done",
                        item_id=item["id"],
                        output_index=oi,
                        content_index=0,
                        part=part,
                    )
                    item["content"] = [part]
                    item["status"] = "completed"
                elif item["type"] == "function_call":
                    yield ev(
                        "response.function_call_arguments.done",
                        item_id=item["id"],
                        output_index=oi,
                        arguments=full,
                    )
                    item["arguments"] = full
                    item["status"] = "completed"
                else:
                    part = {"type": "summary_text", "text": full}
                    yield ev(
                        "response.reasoning_summary_text.done",
                        item_id=item["id"],
                        output_index=oi,
                        summary_index=0,
                        text=full,
                    )
                    yield ev(
                        "response.reasoning_summary_part.done",
                        item_id=item["id"],
                        output_index=oi,
                        summary_index=0,
                        part=part,
                    )
                    item["summary"] = [part]
                yield ev(
                    "response.output_item.done",
                    output_index=oi,
                    item=_pub(item),
                )
            elif isinstance(e, c.ResponseStop):
                final = _response_obj(
                    rid, model, created, [_pub(i) for i in output], req
                ) | _status(e.stop_reason, e.usage)
                kind = (
                    "response.completed"
                    if final["status"] == "completed"
                    else "response.incomplete"
                )
                yield ev(kind, response=final)
                return
            elif isinstance(e, c.ErrorEvent):
                yield ev(
                    "error", code=e.error_type, message=e.message, param=None
                )
                return

    def encode_error(
        self,
        status: int,
        message: str,
        error_type: str | None = None,
        param: str | None = None,
    ) -> dict[str, Any]:
        """Return an OpenAI error object."""
        from swarmpilot.api.styles.openai_chat import OpenAIChatStyle

        return OpenAIChatStyle().encode_error(
            status, message, error_type, param
        )

    # Engine side.

    def encode_request(
        self, req: c.Request, ctx: Context
    ) -> tuple[str, dict[str, Any]]:
        """Build a stateless Responses body."""
        items: list[dict[str, Any]] = []
        for m in req.messages:
            if m.role == "tool":
                for p in m.content:
                    if isinstance(p, c.ToolResultPart):
                        if any(isinstance(x, c.ImagePart) for x in p.content):
                            ctx.drop("image in tool result", semantic=True)
                        items.append(
                            {
                                "type": "function_call_output",
                                "call_id": p.call_id,
                                "output": "".join(
                                    x.text
                                    for x in p.content
                                    if isinstance(x, c.TextPart)
                                ),
                            }
                        )
                continue
            content: list[dict[str, Any]] = []
            for p in m.content:
                if isinstance(p, c.TextPart):
                    kind = (
                        "output_text" if m.role == "assistant" else "input_text"
                    )
                    content.append({"type": kind, "text": p.text})
                elif isinstance(p, c.ImagePart):
                    img: dict[str, Any] = {
                        "type": "input_image",
                        "image_url": p.url,
                    }
                    if p.detail:
                        img["detail"] = p.detail
                    content.append(img)
                elif isinstance(p, c.ToolCallPart):
                    if content:
                        items.append({"role": m.role, "content": content})
                        content = []
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": p.id,
                            "name": p.name,
                            "arguments": p.arguments,
                        }
                    )
            if content:
                items.append({"role": m.role, "content": content})
        body: dict[str, Any] = {"input": items, "store": False}
        if req.model is not None:
            body["model"] = req.model
        if req.system:
            body["instructions"] = req.system_text()
        if req.max_output_tokens is not None:
            body["max_output_tokens"] = req.max_output_tokens
        for key in ("temperature", "top_p", "user", "parallel_tool_calls"):
            v = getattr(req, key)
            if v is not None:
                body[key] = v
        if req.top_k is not None:
            ctx.drop("top_k")
        if req.seed is not None:
            ctx.drop("seed")
        if req.stop:
            ctx.drop("stop", semantic=True)
        if req.tools:
            body["tools"] = [
                {
                    k: v
                    for k, v in (
                        ("type", "function"),
                        ("name", t.name),
                        ("description", t.description),
                        ("parameters", t.parameters),
                        ("strict", t.strict),
                    )
                    if v is not None
                }
                for t in req.tools
            ]
        tc = req.tool_choice
        if tc is not None:
            body["tool_choice"] = (
                {"type": "function", "name": tc.name}
                if tc.mode == "tool"
                else tc.mode
            )
        rf = req.response_format
        if rf is not None and rf.type != "text":
            fmt: dict[str, Any] = {"type": rf.type}
            if rf.type == "json_schema":
                fmt.update(name=rf.name or "response", schema=rf.schema or {})
                if rf.strict is not None:
                    fmt["strict"] = rf.strict
            body["text"] = {"format": fmt}
        if req.reasoning_effort is not None:
            body["reasoning"] = {"effort": req.reasoning_effort}
        elif req.reasoning_budget is not None:
            ctx.drop("reasoning budget")
        if req.stream:
            body["stream"] = True
        return self.generate_path(), body

    def decode_response(self, body: Mapping[str, Any]) -> c.Response:
        """Parse a Responses object."""
        output: list[c.Part] = []
        refused = False
        for it in body.get("output") or []:
            kind = it.get("type")
            if kind == "message":
                for p in it.get("content") or []:
                    if p.get("type") == "output_text":
                        output.append(c.TextPart(str(p.get("text") or "")))
                    elif p.get("type") == "refusal":
                        refused = True
                        output.append(c.TextPart(str(p.get("refusal") or "")))
            elif kind == "function_call":
                output.append(
                    c.ToolCallPart(
                        id=str(it.get("call_id") or new_id("call_")),
                        name=str(it.get("name") or ""),
                        arguments=str(it.get("arguments") or "{}"),
                    )
                )
            elif kind == "reasoning":
                text = "".join(
                    s.get("text", "") for s in it.get("summary") or []
                ) or "".join(s.get("text", "") for s in it.get("content") or [])
                output.append(c.ReasoningPart(text))
        return c.Response(
            id=str(body.get("id") or new_id("resp_")),
            model=body.get("model"),
            output=output,
            stop_reason=_stop_in(body, output, refused),
            usage=_usage_in(body.get("usage")),
            created=int(body.get("created_at") or 0),
        )

    async def decode_stream(
        self, chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[Any]:
        """Turn ``response.*`` events into canonical events."""
        index = -1
        open_part: str | None = None
        started = False
        final: Mapping[str, Any] | None = None
        calls = False
        async for _event, data in iter_sse(chunks):
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            kind = obj.get("type", "")
            if kind == "response.created":
                r = obj.get("response") or {}
                started = True
                yield c.ResponseStart(
                    id=str(r.get("id") or new_id("resp_")),
                    model=r.get("model"),
                    created=int(r.get("created_at") or 0),
                )
            elif kind == "response.output_item.added":
                item = obj.get("item") or {}
                t = item.get("type")
                if t == "function_call":
                    index += 1
                    open_part = "tool_call"
                    calls = True
                    yield c.PartStart(
                        index,
                        "tool_call",
                        id=item.get("call_id"),
                        name=item.get("name"),
                    )
                elif t == "reasoning":
                    index += 1
                    open_part = "reasoning"
                    yield c.PartStart(index, "reasoning")
            elif kind == "response.content_part.added":
                index += 1
                open_part = "text"
                yield c.PartStart(index, "text")
            elif kind == "response.output_text.delta" and open_part == "text":
                yield c.TextDelta(index, str(obj.get("delta") or ""))
            elif kind == "response.function_call_arguments.delta":
                yield c.ToolArgsDelta(index, str(obj.get("delta") or ""))
            elif kind in (
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            ):
                yield c.ReasoningDelta(index, str(obj.get("delta") or ""))
            elif (
                kind == "response.content_part.done" and open_part == "text"
            ) or (
                kind == "response.output_item.done"
                and open_part
                in (
                    "tool_call",
                    "reasoning",
                )
            ):
                open_part = None
                yield c.PartStop(index)
            elif kind in ("response.completed", "response.incomplete"):
                final = obj.get("response") or {}
                break
            elif kind in ("response.failed", "error"):
                r = obj.get("response") or {}
                err = r.get("error") or obj
                yield c.ErrorEvent(str(err.get("message") or "Engine error"))
                return
        if not started:
            yield c.ErrorEvent("Engine stream ended without a response")
            return
        if open_part is not None:
            yield c.PartStop(index)
        final = final or {}
        stop = _stop_in(final, [], False)
        if calls and stop == c.StopReason.END_TURN:
            stop = c.StopReason.TOOL_USE
        yield c.ResponseStop(stop, None, _usage_in(final.get("usage")))


def _item(p: c.Part) -> dict[str, Any]:
    if isinstance(p, c.ToolCallPart):
        return {
            "type": "function_call",
            "id": new_id("fc_"),
            "call_id": p.id,
            "name": p.name,
            "arguments": p.arguments,
            "status": "completed",
        }
    if isinstance(p, c.ReasoningPart):
        return {
            "type": "reasoning",
            "id": new_id("rs_"),
            "summary": [{"type": "summary_text", "text": p.text}],
        }
    text = p.text if isinstance(p, c.TextPart) else ""
    return {
        "type": "message",
        "id": new_id("msg_"),
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _pub(item: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if not k.startswith("_")}


def _response_obj(
    rid: str,
    model: str | None,
    created: int,
    output: list[dict[str, Any]],
    req: c.Request,
) -> dict[str, Any]:
    return {
        "id": rid if rid.startswith("resp_") else f"resp_{rid}",
        "object": "response",
        "created_at": created or now(),
        "model": model or "",
        "output": output,
        "error": None,
        "incomplete_details": None,
        "instructions": req.system_text() or None,
        "max_output_tokens": req.max_output_tokens,
        "parallel_tool_calls": req.parallel_tool_calls is not False,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "tool_choice": "auto",
        "tools": [],
        "text": {"format": {"type": "text"}},
        "store": False,
        "metadata": {},
    }


def _status(stop: c.StopReason, usage: c.Usage) -> dict[str, Any]:
    out: dict[str, Any] = {"status": "completed", "usage": _usage_out(usage)}
    if stop == c.StopReason.MAX_TOKENS:
        out["status"] = "incomplete"
        out["incomplete_details"] = {"reason": "max_output_tokens"}
    elif stop == c.StopReason.CONTENT_FILTER:
        out["status"] = "incomplete"
        out["incomplete_details"] = {"reason": "content_filter"}
    return out


def _stop_in(
    body: Mapping[str, Any], output: list[c.Part], refused: bool
) -> c.StopReason:
    details = body.get("incomplete_details") or {}
    if body.get("status") == "incomplete":
        if details.get("reason") == "max_output_tokens":
            return c.StopReason.MAX_TOKENS
        return c.StopReason.CONTENT_FILTER
    if refused:
        return c.StopReason.CONTENT_FILTER
    if any(isinstance(p, c.ToolCallPart) for p in output):
        return c.StopReason.TOOL_USE
    return c.StopReason.END_TURN


def _usage_in(u: Any) -> c.Usage:
    if not isinstance(u, Mapping):
        return c.Usage()
    idt = u.get("input_tokens_details") or {}
    odt = u.get("output_tokens_details") or {}
    return c.Usage(
        input_tokens=u.get("input_tokens"),
        output_tokens=u.get("output_tokens"),
        cached_input_tokens=idt.get("cached_tokens") if idt else None,
        reasoning_tokens=odt.get("reasoning_tokens") if odt else None,
    )


def _usage_out(u: c.Usage) -> dict[str, Any]:
    i, o = u.input_tokens or 0, u.output_tokens or 0
    return {
        "input_tokens": i,
        "input_tokens_details": {"cached_tokens": u.cached_input_tokens or 0},
        "output_tokens": o,
        "output_tokens_details": {"reasoning_tokens": u.reasoning_tokens or 0},
        "total_tokens": i + o,
    }
