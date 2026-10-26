"""The canonical interaction every API style translates through.

A client may call SwarmPilot in any API style (OpenAI Chat Completions,
OpenAI Responses, Anthropic Messages, ...) and the engine behind it may
speak a different one. Each style converts its own wire format to and from
the types here, so N styles need N converters rather than N x M.

The types cover what the LLM styles share: a system prompt, a list of
messages made of typed parts, the common generation parameters, tools, and
the answer with its stop reason and token usage. A streamed answer is a
sequence of :data:`StreamEvent` values.

Fields a style has but the canonical model does not go to
``Request.extensions[<style name>]``. They reach the engine only when the
engine speaks the client's style (and then the body is passed through
untouched anyway); otherwise the translator drops them or refuses the
request, see :mod:`swarmpilot.api.translate`.

The JSON Schema of these types is checked in as ``canonical.schema.json``
so that the Rust gateway can use the same model::

    python -m swarmpilot.api.canonical > swarmpilot/api/canonical.schema.json
"""

from __future__ import annotations

import enum
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter

SCHEMA_PATH = Path(__file__).with_name("canonical.schema.json")


class StopReason(enum.StrEnum):
    """Why the engine stopped generating."""

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


@dataclass(slots=True)
class TextPart:
    """Plain text."""

    text: str
    type: Literal["text"] = "text"


@dataclass(slots=True)
class ImagePart:
    """An image, as an ``http(s)`` URL or a ``data:`` URL."""

    url: str
    detail: str | None = None
    type: Literal["image"] = "image"


@dataclass(slots=True)
class ToolCallPart:
    """A call the assistant made to a tool.

    Attributes:
        id: Call id; the matching :class:`ToolResultPart` repeats it.
        name: Tool name.
        arguments: Arguments as JSON text, exactly as generated.
    """

    id: str
    name: str
    arguments: str = "{}"
    type: Literal["tool_call"] = "tool_call"


@dataclass(slots=True)
class ToolResultPart:
    """The result of a tool call, sent back by the client."""

    call_id: str
    content: list[TextPart | ImagePart] = field(default_factory=list)
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


@dataclass(slots=True)
class ReasoningPart:
    """Model reasoning ("thinking").

    Attributes:
        text: The reasoning text or its summary.
        signature: Opaque provider signature. Only an engine of the style
            that produced it can check it, so other styles drop it.
    """

    text: str
    signature: str | None = None
    type: Literal["reasoning"] = "reasoning"


Part = Annotated[
    TextPart | ImagePart | ToolCallPart | ToolResultPart | ReasoningPart,
    Field(discriminator="type"),
]


@dataclass(slots=True)
class Message:
    """One turn of the conversation.

    ``tool`` messages hold :class:`ToolResultPart` values; ``assistant``
    messages may hold text, reasoning and tool calls; ``user`` messages
    hold text and images.
    """

    role: Literal["user", "assistant", "tool"]
    content: list[Part] = field(default_factory=list)


@dataclass(slots=True)
class Tool:
    """A function the model may call."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    strict: bool | None = None


@dataclass(slots=True)
class ToolChoice:
    """Whether and which tool the model must call."""

    mode: Literal["auto", "none", "required", "tool"] = "auto"
    name: str | None = None


@dataclass(slots=True)
class ResponseFormat:
    """Constrained output: free text, any JSON, or JSON matching a schema."""

    type: Literal["text", "json_object", "json_schema"] = "text"
    name: str | None = None
    schema: dict[str, Any] | None = None
    strict: bool | None = None


@dataclass(slots=True)
class Request:
    """A request in canonical form.

    Attributes:
        model: Model name the client asked for.
        system: System prompt parts, in order.
        messages: The conversation.
        max_output_tokens: Output token limit, if the client set one.
        temperature: Sampling temperature.
        top_p: Nucleus sampling mass.
        top_k: Top-k sampling.
        stop: Stop sequences.
        seed: Sampling seed.
        tools: Tools the model may call.
        tool_choice: Tool-use policy.
        parallel_tool_calls: Whether several calls may come in one turn.
        response_format: Output constraint.
        reasoning_effort: ``low``, ``medium`` or ``high``, if asked.
        reasoning_budget: Reasoning token budget, if asked.
        stream: Whether the client wants a streamed answer.
        user: End-user id for abuse tracking.
        extensions: Style name to the fields of that style with no
            canonical home.
    """

    messages: list[Message] = field(default_factory=list)
    model: str | None = None
    system: list[TextPart] = field(default_factory=list)
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    tools: list[Tool] = field(default_factory=list)
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool | None = None
    response_format: ResponseFormat | None = None
    reasoning_effort: str | None = None
    reasoning_budget: int | None = None
    stream: bool = False
    user: str | None = None
    extensions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def system_text(self) -> str:
        """Return the system prompt as one string."""
        return "\n\n".join(p.text for p in self.system)


def prompt_text(req: Request) -> str:
    """Return the request's prompt text, whatever style it came in.

    The system prompt, then every message's text (tool results included),
    joined by blank lines. Semantic encoders and other text features read
    this instead of a style's own message shapes.

    Args:
        req: A canonical request.

    Returns:
        The concatenated text.
    """
    chunks = [req.system_text()] if req.system else []
    for m in req.messages:
        for p in m.content:
            if isinstance(p, TextPart):
                chunks.append(p.text)
            elif isinstance(p, ToolResultPart):
                chunks.extend(
                    x.text for x in p.content if isinstance(x, TextPart)
                )
    return "\n\n".join(chunks)


@dataclass(slots=True)
class Usage:
    """Token counts; unknown counts stay None."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(slots=True)
class Response:
    """A complete (non-streamed) answer."""

    id: str
    model: str | None = None
    output: list[Part] = field(default_factory=list)
    stop_reason: StopReason = StopReason.END_TURN
    stop_sequence: str | None = None
    usage: Usage = field(default_factory=Usage)
    created: int = 0


# Stream events. Parts are numbered from 0 in output order, and every part
# is opened with PartStart and closed with PartStop before the next opens.


@dataclass(slots=True)
class ResponseStart:
    """First event of a stream."""

    id: str
    model: str | None = None
    created: int = 0
    type: Literal["response_start"] = "response_start"


@dataclass(slots=True)
class PartStart:
    """A new output part begins.

    Attributes:
        index: Part number, from 0.
        kind: ``text``, ``tool_call`` or ``reasoning``.
        id: Tool call id (tool calls only).
        name: Tool name (tool calls only).
    """

    index: int
    kind: Literal["text", "tool_call", "reasoning"]
    id: str | None = None
    name: str | None = None
    type: Literal["part_start"] = "part_start"


@dataclass(slots=True)
class TextDelta:
    """More text for an open text part."""

    index: int
    text: str
    type: Literal["text_delta"] = "text_delta"


@dataclass(slots=True)
class ToolArgsDelta:
    """More JSON argument text for an open tool call."""

    index: int
    delta: str
    type: Literal["tool_args_delta"] = "tool_args_delta"


@dataclass(slots=True)
class ReasoningDelta:
    """More reasoning text, or the reasoning signature, for an open part."""

    index: int
    text: str = ""
    signature: str | None = None
    type: Literal["reasoning_delta"] = "reasoning_delta"


@dataclass(slots=True)
class PartStop:
    """The open part is complete."""

    index: int
    type: Literal["part_stop"] = "part_stop"


@dataclass(slots=True)
class ResponseStop:
    """Last event of a successful stream."""

    stop_reason: StopReason = StopReason.END_TURN
    stop_sequence: str | None = None
    usage: Usage = field(default_factory=Usage)
    type: Literal["response_stop"] = "response_stop"


@dataclass(slots=True)
class ErrorEvent:
    """The stream failed; nothing follows."""

    message: str
    error_type: str = "api_error"
    type: Literal["error"] = "error"


StreamEvent = Annotated[
    ResponseStart
    | PartStart
    | TextDelta
    | ToolArgsDelta
    | ReasoningDelta
    | PartStop
    | ResponseStop
    | ErrorEvent,
    Field(discriminator="type"),
]


def to_dict(value: Any) -> dict[str, Any]:
    """Return a canonical value as plain JSON-ready data."""
    return asdict(value)


_ADAPTERS: dict[str, TypeAdapter[Any]] = {}


def _adapter(name: str) -> TypeAdapter[Any]:
    if name not in _ADAPTERS:
        _ADAPTERS[name] = TypeAdapter(
            {"request": Request, "response": Response, "event": StreamEvent}[
                name
            ]
        )
    return _ADAPTERS[name]


def request_from_dict(data: dict[str, Any]) -> Request:
    """Build a :class:`Request` from :func:`to_dict` output."""
    return _adapter("request").validate_python(data)


def response_from_dict(data: dict[str, Any]) -> Response:
    """Build a :class:`Response` from :func:`to_dict` output."""
    return _adapter("response").validate_python(data)


def event_from_dict(data: dict[str, Any]) -> Any:
    """Build a stream event from :func:`to_dict` output."""
    return _adapter("event").validate_python(data)


def json_schema() -> dict[str, Any]:
    """Return the JSON Schema of the canonical request, response and events."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SwarmPilot canonical interaction",
        "type": "object",
        "properties": {
            "request": _adapter("request").json_schema(),
            "response": _adapter("response").json_schema(),
            "event": _adapter("event").json_schema(),
        },
    }


if __name__ == "__main__":  # pragma: no cover
    json.dump(json_schema(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
