"""Translate one request between a client's API style and an engine's.

:func:`plan_request` looks at the path a client called and the styles the
engine speaks natively (``EngineAdapter.native_styles()``):

* the path belongs to no style: not ours to translate (None);
* the engine speaks the client's style: pass the body through untouched,
  at no cost and with nothing lost;
* otherwise: decode with the client's style, encode with the engine's,
  and convert the answer back with :class:`Translation`.

Fields that cannot be carried across are either dropped and listed in the
``x-swarmpilot-dropped`` response header, or, when leaving them out would
change the answer and the deployment is strict (the default), refused with
a 400 that names the field.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger

from swarmpilot.api import canonical as c
from swarmpilot.api.base import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    ApiStyle,
    Context,
    TranslationError,
)
from swarmpilot.api.registry import find_style, style_for_path

#: Set to ``lenient`` to drop answer-changing fields instead of refusing.
POLICY_ENV = "SWARMPILOT_API_TRANSLATION"
DROPPED_HEADER = "x-swarmpilot-dropped"
# Preferred engine style when an engine speaks several.
_ENGINE_PREFERENCE = ("openai_chat", "anthropic_messages", "openai_responses")


def strict_default() -> bool:
    """Return whether translation is strict unless told otherwise."""
    return os.getenv(POLICY_ENV, "strict").strip().lower() != "lenient"


@dataclass
class Translation:
    """How one request travels to the engine and its answer back.

    Attributes:
        client: Style the client speaks.
        engine: Style the engine speaks; the same object as ``client``
            when the body passes through.
        request: The canonical request (None when passing through).
        path: Path to send to the engine.
        body: Body to send to the engine.
        ctx: Translation state, including dropped fields.
    """

    client: ApiStyle
    engine: ApiStyle
    request: c.Request | None
    path: str
    body: dict[str, Any]
    ctx: Context

    @property
    def passthrough(self) -> bool:
        """Whether the engine gets the client's body unchanged."""
        return self.request is None

    @property
    def stream(self) -> bool:
        """Whether the client asked for a stream."""
        if self.request is None:
            return self.client.is_stream(self.body)
        return self.request.stream

    def headers(self) -> dict[str, str]:
        """Return extra response headers (the dropped-field list)."""
        if not self.ctx.dropped:
            return {}
        return {DROPPED_HEADER: ",".join(self.ctx.dropped)}

    def response(self, status: int, body: Any) -> tuple[int, Any]:
        """Convert an engine's non-streamed answer to the client's style."""
        if self.request is None:
            return status, body
        if status >= 400:
            return status, self.client.encode_error(
                status, self.engine.decode_error(body)
            )
        if not isinstance(body, Mapping):
            return 502, self.client.encode_error(
                502, "Engine returned a malformed response", "bad_gateway"
            )
        resp = self.engine.decode_response(body)
        return status, self.client.encode_response(resp, self.request)

    def stream_response(
        self, chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[bytes]:
        """Convert an engine's SSE stream to the client's style."""
        if self.request is None:
            return chunks
        return self.client.encode_stream(
            _guard(self.engine.decode_stream(chunks), chunks), self.request
        )

    def stream_from_body(self, body: Mapping[str, Any]) -> AsyncIterator[bytes]:
        """Stream a complete answer to a client that asked for a stream.

        Used when the engine ignored ``stream`` and answered with one body.
        """
        assert self.request is not None
        resp = self.engine.decode_response(body)
        return self.client.encode_stream(
            events_from_response(resp), self.request
        )

    def error(
        self, status: int, message: str, error_type: str | None = None
    ) -> dict[str, Any]:
        """Return SwarmPilot's own error in the client's style."""
        return self.client.encode_error(status, message, error_type)


def plan_request(
    path: str,
    body: Mapping[str, Any],
    native_styles: Sequence[str],
    *,
    strict: bool | None = None,
    default_max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> Translation | None:
    """Decide how a request reaches the engine.

    Args:
        path: Path the client called, without a leading slash.
        body: Parsed JSON body.
        native_styles: Styles the engine serves natively.
        strict: Refuse answer-changing drops; defaults to
            :func:`strict_default`.
        default_max_tokens: Output limit for engine styles that need one.

    Returns:
        The translation, or None when ``path`` belongs to no style or the
        engine declares none.

    Raises:
        TranslationError: The request cannot be translated, or the engine
            speaks no style of the client's family.
    """
    client = style_for_path(path)
    if client is None or not native_styles:
        # An engine that declares no style (the passthrough adapter, a
        # generic HTTP server) gets whatever the client sent.
        return None
    ctx = Context(
        strict=strict_default() if strict is None else strict,
        default_max_tokens=default_max_tokens,
    )
    if client.name in native_styles:
        return Translation(client, client, None, path, dict(body), ctx)
    engine = _engine_style(client, native_styles)
    if engine is None:
        raise TranslationError(
            f"This model's engine cannot serve /{path}",
            status=404,
            error_type="not_found_error",
        )
    req = client.decode_request(body, ctx)
    for key in req.extensions.get(client.name, {}):
        ctx.drop(key, semantic=key in client.semantic_extensions)
    epath, ebody = engine.encode_request(req, ctx)
    if ctx.dropped:
        logger.debug(f"[API] /{path} -> /{epath} dropped {ctx.dropped}")
    return Translation(client, engine, req, epath, ebody, ctx)


def _engine_style(
    client: ApiStyle, native_styles: Sequence[str]
) -> ApiStyle | None:
    ordered = sorted(
        native_styles,
        key=lambda n: (
            _ENGINE_PREFERENCE.index(n)
            if n in _ENGINE_PREFERENCE
            else len(_ENGINE_PREFERENCE)
        ),
    )
    for name in ordered:
        style = find_style(name)
        if style is not None and style.family == client.family:
            return style
    return None


async def _guard(
    events: AsyncIterator[Any], source: AsyncIterator[bytes]
) -> AsyncIterator[Any]:
    """Turn a failing engine stream into an error event; always close it."""
    try:
        async for ev in events:
            yield ev
    except Exception as e:  # engine connection dropped, bad framing, ...
        logger.warning(f"[API] Engine stream failed: {e}")
        yield c.ErrorEvent("Engine stream failed")
    finally:
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            await aclose()


async def events_from_response(resp: c.Response) -> AsyncIterator[Any]:
    """Yield the stream events equivalent to a complete answer."""
    yield c.ResponseStart(resp.id, resp.model, resp.created)
    for i, p in enumerate(resp.output):
        if isinstance(p, c.TextPart):
            yield c.PartStart(i, "text")
            yield c.TextDelta(i, p.text)
        elif isinstance(p, c.ReasoningPart):
            yield c.PartStart(i, "reasoning")
            yield c.ReasoningDelta(i, p.text, p.signature)
        elif isinstance(p, c.ToolCallPart):
            yield c.PartStart(i, "tool_call", id=p.id, name=p.name)
            yield c.ToolArgsDelta(i, p.arguments)
        else:
            continue
        yield c.PartStop(i)
    yield c.ResponseStop(resp.stop_reason, resp.stop_sequence, resp.usage)
