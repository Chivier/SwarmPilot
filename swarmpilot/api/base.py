"""The API style plugin interface.

An :class:`ApiStyle` is one wire format for talking to a model, such as
OpenAI Chat Completions or Anthropic Messages. It converts in both
directions, because SwarmPilot sits between a client and an engine that may
each speak any style:

* client side: ``decode_request`` turns the client's body into a canonical
  :class:`~swarmpilot.api.canonical.Request`; ``encode_response``,
  ``encode_stream`` and ``encode_error`` write the answer back in the
  client's style;
* engine side: ``encode_request`` builds the body an engine of this style
  expects; ``decode_response``, ``decode_stream`` and ``decode_error`` read
  its answer.

Styles are discovered through the ``swarmpilot.api_styles`` entry-point
group (see :mod:`swarmpilot.api.registry`), like engine adapters.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from swarmpilot.api import canonical as c

#: Output limit sent to engines whose style requires one when the client
#: set none.
DEFAULT_MAX_OUTPUT_TOKENS = 4096


class TranslationError(Exception):
    """A request cannot be translated faithfully.

    Attributes:
        status: HTTP status to answer with.
        message: Human-readable reason.
        param: The request field at fault, if any.
        error_type: Short machine-readable kind.
    """

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        status: int = 400,
        error_type: str = "invalid_request_error",
    ) -> None:
        """Initialize the error."""
        super().__init__(message)
        self.message = message
        self.param = param
        self.status = status
        self.error_type = error_type


@dataclass
class Context:
    """Per-request translation state.

    Attributes:
        strict: Refuse fields that would change the answer instead of
            dropping them.
        dropped: Fields left out of the engine request, reported to the
            client in the ``x-swarmpilot-dropped`` header.
        default_max_tokens: Output limit for engine styles that need one.
    """

    strict: bool = True
    dropped: list[str] = field(default_factory=list)
    default_max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS

    def drop(self, name: str, *, semantic: bool = False) -> None:
        """Record that ``name`` is not sent to the engine.

        Args:
            name: Field name as the client spelled it.
            semantic: Whether leaving it out changes the answer. In strict
                mode such fields are refused.

        Raises:
            TranslationError: ``semantic`` in strict mode.
        """
        if semantic and self.strict:
            raise TranslationError(
                f"'{name}' is not supported for this model's engine",
                param=name,
            )
        if name not in self.dropped:
            self.dropped.append(name)


def unsupported(name: str, why: str = "") -> TranslationError:
    """Return the error for a field no translation can honour."""
    reason = f": {why}" if why else ""
    return TranslationError(f"'{name}' is not supported{reason}", param=name)


@dataclass(frozen=True)
class StyleRoute:
    """A public route that speaks a style.

    Attributes:
        path: Path without a leading slash.
        methods: Allowed HTTP methods.
        streaming: Whether the route can stream.
    """

    path: str
    methods: tuple[str, ...] = ("POST",)
    streaming: bool = True


class ApiStyle(abc.ABC):
    """One API wire format, converted to and from the canonical model."""

    #: Registry name, e.g. ``"openai_chat"``.
    name: ClassVar[str]
    #: Workload family the style serves; only ``"llm"`` exists so far.
    family: ClassVar[str] = "llm"
    #: Fields with no canonical home that change the answer when left out.
    #: Translating them is refused in strict mode.
    semantic_extensions: ClassVar[frozenset[str]] = frozenset()
    #: Extension fields the style itself honours on the way back (so they
    #: are neither sent to the engine nor reported as dropped).
    handled_extensions: ClassVar[frozenset[str]] = frozenset()

    @abc.abstractmethod
    def routes(self) -> tuple[StyleRoute, ...]:
        """Return the routes this style owns."""

    def generate_path(self) -> str:
        """Return the route an engine of this style generates on."""
        return self.routes()[0].path

    def feature_extractors(
        self, default_max_tokens: int
    ) -> tuple[tuple[str, str], ...]:
        """Return the LLM features as extractor expressions over this body.

        The names are the LLM workload's ``prompt_length`` and
        ``max_tokens``, so the predictor sees the same features whatever
        style the client used (see :mod:`swarmpilot.engines.extractors`).

        Args:
            default_max_tokens: Used when the request sets no limit.
        """
        return ()

    def is_stream(self, body: Mapping[str, Any]) -> bool:
        """Return whether the client asked for a streamed answer."""
        return isinstance(body, Mapping) and body.get("stream") is True

    # Client side.

    @abc.abstractmethod
    def decode_request(
        self, body: Mapping[str, Any], ctx: Context
    ) -> c.Request:
        """Parse a client body into a canonical request.

        Raises:
            TranslationError: The body is invalid or uses a feature that
                cannot be translated.
        """

    @abc.abstractmethod
    def encode_response(
        self, resp: c.Response, req: c.Request
    ) -> dict[str, Any]:
        """Write a canonical answer as this style's response body."""

    @abc.abstractmethod
    def encode_stream(
        self, events: AsyncIterator[Any], req: c.Request
    ) -> AsyncIterator[bytes]:
        """Write canonical stream events as this style's SSE bytes."""

    @abc.abstractmethod
    def encode_error(
        self,
        status: int,
        message: str,
        error_type: str | None = None,
        param: str | None = None,
    ) -> dict[str, Any]:
        """Return this style's error body."""

    # Engine side.

    @abc.abstractmethod
    def encode_request(
        self, req: c.Request, ctx: Context
    ) -> tuple[str, dict[str, Any]]:
        """Build ``(path, body)`` for an engine that speaks this style.

        Raises:
            TranslationError: The request needs something this style
                cannot express.
        """

    @abc.abstractmethod
    def decode_response(self, body: Mapping[str, Any]) -> c.Response:
        """Parse an engine's response body."""

    @abc.abstractmethod
    def decode_stream(self, chunks: AsyncIterator[bytes]) -> AsyncIterator[Any]:
        """Parse an engine's SSE byte stream into canonical events."""

    def decode_error(self, body: Any) -> str:
        """Return the message of an engine's error body."""
        if isinstance(body, Mapping):
            err = body.get("error")
            if isinstance(err, Mapping) and err.get("message"):
                return str(err["message"])
            if isinstance(err, str):
                return err
            if body.get("message"):
                return str(body["message"])
            if body.get("detail"):
                return str(body["detail"])
        return "Engine request failed"
