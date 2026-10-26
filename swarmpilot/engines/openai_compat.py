"""Shared pieces for engines that speak the OpenAI HTTP API.

vLLM and SGLang both serve ``/v1/chat/completions`` and
``/v1/completions`` and stream with SSE. This module holds the request
schemas, the feature extraction and the Prometheus parsing they share.

The schemas list the fields SwarmPilot knows about. They are declared now
(plan item 1.2) and enforced at the public boundary later (plan item 6.3),
so for now they only document the surface.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, ConfigDict

from swarmpilot.engines.base import (
    Capacity,
    EngineAdapter,
    RouteSpec,
    RuntimeMetrics,
    WorkloadSpec,
    WorkloadType,
)


class _Request(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ChatMessage(BaseModel):
    """One chat message; content is text or a list of content parts."""

    model_config = ConfigDict(extra="ignore")

    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None


class ChatCompletionRequest(_Request):
    """Fields of an OpenAI chat completion request SwarmPilot forwards."""

    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    stream_options: dict[str, Any] | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    user: str | None = None


class CompletionRequest(_Request):
    """Fields of an OpenAI text completion request SwarmPilot forwards."""

    model: str | None = None
    prompt: str | list[str] | list[int] | list[list[int]]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    stream_options: dict[str, Any] | None = None
    seed: int | None = None
    logprobs: int | None = None
    echo: bool | None = None
    user: str | None = None


LLM_WORKLOAD = WorkloadSpec(
    type=WorkloadType.LLM_CHAT,
    feature_names=("prompt_length", "max_tokens"),
    latency_metric="e2e",
)

LLM_ROUTES: tuple[RouteSpec, ...] = (
    RouteSpec(
        "v1/chat/completions",
        request_model=ChatCompletionRequest,
        streaming=True,
    ),
    RouteSpec(
        "v1/completions", request_model=CompletionRequest, streaming=True
    ),
    RouteSpec("v1/models", methods=("GET",), lightweight=True, retry_safe=True),
    RouteSpec("health", methods=("GET",), lightweight=True, retry_safe=True),
    RouteSpec("version", methods=("GET",), lightweight=True, retry_safe=True),
)


def _text_length(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(
            len(p.get("text", ""))
            for p in content
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return 0


def llm_features(
    body: Mapping[str, Any], default_max_tokens: int
) -> dict[str, float]:
    """Return ``prompt_length`` (characters) and ``max_tokens`` for a request.

    ``prompt_length`` counts characters, as the proxy did before adapters,
    so models trained on earlier data keep their meaning.

    Args:
        body: Chat or completion request body.
        default_max_tokens: Used when the request sets no output limit.

    Returns:
        Feature dict with both keys always present.
    """
    messages = body.get("messages")
    if isinstance(messages, list):
        plen = sum(
            _text_length(m.get("content"))
            for m in messages
            if isinstance(m, dict)
        )
    else:
        prompt = body.get("prompt")
        if isinstance(prompt, str):
            plen = len(prompt)
        elif isinstance(prompt, list):
            plen = sum(
                len(p) if isinstance(p, str | list) else 1 for p in prompt
            )
        else:
            plen = 0
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    try:
        mt = float(max_tokens) if max_tokens is not None else default_max_tokens
    except (TypeError, ValueError):
        mt = float(default_max_tokens)
    return {"prompt_length": float(plen), "max_tokens": mt}


_PROM_LINE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)"
)


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum Prometheus samples by metric name, ignoring labels.

    Args:
        text: Prometheus exposition text.

    Returns:
        Metric name to summed value.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE.match(line)
        if not m:
            continue
        try:
            value = float(m.group(3))
        except ValueError:
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + value
    return out


class OpenAICompatibleAdapter(EngineAdapter):
    """Base for LLM engines with an OpenAI-compatible HTTP server."""

    workload: ClassVar[WorkloadSpec] = LLM_WORKLOAD
    #: Output length assumed when a request sets none.
    default_max_tokens: ClassVar[int] = 1024
    #: In-flight requests per instance; continuous batching takes many.
    max_in_flight: ClassVar[int] = 256
    #: Prometheus metric names: running, waiting, KV-cache usage (0..1).
    metric_names: ClassVar[tuple[tuple[str, ...], ...]] = ((), (), ())

    def routes(self) -> tuple[RouteSpec, ...]:
        """Return the OpenAI routes the engine serves."""
        return LLM_ROUTES

    def features(self, path: str, body: Mapping[str, Any]) -> dict[str, float]:
        """Return prompt length and max tokens."""
        return llm_features(body, self.default_max_tokens)

    def example_requests(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        """Return a chat and a completion request."""
        return (
            (
                "v1/chat/completions",
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "Hello there"}],
                    "max_tokens": 32,
                },
            ),
            ("v1/completions", {"model": "m", "prompt": "Once upon a time"}),
        )

    def capacity(self) -> Capacity:
        """Continuous batching: many requests in flight per instance."""
        return Capacity(max_in_flight=self.max_in_flight, batching="continuous")

    async def runtime_metrics(
        self, client: httpx.AsyncClient, endpoint: str
    ) -> RuntimeMetrics | None:
        """Read running/waiting requests and KV-cache use from ``/metrics``."""
        try:
            r = await client.get(f"{endpoint.rstrip('/')}/metrics", timeout=5)
        except httpx.HTTPError:
            return None
        if not r.is_success:
            return None
        values = parse_prometheus(r.text)

        def first(names: tuple[str, ...]) -> float | None:
            for n in names:
                if n in values:
                    return values[n]
            return None

        running, waiting, kv = (first(n) for n in self.metric_names)
        return RuntimeMetrics(
            running_requests=int(running) if running is not None else None,
            waiting_requests=int(waiting) if waiting is not None else None,
            kv_cache_usage=kv,
        )
