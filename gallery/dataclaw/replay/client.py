"""Async OpenAI-compatible client for dual-endpoint replay requests."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

from openai import AsyncOpenAI  # type: ignore[import-untyped]

from replay.models import ModelEndpoint, RequestMetrics


class ReplayClient:
    """Sends replay requests to large/small model endpoints.

    Args:
        large: Configuration for the large model endpoint.
        small: Configuration for the small model endpoint.
        timeout_s: Per-request timeout in seconds.
        max_tokens: Max tokens to generate per request.
    """

    def __init__(
        self,
        large: ModelEndpoint,
        small: ModelEndpoint,
        timeout_s: float = 120.0,
        max_tokens: int = 1,
    ) -> None:
        self._clients: dict[str, AsyncOpenAI] = {
            "large": AsyncOpenAI(base_url=large.base_url, api_key=large.api_key),
            "small": AsyncOpenAI(base_url=small.base_url, api_key=small.api_key),
        }
        self._models: dict[str, str] = {
            "large": large.model,
            "small": small.model,
        }
        self._timeout = timeout_s
        self._max_tokens = max_tokens

    async def send(
        self,
        messages: list[dict[str, Any]],
        model_size: Literal["large", "small"],
        group_id: str,
        step_index: int,
    ) -> RequestMetrics:
        """Send a chat completion request and return latency metrics.

        Args:
            messages: Cumulative message history to send.
            model_size: Which endpoint to use ("large" or "small").
            group_id: Replay group identifier for metrics.
            step_index: Step position within the group.

        Returns:
            RequestMetrics with timing and status information.
        """
        client = self._clients[model_size]
        model = self._models[model_size]
        start = time.monotonic()

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=self._max_tokens,
                ),
                timeout=self._timeout,
            )
            end = time.monotonic()
            usage = getattr(response, "usage", None)
            return RequestMetrics(
                group_id=group_id,
                step_index=step_index,
                model_size=model_size,
                start_time=start,
                end_time=end,
                latency_ms=(end - start) * 1000,
                status="success",
                input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            )
        except asyncio.TimeoutError:
            end = time.monotonic()
            return RequestMetrics(
                group_id=group_id,
                step_index=step_index,
                model_size=model_size,
                start_time=start,
                end_time=end,
                latency_ms=(end - start) * 1000,
                status="timeout",
                error_message=f"Request timed out after {self._timeout}s",
            )
        except Exception as exc:
            end = time.monotonic()
            return RequestMetrics(
                group_id=group_id,
                step_index=step_index,
                model_size=model_size,
                start_time=start,
                end_time=end,
                latency_ms=(end - start) * 1000,
                status="error",
                error_message=str(exc),
            )

    async def close(self) -> None:
        """Close underlying HTTP clients."""
        for client in self._clients.values():
            await client.close()
