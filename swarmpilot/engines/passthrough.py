"""Fallback adapter for instances whose engine has no adapter.

Online API endpoints and ad-hoc test servers register with a platform
``software_name`` that matches no adapter. They still need requests sent,
so the Scheduler uses this pass-through: plain HTTP JSON or SSE with one
request in flight, exactly as before adapters existed. It is not
registered, so it never shows up as a launchable engine.

Its features are the ones the proxy extracted before adapters, and only
the keys the request actually carries, so online endpoints keep routing
the way they did. That is why it is exempt from the contract suite.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from swarmpilot.engines.base import (
    EngineAdapter,
    LaunchSpec,
    RouteSpec,
    WorkloadSpec,
    WorkloadType,
)


class PassthroughAdapter(EngineAdapter):
    """Send requests as-is; cannot launch anything."""

    name: ClassVar[str] = "passthrough"
    workload: ClassVar[WorkloadSpec] = WorkloadSpec(
        type=WorkloadType.GENERIC_JOB,
        feature_names=("prompt_length", "max_tokens"),
    )

    def launch_spec(
        self,
        model_id: str,
        *,
        gpus: int = 1,
        extra_args: tuple[str, ...] = (),
    ) -> LaunchSpec:
        """Pass-through instances are started outside SwarmPilot."""
        raise NotImplementedError("passthrough instances are not launched")

    def routes(self) -> tuple[RouteSpec, ...]:
        """Declares no routes; the proxy forwards whatever it receives."""
        return ()

    def features(self, path: str, body: Mapping[str, Any]) -> dict[str, float]:
        """Return prompt length and max tokens when the body has them."""
        feats: dict[str, float] = {}
        if "max_tokens" in body:
            feats["max_tokens"] = body["max_tokens"]
        messages = body.get("messages")
        if isinstance(messages, list):
            feats["prompt_length"] = sum(
                len(m.get("content", ""))
                for m in messages
                if isinstance(m, dict)
            )
        elif "prompt" in body:
            feats["prompt_length"] = len(body["prompt"])
        return feats
