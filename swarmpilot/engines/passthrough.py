"""Fallback adapter for instances whose engine has no adapter.

Online API endpoints and ad-hoc test servers register with a platform
``software_name`` that matches no adapter. They still need requests sent,
so the Scheduler uses this pass-through: plain HTTP JSON or SSE with one
request in flight, exactly as before adapters existed. It is not
registered, so it never shows up as a launchable engine.
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
        type=WorkloadType.GENERIC_JOB, feature_names=()
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
        """No features of its own."""
        return {}
