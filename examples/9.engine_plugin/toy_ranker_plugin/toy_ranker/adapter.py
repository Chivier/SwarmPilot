"""SwarmPilot engine adapter for the toy ranking server.

This file is all an out-of-tree engine needs. The package's
``pyproject.toml`` registers it in the ``swarmpilot.engines`` entry-point
group, so once the package is installed next to SwarmPilot the Planner can
launch it, the Scheduler can route to it and the predictor learns its
runtime, without any change to SwarmPilot itself.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any, ClassVar

import httpx

from swarmpilot.engines import (
    PORT,
    Capacity,
    EngineAdapter,
    LaunchSpec,
    RouteSpec,
    WorkloadSpec,
    WorkloadType,
)


class ToyRankerAdapter(EngineAdapter):
    """Launch, route and predict for ``toy_ranker.server``."""

    name: ClassVar[str] = "toy-ranker"
    workload: ClassVar[WorkloadSpec] = WorkloadSpec(
        type=WorkloadType.GENERIC_JOB,
        feature_names=("n_docs", "query_chars"),
    )
    health_path: ClassVar[str] = "/health"

    def launch_spec(
        self,
        model_id: str,
        *,
        gpus: int = 1,
        extra_args: tuple[str, ...] = (),
    ) -> LaunchSpec:
        """Start the server with this Python interpreter."""
        return LaunchSpec(
            argv=(
                sys.executable,
                "-m",
                "toy_ranker.server",
                "--model",
                model_id,
                "--port",
                PORT,
                *extra_args,
            ),
            gpus=gpus,
            ready_timeout_s=60,
        )

    def routes(self) -> tuple[RouteSpec, ...]:
        """Expose the ranking route."""
        return (RouteSpec(path="v1/rank", retry_safe=True),)

    def features(self, path: str, body: Mapping[str, Any]) -> dict[str, float]:
        """Runtime grows with document count and query length."""
        docs = body.get("documents")
        query = body.get("query")
        return {
            "n_docs": float(len(docs)) if isinstance(docs, list) else 0.0,
            "query_chars": float(len(query)) if isinstance(query, str) else 0.0,
        }

    def capacity(self) -> Capacity:
        """The server is threaded; four requests at a time is safe."""
        return Capacity(max_in_flight=4, batching="none")

    async def version(self, client: httpx.AsyncClient, endpoint: str) -> str:
        """Read ``build.version`` from ``/info``."""
        try:
            r = await client.get(f"{endpoint.rstrip('/')}/info", timeout=5)
            return str(r.json()["build"]["version"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return "unknown"
