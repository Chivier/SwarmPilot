"""Adapter manifests: an engine adapter as plain data.

The Python Scheduler calls adapters directly. A data plane written in
another language (the Rust gateway of the migration plan) cannot, so every
adapter can also describe itself as a manifest: its routes and their
request schemas, its capacity, its transport and its predictor features as
declarative extractor expressions. Any implementation of the extractor
grammar in :mod:`swarmpilot.engines.extractors` then computes the same
features the Python adapter does; the contract suite checks that it does.

An adapter whose features need code the extractors cannot express has
``features: null`` and ``python_features: true``; its routes must run on a
Python lane.

The manifest is validated by :class:`EngineManifest`, and the JSON Schema
in ``manifest.schema.json`` next to this file is generated from it::

    python -m swarmpilot.engines.manifest --schema > swarmpilot/engines/manifest.schema.json
    python -m swarmpilot.engines.manifest vllm      # print one manifest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from swarmpilot.engines.extractors import compile_extractor

if TYPE_CHECKING:
    from swarmpilot.engines.base import EngineAdapter

MANIFEST_VERSION = 1
SCHEMA_PATH = Path(__file__).with_name("manifest.schema.json")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RouteManifest(_Strict):
    """One public route.

    ``drop_fields`` and ``clamps`` are filled by the request hardening of
    plan item 6.3; until then they are empty.
    """

    path: str
    methods: list[str]
    streaming: bool = False
    lightweight: bool = False
    retry_safe: bool = False
    request_schema: dict[str, Any] | None = None
    drop_fields: list[str] = Field(default_factory=list)
    clamps: dict[str, tuple[float, float]] = Field(default_factory=dict)


class FeatureManifest(_Strict):
    """One predictor feature as an extractor expression."""

    name: str
    expr: str


class WorkloadManifest(_Strict):
    """What the engine does; keys predictor models."""

    type: str
    feature_names: list[str]
    feature_schema_version: int
    latency_metric: Literal["e2e", "ttft"]
    runtime_unit: Literal["ms"]


class TransportManifest(_Strict):
    """How requests reach an instance."""

    kind: Literal["http_json", "sse", "grpc"]
    health_path: str


class CapacityManifest(_Strict):
    """Concurrent requests per instance."""

    max_in_flight: int = Field(ge=1)
    batching: Literal["continuous", "static", "none"]


class EngineManifest(_Strict):
    """A whole adapter manifest."""

    manifest_version: Literal[1] = MANIFEST_VERSION
    name: str
    workload: WorkloadManifest
    transport: TransportManifest
    capacity: CapacityManifest
    routes: list[RouteManifest]
    features: list[FeatureManifest] | None
    python_features: bool
    #: API styles the engine serves itself (passed through untouched).
    native_styles: list[str] = Field(default_factory=list)
    #: For each client API style that is translated into a native one,
    #: the same features as expressions over that style's body.
    style_features: dict[str, list[FeatureManifest]] = Field(
        default_factory=dict
    )


def build_manifest(adapter: EngineAdapter) -> dict[str, Any]:
    """Describe ``adapter`` as a validated manifest dict.

    Args:
        adapter: Any engine adapter.

    Returns:
        JSON-ready manifest.
    """
    spec = adapter.workload
    routes = adapter.routes()
    cap = adapter.capacity()
    pairs = adapter.feature_extractors()
    manifest = EngineManifest(
        name=adapter.name,
        workload=WorkloadManifest(
            type=str(spec.type),
            feature_names=list(spec.feature_names),
            feature_schema_version=spec.feature_schema_version,
            latency_metric=spec.latency_metric,
            runtime_unit=spec.runtime_unit,
        ),
        transport=TransportManifest(
            kind="sse" if any(r.streaming for r in routes) else "http_json",
            health_path=adapter.health_path,
        ),
        capacity=CapacityManifest(
            max_in_flight=cap.max_in_flight, batching=cap.batching
        ),
        routes=[
            RouteManifest(
                path=r.path,
                methods=list(r.methods),
                streaming=r.streaming,
                lightweight=r.lightweight,
                retry_safe=r.retry_safe,
                request_schema=(
                    r.request_model.model_json_schema()
                    if r.request_model is not None
                    else None
                ),
            )
            for r in routes
        ],
        features=(
            [FeatureManifest(name=n, expr=e) for n, e in pairs]
            if pairs
            else None
        ),
        python_features=not pairs,
        native_styles=list(adapter.native_styles()),
        style_features=_style_features(adapter),
    )
    return manifest.model_dump(mode="json")


def _style_features(adapter: EngineAdapter) -> dict[str, list[FeatureManifest]]:
    """Return per-style feature expressions for translated requests."""
    from swarmpilot.api.base import DEFAULT_MAX_OUTPUT_TOKENS
    from swarmpilot.api.registry import get_style, style_names

    native = adapter.native_styles()
    if not native:
        return {}
    default = getattr(adapter, "default_max_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
    out: dict[str, list[FeatureManifest]] = {}
    for name in style_names():
        pairs = get_style(name).feature_extractors(default)
        if (
            name not in native
            and pairs
            and tuple(n for n, _ in pairs)
            == tuple(adapter.workload.feature_names)
        ):
            out[name] = [FeatureManifest(name=n, expr=e) for n, e in pairs]
    return out


def manifest_features(
    manifest: dict[str, Any], body: dict[str, Any]
) -> dict[str, float] | None:
    """Compute features from a manifest alone, as a gateway would.

    Args:
        manifest: Output of :func:`build_manifest`.
        body: Request body.

    Returns:
        Feature values, or None when the manifest has no extractors.
    """
    feats = manifest.get("features")
    if not feats:
        return None
    return {f["name"]: compile_extractor(f["expr"])(body) for f in feats}


def json_schema() -> dict[str, Any]:
    """Return the JSON Schema every manifest satisfies."""
    schema = EngineManifest.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def main(argv: list[str] | None = None) -> None:
    """Print the schema or an adapter's manifest."""
    from swarmpilot.engines.registry import adapter_names, get_adapter

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("engine", nargs="?", help="adapter name")
    parser.add_argument("--schema", action="store_true")
    args = parser.parse_args(argv)
    if args.schema:
        out = json_schema()
    elif args.engine:
        out = get_adapter(args.engine).manifest()
    else:
        out = {name: get_adapter(name).manifest() for name in adapter_names()}
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
