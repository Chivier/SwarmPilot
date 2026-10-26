"""Generic HTTP engine adapter, configured in YAML.

For private inference servers that speak plain HTTP and JSON. Instead of
writing an adapter class, describe the server in a YAML file::

    name: toy-ranker
    workload: generic_job
    launch:
      argv: [python, -m, toy_ranker, --model, "{model}", --port, "{port}"]
      gpu_args: [--devices, "{gpus}"]      # appended when gpus > 1
      health_path: /health
      ready_timeout_s: 120
    version: {path: /info, field: build.version}
    routes:
      - {path: v1/rank, methods: [POST], retry_safe: true}
    features:
      - {name: n_docs, expr: "len(documents)"}
      - {name: query_chars, expr: "chars(query)"}
    capacity: {max_in_flight: 8, batching: static}

``{model}``, ``{gpus}`` and ``{port}`` are filled in by SwarmPilot; the
model id always stays inside its own argument. Feature expressions are
described in :mod:`swarmpilot.engines.extractors`.

Point ``SWARMPILOT_ENGINE_CONFIGS`` at one or more such files, or at
directories of them (separated by ``:``), and every SwarmPilot service
registers them as engines under their ``name``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from swarmpilot.engines.base import (
    PORT,
    Capacity,
    EngineAdapter,
    LaunchSpec,
    RouteSpec,
    WorkloadSpec,
    WorkloadType,
)
from swarmpilot.engines.extractors import compile_extractor

CONFIG_ENV = "SWARMPILOT_ENGINE_CONFIGS"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LaunchConfig(_Strict):
    """How to start the server. See the module docstring."""

    argv: list[str] = Field(min_length=1)
    gpu_args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    health_path: str = "/health"
    ready_timeout_s: float = 900.0
    cpus: int = 1
    memory_mb: int = 512

    @field_validator("argv")
    @classmethod
    def _needs_port(cls, argv: list[str]) -> list[str]:
        if not any(PORT in a for a in argv):
            raise ValueError(f"launch.argv must use {PORT} for the port")
        return argv


class VersionConfig(_Strict):
    """Where a live instance reports its version."""

    path: str = "/version"
    field: str = "version"


class RouteConfig(_Strict):
    """One public route."""

    path: str
    methods: list[str] = Field(default_factory=lambda: ["POST"])
    streaming: bool = False
    lightweight: bool = False
    retry_safe: bool = False


class FeatureConfig(_Strict):
    """One predictor feature."""

    name: str
    expr: str
    default: float = 0.0


class CapacityConfig(_Strict):
    """Concurrent requests per instance."""

    max_in_flight: int = Field(default=1, ge=1)
    batching: Literal["continuous", "static", "none"] = "none"


class GenericEngineConfig(_Strict):
    """A whole engine description."""

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    workload: WorkloadType = WorkloadType.GENERIC_JOB
    feature_schema_version: int = 1
    latency_metric: Literal["e2e", "ttft"] = "e2e"
    launch: LaunchConfig
    version: VersionConfig = Field(default_factory=VersionConfig)
    routes: list[RouteConfig] = Field(min_length=1)
    features: list[FeatureConfig] = Field(min_length=1)
    capacity: CapacityConfig = Field(default_factory=CapacityConfig)


class GenericHTTPAdapter(EngineAdapter):
    """Adapter driven by a :class:`GenericEngineConfig`.

    Use :func:`make_adapter_class` to get a registrable subclass for one
    configuration.
    """

    config: ClassVar[GenericEngineConfig]
    _extractors: ClassVar[tuple[tuple[str, str], ...]]
    _routes: ClassVar[tuple[RouteSpec, ...]]

    def launch_spec(
        self,
        model_id: str,
        *,
        gpus: int = 1,
        extra_args: tuple[str, ...] = (),
    ) -> LaunchSpec:
        """Fill ``{model}`` and ``{gpus}`` into the configured argv."""
        launch = self.config.launch
        argv = list(launch.argv)
        if gpus > 1:
            argv += launch.gpu_args
        argv = [
            a.replace("{model}", model_id).replace("{gpus}", str(gpus))
            if "{" in a
            else a
            for a in argv
        ]
        return LaunchSpec(
            argv=(*argv, *extra_args),
            env=dict(launch.env),
            health_path=launch.health_path,
            gpus=gpus,
            cpus=launch.cpus,
            memory_mb=launch.memory_mb,
            ready_timeout_s=launch.ready_timeout_s,
        )

    def routes(self) -> tuple[RouteSpec, ...]:
        """Return the configured routes."""
        return self._routes

    def feature_extractors(self) -> tuple[tuple[str, str], ...]:
        """Return the configured feature expressions.

        :meth:`features` evaluates them; there is no hand-written copy.
        """
        return self._extractors

    def capacity(self) -> Capacity:
        """Return the configured capacity."""
        cap = self.config.capacity
        return Capacity(max_in_flight=cap.max_in_flight, batching=cap.batching)

    async def version(self, client: httpx.AsyncClient, endpoint: str) -> str:
        """Read the version from the configured path and field."""
        cfg = self.config.version
        try:
            r = await client.get(
                f"{endpoint.rstrip('/')}/{cfg.path.lstrip('/')}", timeout=5
            )
            if r.is_success:
                value: Any = r.json()
                for key in cfg.field.split("."):
                    value = value[key]
                if value not in (None, ""):
                    return str(value)
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            pass
        return "unknown"


def _with_default(expr: str, default: float) -> str:
    """Fold a feature's default into its expression, checking it parses."""
    out = expr if default == 0 else f"coalesce({expr}, {default!r})"
    compile_extractor(out)
    return out


def make_adapter_class(
    config: GenericEngineConfig | Mapping[str, Any],
) -> type[GenericHTTPAdapter]:
    """Build an adapter class for one engine description.

    Args:
        config: Parsed config, or the raw mapping from YAML.

    Returns:
        A :class:`GenericHTTPAdapter` subclass named after the engine.

    Raises:
        pydantic.ValidationError: If the config is invalid.
        swarmpilot.engines.extractors.ExtractorError: If a feature
            expression does not parse.
    """
    if not isinstance(config, GenericEngineConfig):
        config = GenericEngineConfig.model_validate(config)
    names = [f.name for f in config.features]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate feature names in {config.name!r}")
    attrs = {
        "name": config.name,
        "config": config,
        "health_path": config.launch.health_path,
        "workload": WorkloadSpec(
            type=config.workload,
            feature_names=tuple(names),
            latency_metric=config.latency_metric,
            feature_schema_version=config.feature_schema_version,
        ),
        "_extractors": tuple(
            (f.name, _with_default(f.expr, f.default)) for f in config.features
        ),
        "_routes": tuple(
            RouteSpec(
                path=r.path.lstrip("/"),
                methods=tuple(m.upper() for m in r.methods),
                streaming=r.streaming,
                lightweight=r.lightweight,
                retry_safe=r.retry_safe,
            )
            for r in config.routes
        ),
    }
    cls_name = "".join(
        p.capitalize() for p in config.name.replace("-", "_").split("_")
    )
    return type(f"{cls_name}Adapter", (GenericHTTPAdapter,), attrs)


def load_config_file(path: str | Path) -> type[GenericHTTPAdapter]:
    """Load one YAML engine description.

    Args:
        path: YAML file.

    Returns:
        The adapter class.
    """
    with open(path, encoding="utf-8") as f:
        return make_adapter_class(yaml.safe_load(f) or {})


def config_files(spec: str | None = None) -> list[Path]:
    """List the YAML files named by ``SWARMPILOT_ENGINE_CONFIGS``.

    Args:
        spec: ``:``-separated files or directories; defaults to the
            environment variable.

    Returns:
        YAML files, directories expanded to their ``*.yaml``/``*.yml``.
    """
    spec = os.getenv(CONFIG_ENV, "") if spec is None else spec
    files: list[Path] = []
    for part in filter(None, spec.split(os.pathsep)):
        p = Path(part).expanduser()
        if p.is_dir():
            files += sorted([*p.glob("*.yaml"), *p.glob("*.yml")])
        else:
            files.append(p)
    return files
