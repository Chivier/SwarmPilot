"""Engine and workload plugin interfaces.

SwarmPilot routes and scales work for any inference engine (vLLM, SGLang,
a diffusion runtime, a private library) through one small interface, the
:class:`EngineAdapter`. An adapter says how to launch the engine, which
routes it serves, how to turn a request into predictor features, how to
send a request to it and how much concurrent work it takes. Core code never
names an engine; it asks the adapter.

Adapters are discovered through the ``swarmpilot.engines`` entry-point group
(see :mod:`swarmpilot.engines.registry`), so a new engine ships as its own
package with no change to ``swarmpilot/``.

Cluster backends (PyLet now, Ray and Kubernetes later) get the same
treatment through :class:`ClusterBackend`.
"""

from __future__ import annotations

import abc
import enum
import json
import re
import shlex
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import aiohttp
import httpx
from pydantic import BaseModel

# Placeholder for the port the cluster backend assigns at launch time.
PORT = "{port}"

_VERSION_RE = re.compile(r"^\s*v?(\d+)(?:\.(\d+))?")


class WorkloadType(enum.StrEnum):
    """Kind of work an engine does; sets the runtime unit and SLO metric."""

    LLM_CHAT = "llm_chat"
    LLM_COMPLETION = "llm_completion"
    EMBEDDING = "embedding"
    DIFFUSION_IMAGE = "diffusion_image"
    DIFFUSION_VIDEO = "diffusion_video"
    GENERIC_JOB = "generic_job"


@dataclass(frozen=True)
class WorkloadSpec:
    """What the predictor learns and what SLOs measure for a workload.

    Attributes:
        type: The workload type.
        runtime_unit: Unit of the runtime the predictor learns.
        latency_metric: Latency SLOs are written against: end-to-end
            ("e2e") or time to first token ("ttft").
        feature_names: Numeric features ``features()`` returns, in order.
        feature_schema_version: Bumped whenever ``features()`` changes
            meaning, so models trained on the old schema are not reused.
    """

    type: WorkloadType
    feature_names: tuple[str, ...]
    runtime_unit: Literal["ms"] = "ms"
    latency_metric: Literal["e2e", "ttft"] = "e2e"
    feature_schema_version: int = 1


@dataclass(frozen=True)
class LaunchSpec:
    """How to start one engine process.

    The command is always an argv list, never a shell string. Use
    :data:`PORT` where the assigned port goes.

    Attributes:
        argv: Program and arguments.
        env: Extra environment variables.
        health_path: Path that answers 2xx once the engine is ready.
        gpus: GPUs per instance.
        cpus: CPU cores per instance.
        memory_mb: Memory per instance.
        ready_timeout_s: How long startup may take.
    """

    argv: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)
    health_path: str = "/health"
    gpus: int = 1
    cpus: int = 1
    memory_mb: int = 512
    ready_timeout_s: float = 900.0

    def render_argv(self, port: int) -> list[str]:
        """Return argv with the port filled in."""
        return [a.replace(PORT, str(port)) for a in self.argv]

    def render_shell(self, port_expr: str = '"$PORT"') -> str:
        """Return a shell-safe command line for backends that need one.

        Every argument is quoted with :func:`shlex.quote`; only the port
        placeholder is replaced by ``port_expr`` (by default the ``$PORT``
        variable the backend sets).

        Args:
            port_expr: Shell expression substituted for :data:`PORT`.

        Returns:
            A command line that runs exactly ``argv``.
        """
        parts = []
        for arg in self.argv:
            pieces = arg.split(PORT)
            parts.append(
                port_expr.join(shlex.quote(p) if p else "" for p in pieces)
                if len(pieces) > 1
                else shlex.quote(arg)
            )
        return " ".join(parts)


@dataclass(frozen=True)
class RouteSpec:
    """A public route an engine serves.

    Attributes:
        path: Path without a leading slash, e.g. ``v1/chat/completions``.
        methods: Allowed HTTP methods.
        request_model: Pydantic model of the request body, or None for
            routes without a body. Unknown fields are dropped by the
            model's config.
        streaming: Whether the route can answer with SSE.
        lightweight: Metadata routes that skip the queue.
        retry_safe: Whether re-sending after a lost response is harmless.
    """

    path: str
    methods: tuple[str, ...] = ("POST",)
    request_model: type[BaseModel] | None = None
    streaming: bool = False
    lightweight: bool = False
    retry_safe: bool = False


@dataclass(frozen=True)
class Capacity:
    """How much concurrent work one instance takes.

    Attributes:
        max_in_flight: Requests sent to one instance at the same time.
        batching: How the engine batches concurrent requests.
    """

    max_in_flight: int = 1
    batching: Literal["continuous", "static", "none"] = "none"


@dataclass
class RuntimeMetrics:
    """Optional live load reported by an engine. Unknown fields stay None."""

    running_requests: int | None = None
    waiting_requests: int | None = None
    kv_cache_usage: float | None = None
    gpu_utilization: float | None = None
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class EngineRequest:
    """One request to forward to an engine instance.

    Attributes:
        path: Route path without a leading slash.
        method: HTTP method.
        body: Parsed JSON body (empty for GET).
        headers: Headers to forward.
        stream: Whether the client asked for a streamed response.
        proxy: HTTP proxy URL to send through, if any.
    """

    path: str
    method: str = "POST"
    body: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    stream: bool = False
    proxy: str | None = None


@dataclass
class EngineResponse:
    """An engine's answer.

    Exactly one of ``body`` and ``chunks`` is set. ``chunks`` is an async
    iterator of decoded body bytes for streamed responses; the caller must
    call ``close`` when done with it, whether or not it was consumed.

    Attributes:
        status_code: HTTP status.
        headers: Response headers.
        body: Parsed JSON body (non-streamed).
        chunks: Body chunks (streamed).
        close: Releases the connection of a streamed response.
    """

    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    body: Any = None
    chunks: AsyncIterator[bytes] | None = None
    close: Callable[[], Awaitable[None]] | None = None


def major_minor(version: str) -> str:
    """Reduce a version string to ``major.minor`` (``"0.11.0rc1"`` -> ``"0.11"``).

    Args:
        version: Any version string.

    Returns:
        ``major.minor``, ``major`` alone, or the input when unparseable.
    """
    m = _VERSION_RE.match(version or "")
    if not m:
        return version or "unknown"
    return m.group(1) if m.group(2) is None else f"{m.group(1)}.{m.group(2)}"


class EngineAdapter(abc.ABC):
    """Everything SwarmPilot needs to know about one engine.

    Subclasses set :attr:`name` and :attr:`workload` and implement
    :meth:`launch_spec`, :meth:`routes` and :meth:`features`. The other
    hooks have defaults that fit an HTTP JSON engine.
    """

    #: Registry name, e.g. ``"vllm"``. Also the platform ``software_name``.
    name: ClassVar[str]
    #: What the engine does.
    workload: ClassVar[WorkloadSpec]
    #: Version ranges this adapter is known to work with (informational
    #: until plan item 8.2 enforces it), e.g. ``(">=0.11,<0.12",)``.
    supported_versions: ClassVar[tuple[str, ...]] = ()
    #: Path that answers 2xx when an instance is ready.
    health_path: ClassVar[str] = "/health"

    @abc.abstractmethod
    def launch_spec(
        self,
        model_id: str,
        *,
        gpus: int = 1,
        extra_args: tuple[str, ...] = (),
    ) -> LaunchSpec:
        """Return the argv and resources to start one instance.

        Args:
            model_id: Model to serve.
            gpus: GPUs for this instance.
            extra_args: Extra engine arguments, already split into argv.

        Returns:
            The launch spec.
        """

    @abc.abstractmethod
    def routes(self) -> tuple[RouteSpec, ...]:
        """Return the public routes this engine serves."""

    def feature_extractors(self) -> tuple[tuple[str, str], ...]:
        """Return ``(feature name, extractor expression)`` pairs.

        The expressions (see :mod:`swarmpilot.engines.extractors`) go into
        the adapter's :meth:`manifest`, so any gateway can compute the same
        features without running Python. Return ``()`` when the features
        need code the extractors cannot express; :meth:`features` is then
        the only definition.
        """
        return ()

    def features(self, path: str, body: Mapping[str, Any]) -> dict[str, float]:
        """Turn a request into numeric predictor features.

        Must be cheap (it runs on every request) and must return exactly
        ``workload.feature_names`` as keys. The default evaluates
        :meth:`feature_extractors`. Adapters may override it with
        hand-written code for speed, and the contract suite then checks
        that both give the same values.

        Args:
            path: Route path without a leading slash.
            body: Parsed JSON body.

        Returns:
            Feature name to value.
        """
        compiled = self.__dict__.get("_compiled_features")
        if compiled is None:
            from swarmpilot.engines.extractors import compile_extractor

            pairs = self.feature_extractors()
            if not pairs:
                raise NotImplementedError(
                    f"{type(self).__name__} declares no feature_extractors() "
                    "and does not override features()"
                )
            compiled = tuple(
                (name, compile_extractor(expr)) for name, expr in pairs
            )
            self.__dict__["_compiled_features"] = compiled
        return {name: fn(body) for name, fn in compiled}

    def manifest(self) -> dict[str, Any]:
        """Return this adapter as plain data for non-Python data planes.

        See :mod:`swarmpilot.engines.manifest` for the format.
        """
        from swarmpilot.engines.manifest import build_manifest

        return build_manifest(self)

    def capacity(self) -> Capacity:
        """Return how many requests one instance takes at once."""
        return Capacity()

    def example_requests(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        """Return sample (path, body) pairs for the contract test suite.

        Every route with a request model should have at least one.
        """
        return ()

    def native_styles(self) -> tuple[str, ...]:
        """Return the API styles this engine serves itself.

        A client calling in one of these styles is passed straight
        through; any other style is translated into the first of them
        (see :mod:`swarmpilot.api.translate`). The default lists every
        registered style whose generation route the engine declares.

        Returns:
            Style names, e.g. ``("openai_chat",)``.
        """
        from swarmpilot.api.registry import find_style, style_names

        paths = {r.path for r in self.routes()}
        found = []
        for name in style_names():
            style = find_style(name)
            if style is not None and style.generate_path() in paths:
                found.append(name)
        return tuple(found)

    def route(self, path: str) -> RouteSpec | None:
        """Return the route spec for ``path``, or None if not declared."""
        path = path.lstrip("/")
        for r in self.routes():
            if r.path == path:
                return r
        return None

    def platform_info(
        self, version: str, hardware_name: str = "unknown"
    ) -> dict[str, str]:
        """Return the platform info that keys predictor models.

        Args:
            version: Engine version as reported by :meth:`version`.
            hardware_name: GPU type or other hardware label.

        Returns:
            Dict with software_name, software_version (major.minor),
            hardware_name, workload and feature_schema. The predictor keys
            its models on all five, so a new engine version or a change in
            what :meth:`features` means starts a fresh model.
        """
        return {
            "software_name": self.name,
            "software_version": major_minor(version),
            "hardware_name": hardware_name,
            **self.workload_keys(),
        }

    @classmethod
    def workload_keys(cls) -> dict[str, str]:
        """Return the workload type and feature schema version as strings.

        Returns:
            Dict with ``workload`` and ``feature_schema``.
        """
        return {
            "workload": str(cls.workload.type),
            "feature_schema": str(cls.workload.feature_schema_version),
        }

    async def version(self, client: httpx.AsyncClient, endpoint: str) -> str:
        """Read the engine version from a live instance.

        The default asks ``GET /version`` and reads ``version``.

        Args:
            client: Shared async HTTP client.
            endpoint: Instance base URL.

        Returns:
            The version string, or ``"unknown"``.
        """
        try:
            r = await client.get(f"{endpoint.rstrip('/')}/version", timeout=5)
            if r.is_success:
                v = r.json().get("version")
                if v:
                    return str(v)
        except (httpx.HTTPError, ValueError, AttributeError):
            pass
        return "unknown"

    async def health(self, client: httpx.AsyncClient, endpoint: str) -> bool:
        """Return whether the instance answers its health path."""
        url = f"{endpoint.rstrip('/')}{self.health_path}"
        try:
            r = await client.get(url, timeout=5)
        except httpx.HTTPError:
            return False
        return r.is_success

    async def dispatch(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        request: EngineRequest,
    ) -> EngineResponse:
        """Send one request to an instance.

        This is the hot path, so it uses aiohttp: at 64 concurrent
        requests httpx's connection pool managed about 5% of aiohttp's
        throughput on the same machine. The default sends JSON over HTTP
        and streams the response when the request asks for it and the
        engine answers with SSE.

        Args:
            session: Shared aiohttp session.
            endpoint: Instance base URL.
            request: The request to send.

        Returns:
            The engine's response.
        """
        url = f"{endpoint.rstrip('/')}/{request.path.lstrip('/')}"
        kwargs: dict[str, Any] = {"headers": request.headers}
        if request.method in ("POST", "PUT", "PATCH"):
            kwargs["json"] = request.body
        if request.proxy:
            kwargs["proxy"] = request.proxy
        resp = await session.request(request.method, url, **kwargs)
        headers = dict(resp.headers)
        if (
            not request.stream
            or resp.status >= 400
            or "text/event-stream" not in resp.headers.get("Content-Type", "")
        ):
            # Non-streamed requests, errors, and engines that ignore
            # stream=true all answer with one body.
            try:
                raw = await resp.read()
            finally:
                resp.release()
            try:
                body: Any = json.loads(raw) if raw else {}
            except ValueError:
                body = {"error": raw.decode(errors="replace")}
            return EngineResponse(resp.status, headers, body=body)

        async def close() -> None:
            # A fully read response can return its connection to the pool;
            # one abandoned midway must drop it.
            if resp.content.at_eof():
                resp.release()
            else:
                resp.close()

        return EngineResponse(
            resp.status,
            headers,
            chunks=resp.content.iter_any(),
            close=close,
        )

    async def runtime_metrics(
        self, client: httpx.AsyncClient, endpoint: str
    ) -> RuntimeMetrics | None:
        """Return live load from the instance, or None if not supported."""
        return None


@dataclass
class ClusterInstance:
    """An engine process as a cluster backend sees it.

    Attributes:
        id: Backend-specific id.
        name: Instance name.
        status: Backend status string (e.g. PENDING, RUNNING, FAILED).
        endpoint: ``host:port`` once running.
        labels: Labels attached at submit time.
    """

    id: str
    name: str
    status: str
    endpoint: str | None = None
    labels: dict[str, str] = field(default_factory=dict)


class ClusterBackend(abc.ABC):
    """Where engine processes run: PyLet now, Ray and Kubernetes later."""

    name: ClassVar[str]

    @abc.abstractmethod
    def submit(
        self,
        spec: LaunchSpec,
        *,
        name: str,
        labels: Mapping[str, str] | None = None,
        target_worker: str | None = None,
    ) -> ClusterInstance:
        """Start one instance from a launch spec."""

    @abc.abstractmethod
    def get(self, instance_id: str) -> ClusterInstance:
        """Return the current state of an instance."""

    @abc.abstractmethod
    def cancel(self, instance_id: str) -> None:
        """Stop an instance."""

    @abc.abstractmethod
    def list(
        self, labels: Mapping[str, str] | None = None
    ) -> list[ClusterInstance]:
        """List instances, optionally filtered by labels."""
