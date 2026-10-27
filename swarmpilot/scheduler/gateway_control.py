"""Control channel for ``swarmpilot-gateway`` (plan item 5.0, phase R3).

With ``SCHEDULER_GATEWAY_SOCKET`` set, the Scheduler serves the
``GatewayControl`` gRPC service of
``proto/swarmpilot/gateway/v1/control.proto`` on that Unix socket. The Rust
gateway then serves the public port: it watches the routing state built
here (instances, strategy, adapter manifests, predictor model keys) and
reports every request it finished, which this module turns into what the
Python proxy's result callback does: instance statistics, throughput for
the Planner, training samples for the predictor, and arrival counts. An
instance the gateway cannot reach is handled as the Python worker queue
handles it.

The state is rebuilt every :data:`POLL_S` seconds and sent only when it
changed, so the gateway follows registrations, drains, health changes,
strategy switches and new models without hooks in every endpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import grpc
import numpy as np
from loguru import logger

from swarmpilot.gateway.v1 import control_pb2 as pb, control_pb2_grpc as pb_grpc
from swarmpilot.scheduler.models.status import InstanceStatus

PROTOCOL_VERSION = 1
POLL_S = 0.2

_STATUS = {
    InstanceStatus.ACTIVE: pb.Instance.ACTIVE,
    InstanceStatus.DRAINING: pb.Instance.DRAINING,
    InstanceStatus.INITIALIZING: pb.Instance.INITIALIZING,
    InstanceStatus.REMOVING: pb.Instance.REMOVING,
}

#: ``RandomState(42).normal(size=2000)``: the draws behind every
#: experiment-mode prediction (``predictor/utils/experiment.py``).
EXPERIMENT_NORMALS = np.random.RandomState(42).normal(size=2000).tolist()


@dataclass
class InstanceView:
    """What the gateway needs to know about one registered instance."""

    instance_id: str
    endpoint: str
    model_id: str
    platform_info: dict[str, str]
    status: Any
    max_in_flight: int
    servers: int
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class Sources:
    """Where the state comes from and where reports go.

    The Scheduler wires these to its globals; tests pass fakes.
    """

    instances: Callable[[], Awaitable[list[InstanceView]]]
    strategy: Callable[[], str]
    model_id: Callable[[], str]
    proxy: Callable[[], pb.ProxySettings]
    manifest: Callable[[str], dict[str, Any]]
    storage_dir: Callable[[], str]
    sketch_quantiles: Callable[[], list[float]]
    model_key: Callable[[str, dict[str, str], str], str]
    hardware_features: Callable[[dict[str, str]], dict[str, float]]
    preprocessor_rules: Callable[[str], bool]
    style_paths: Callable[[], dict[str, str]]
    on_completion: Callable[[pb.Completion], Awaitable[None]]
    on_arrivals: Callable[[int, dict[str, int]], None]
    on_unreachable: Callable[[str, str], Awaitable[None]]


def platform_key(platform_info: dict[str, str]) -> str:
    """Group key of the predictor client (``json.dumps``, sorted keys)."""
    return json.dumps(platform_info, sort_keys=True)


def _is_experiment(platform_info: dict[str, str]) -> bool:
    return all(
        platform_info.get(k) == "exp"
        for k in ("software_name", "software_version", "hardware_name")
    )


async def build_state(src: Sources, version: int) -> pb.State:
    """Build the gateway's state from the Scheduler's.

    Args:
        src: State sources.
        version: Version number to stamp on it.

    Returns:
        The state message.
    """
    instances = await src.instances()
    model_id = instances[0].model_id if instances else src.model_id()
    state = pb.State(
        version=version,
        protocol_version=PROTOCOL_VERSION,
        model_id=model_id or "",
        strategy=src.strategy(),
        proxy=src.proxy(),
        api_style_paths=src.style_paths(),
    )
    softwares = {""}
    platforms: dict[str, dict[str, str]] = {}
    for inst in instances:
        key = platform_key(inst.platform_info)
        platforms.setdefault(key, inst.platform_info)
        softwares.add(inst.platform_info.get("software_name") or "")
        state.instances.append(
            pb.Instance(
                instance_id=inst.instance_id,
                endpoint=inst.endpoint,
                model_id=inst.model_id,
                platform_info=inst.platform_info,
                status=_STATUS.get(inst.status, pb.Instance.UNHEALTHY),
                max_in_flight=max(1, inst.max_in_flight),
                servers=max(1, inst.servers),
                extra_headers=inst.extra_headers,
                platform_key=key,
            )
        )
    for software in sorted(softwares):
        state.manifests[software] = json.dumps(
            src.manifest(software), sort_keys=True
        )
    predictor = pb.PredictorSettings(
        storage_dir=src.storage_dir(),
        sketch_quantiles=src.sketch_quantiles(),
        preprocessor_rules=bool(model_id) and src.preprocessor_rules(model_id),
        experiment_normals=EXPERIMENT_NORMALS,
    )
    for key, info in platforms.items():
        try:
            hardware = src.hardware_features(info)
        except (TypeError, ValueError) as e:
            logger.debug(f"[GATEWAY] no hardware features for {key}: {e}")
            hardware = {}
        predictor.platforms.append(
            pb.Platform(
                platform_key=key,
                quantile_model_key=_safe_key(src, model_id, info, "quantile"),
                expect_error_model_key=_safe_key(
                    src, model_id, info, "expect_error"
                ),
                hardware_features=hardware,
                experiment=_is_experiment(info),
            )
        )
    state.predictor.CopyFrom(predictor)
    return state


def _safe_key(
    src: Sources, model_id: str, info: dict[str, str], ptype: str
) -> str:
    try:
        return src.model_key(model_id, info, ptype)
    except (KeyError, TypeError, ValueError):
        return ""


def _fingerprint(state: pb.State) -> bytes:
    copy = pb.State()
    copy.CopyFrom(state)
    copy.version = 0
    return copy.SerializeToString(deterministic=True)


class GatewayControlService(pb_grpc.GatewayControlServicer):
    """The gRPC service the gateway connects to."""

    def __init__(self, src: Sources, poll_s: float = POLL_S):
        """Create the service.

        Args:
            src: State sources and report sinks.
            poll_s: How often the state is rebuilt to look for changes.
        """
        self._src = src
        self._poll_s = poll_s
        self._version = 0
        self._last: bytes | None = None

    async def WatchState(  # noqa: N802 - gRPC method name
        self, request: pb.WatchStateRequest, context: Any
    ) -> AsyncIterator[pb.State]:
        """Send the state now and whenever it changes."""
        logger.info(
            f"[GATEWAY] {request.gateway_id or 'gateway'} connected "
            f"(protocol {request.protocol_version})"
        )
        sent: bytes | None = None
        while True:
            state = await build_state(self._src, 0)
            fp = _fingerprint(state)
            if fp != sent:
                if fp != self._last:
                    self._version += 1
                    self._last = fp
                state.version = self._version
                sent = fp
                yield state
            await asyncio.sleep(self._poll_s)

    async def ReportCompletions(  # noqa: N802 - gRPC method name
        self, request_iterator: AsyncIterator[pb.CompletionBatch], context: Any
    ) -> pb.ReportAck:
        """Take finished requests for as long as the gateway sends them."""
        async for batch in request_iterator:
            if batch.arrivals or batch.calls_from:
                self._src.on_arrivals(batch.arrivals, dict(batch.calls_from))
            for completion in batch.completions:
                try:
                    await self._src.on_completion(completion)
                except Exception as e:  # one bad record must not stop the rest
                    logger.opt(exception=True).warning(
                        f"[GATEWAY] completion {completion.task_id} failed: {e}"
                    )
        return pb.ReportAck()

    async def ReportUnreachable(  # noqa: N802 - gRPC method name
        self, request: pb.UnreachableReport, context: Any
    ) -> pb.ReportAck:
        """Handle an instance the gateway could not connect to."""
        await self._src.on_unreachable(request.instance_id, request.error)
        return pb.ReportAck()


async def start_server(src: Sources, socket_path: str) -> grpc.aio.Server:
    """Serve the control channel on a Unix socket.

    Args:
        src: State sources and report sinks.
        socket_path: Path of the socket; a stale file there is replaced.

    Returns:
        The started server; stop it with ``await server.stop(grace)``.
    """
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
    server = grpc.aio.server()
    pb_grpc.add_GatewayControlServicer_to_server(
        GatewayControlService(src), server
    )
    server.add_insecure_port(f"unix:{path}")
    await server.start()
    # Only this user may talk to the Scheduler's control plane.
    os.chmod(path, 0o600)
    logger.info(f"[GATEWAY] control channel on unix:{path}")
    return server
