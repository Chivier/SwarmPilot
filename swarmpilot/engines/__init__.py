"""Pluggable inference engines and cluster backends.

See :mod:`swarmpilot.engines.base` for the interfaces and
:mod:`swarmpilot.engines.registry` for discovery.
"""

from swarmpilot.engines.base import (
    PORT,
    Capacity,
    ClusterBackend,
    ClusterInstance,
    EngineAdapter,
    EngineRequest,
    EngineResponse,
    LaunchSpec,
    RouteSpec,
    RuntimeMetrics,
    WorkloadSpec,
    WorkloadType,
    major_minor,
)
from swarmpilot.engines.registry import (
    adapter_names,
    find_adapter,
    get_adapter,
    register_adapter,
    unregister_adapter,
)

__all__ = [
    "PORT",
    "Capacity",
    "ClusterBackend",
    "ClusterInstance",
    "EngineAdapter",
    "EngineRequest",
    "EngineResponse",
    "LaunchSpec",
    "RouteSpec",
    "RuntimeMetrics",
    "WorkloadSpec",
    "WorkloadType",
    "adapter_names",
    "find_adapter",
    "get_adapter",
    "major_minor",
    "register_adapter",
    "unregister_adapter",
]
