"""Contract checks every engine adapter must pass.

Run them for all installed adapters with
``uv run pytest tests/engines/test_contract.py``. A plugin package can run
the same checks on its own adapter::

    from swarmpilot.engines.contract import check_adapter


    def test_my_adapter_contract():
        assert check_adapter(MyAdapter()) == []
"""

from __future__ import annotations

import math
import shlex

from pydantic import BaseModel, ValidationError

from swarmpilot.engines.base import (
    PORT,
    Capacity,
    EngineAdapter,
    LaunchSpec,
    RouteSpec,
    WorkloadSpec,
)

_HOSTILE_MODEL_ID = "org/model; touch /tmp/pwned #$(id)`id`"
_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


def _check_launch(adapter: EngineAdapter) -> list[str]:
    problems = []
    for gpus in (1, 2):
        spec = adapter.launch_spec(_HOSTILE_MODEL_ID, gpus=gpus)
        if not isinstance(spec, LaunchSpec):
            return [f"launch_spec() returned {type(spec).__name__}"]
        if not spec.argv or not all(isinstance(a, str) for a in spec.argv):
            problems.append("launch_spec().argv must be a non-empty str tuple")
            continue
        if spec.gpus != gpus:
            problems.append(f"launch_spec(gpus={gpus}).gpus is {spec.gpus}")
        if _HOSTILE_MODEL_ID not in spec.argv:
            problems.append("model_id must be passed as one argv element")
        if not any(PORT in a for a in spec.argv) and PORT not in "".join(
            spec.env.values()
        ):
            problems.append("launch spec never uses the PORT placeholder")
        # The shell rendering must run exactly argv, whatever the model id.
        rendered = shlex.split(spec.render_shell(port_expr="12345"))
        if rendered != spec.render_argv(12345):
            problems.append("render_shell() does not round-trip to argv")
    extra = adapter.launch_spec("org/model", extra_args=("--flag", "a b"))
    if tuple(extra.argv[-2:]) != ("--flag", "a b"):
        problems.append("extra_args must be appended to argv unchanged")
    return problems


def _check_routes(adapter: EngineAdapter) -> list[str]:
    problems = []
    routes = adapter.routes()
    if not routes:
        return ["routes() is empty"]
    paths = [r.path for r in routes]
    if len(set(paths)) != len(paths):
        problems.append(f"duplicate route paths: {paths}")
    for r in routes:
        if not isinstance(r, RouteSpec):
            problems.append(f"route {r!r} is not a RouteSpec")
            continue
        if r.path.startswith("/") or not r.path:
            problems.append(f"route path {r.path!r} must not start with '/'")
        if not set(r.methods) <= _METHODS:
            problems.append(f"route {r.path} has bad methods {r.methods}")
        if r.request_model is not None and not (
            isinstance(r.request_model, type)
            and issubclass(r.request_model, BaseModel)
        ):
            problems.append(f"route {r.path} request_model is not pydantic")
        if adapter.route("/" + r.path) != r:
            problems.append(f"route('/{r.path}') does not find the route")
    return problems


def _check_features(adapter: EngineAdapter, spec: WorkloadSpec) -> list[str]:
    problems = []
    examples = adapter.example_requests()
    with_model = {r.path for r in adapter.routes() if r.request_model}
    missing = with_model - {p for p, _ in examples}
    if missing:
        problems.append(f"no example_requests() for routes {sorted(missing)}")
    for path, body in examples:
        route = adapter.route(path)
        if route is None:
            problems.append(f"example for undeclared route {path}")
            continue
        if route.request_model is not None:
            try:
                route.request_model.model_validate(body)
            except ValidationError as e:
                problems.append(f"example for {path} fails its schema: {e}")
        for b in (body, {}):
            feats = adapter.features(path, b)
            if set(feats) != set(spec.feature_names):
                problems.append(
                    f"features({path}) keys {sorted(feats)} != "
                    f"{sorted(spec.feature_names)}"
                )
            elif not all(
                isinstance(v, int | float) and math.isfinite(v)
                for v in feats.values()
            ):
                problems.append(f"features({path}) has non-finite values")
    return problems


def check_adapter(adapter: EngineAdapter) -> list[str]:
    """Return every contract violation of ``adapter`` (empty when it passes).

    Args:
        adapter: Adapter instance to check.

    Returns:
        Human-readable problems.
    """
    problems: list[str] = []
    name = getattr(adapter, "name", None)
    if not isinstance(name, str) or not name:
        problems.append("adapter.name must be a non-empty string")
    spec = getattr(adapter, "workload", None)
    if not isinstance(spec, WorkloadSpec):
        return [*problems, "adapter.workload must be a WorkloadSpec"]
    if not spec.feature_names:
        problems.append("workload.feature_names is empty")

    problems += _check_launch(adapter)
    problems += _check_routes(adapter)
    problems += _check_features(adapter, spec)

    cap = adapter.capacity()
    if not isinstance(cap, Capacity) or cap.max_in_flight < 1:
        problems.append("capacity().max_in_flight must be >= 1")

    pinfo = adapter.platform_info("1.2.3rc1", "H100")
    expected = {
        "software_name": name,
        "software_version": "1.2",
        "hardware_name": "H100",
    }
    if pinfo != expected:
        problems.append(f"platform_info() returned {pinfo}, want {expected}")
    return problems
