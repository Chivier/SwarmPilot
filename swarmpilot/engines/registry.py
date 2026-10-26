"""Discovery of engine adapters and cluster backends.

Adapters come from two places:

1. The ``swarmpilot.engines`` entry-point group. Any installed package can
   add an engine by declaring, in its ``pyproject.toml``::

       [project.entry-points."swarmpilot.engines"]
       my-engine = "my_package.adapter:MyEngineAdapter"

2. :func:`register_adapter`, for tests and in-process plugins.

Built-in adapters are declared as entry points of ``swarmpilot`` itself and
also listed in :data:`BUILTIN_ADAPTERS`, so they work from a source checkout
that was never installed.
"""

from __future__ import annotations

import threading
from importlib import import_module
from importlib.metadata import entry_points

from loguru import logger

from swarmpilot.engines.base import EngineAdapter

ENTRY_POINT_GROUP = "swarmpilot.engines"

BUILTIN_ADAPTERS: dict[str, str] = {
    "vllm": "swarmpilot.engines.vllm:VLLMAdapter",
    "sglang": "swarmpilot.engines.sglang:SGLangAdapter",
}

_lock = threading.Lock()
_classes: dict[str, type[EngineAdapter]] = {}
_instances: dict[str, EngineAdapter] = {}
_discovered = False


def _load(target: str) -> type[EngineAdapter]:
    module, _, attr = target.partition(":")
    cls = getattr(import_module(module), attr)
    if not (isinstance(cls, type) and issubclass(cls, EngineAdapter)):
        raise TypeError(f"{target} is not an EngineAdapter subclass")
    return cls


def _discover() -> None:
    global _discovered
    if _discovered:
        return
    found: dict[str, str] = dict(BUILTIN_ADAPTERS)
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name in found and found[ep.name] != ep.value:
            # A plugin may replace a built-in on purpose.
            logger.info(f"Engine adapter {ep.name!r} overridden by {ep.value}")
        found[ep.name] = ep.value
    for name, target in found.items():
        if name in _classes:
            continue
        try:
            _classes[name] = _load(target)
        except Exception as e:
            logger.warning(f"Skipping engine adapter {name!r} ({target}): {e}")
    _discovered = True


def register_adapter(
    cls: type[EngineAdapter], name: str | None = None
) -> type[EngineAdapter]:
    """Register an adapter class under ``name`` (default ``cls.name``).

    Usable as a decorator.

    Args:
        cls: Adapter class.
        name: Registry name; defaults to the class's ``name``.

    Returns:
        The class, unchanged.
    """
    key = name or cls.name
    with _lock:
        _classes[key] = cls
        _instances.pop(key, None)
    return cls


def unregister_adapter(name: str) -> None:
    """Remove an adapter registered with :func:`register_adapter`."""
    with _lock:
        _classes.pop(name, None)
        _instances.pop(name, None)


def adapter_names() -> list[str]:
    """Return the names of all available adapters, sorted."""
    with _lock:
        _discover()
        return sorted(_classes)


def get_adapter(name: str) -> EngineAdapter:
    """Return the shared adapter instance for ``name``.

    Args:
        name: Adapter name, e.g. ``"vllm"``.

    Returns:
        The adapter.

    Raises:
        KeyError: If no adapter has that name.
    """
    with _lock:
        _discover()
        if name not in _instances:
            if name not in _classes:
                raise KeyError(
                    f"Unknown engine {name!r}. Available: {sorted(_classes)}"
                )
            _instances[name] = _classes[name]()
        return _instances[name]


def reset_registry() -> None:
    """Forget everything discovered or registered (for tests)."""
    global _discovered
    with _lock:
        _classes.clear()
        _instances.clear()
        _discovered = False
