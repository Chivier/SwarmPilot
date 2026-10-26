"""Find API styles by name or by route.

Styles come from two places:

1. The ``swarmpilot.api_styles`` entry-point group. Any installed package
   can add one::

       [project.entry-points."swarmpilot.api_styles"]
       gemini = "my_pkg.gemini:GeminiStyle"

2. :func:`register_style`, for tests and embedded use.

A plugin may replace a built-in style of the same name.
"""

from __future__ import annotations

import threading
from importlib import import_module
from importlib.metadata import entry_points

from loguru import logger

from swarmpilot.api.base import ApiStyle

ENTRY_POINT_GROUP = "swarmpilot.api_styles"

BUILTIN_STYLES: dict[str, str] = {
    "openai_chat": "swarmpilot.api.styles.openai_chat:OpenAIChatStyle",
    "anthropic_messages": (
        "swarmpilot.api.styles.anthropic_messages:AnthropicMessagesStyle"
    ),
    "openai_responses": (
        "swarmpilot.api.styles.openai_responses:OpenAIResponsesStyle"
    ),
}

_lock = threading.Lock()
_styles: dict[str, ApiStyle] = {}
_by_path: dict[str, ApiStyle] = {}
_discovered = False


def _load(target: str) -> ApiStyle:
    module, _, attr = target.partition(":")
    cls = getattr(import_module(module), attr)
    if not (isinstance(cls, type) and issubclass(cls, ApiStyle)):
        raise TypeError(f"{target} is not an ApiStyle subclass")
    return cls()


def _index() -> None:
    _by_path.clear()
    for style in _styles.values():
        for route in style.routes():
            _by_path[route.path] = style


def _discover() -> None:
    global _discovered
    if _discovered:
        return
    with _lock:
        if _discovered:
            return
        found = dict(BUILTIN_STYLES)
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            if ep.name in found and found[ep.name] != ep.value:
                logger.info(f"API style {ep.name!r} overridden by {ep.value}")
            found[ep.name] = ep.value
        for name, target in found.items():
            if name in _styles:
                continue
            try:
                _styles[name] = _load(target)
            except Exception as e:
                logger.warning(f"Skipping API style {name!r} ({target}): {e}")
        _index()
        _discovered = True


def register_style(style: ApiStyle) -> ApiStyle:
    """Register a style instance under its ``name``."""
    _discover()
    with _lock:
        _styles[style.name] = style
        _index()
    return style


def unregister_style(name: str) -> None:
    """Remove a style."""
    _discover()
    with _lock:
        _styles.pop(name, None)
        _index()


def style_names() -> list[str]:
    """Return the names of all registered styles."""
    _discover()
    return sorted(_styles)


def get_style(name: str) -> ApiStyle:
    """Return the style called ``name``.

    Raises:
        KeyError: No such style.
    """
    _discover()
    return _styles[name]


def find_style(name: str) -> ApiStyle | None:
    """Return the style called ``name``, or None."""
    _discover()
    return _styles.get(name)


def style_for_path(path: str) -> ApiStyle | None:
    """Return the style that owns ``path`` (no leading slash), or None."""
    _discover()
    return _by_path.get(path.lstrip("/"))


def reset_registry() -> None:
    """Forget every style so the next lookup rediscovers them (tests)."""
    global _discovered
    with _lock:
        _styles.clear()
        _by_path.clear()
        _discovered = False
