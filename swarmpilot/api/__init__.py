"""API styles: serve any client API on any engine.

See :mod:`swarmpilot.api.canonical` for the shared model,
:mod:`swarmpilot.api.base` for the plugin interface and
:mod:`swarmpilot.api.translate` for how the proxy uses them.
"""

from swarmpilot.api.base import ApiStyle, Context, StyleRoute, TranslationError
from swarmpilot.api.registry import (
    find_style,
    get_style,
    register_style,
    style_for_path,
    style_names,
)
from swarmpilot.api.translate import Translation, plan_request

__all__ = [
    "ApiStyle",
    "Context",
    "StyleRoute",
    "Translation",
    "TranslationError",
    "find_style",
    "get_style",
    "plan_request",
    "register_style",
    "style_for_path",
    "style_names",
]
