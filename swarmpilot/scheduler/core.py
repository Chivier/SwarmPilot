"""Which implementation runs the scheduling core: Python or Rust.

The queue sketches, the routing strategies' selection and the feature
extractor grammar exist twice: in Python, and in the Rust crate
``rust/crates/core`` (Python module ``swarmpilot_core``, installed with
``uv sync --extra core``). ``SCHEDULER_CORE`` picks one:

``python`` (default)
    The Python code everywhere.
``rust``
    The Python code calls the Rust functions instead. Decisions are the
    same: the strategies still draw from their own numpy generators and
    pass the draws to Rust, and the golden vectors and differential tests
    under ``tests/golden/`` and ``tests/core/`` check that every number
    matches.

Setting ``rust`` without the module installed fails at import, so a
deployment never runs the Python core while it believes it runs Rust.
"""

from __future__ import annotations

import os
from types import ModuleType

CORES = ("python", "rust")

_rust: ModuleType | None = None


def _load_rust() -> ModuleType:
    try:
        import swarmpilot_core
    except ImportError as e:
        raise RuntimeError(
            "SCHEDULER_CORE=rust needs the swarmpilot_core module; install it "
            "with `uv sync --extra core` (needs a Rust toolchain)"
        ) from e
    return swarmpilot_core


def set_core(name: str) -> None:
    """Switch the scheduling core.

    Args:
        name: "python" or "rust".

    Raises:
        ValueError: For an unknown name.
        RuntimeError: For "rust" when the module is not installed.
    """
    global _rust
    if name not in CORES:
        raise ValueError(f"SCHEDULER_CORE must be one of {CORES}, not {name!r}")
    _rust = _load_rust() if name == "rust" else None


def core_name() -> str:
    """Return the active core's name."""
    return "rust" if _rust is not None else "python"


def rust() -> ModuleType | None:
    """Return the Rust module when the Rust core is active, else None."""
    return _rust


def rust_available() -> bool:
    """Return whether the Rust module can be imported."""
    try:
        _load_rust()
    except RuntimeError:
        return False
    return True


set_core(os.getenv("SCHEDULER_CORE", "python").strip().lower() or "python")
