"""Fixtures for the Python-vs-Rust core parity tests.

The tests need the ``swarmpilot_core`` module (``uv sync --extra core``).
Without it they skip, unless ``SWARMPILOT_REQUIRE_RUST_CORE=1`` (set in
CI), which turns a missing module into a failure.
"""

import os

import pytest

from swarmpilot.scheduler import core


def _module():
    if os.getenv("SWARMPILOT_REQUIRE_RUST_CORE") == "1":
        import swarmpilot_core

        return swarmpilot_core
    return pytest.importorskip("swarmpilot_core")


@pytest.fixture
def rc():
    """The Rust module, with the Python core active."""
    module = _module()
    before = core.core_name()
    core.set_core("python")
    yield module
    core.set_core(before)


@pytest.fixture
def use_core():
    """Switch the active core inside a test; restored afterwards."""
    _module()
    before = core.core_name()
    yield core.set_core
    core.set_core(before)
