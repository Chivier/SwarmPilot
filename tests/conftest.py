"""Shared pytest configuration for all tests.

This conftest provides fixtures and configuration shared across
unit, integration, and performance tests.

PYLET-012: Integration Testing & Validation
"""

import os

import pytest


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--pylet-head",
        action="store",
        default=os.getenv("PYLET_HEAD", "http://localhost:5100"),
        help="PyLet head node address for integration tests",
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires PyLet cluster)",
    )


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires PyLet cluster)",
    )
    config.addinivalue_line(
        "markers",
        "performance: mark test as performance benchmark",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration/performance tests unless explicitly enabled.

    Only tests carrying an explicit ``@pytest.mark.integration``
    or ``@pytest.mark.performance`` decorator are skipped.  The
    check uses ``get_closest_marker`` so that directory names
    (e.g. ``tests/integration/``) do not trigger a false skip.
    """
    if not config.getoption("--run-integration"):
        skip_marker = pytest.mark.skip(
            reason="Need --run-integration flag and PyLet cluster"
        )
        for item in items:
            if item.get_closest_marker(
                "integration"
            ) or item.get_closest_marker("performance"):
                item.add_marker(skip_marker)


@pytest.fixture
def pylet_head_address(request):
    """Get PyLet head address from command line or environment."""
    return request.config.getoption("--pylet-head")
