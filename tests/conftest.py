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
    """Skip integration/performance tests unless explicitly enabled."""
    if not config.getoption("--run-integration"):
        skip_marker = pytest.mark.skip(
            reason="Need --run-integration flag and PyLet cluster"
        )
        for item in items:
            if "integration" in item.keywords or "performance" in item.keywords:
                item.add_marker(skip_marker)


@pytest.fixture
def pylet_head_address(request):
    """Get PyLet head address from command line or environment."""
    return request.config.getoption("--pylet-head")


@pytest.fixture
async def pylet_client(pylet_head_address):
    """Create PyLet client for integration tests."""
    try:
        from planner.src.pylet import PyLetClient
    except ImportError:
        pytest.skip("planner.src.pylet not available")

    client = PyLetClient(head_address=pylet_head_address)
    try:
        await client.connect()
        yield client
    finally:
        await client.close()


@pytest.fixture
async def pylet_wrapper(pylet_head_address):
    """Create PyLet wrapper for integration tests."""
    try:
        from planner.src.pylet import PyLetServiceWrapper
    except ImportError:
        pytest.skip("planner.src.pylet not available")

    wrapper = PyLetServiceWrapper(head_address=pylet_head_address)
    async with wrapper:
        yield wrapper

