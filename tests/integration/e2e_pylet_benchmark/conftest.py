"""Pytest configuration for E2E PyLet Benchmark tests.

This module provides fixtures and configuration for running the E2E
benchmark tests via pytest.
"""

import os
import pytest
from pathlib import Path


def pytest_addoption(parser):
    """Add custom command-line options for E2E tests."""
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run E2E benchmark tests (requires full service stack)",
    )
    parser.addoption(
        "--e2e-qps",
        type=float,
        default=5.0,
        help="QPS for E2E benchmark test",
    )
    parser.addoption(
        "--e2e-duration",
        type=float,
        default=60.0,
        help="Duration in seconds for E2E benchmark test",
    )
    parser.addoption(
        "--e2e-output-dir",
        type=str,
        default="./e2e_benchmark_results",
        help="Output directory for E2E benchmark results",
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "e2e: mark test as E2E benchmark test"
    )


def pytest_collection_modifyitems(config, items):
    """Skip E2E tests unless --run-e2e flag is provided."""
    if not config.getoption("--run-e2e"):
        skip_e2e = pytest.mark.skip(
            reason="Need --run-e2e flag to run E2E benchmark tests"
        )
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)


@pytest.fixture
def e2e_qps(request) -> float:
    """Get QPS from command line option."""
    return request.config.getoption("--e2e-qps")


@pytest.fixture
def e2e_duration(request) -> float:
    """Get duration from command line option."""
    return request.config.getoption("--e2e-duration")


@pytest.fixture
def e2e_output_dir(request) -> Path:
    """Get output directory from command line option."""
    return Path(request.config.getoption("--e2e-output-dir"))


@pytest.fixture
def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent.parent


@pytest.fixture
def log_dir(tmp_path) -> Path:
    """Create temporary log directory."""
    log_path = tmp_path / "e2e_logs"
    log_path.mkdir(parents=True, exist_ok=True)
    return log_path
