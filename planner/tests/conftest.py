"""Shared test fixtures for all test modules."""

import os

import pytest
from typing import Dict, List
from fastapi.testclient import TestClient


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


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless explicitly enabled."""
    if not config.getoption("--run-integration"):
        skip_marker = pytest.mark.skip(
            reason="Need --run-integration flag and PyLet cluster"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_marker)


@pytest.fixture
def sample_planner_input() -> Dict:
    """Sample valid PlannerInput data."""
    return {
        "M": 4,
        "N": 3,
        "B": [
            [10.0, 5.0, 0.0],
            [8.0, 6.0, 4.0],
            [0.0, 10.0, 8.0],
            [6.0, 0.0, 12.0]
        ],
        "initial": [0, 1, 2, 2],
        "a": 0.5,
        "target": [20.0, 30.0, 25.0],
        "algorithm": "simulated_annealing",
        "objective_method": "relative_error",
        "verbose": False,
        "max_iterations": 100
    }


@pytest.fixture
def sample_instances() -> List[Dict]:
    """Sample instance information."""
    return [
        {"endpoint": "http://instance-1:8080", "current_model": "model_0"},
        {"endpoint": "http://instance-2:8080", "current_model": "model_1"},
        {"endpoint": "http://instance-3:8080", "current_model": "model_2"},
        {"endpoint": "http://instance-4:8080", "current_model": "model_2"}
    ]


@pytest.fixture
def mock_instance_responses() -> Dict:
    """Mock HTTP responses from instance API."""
    return {
        "info": {
            "success": True,
            "instance": {
                "instance_id": "test-instance",
                "status": "running",
                "current_model": {
                    "model_id": "model_0",
                    "started_at": "2025-10-31T10:00:00Z",
                    "parameters": {}
                },
                "task_queue": {
                    "total": 0,
                    "queued": 0,
                    "running": 0,
                    "completed": 0,
                    "failed": 0
                }
            }
        },
        "stop": {
            "success": True,
            "message": "Model stopped successfully",
            "model_id": "model_0"
        },
        "start": {
            "success": True,
            "message": "Model started successfully",
            "model_id": "model_1",
            "status": "running"
        },
        "deregister": {
            "success": True,
            "message": "Model deregistered successfully"
        },
        "register": {
            "success": True,
            "message": "Model registered successfully"
        }
    }


@pytest.fixture
def client():
    """Create FastAPI test client."""
    from swarmpilot.planner.api import app
    return TestClient(app)
