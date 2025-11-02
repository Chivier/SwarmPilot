"""Shared test fixtures for all test modules."""

import pytest
from typing import Dict, List


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
def sample_deployment_input(sample_instances, sample_planner_input) -> Dict:
    """Sample DeploymentInput data."""
    return {
        "instances": sample_instances,
        "planner_input": sample_planner_input
    }


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
        }
    }
