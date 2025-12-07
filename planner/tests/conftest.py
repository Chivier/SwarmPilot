"""Shared test fixtures for all test modules."""

import pytest
from typing import Dict, List
from fastapi.testclient import TestClient


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
        "planner_input": sample_planner_input,
        "scheduler_url": "http://scheduler:8100"
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
def mock_migration_info_responses() -> Dict:
    """Mock HTTP responses for migration operations."""
    return {
        "original_info": {
            "success": True,
            "instance": {
                "instance_id": "original-instance",
                "status": "running",
                "current_model": {
                    "model_id": "model_0",
                    "started_at": "2025-10-31T10:00:00Z",
                    "parameters": {}
                }
            }
        },
        "target_info": {
            "success": True,
            "instance": {
                "instance_id": "target-instance",
                "status": "running",
                "current_model": {
                    "model_id": "model_1",
                    "started_at": "2025-10-31T10:00:00Z",
                    "parameters": {}
                }
            }
        }
    }


@pytest.fixture
def client():
    """Create FastAPI test client."""
    from src.api import app
    return TestClient(app)


@pytest.fixture
def reset_throughput_state():
    """Reset throughput-related global state before and after each test."""
    from src import api as api_module

    # Save original state
    original_mapping = api_module._stored_model_mapping
    original_reverse = api_module._stored_reverse_mapping
    original_target = api_module._current_target
    original_submitted = api_module._submitted_models.copy() if api_module._submitted_models else set()
    original_running = api_module._auto_optimize_running
    original_deployment_input = api_module._stored_deployment_input
    original_first_data_received = api_module._first_data_received
    original_first_migration_done = api_module._first_migration_done
    original_optimization_timer_start = api_module._optimization_timer_start

    # Save throughput data if it exists
    original_throughput_data = getattr(api_module, '_throughput_data', {}).copy() if hasattr(api_module, '_throughput_data') else {}

    yield

    # Restore original state
    api_module._stored_model_mapping = original_mapping
    api_module._stored_reverse_mapping = original_reverse
    api_module._current_target = original_target
    api_module._submitted_models = original_submitted
    api_module._auto_optimize_running = original_running
    api_module._stored_deployment_input = original_deployment_input
    api_module._first_data_received = original_first_data_received
    api_module._first_migration_done = original_first_migration_done
    api_module._optimization_timer_start = original_optimization_timer_start

    # Restore throughput data if it exists
    if hasattr(api_module, '_throughput_data'):
        api_module._throughput_data = original_throughput_data


@pytest.fixture
def setup_throughput_deployment_state(reset_throughput_state):
    """Set up deployment state with instances for throughput testing."""
    from src.models import DeploymentInput, PlannerInput, InstanceInfo
    from src import api as api_module

    mock_input = DeploymentInput(
        instances=[
            InstanceInfo(endpoint="http://inst1:8080", current_model="model_a"),
            InstanceInfo(endpoint="http://inst2:8080", current_model="model_b"),
            InstanceInfo(endpoint="http://inst3:8080", current_model="model_c"),
        ],
        planner_input=PlannerInput(
            M=3, N=3,
            B=[[10.0, 5.0, 2.0], [5.0, 10.0, 5.0], [2.0, 5.0, 10.0]],
            a=0.5,
            target=[100.0, 200.0, 300.0],
            algorithm="simulated_annealing",
            max_iterations=10,
            verbose=False
        ),
        scheduler_mapping={}
    )
    api_module._stored_deployment_input = mock_input
    api_module._stored_model_mapping = {"model_a": 0, "model_b": 1, "model_c": 2}
    api_module._stored_reverse_mapping = {0: "model_a", 1: "model_b", 2: "model_c"}
    api_module._current_target = [0.0, 0.0, 0.0]

    # Initialize throughput data if it exists
    if hasattr(api_module, '_throughput_data'):
        api_module._throughput_data = {}
