"""Shared pytest fixtures for scheduler tests.

This module provides reusable fixtures for testing all components of the
scheduler system including registries, clients, and sample data.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Import components
from src.instance_registry import InstanceRegistry

# Import models
from src.model import (
    Instance,
    InstanceQueueBase,
    InstanceQueueProbabilistic,
    Task,
)
from src.predictor_client import Prediction, PredictorClient
from src.task_dispatcher import TaskDispatcher
from src.task_registry import TaskRegistry
from src.websocket_manager import ConnectionManager

# ============================================================================
# Sample Data Fixtures
# ============================================================================


@pytest.fixture
def sample_instance() -> Instance:
    """Create a sample instance for testing."""
    return Instance(
        instance_id="test-instance-1",
        model_id="test-model",
        endpoint="http://localhost:8001",
        platform_info={
            "software_name": "docker",
            "software_version": "20.10",
            "hardware_name": "test-hardware",
        },
    )


@pytest.fixture
def sample_instances() -> list[Instance]:
    """Create multiple sample instances for testing."""
    return [
        Instance(
            instance_id=f"instance-{i}",
            model_id="test-model",
            endpoint=f"http://localhost:800{i}",
            platform_info={
                "software_name": "docker",
                "software_version": "20.10",
                "hardware_name": f"test-hardware-{i}",
            },
        )
        for i in range(1, 4)
    ]


@pytest.fixture
def sample_task() -> Task:
    """Create a sample task for testing."""
    return Task(
        task_id="test-task-1",
        model_id="test-model",
        task_input={"prompt": "test prompt"},
        metadata={"priority": "high"},
    )


@pytest.fixture
def sample_tasks() -> list[Task]:
    """Create multiple sample tasks for testing."""
    return [
        Task(
            task_id=f"task-{i}",
            model_id="test-model",
            task_input={"prompt": f"test prompt {i}"},
            metadata={"priority": "normal"},
        )
        for i in range(1, 4)
    ]


@pytest.fixture
def sample_queue_info() -> InstanceQueueBase:
    """Create sample queue info for testing."""
    return InstanceQueueBase(instance_id="test-instance-1")


@pytest.fixture
def sample_probabilistic_queue() -> InstanceQueueProbabilistic:
    """Create sample probabilistic queue info for testing."""
    return InstanceQueueProbabilistic(
        instance_id="test-instance-1",
        quantiles=[0.5, 0.9, 0.95, 0.99],
        values=[100.0, 200.0, 300.0, 500.0],
    )


@pytest.fixture
def sample_prediction() -> Prediction:
    """Create a sample prediction for testing."""
    return Prediction(
        instance_id="test-instance-1",
        predicted_time_ms=150.0,
        confidence=0.95,
        quantiles={0.5: 100.0, 0.9: 200.0, 0.95: 300.0, 0.99: 500.0},
    )


@pytest.fixture
def sample_predictions() -> list[Prediction]:
    """Create multiple sample predictions for testing."""
    return [
        Prediction(
            instance_id=f"instance-{i}",
            predicted_time_ms=100.0 * i,
            confidence=0.9,
            quantiles={
                0.5: 50.0 * i,
                0.9: 100.0 * i,
                0.95: 150.0 * i,
                0.99: 200.0 * i,
            },
        )
        for i in range(1, 4)
    ]


# ============================================================================
# Registry Fixtures
# ============================================================================


@pytest.fixture
def instance_registry() -> InstanceRegistry:
    """Create a fresh instance registry for testing."""
    return InstanceRegistry()


@pytest.fixture
def instance_registry_with_instances(
    instance_registry: InstanceRegistry, sample_instances: list[Instance]
) -> InstanceRegistry:
    """Create an instance registry pre-populated with instances."""
    for instance in sample_instances:
        instance_registry.register(instance)
    return instance_registry


@pytest.fixture
def task_registry() -> TaskRegistry:
    """Create a fresh task registry for testing."""
    return TaskRegistry()


@pytest.fixture
def task_registry_with_tasks(
    task_registry: TaskRegistry, sample_tasks: list[Task]
) -> TaskRegistry:
    """Create a task registry pre-populated with tasks."""
    for task in sample_tasks:
        task_registry.create_task(
            task_id=task.task_id,
            model_id=task.model_id,
            task_input=task.task_input,
            metadata=task.metadata,
            assigned_instance="instance-1",
        )
    return task_registry


# ============================================================================
# WebSocket Fixtures
# ============================================================================


@pytest.fixture
def websocket_manager() -> ConnectionManager:
    """Create a fresh connection manager for testing."""
    return ConnectionManager()


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = MagicMock()
    ws.send_json = AsyncMock()
    ws.send_text = AsyncMock()
    ws.receive_json = AsyncMock()
    ws.receive_text = AsyncMock()
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def mock_websockets(request):
    """Create multiple mock WebSocket connections.

    Usage: @pytest.mark.parametrize("mock_websockets", [3], indirect=True)
    """
    count = getattr(request, "param", 2)
    return [MagicMock() for _ in range(count)]


# ============================================================================
# Predictor Client Fixtures
# ============================================================================


@pytest.fixture
def mock_predictor_client() -> PredictorClient:
    """Create a mock predictor client for testing."""
    client = MagicMock(spec=PredictorClient)
    client.predict = AsyncMock()
    client.health_check = AsyncMock()
    return client


@pytest.fixture
def predictor_client_with_response(
    mock_predictor_client: PredictorClient, sample_predictions: list[Prediction]
) -> PredictorClient:
    """Create a mock predictor client with preset predictions."""
    mock_predictor_client.predict.return_value = sample_predictions
    mock_predictor_client.health_check.return_value = True
    return mock_predictor_client


# ============================================================================
# Task Dispatcher Fixtures
# ============================================================================


@pytest.fixture
def task_dispatcher(
    task_registry: TaskRegistry,
    instance_registry: InstanceRegistry,
    websocket_manager: ConnectionManager,
) -> TaskDispatcher:
    """Create a task dispatcher with fresh registries."""
    return TaskDispatcher(
        task_registry=task_registry,
        instance_registry=instance_registry,
        websocket_manager=websocket_manager,
        timeout=30.0,
    )


@pytest.fixture
def task_dispatcher_with_data(
    task_registry_with_tasks: TaskRegistry,
    instance_registry_with_instances: InstanceRegistry,
    websocket_manager: ConnectionManager,
) -> TaskDispatcher:
    """Create a task dispatcher with pre-populated registries."""
    return TaskDispatcher(
        task_registry=task_registry_with_tasks,
        instance_registry=instance_registry_with_instances,
        websocket_manager=websocket_manager,
        timeout=30.0,
    )


# ============================================================================
# API Testing Fixtures
# ============================================================================


@pytest.fixture
def test_app():
    """Create a FastAPI test application.

    This fixture imports the app lazily to avoid circular dependencies.
    """
    from src.api import app

    return app


@pytest.fixture
def test_client(test_app):
    """Create a FastAPI test client."""
    from fastapi.testclient import TestClient

    return TestClient(test_app)


@pytest.fixture
def default_platform_info() -> dict[str, str]:
    """Provide default platform info for testing."""
    return {
        "software_name": "docker",
        "software_version": "20.10",
        "hardware_name": "test-hardware",
    }


def make_register_request(
    instance_id: str,
    model_id: str,
    endpoint: str,
    platform_info: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Helper function to create instance registration request data."""
    if platform_info is None:
        platform_info = {
            "software_name": "docker",
            "software_version": "20.10",
            "hardware_name": "test-hardware",
        }
    return {
        "instance_id": instance_id,
        "model_id": model_id,
        "endpoint": endpoint,
        "platform_info": platform_info,
    }


# ============================================================================
# HTTP Mocking Fixtures
# ============================================================================


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx AsyncClient."""
    client = MagicMock()
    client.post = AsyncMock()
    client.get = AsyncMock()
    return client


@pytest.fixture
def successful_task_response():
    """Create a mock successful task execution response."""
    return {
        "success": True,
        "result": {"output": "test output", "tokens": 100},
        "execution_time_ms": 150,
    }


@pytest.fixture
def failed_task_response():
    """Create a mock failed task execution response."""
    return {"success": False, "error": "Task execution failed"}


# ============================================================================
# Utility Fixtures
# ============================================================================


@pytest.fixture
def current_timestamp() -> str:
    """Get current ISO formatted timestamp."""
    return datetime.now(UTC).isoformat()


@pytest.fixture
def freeze_time(monkeypatch):
    """Fixture to freeze time for consistent timestamp testing."""
    fixed_time = datetime(2024, 1, 1, 12, 0, 0)

    class MockDatetime:
        @staticmethod
        def utcnow():
            return fixed_time

        @staticmethod
        def now():
            return fixed_time

    monkeypatch.setattr("src.task_registry.datetime", MockDatetime)
    return fixed_time


@pytest.fixture(autouse=True)
def reset_round_robin_counter():
    """Reset the round-robin counter before each test."""
    from src.scheduler import RoundRobinStrategy

    # Reset the class-level counter if it exists
    if hasattr(RoundRobinStrategy, "_counter"):
        RoundRobinStrategy._counter = 0
    yield
    # Clean up after test
    if hasattr(RoundRobinStrategy, "_counter"):
        RoundRobinStrategy._counter = 0


@pytest.fixture(autouse=True)
def reset_global_registries():
    """Reset global registries before each test to ensure test isolation."""
    import asyncio

    from src.api import instance_registry, task_registry

    # Clear registries before test
    async def clear_registries():
        await task_registry.clear_all()
        await instance_registry.clear_all()

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a new task
            pass  # Can't easily clear in this case
        else:
            loop.run_until_complete(clear_registries())
    except RuntimeError:
        # No event loop, create one
        asyncio.run(clear_registries())

    yield

    # Clear after test as well
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            loop.run_until_complete(clear_registries())
    except RuntimeError:
        asyncio.run(clear_registries())
