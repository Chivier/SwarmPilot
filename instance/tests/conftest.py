"""
Pytest configuration and shared fixtures for all tests.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, MagicMock
import pytest
from fastapi.testclient import TestClient

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models import Task, TaskStatus, InstanceStatus, ModelInfo, ModelRegistryEntry
from src.config import Config


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def sample_config(monkeypatch):
    """Fixture providing a test configuration."""
    monkeypatch.setenv("INSTANCE_ID", "test-instance")
    monkeypatch.setenv("INSTANCE_PORT", "8000")
    monkeypatch.setenv("BASE_DIR", "/tmp/test_base")
    monkeypatch.setenv("DOCKERS_DIR", "test_dockers")
    monkeypatch.setenv("REGISTRY_PATH", "test_registry.yaml")
    return Config()


@pytest.fixture
def mock_config(monkeypatch):
    """Fixture providing a mock Config object."""
    config = Mock(spec=Config)
    config.instance_id = "test-instance"
    config.instance_port = 8000
    config.model_port = 9000
    config.base_dir = Path("/tmp/test_base")
    config.dockers_dir = Path("/tmp/test_base/test_dockers")
    config.registry_path = Path("/tmp/test_base/test_registry.yaml")
    config.log_level = "INFO"
    config.log_dir = "logs"
    config.enable_json_logs = False
    config.get_model_directory = Mock(return_value=Path("/tmp/test_base/test_dockers/test_model"))
    config.get_model_container_name = Mock(return_value="test-instance-test_model")
    return config


# ============================================================================
# Model and Task Fixtures
# ============================================================================

@pytest.fixture
def sample_task():
    """Fixture providing a sample Task object."""
    return Task(
        task_id="task-123",
        model_id="test-model",
        task_input={"prompt": "Hello, world!"},
        status=TaskStatus.QUEUED,
    )


@pytest.fixture
def sample_running_task():
    """Fixture providing a running Task object."""
    task = Task(
        task_id="task-running",
        model_id="test-model",
        task_input={"prompt": "Running task"},
        status=TaskStatus.RUNNING,
    )
    task.started_at = datetime.now(timezone.utc).isoformat() + "Z"
    return task


@pytest.fixture
def sample_completed_task():
    """Fixture providing a completed Task object."""
    task = Task(
        task_id="task-completed",
        model_id="test-model",
        task_input={"prompt": "Completed task"},
        status=TaskStatus.COMPLETED,
    )
    task.started_at = datetime.now(timezone.utc).isoformat() + "Z"
    task.completed_at = datetime.now(timezone.utc).isoformat() + "Z"
    task.result = {"output": "Task completed successfully"}
    return task


@pytest.fixture
def sample_failed_task():
    """Fixture providing a failed Task object."""
    task = Task(
        task_id="task-failed",
        model_id="test-model",
        task_input={"prompt": "Failed task"},
        status=TaskStatus.FAILED,
    )
    task.started_at = datetime.now(timezone.utc).isoformat() + "Z"
    task.completed_at = datetime.now(timezone.utc).isoformat() + "Z"
    task.error = "Task failed due to timeout"
    return task


@pytest.fixture
def sample_model_info():
    """Fixture providing a sample ModelInfo object."""
    return ModelInfo(
        model_id="test-model",
        parameters={"temperature": 0.7, "max_tokens": 100},
        started_at=datetime.now(timezone.utc).isoformat() + "Z",
    )


@pytest.fixture
def sample_registry_entry():
    """Fixture providing a sample ModelRegistryEntry."""
    return ModelRegistryEntry(
        model_id="test-model",
        name="Test Model",
        directory="test_model",
        resource_requirements={},
    )


# ============================================================================
# Registry Data Fixtures
# ============================================================================

@pytest.fixture
def sample_registry_data():
    """Fixture providing sample registry data."""
    return {
        "models": [
            {
                "model_id": "test-model",
                "name": "Test Model",
                "directory": "test_model",
                "resource_requirements": {},
            },
            {
                "model_id": "another-model",
                "name": "Another Test Model",
                "directory": "another_model",
                "resource_requirements": {},
            },
        ]
    }


@pytest.fixture
def invalid_registry_data():
    """Fixture providing invalid registry data (missing 'models' key)."""
    return {
        "invalid_key": [
            {
                "model_id": "test-model",
                "description": "A test model",
                "directory": "test_model",
            }
        ]
    }


@pytest.fixture
def sample_registry_yaml():
    """Fixture providing sample registry YAML content."""
    return """
models:
  - model_id: test-model
    name: Test Model
    directory: test_model
    resource_requirements: {}
  - model_id: another-model
    name: Another Test Model
    directory: another_model
    resource_requirements: {}
"""


# ============================================================================
# Mock Component Fixtures
# ============================================================================

@pytest.fixture
def mock_model_registry():
    """Fixture providing a mock ModelRegistry."""
    registry = Mock()
    registry.get_model = Mock(return_value=ModelRegistryEntry(
        model_id="test-model",
        name="Test Model",
        directory="test_model",
        resource_requirements={}
    ))
    registry.list_models = Mock(return_value=[
        ModelRegistryEntry(
            model_id="test-model",
            name="Test Model",
            directory="test_model",
            resource_requirements={}
        )
    ])
    registry.model_exists = Mock(return_value=True)
    registry.get_model_directory = Mock(return_value=Path("/tmp/test_base/test_dockers/test_model"))
    registry.reload = Mock()
    return registry


@pytest.fixture
def mock_docker_manager():
    """Fixture providing a mock DockerManager."""
    manager = Mock()
    manager.start_model = AsyncMock(return_value=ModelInfo(
        model_id="test-model",
        parameters={},
        started_at=datetime.now(timezone.utc).isoformat() + "Z",
    ))
    manager.stop_model = AsyncMock(return_value="test-model")
    manager.get_current_model = AsyncMock(return_value=None)
    manager.is_model_running = AsyncMock(return_value=False)
    manager.check_model_health = AsyncMock(return_value=True)
    manager.invoke_inference = AsyncMock(return_value={"output": "Inference result"})
    manager.close = AsyncMock()
    return manager


@pytest.fixture
def mock_task_queue():
    """Fixture providing a mock TaskQueue."""
    queue = Mock()
    queue.submit_task = AsyncMock(return_value=1)
    queue.get_task = AsyncMock(return_value=None)
    queue.list_tasks = AsyncMock(return_value=[])
    queue.delete_task = AsyncMock(return_value=True)
    queue.get_queue_stats = AsyncMock(return_value={
        "total": 0,
        "queued": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
    })
    queue.clear_all_tasks = AsyncMock(return_value={
        "queued": 0,
        "completed": 0,
        "failed": 0,
        "total": 0,
    })
    queue.stop_processing = AsyncMock()
    return queue


@pytest.fixture
def mock_scheduler_client():
    """Fixture providing a mock SchedulerClient."""
    client = Mock()
    client.is_enabled = False  # Disabled by default to simplify tests
    client.scheduler_url = None
    client._registered = False
    client.register_instance = AsyncMock(return_value=True)
    client.deregister_instance = AsyncMock(return_value=True)
    client.drain_instance = AsyncMock(return_value={
        "success": True,
        "status": "draining",
        "pending_tasks": 0
    })
    return client


# ============================================================================
# HTTP Client Fixtures
# ============================================================================

@pytest.fixture
def mock_httpx_client():
    """Fixture providing a mock httpx AsyncClient."""
    client = AsyncMock()

    # Mock health check response
    health_response = AsyncMock()
    health_response.status_code = 200
    health_response.json = Mock(return_value={"status": "healthy"})

    # Mock inference response
    inference_response = AsyncMock()
    inference_response.status_code = 200
    inference_response.json = Mock(return_value={"output": "Inference result"})

    client.get = AsyncMock(return_value=health_response)
    client.post = AsyncMock(return_value=inference_response)
    client.aclose = AsyncMock()

    return client


# ============================================================================
# API Test Client Fixtures
# ============================================================================

@pytest.fixture
def api_client(mock_docker_manager, mock_task_queue, mock_model_registry, mock_scheduler_client, monkeypatch):
    """Fixture providing a FastAPI TestClient with mocked dependencies."""
    # Mock the global instances
    monkeypatch.setattr("src.api.get_docker_manager", lambda: mock_docker_manager)
    monkeypatch.setattr("src.api.get_task_queue", lambda: mock_task_queue)
    monkeypatch.setattr("src.api.get_registry", lambda: mock_model_registry)
    monkeypatch.setattr("src.api.get_scheduler_client", lambda: mock_scheduler_client)

    # Import API after mocking
    from src.api import app

    return TestClient(app)


# ============================================================================
# File System Fixtures
# ============================================================================

@pytest.fixture
def temp_registry_file(tmp_path, sample_registry_yaml):
    """Fixture providing a temporary registry file."""
    registry_file = tmp_path / "test_registry.yaml"
    registry_file.write_text(sample_registry_yaml)
    return registry_file


@pytest.fixture
def temp_model_directory(tmp_path):
    """Fixture providing a temporary model directory with Dockerfile."""
    model_dir = tmp_path / "test_dockers" / "test_model"
    model_dir.mkdir(parents=True)

    # Create a sample Dockerfile
    dockerfile = model_dir / "Dockerfile"
    dockerfile.write_text("""
FROM python:3.11-slim

WORKDIR /app

EXPOSE 8000

CMD ["python", "-m", "http.server", "8000"]
""")

    return model_dir


# ============================================================================
# Async Event Loop Fixtures
# ============================================================================

@pytest.fixture
def event_loop():
    """Fixture providing an event loop for async tests."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # This ensures clean state between tests
    import src.model_registry
    import src.task_queue
    import src.docker_manager
    import src.api

    # Reset module-level singleton variables
    src.model_registry._registry_instance = None
    src.task_queue._task_queue_instance = None
    src.docker_manager._docker_manager_instance = None

    # Reset API-level restart operations
    if hasattr(src.api, '_restart_operations'):
        src.api._restart_operations.clear()

    yield

    # Cleanup after test
    src.model_registry._registry_instance = None
    src.task_queue._task_queue_instance = None
    src.docker_manager._docker_manager_instance = None

    # Clear restart operations
    if hasattr(src.api, '_restart_operations'):
        src.api._restart_operations.clear()
