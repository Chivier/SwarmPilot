"""Unit tests for Model Pull API routes.

Tests follow TDD principle - written before implementation.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory."""
    data_dir = tmp_path / "models"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def app(temp_data_dir):
    """Create a FastAPI app with model pull routes."""
    from src.api.routes import models as models_module
    from src.api.routes import tasks as tasks_module
    from src.services.model_storage import ModelStorageService
    from src.services.task_tracking import TaskTrackingService

    app = FastAPI()
    # Include tasks router FIRST to avoid {model_id} catching "tasks"
    app.include_router(tasks_module.router, prefix="/v1/models/tasks")
    app.include_router(models_module.router, prefix="/v1/models")

    model_service = ModelStorageService(data_dir=temp_data_dir)
    task_service = TaskTrackingService()

    # Set services on both routers
    models_module.set_model_storage_service(model_service)
    models_module.set_task_tracking_service(task_service)
    tasks_module.set_task_tracking_service(task_service)

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestModelPullEndpoint:
    """Tests for POST /v1/models/pull - EDI-45."""

    def test_pull_returns_202(self, client):
        """Test that pull endpoint returns 202 Accepted."""
        response = client.post(
            "/v1/models/pull",
            json={
                "name": "llama-3.1-8b",
                "type": "llm",
                "source": {
                    "repo": "meta-llama/Llama-3.1-8B",
                },
            },
        )
        assert response.status_code == 202

    def test_pull_returns_model_response(self, client):
        """Test that pull returns proper response format."""
        response = client.post(
            "/v1/models/pull",
            json={
                "name": "test-model",
                "type": "llm",
                "source": {"repo": "test/model"},
            },
        )
        data = response.json()

        assert "model_id" in data
        assert data["model_id"].startswith("model_")
        assert "task_id" in data
        assert data["task_id"].startswith("task_")
        assert data["name"] == "test-model"
        assert data["type"] == "llm"
        assert data["status"] == "pulling"
        assert "message" in data

    def test_pull_requires_name(self, client):
        """Test that name is required."""
        response = client.post(
            "/v1/models/pull",
            json={
                "type": "llm",
                "source": {"repo": "test/model"},
            },
        )
        assert response.status_code == 422

    def test_pull_requires_type(self, client):
        """Test that type is required."""
        response = client.post(
            "/v1/models/pull",
            json={
                "name": "test-model",
                "source": {"repo": "test/model"},
            },
        )
        assert response.status_code == 422

    def test_pull_requires_source(self, client):
        """Test that source is required."""
        response = client.post(
            "/v1/models/pull",
            json={
                "name": "test-model",
                "type": "llm",
            },
        )
        assert response.status_code == 422

    def test_pull_requires_repo_in_source(self, client):
        """Test that source requires repo."""
        response = client.post(
            "/v1/models/pull",
            json={
                "name": "test-model",
                "type": "llm",
                "source": {},
            },
        )
        assert response.status_code == 422

    def test_pull_validates_model_type(self, client):
        """Test that model type is validated."""
        response = client.post(
            "/v1/models/pull",
            json={
                "name": "test-model",
                "type": "invalid_type",
                "source": {"repo": "test/model"},
            },
        )
        assert response.status_code == 422

    def test_pull_accepts_optional_revision(self, client):
        """Test that optional revision is accepted."""
        response = client.post(
            "/v1/models/pull",
            json={
                "name": "test-model",
                "type": "llm",
                "source": {
                    "repo": "test/model",
                    "revision": "v1.0",
                },
            },
        )
        assert response.status_code == 202

    def test_pull_accepts_optional_endpoint(self, client):
        """Test that optional endpoint is accepted."""
        response = client.post(
            "/v1/models/pull",
            json={
                "name": "test-model",
                "type": "llm",
                "source": {
                    "repo": "test/model",
                    "endpoint": "https://custom-hf.example.com",
                },
            },
        )
        assert response.status_code == 202

    def test_pull_accepts_optional_token(self, client):
        """Test that optional token is accepted (for private repos)."""
        response = client.post(
            "/v1/models/pull",
            json={
                "name": "test-model",
                "type": "llm",
                "source": {
                    "repo": "test/private-model",
                    "token": "hf_secret_token",
                },
            },
        )
        assert response.status_code == 202

    def test_pull_creates_model_entry(self, client, temp_data_dir):
        """Test that pull creates a model entry in storage."""
        response = client.post(
            "/v1/models/pull",
            json={
                "name": "test-model",
                "type": "llm",
                "source": {"repo": "test/model"},
            },
        )
        data = response.json()
        model_id = data["model_id"]

        # Verify model was created
        model_response = client.get(f"/v1/models/{model_id}")
        assert model_response.status_code == 200

        model_data = model_response.json()
        assert model_data["name"] == "test-model"
        assert model_data["status"] == "pulling"


class TestTaskProgressEndpoint:
    """Tests for GET /v1/models/tasks/{task_id} - EDI-49."""

    def test_task_returns_200(self, client):
        """Test that task endpoint returns 200 for existing task."""
        # First create a task via pull
        pull_response = client.post(
            "/v1/models/pull",
            json={
                "name": "test-model",
                "type": "llm",
                "source": {"repo": "test/model"},
            },
        )
        task_id = pull_response.json()["task_id"]

        # Then get task progress
        response = client.get(f"/v1/models/tasks/{task_id}")
        assert response.status_code == 200

    def test_task_returns_progress_info(self, client):
        """Test that task returns progress information."""
        # Create a task
        pull_response = client.post(
            "/v1/models/pull",
            json={
                "name": "test-model",
                "type": "llm",
                "source": {"repo": "test/model"},
            },
        )
        task_id = pull_response.json()["task_id"]

        # Get task progress
        response = client.get(f"/v1/models/tasks/{task_id}")
        data = response.json()

        assert data["task_id"] == task_id
        assert "model_id" in data
        assert data["type"] == "llm"
        assert data["operation"] == "pull"
        assert "status" in data
        assert "progress_percent" in data
        assert "current_step" in data
        assert "bytes_completed" in data
        assert "bytes_total" in data

    def test_task_not_found(self, client):
        """Test 404 for non-existent task."""
        response = client.get("/v1/models/tasks/task_nonexistent")
        assert response.status_code == 404

    def test_task_error_response(self, client):
        """Test error response format."""
        response = client.get("/v1/models/tasks/task_nonexistent")
        data = response.json()

        assert "detail" in data
        assert data["detail"]["error"] == "task_not_found"


class TestTaskListEndpoint:
    """Tests for GET /v1/models/tasks - list all tasks."""

    def test_list_tasks_returns_200(self, client):
        """Test that list tasks endpoint returns 200."""
        response = client.get("/v1/models/tasks")
        assert response.status_code == 200

    def test_list_tasks_returns_array(self, client):
        """Test that list tasks returns array format."""
        response = client.get("/v1/models/tasks")
        data = response.json()

        assert "tasks" in data
        assert isinstance(data["tasks"], list)

    def test_list_tasks_includes_created_tasks(self, client):
        """Test that created tasks appear in list."""
        # Create a task
        client.post(
            "/v1/models/pull",
            json={
                "name": "test-model",
                "type": "llm",
                "source": {"repo": "test/model"},
            },
        )

        response = client.get("/v1/models/tasks")
        data = response.json()

        assert len(data["tasks"]) >= 1

    def test_list_tasks_filter_by_status(self, client):
        """Test filtering tasks by status."""
        # Create a task (will be in pending/pulling status)
        client.post(
            "/v1/models/pull",
            json={
                "name": "test-model",
                "type": "llm",
                "source": {"repo": "test/model"},
            },
        )

        response = client.get("/v1/models/tasks?status=pulling")
        data = response.json()

        for task in data["tasks"]:
            assert task["status"] == "pulling"
