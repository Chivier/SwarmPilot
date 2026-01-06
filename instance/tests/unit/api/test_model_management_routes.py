"""Unit tests for Model Management API routes.

Tests for model config and deletion endpoints.
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
    """Create a FastAPI app with model management routes."""
    from src.api.routes import models as models_module
    from src.services.model_storage import ModelStorageService
    from src.services.task_tracking import TaskTrackingService

    app = FastAPI()
    app.include_router(models_module.router, prefix="/v1/models")

    model_service = ModelStorageService(data_dir=temp_data_dir)
    task_service = TaskTrackingService()

    models_module.set_model_storage_service(model_service)
    models_module.set_task_tracking_service(task_service)

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
async def sample_model(temp_data_dir):
    """Create a sample model for testing."""
    from src.services.model_storage import ModelStorageService

    service = ModelStorageService(data_dir=temp_data_dir)
    model_id = await service.create_model_entry(
        name="test-model",
        model_type="llm",
        source={"type": "huggingface", "repo": "test/repo"},
    )
    await service.update_model_status(model_id, "ready")
    return model_id


class TestModelConfigEndpoint:
    """Tests for PUT /v1/models/{model_id}/config - EDI-50."""

    @pytest.mark.asyncio
    async def test_set_config_returns_200(self, client, temp_data_dir):
        """Test that setting config returns 200 OK."""
        from src.api.routes import models as models_module
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        models_module.set_model_storage_service(service)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )

        response = client.put(
            f"/v1/models/{model_id}/config",
            json={"tensor_parallel_size": 2},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_set_config_saves_config(self, client, temp_data_dir):
        """Test that config is saved correctly."""
        from src.api.routes import models as models_module
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        models_module.set_model_storage_service(service)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )

        config = {"tensor_parallel_size": 4, "max_model_len": 8192}
        response = client.put(f"/v1/models/{model_id}/config", json=config)
        data = response.json()

        assert data["model_id"] == model_id
        assert data["default_config"]["tensor_parallel_size"] == 4
        assert data["default_config"]["max_model_len"] == 8192

    @pytest.mark.asyncio
    async def test_set_config_response_format(self, client, temp_data_dir):
        """Test config response format."""
        from src.api.routes import models as models_module
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        models_module.set_model_storage_service(service)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )

        response = client.put(
            f"/v1/models/{model_id}/config",
            json={"tensor_parallel_size": 2},
        )
        data = response.json()

        assert "model_id" in data
        assert "default_config" in data
        assert "message" in data

    def test_set_config_model_not_found(self, client):
        """Test 404 for non-existent model."""
        response = client.put(
            "/v1/models/model_nonexistent/config",
            json={"tensor_parallel_size": 2},
        )
        assert response.status_code == 404

    def test_set_config_error_response(self, client):
        """Test error response format."""
        response = client.put(
            "/v1/models/model_nonexistent/config",
            json={"tensor_parallel_size": 2},
        )
        data = response.json()

        assert "detail" in data
        assert data["detail"]["error"] == "model_not_found"


class TestModelConfigDeleteEndpoint:
    """Tests for DELETE /v1/models/{model_id}/config - EDI-50."""

    @pytest.mark.asyncio
    async def test_delete_config_returns_200(self, client, temp_data_dir):
        """Test that deleting config returns 200 OK."""
        from src.api.routes import models as models_module
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        models_module.set_model_storage_service(service)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await service.save_default_config(model_id, {"tensor_parallel_size": 2})

        response = client.delete(f"/v1/models/{model_id}/config")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_config_removes_config(self, client, temp_data_dir):
        """Test that config is deleted."""
        from src.api.routes import models as models_module
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        models_module.set_model_storage_service(service)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await service.save_default_config(model_id, {"tensor_parallel_size": 2})

        client.delete(f"/v1/models/{model_id}/config")

        # Verify config is gone
        config = await service.get_default_config(model_id)
        assert config is None

    def test_delete_config_model_not_found(self, client):
        """Test 404 for non-existent model."""
        response = client.delete("/v1/models/model_nonexistent/config")
        assert response.status_code == 404


class TestModelDeleteEndpoint:
    """Tests for DELETE /v1/models/{model_id} - EDI-51."""

    @pytest.mark.asyncio
    async def test_delete_model_returns_200(self, client, temp_data_dir):
        """Test that deleting model returns 200 OK."""
        from src.api.routes import models as models_module
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        models_module.set_model_storage_service(service)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )

        response = client.delete(f"/v1/models/{model_id}")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_model_response_format(self, client, temp_data_dir):
        """Test delete model response format."""
        from src.api.routes import models as models_module
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        models_module.set_model_storage_service(service)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )

        response = client.delete(f"/v1/models/{model_id}")
        data = response.json()

        assert data["deleted"] is True
        assert data["model_id"] == model_id
        assert "disk_freed_bytes" in data
        assert isinstance(data["disk_freed_bytes"], int)

    @pytest.mark.asyncio
    async def test_delete_model_removes_from_storage(self, client, temp_data_dir):
        """Test that model is removed from storage."""
        from src.api.routes import models as models_module
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        models_module.set_model_storage_service(service)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )

        client.delete(f"/v1/models/{model_id}")

        # Verify model is gone
        model = await service.get_model(model_id)
        assert model is None

    def test_delete_model_not_found(self, client):
        """Test 404 for non-existent model."""
        response = client.delete("/v1/models/model_nonexistent")
        assert response.status_code == 404

    def test_delete_model_error_response(self, client):
        """Test error response format."""
        response = client.delete("/v1/models/model_nonexistent")
        data = response.json()

        assert "detail" in data
        assert data["detail"]["error"] == "model_not_found"

    @pytest.mark.asyncio
    async def test_delete_model_cannot_delete_running(self, client, temp_data_dir):
        """Test that running models cannot be deleted."""
        from src.api.routes import models as models_module
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        models_module.set_model_storage_service(service)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await service.update_model_status(model_id, "running")

        response = client.delete(f"/v1/models/{model_id}")
        assert response.status_code == 409  # Conflict

    @pytest.mark.asyncio
    async def test_delete_model_running_error_message(self, client, temp_data_dir):
        """Test error message for running model deletion."""
        from src.api.routes import models as models_module
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        models_module.set_model_storage_service(service)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await service.update_model_status(model_id, "running")

        response = client.delete(f"/v1/models/{model_id}")
        data = response.json()

        assert data["detail"]["error"] == "model_in_use"
