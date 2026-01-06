"""Unit tests for Model Lifecycle API routes.

Tests for start, stop, and switch endpoints.
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
    """Create a FastAPI app with model lifecycle routes."""
    from src.api.routes import models as models_module
    from src.services.inference_manager import InferenceManagerService
    from src.services.model_storage import ModelStorageService
    from src.services.task_tracking import TaskTrackingService

    app = FastAPI()
    app.include_router(models_module.router, prefix="/v1/models")

    model_service = ModelStorageService(data_dir=temp_data_dir)
    task_service = TaskTrackingService()
    inference_service = InferenceManagerService(model_storage=model_service)

    models_module.set_model_storage_service(model_service)
    models_module.set_task_tracking_service(task_service)
    models_module.set_inference_manager_service(inference_service)

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestModelStartEndpoint:
    """Tests for POST /v1/models/{model_id}/start - EDI-53."""

    @pytest.mark.asyncio
    async def test_start_returns_200(self, client, temp_data_dir):
        """Test that start endpoint returns 200 OK."""
        from src.api.routes import models as models_module
        from src.services.inference_manager import InferenceManagerService
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        inference = InferenceManagerService(model_storage=service)
        models_module.set_model_storage_service(service)
        models_module.set_inference_manager_service(inference)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await service.update_model_status(model_id, "ready")

        response = client.post(
            f"/v1/models/{model_id}/start",
            json={"gpu_ids": [0]},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_start_response_format(self, client, temp_data_dir):
        """Test start response format."""
        from src.api.routes import models as models_module
        from src.services.inference_manager import InferenceManagerService
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        inference = InferenceManagerService(model_storage=service)
        models_module.set_model_storage_service(service)
        models_module.set_inference_manager_service(inference)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await service.update_model_status(model_id, "ready")

        response = client.post(
            f"/v1/models/{model_id}/start",
            json={"gpu_ids": [0]},
        )
        data = response.json()

        assert data["model_id"] == model_id
        assert "type" in data
        assert "status" in data
        assert "message" in data

    @pytest.mark.asyncio
    async def test_start_with_config(self, client, temp_data_dir):
        """Test start with custom configuration."""
        from src.api.routes import models as models_module
        from src.services.inference_manager import InferenceManagerService
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        inference = InferenceManagerService(model_storage=service)
        models_module.set_model_storage_service(service)
        models_module.set_inference_manager_service(inference)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await service.update_model_status(model_id, "ready")

        response = client.post(
            f"/v1/models/{model_id}/start",
            json={
                "gpu_ids": [0, 1],
                "config": {"tensor_parallel_size": 2},
            },
        )
        assert response.status_code == 200

    def test_start_model_not_found(self, client):
        """Test 404 for non-existent model."""
        response = client.post(
            "/v1/models/model_nonexistent/start",
            json={"gpu_ids": [0]},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_start_model_not_ready(self, client, temp_data_dir):
        """Test 409 for model not in ready state."""
        from src.api.routes import models as models_module
        from src.services.inference_manager import InferenceManagerService
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        inference = InferenceManagerService(model_storage=service)
        models_module.set_model_storage_service(service)
        models_module.set_inference_manager_service(inference)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        # Model is still in 'pulling' state

        response = client.post(
            f"/v1/models/{model_id}/start",
            json={"gpu_ids": [0]},
        )
        assert response.status_code == 409


class TestModelStopEndpoint:
    """Tests for POST /v1/models/{model_id}/stop - EDI-54."""

    @pytest.mark.asyncio
    async def test_stop_returns_200(self, client, temp_data_dir):
        """Test that stop endpoint returns 200 OK."""
        from src.api.routes import models as models_module
        from src.services.inference_manager import InferenceManagerService
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        inference = InferenceManagerService(model_storage=service)
        models_module.set_model_storage_service(service)
        models_module.set_inference_manager_service(inference)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await service.update_model_status(model_id, "ready")

        # Start the model first
        await inference.start_model(model_id=model_id, gpu_ids=[0], config=None)

        response = client.post(
            f"/v1/models/{model_id}/stop",
            json={"force": False},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_stop_response_format(self, client, temp_data_dir):
        """Test stop response format."""
        from src.api.routes import models as models_module
        from src.services.inference_manager import InferenceManagerService
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        inference = InferenceManagerService(model_storage=service)
        models_module.set_model_storage_service(service)
        models_module.set_inference_manager_service(inference)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await service.update_model_status(model_id, "ready")
        await inference.start_model(model_id=model_id, gpu_ids=[0], config=None)

        response = client.post(
            f"/v1/models/{model_id}/stop",
            json={"force": False},
        )
        data = response.json()

        assert data["model_id"] == model_id
        assert "status" in data
        assert "message" in data

    def test_stop_model_not_found(self, client):
        """Test 404 for non-existent model."""
        response = client.post(
            "/v1/models/model_nonexistent/stop",
            json={"force": False},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_stop_model_not_running(self, client, temp_data_dir):
        """Test 409 for model not running."""
        from src.api.routes import models as models_module
        from src.services.inference_manager import InferenceManagerService
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        inference = InferenceManagerService(model_storage=service)
        models_module.set_model_storage_service(service)
        models_module.set_inference_manager_service(inference)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await service.update_model_status(model_id, "ready")

        response = client.post(
            f"/v1/models/{model_id}/stop",
            json={"force": False},
        )
        assert response.status_code == 409


class TestModelSwitchEndpoint:
    """Tests for POST /v1/models/switch - EDI-55."""

    @pytest.mark.asyncio
    async def test_switch_returns_200(self, client, temp_data_dir):
        """Test that switch endpoint returns 200 OK."""
        from src.api.routes import models as models_module
        from src.services.inference_manager import InferenceManagerService
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        inference = InferenceManagerService(model_storage=service)
        models_module.set_model_storage_service(service)
        models_module.set_inference_manager_service(inference)

        model1_id = await service.create_model_entry(
            name="model1", model_type="llm", source={"type": "upload"}
        )
        await service.update_model_status(model1_id, "ready")

        model2_id = await service.create_model_entry(
            name="model2", model_type="llm", source={"type": "upload"}
        )
        await service.update_model_status(model2_id, "ready")

        # Start first model
        await inference.start_model(model_id=model1_id, gpu_ids=[0], config=None)

        response = client.post(
            "/v1/models/switch",
            json={
                "target_model_id": model2_id,
                "gpu_ids": [0],
            },
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_switch_response_format(self, client, temp_data_dir):
        """Test switch response format."""
        from src.api.routes import models as models_module
        from src.services.inference_manager import InferenceManagerService
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        inference = InferenceManagerService(model_storage=service)
        models_module.set_model_storage_service(service)
        models_module.set_inference_manager_service(inference)

        model1_id = await service.create_model_entry(
            name="model1", model_type="llm", source={"type": "upload"}
        )
        await service.update_model_status(model1_id, "ready")

        model2_id = await service.create_model_entry(
            name="model2", model_type="llm", source={"type": "upload"}
        )
        await service.update_model_status(model2_id, "ready")

        await inference.start_model(model_id=model1_id, gpu_ids=[0], config=None)

        response = client.post(
            "/v1/models/switch",
            json={
                "target_model_id": model2_id,
                "gpu_ids": [0],
            },
        )
        data = response.json()

        assert data["current_model_id"] == model2_id
        assert data["previous_model_id"] == model1_id
        assert "status" in data
        assert "message" in data

    def test_switch_target_not_found(self, client):
        """Test 404 for non-existent target model."""
        response = client.post(
            "/v1/models/switch",
            json={
                "target_model_id": "model_nonexistent",
                "gpu_ids": [0],
            },
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_switch_no_active_model(self, client, temp_data_dir):
        """Test switch when no model is currently running."""
        from src.api.routes import models as models_module
        from src.services.inference_manager import InferenceManagerService
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        inference = InferenceManagerService(model_storage=service)
        models_module.set_model_storage_service(service)
        models_module.set_inference_manager_service(inference)

        model_id = await service.create_model_entry(
            name="model1", model_type="llm", source={"type": "upload"}
        )
        await service.update_model_status(model_id, "ready")

        response = client.post(
            "/v1/models/switch",
            json={
                "target_model_id": model_id,
                "gpu_ids": [0],
            },
        )

        data = response.json()
        assert data["current_model_id"] == model_id
        assert data["previous_model_id"] is None
