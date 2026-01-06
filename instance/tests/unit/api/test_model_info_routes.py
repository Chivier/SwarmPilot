"""Unit tests for Model Info API routes.

Tests for GET /v1/model/info endpoint - EDI-56.
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
    """Create a FastAPI app with model info route."""
    from src.api.routes import model_info as model_info_module
    from src.services.inference_manager import InferenceManagerService
    from src.services.model_storage import ModelStorageService

    app = FastAPI()
    app.include_router(model_info_module.router, prefix="/v1/model")

    model_service = ModelStorageService(data_dir=temp_data_dir)
    inference_service = InferenceManagerService(model_storage=model_service)

    model_info_module.set_inference_manager_service(inference_service)

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestModelInfoEndpoint:
    """Tests for GET /v1/model/info - EDI-56."""

    def test_info_returns_200(self, client):
        """Test that info endpoint returns 200 OK."""
        response = client.get("/v1/model/info")
        assert response.status_code == 200

    def test_info_not_serving(self, client):
        """Test response when no model is serving."""
        response = client.get("/v1/model/info")
        data = response.json()

        assert data["serving"] is False
        assert "message" in data
        assert data["message"] == "No model currently serving"

    @pytest.mark.asyncio
    async def test_info_serving_model(self, client, temp_data_dir):
        """Test response when model is serving."""
        from src.api.routes import model_info as model_info_module
        from src.services.inference_manager import InferenceManagerService
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        inference = InferenceManagerService(model_storage=service)
        model_info_module.set_inference_manager_service(inference)

        # Create and start a model
        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "huggingface", "repo": "test/repo"},
        )
        await service.update_model_status(model_id, "ready")
        await inference.start_model(model_id=model_id, gpu_ids=[0], config=None)

        response = client.get("/v1/model/info")
        data = response.json()

        assert data["serving"] is True
        assert "model" in data
        assert data["model"]["model_id"] == model_id
        assert data["model"]["name"] == "test-model"
        assert data["model"]["type"] == "llm"

    @pytest.mark.asyncio
    async def test_info_includes_resources(self, client, temp_data_dir):
        """Test that response includes resource information."""
        from src.api.routes import model_info as model_info_module
        from src.services.inference_manager import InferenceManagerService
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        inference = InferenceManagerService(model_storage=service)
        model_info_module.set_inference_manager_service(inference)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await service.update_model_status(model_id, "ready")
        await inference.start_model(model_id=model_id, gpu_ids=[0, 1], config=None)

        response = client.get("/v1/model/info")
        data = response.json()

        assert "resources" in data
        assert "gpu" in data["resources"]
        assert data["resources"]["gpu"]["gpu_ids"] == [0, 1]
        assert data["resources"]["gpu"]["gpu_count"] == 2

    @pytest.mark.asyncio
    async def test_info_includes_parameters(self, client, temp_data_dir):
        """Test that response includes model parameters."""
        from src.api.routes import model_info as model_info_module
        from src.services.inference_manager import InferenceManagerService
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        inference = InferenceManagerService(model_storage=service)
        model_info_module.set_inference_manager_service(inference)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await service.update_model_status(model_id, "ready")
        await inference.start_model(
            model_id=model_id,
            gpu_ids=[0],
            config={"tensor_parallel_size": 2},
        )

        response = client.get("/v1/model/info")
        data = response.json()

        assert "parameters" in data
        assert "runtime_config" in data["parameters"]

    @pytest.mark.asyncio
    async def test_info_includes_stats(self, client, temp_data_dir):
        """Test that response includes runtime statistics."""
        from src.api.routes import model_info as model_info_module
        from src.services.inference_manager import InferenceManagerService
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        inference = InferenceManagerService(model_storage=service)
        model_info_module.set_inference_manager_service(inference)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await service.update_model_status(model_id, "ready")
        await inference.start_model(model_id=model_id, gpu_ids=[0], config=None)

        response = client.get("/v1/model/info")
        data = response.json()

        assert "stats" in data
        assert "loaded_at" in data["stats"]
        assert "uptime_seconds" in data["stats"]
        assert "requests_served" in data["stats"]
