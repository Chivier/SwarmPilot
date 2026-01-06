"""Unit tests for Models API routes.

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
    """Create a FastAPI app with model routes."""
    from src.api.routes.models import router, set_model_storage_service
    from src.services.model_storage import ModelStorageService

    app = FastAPI()
    app.include_router(router, prefix="/v1/models")

    service = ModelStorageService(data_dir=temp_data_dir)
    set_model_storage_service(service)

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
async def ready_model(temp_data_dir):
    """Create a model with ready status for testing."""
    from src.services.model_storage import ModelStorageService

    service = ModelStorageService(data_dir=temp_data_dir)
    model_id = await service.create_model_entry(
        name="llama-3.1-70b",
        model_type="llm",
        source={"type": "huggingface", "repo": "meta-llama/Llama-3.1-70B"},
    )
    await service.update_model_status(model_id, "ready")
    return model_id


class TestOpenAIModelsEndpoint:
    """Tests for GET /v1/models (OpenAI-compatible) - EDI-42."""

    def test_models_returns_200(self, client):
        """Test that models endpoint returns 200 OK."""
        response = client.get("/v1/models")
        assert response.status_code == 200

    def test_models_returns_openai_format(self, client):
        """Test that response matches OpenAI format."""
        response = client.get("/v1/models")
        data = response.json()

        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)

    @pytest.mark.asyncio
    async def test_models_returns_ready_models(self, client, temp_data_dir):
        """Test that only ready/running models are returned."""
        from src.api.routes.models import set_model_storage_service
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        set_model_storage_service(service)

        # Create ready model
        model_id = await service.create_model_entry(
            name="ready-model", model_type="llm", source={"type": "upload"}
        )
        await service.update_model_status(model_id, "ready")

        # Create pulling model (should not appear)
        await service.create_model_entry(
            name="pulling-model", model_type="llm", source={"type": "upload"}
        )

        response = client.get("/v1/models")
        data = response.json()

        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "ready-model"

    @pytest.mark.asyncio
    async def test_models_item_format(self, client, temp_data_dir):
        """Test that each model item has correct format."""
        from src.api.routes.models import set_model_storage_service
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        set_model_storage_service(service)

        model_id = await service.create_model_entry(
            name="test-model", model_type="llm", source={"type": "upload"}
        )
        await service.update_model_status(model_id, "running")

        response = client.get("/v1/models")
        data = response.json()

        model = data["data"][0]
        assert model["id"] == "test-model"
        assert model["object"] == "model"
        assert "created" in model
        assert isinstance(model["created"], int)
        assert model["owned_by"] == "swarmx"


class TestDetailedModelsListEndpoint:
    """Tests for GET /v1/models/list (detailed format) - EDI-43."""

    def test_list_returns_200(self, client):
        """Test that list endpoint returns 200 OK."""
        response = client.get("/v1/models/list")
        assert response.status_code == 200

    def test_list_returns_structure(self, client):
        """Test that list returns proper structure."""
        response = client.get("/v1/models/list")
        data = response.json()

        assert "models" in data
        assert isinstance(data["models"], list)

    @pytest.mark.asyncio
    async def test_list_returns_all_models(self, client, temp_data_dir):
        """Test that list returns all models regardless of status."""
        from src.api.routes.models import set_model_storage_service
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        set_model_storage_service(service)

        await service.create_model_entry(
            name="model1", model_type="llm", source={"type": "upload"}
        )
        model_id = await service.create_model_entry(
            name="model2", model_type="llm", source={"type": "upload"}
        )
        await service.update_model_status(model_id, "ready")

        response = client.get("/v1/models/list")
        data = response.json()

        assert len(data["models"]) == 2

    @pytest.mark.asyncio
    async def test_list_filter_by_type(self, client, temp_data_dir):
        """Test filtering by model type."""
        from src.api.routes.models import set_model_storage_service
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        set_model_storage_service(service)

        await service.create_model_entry(
            name="model1", model_type="llm", source={"type": "upload"}
        )

        response = client.get("/v1/models/list?type=llm")
        data = response.json()

        assert len(data["models"]) == 1
        assert data["models"][0]["type"] == "llm"

    @pytest.mark.asyncio
    async def test_list_filter_by_status(self, client, temp_data_dir):
        """Test filtering by status."""
        from src.api.routes.models import set_model_storage_service
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        set_model_storage_service(service)

        model_id = await service.create_model_entry(
            name="model1", model_type="llm", source={"type": "upload"}
        )
        await service.update_model_status(model_id, "ready")

        await service.create_model_entry(
            name="model2", model_type="llm", source={"type": "upload"}
        )

        response = client.get("/v1/models/list?status=ready")
        data = response.json()

        assert len(data["models"]) == 1
        assert data["models"][0]["status"] == "ready"

    @pytest.mark.asyncio
    async def test_list_detailed_model_info(self, client, temp_data_dir):
        """Test that detailed info is included."""
        from src.api.routes.models import set_model_storage_service
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        set_model_storage_service(service)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "huggingface", "repo": "test/repo"},
        )
        await service.update_model_status(model_id, "ready")

        response = client.get("/v1/models/list")
        data = response.json()

        model = data["models"][0]
        assert "model_id" in model
        assert "name" in model
        assert "type" in model
        assert "status" in model
        assert "source" in model
        assert "size_bytes" in model
        assert "created_at" in model


class TestModelDetailEndpoint:
    """Tests for GET /v1/models/{model_id} - EDI-44."""

    @pytest.mark.asyncio
    async def test_get_model_returns_200(self, client, temp_data_dir):
        """Test that getting model detail returns 200."""
        from src.api.routes.models import set_model_storage_service
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        set_model_storage_service(service)

        model_id = await service.create_model_entry(
            name="test-model", model_type="llm", source={"type": "upload"}
        )

        response = client.get(f"/v1/models/{model_id}")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_model_returns_detail(self, client, temp_data_dir):
        """Test that model detail is returned."""
        from src.api.routes.models import set_model_storage_service
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        set_model_storage_service(service)

        model_id = await service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "huggingface", "repo": "test/repo"},
        )

        response = client.get(f"/v1/models/{model_id}")
        data = response.json()

        assert data["model_id"] == model_id
        assert data["name"] == "test-model"
        assert data["type"] == "llm"
        assert data["source"]["type"] == "huggingface"

    def test_get_model_not_found(self, client):
        """Test 404 for non-existent model."""
        response = client.get("/v1/models/model_nonexistent")
        assert response.status_code == 404

    def test_get_model_error_response(self, client):
        """Test error response format."""
        response = client.get("/v1/models/model_nonexistent")
        data = response.json()

        assert "detail" in data
        assert data["detail"]["error"] == "model_not_found"

    @pytest.mark.asyncio
    async def test_get_model_includes_default_config(self, client, temp_data_dir):
        """Test that default config is included if set."""
        from src.api.routes.models import set_model_storage_service
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        set_model_storage_service(service)

        model_id = await service.create_model_entry(
            name="test-model", model_type="llm", source={"type": "upload"}
        )
        await service.save_default_config(
            model_id, {"tensor_parallel_size": 2}
        )

        response = client.get(f"/v1/models/{model_id}")
        data = response.json()

        assert data["default_config"] == {"tensor_parallel_size": 2}
