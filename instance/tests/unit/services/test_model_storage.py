"""Unit tests for ModelStorageService.

Tests follow TDD principle - written before implementation.
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory."""
    data_dir = tmp_path / "models"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def storage_service(temp_data_dir):
    """Create a ModelStorageService instance."""
    from src.services.model_storage import ModelStorageService

    return ModelStorageService(data_dir=temp_data_dir, min_free_disk_gb=10)


class TestModelStorageServiceInit:
    """Tests for ModelStorageService initialization."""

    def test_init_creates_data_dir(self, tmp_path):
        """Test that init creates data directory if not exists."""
        from src.services.model_storage import ModelStorageService

        data_dir = tmp_path / "nonexistent"
        service = ModelStorageService(data_dir=data_dir)
        assert data_dir.exists()

    def test_init_with_existing_dir(self, temp_data_dir):
        """Test that init works with existing directory."""
        from src.services.model_storage import ModelStorageService

        service = ModelStorageService(data_dir=temp_data_dir)
        assert service.data_dir == temp_data_dir


class TestCreateModelEntry:
    """Tests for create_model_entry method."""

    @pytest.mark.asyncio
    async def test_create_model_returns_model_id(self, storage_service):
        """Test that create_model_entry returns a model_id."""
        model_id = await storage_service.create_model_entry(
            name="llama-3.1-70b",
            model_type="llm",
            source={"type": "huggingface", "repo": "meta-llama/Llama-3.1-70B"},
        )

        assert model_id.startswith("model_")

    @pytest.mark.asyncio
    async def test_create_model_creates_directory(
        self, storage_service, temp_data_dir
    ):
        """Test that create_model_entry creates model directory."""
        model_id = await storage_service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )

        model_dir = temp_data_dir / model_id
        assert model_dir.exists()

    @pytest.mark.asyncio
    async def test_create_model_creates_metadata(
        self, storage_service, temp_data_dir
    ):
        """Test that create_model_entry creates metadata.json."""
        model_id = await storage_service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "huggingface", "repo": "test/repo"},
        )

        metadata_path = temp_data_dir / model_id / "metadata.json"
        assert metadata_path.exists()

        metadata = json.loads(metadata_path.read_text())
        assert metadata["name"] == "test-model"
        assert metadata["type"] == "llm"
        assert metadata["source"]["type"] == "huggingface"

    @pytest.mark.asyncio
    async def test_create_model_initial_status(self, storage_service):
        """Test that new models have 'pulling' status."""
        model_id = await storage_service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "huggingface", "repo": "test/repo"},
        )

        model = await storage_service.get_model(model_id)
        assert model.status == "pulling"


class TestGetModel:
    """Tests for get_model method."""

    @pytest.mark.asyncio
    async def test_get_model_returns_model_info(self, storage_service):
        """Test that get_model returns ModelInfo."""
        from src.api.schemas import ModelDetailResponse

        model_id = await storage_service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )

        model = await storage_service.get_model(model_id)
        assert model is not None
        assert model.model_id == model_id
        assert model.name == "test-model"

    @pytest.mark.asyncio
    async def test_get_model_not_found(self, storage_service):
        """Test that get_model returns None for non-existent model."""
        model = await storage_service.get_model("model_nonexistent")
        assert model is None

    @pytest.mark.asyncio
    async def test_get_model_includes_source(self, storage_service):
        """Test that get_model includes source info."""
        model_id = await storage_service.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "huggingface", "repo": "test/repo"},
        )

        model = await storage_service.get_model(model_id)
        assert model.source.type == "huggingface"
        assert model.source.repo == "test/repo"


class TestListModels:
    """Tests for list_models method."""

    @pytest.mark.asyncio
    async def test_list_models_empty(self, storage_service):
        """Test list_models returns empty list when no models."""
        models = await storage_service.list_models(model_type=None, status=None)
        assert models == []

    @pytest.mark.asyncio
    async def test_list_models_returns_all(self, storage_service):
        """Test list_models returns all models."""
        await storage_service.create_model_entry(
            name="model1", model_type="llm", source={"type": "upload"}
        )
        await storage_service.create_model_entry(
            name="model2", model_type="llm", source={"type": "upload"}
        )

        models = await storage_service.list_models(model_type=None, status=None)
        assert len(models) == 2

    @pytest.mark.asyncio
    async def test_list_models_filter_by_type(self, storage_service):
        """Test list_models filters by model_type."""
        await storage_service.create_model_entry(
            name="model1", model_type="llm", source={"type": "upload"}
        )

        models = await storage_service.list_models(model_type="llm", status=None)
        assert len(models) == 1
        assert models[0].type == "llm"

    @pytest.mark.asyncio
    async def test_list_models_filter_by_status(self, storage_service):
        """Test list_models filters by status."""
        model_id = await storage_service.create_model_entry(
            name="model1", model_type="llm", source={"type": "upload"}
        )
        await storage_service.update_model_status(model_id, "ready")

        models = await storage_service.list_models(model_type=None, status="ready")
        assert len(models) == 1
        assert models[0].status == "ready"


class TestUpdateModelStatus:
    """Tests for update_model_status method."""

    @pytest.mark.asyncio
    async def test_update_status_changes_status(self, storage_service):
        """Test that update_model_status changes the status."""
        model_id = await storage_service.create_model_entry(
            name="test-model", model_type="llm", source={"type": "upload"}
        )

        await storage_service.update_model_status(model_id, "ready")

        model = await storage_service.get_model(model_id)
        assert model.status == "ready"

    @pytest.mark.asyncio
    async def test_update_status_not_found(self, storage_service):
        """Test update_model_status raises for non-existent model."""
        from src.services.model_storage import ModelNotFoundError

        with pytest.raises(ModelNotFoundError):
            await storage_service.update_model_status("model_nonexistent", "ready")


class TestDeleteModel:
    """Tests for delete_model method."""

    @pytest.mark.asyncio
    async def test_delete_model_returns_freed_bytes(self, storage_service):
        """Test that delete_model returns freed bytes."""
        model_id = await storage_service.create_model_entry(
            name="test-model", model_type="llm", source={"type": "upload"}
        )

        freed = await storage_service.delete_model(model_id)
        assert isinstance(freed, int)
        assert freed >= 0

    @pytest.mark.asyncio
    async def test_delete_model_removes_directory(
        self, storage_service, temp_data_dir
    ):
        """Test that delete_model removes model directory."""
        model_id = await storage_service.create_model_entry(
            name="test-model", model_type="llm", source={"type": "upload"}
        )

        model_dir = temp_data_dir / model_id
        assert model_dir.exists()

        await storage_service.delete_model(model_id)
        assert not model_dir.exists()

    @pytest.mark.asyncio
    async def test_delete_model_not_found(self, storage_service):
        """Test delete_model raises for non-existent model."""
        from src.services.model_storage import ModelNotFoundError

        with pytest.raises(ModelNotFoundError):
            await storage_service.delete_model("model_nonexistent")


class TestDefaultConfig:
    """Tests for default config management."""

    @pytest.mark.asyncio
    async def test_save_default_config(self, storage_service, temp_data_dir):
        """Test saving default configuration."""
        model_id = await storage_service.create_model_entry(
            name="test-model", model_type="llm", source={"type": "upload"}
        )

        config = {"tensor_parallel_size": 2, "max_model_len": 8192}
        await storage_service.save_default_config(model_id, config)

        config_path = temp_data_dir / model_id / "default_config.json"
        assert config_path.exists()

    @pytest.mark.asyncio
    async def test_get_default_config(self, storage_service):
        """Test getting default configuration."""
        model_id = await storage_service.create_model_entry(
            name="test-model", model_type="llm", source={"type": "upload"}
        )

        config = {"tensor_parallel_size": 2}
        await storage_service.save_default_config(model_id, config)

        retrieved = await storage_service.get_default_config(model_id)
        assert retrieved == config

    @pytest.mark.asyncio
    async def test_get_default_config_none(self, storage_service):
        """Test get_default_config returns None when not set."""
        model_id = await storage_service.create_model_entry(
            name="test-model", model_type="llm", source={"type": "upload"}
        )

        config = await storage_service.get_default_config(model_id)
        assert config is None

    @pytest.mark.asyncio
    async def test_delete_default_config(self, storage_service, temp_data_dir):
        """Test deleting default configuration."""
        model_id = await storage_service.create_model_entry(
            name="test-model", model_type="llm", source={"type": "upload"}
        )

        await storage_service.save_default_config(model_id, {"test": 1})
        await storage_service.delete_default_config(model_id)

        config_path = temp_data_dir / model_id / "default_config.json"
        assert not config_path.exists()


class TestModelSizeCalculation:
    """Tests for model size calculation."""

    @pytest.mark.asyncio
    async def test_size_calculated_from_files(
        self, storage_service, temp_data_dir
    ):
        """Test that size_bytes is calculated from weight files."""
        model_id = await storage_service.create_model_entry(
            name="test-model", model_type="llm", source={"type": "upload"}
        )

        # Create fake weight files
        model_dir = temp_data_dir / model_id
        (model_dir / "model.safetensors").write_bytes(b"x" * 1000)
        (model_dir / "config.json").write_bytes(b"{}")

        # Recalculate size
        await storage_service.update_model_size(model_id)

        model = await storage_service.get_model(model_id)
        assert model.size_bytes >= 1000


class TestGetModelPath:
    """Tests for get_model_path method."""

    @pytest.mark.asyncio
    async def test_get_model_path_returns_path(
        self, storage_service, temp_data_dir
    ):
        """Test that get_model_path returns the model directory."""
        model_id = await storage_service.create_model_entry(
            name="test-model", model_type="llm", source={"type": "upload"}
        )

        path = await storage_service.get_model_path(model_id)
        assert path == temp_data_dir / model_id

    @pytest.mark.asyncio
    async def test_get_model_path_not_found(self, storage_service):
        """Test get_model_path raises for non-existent model."""
        from src.services.model_storage import ModelNotFoundError

        with pytest.raises(ModelNotFoundError):
            await storage_service.get_model_path("model_nonexistent")
