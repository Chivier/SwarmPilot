"""Unit tests for InferenceManagerService.

Tests follow TDD principle - written before implementation.
"""

import pytest


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory."""
    data_dir = tmp_path / "models"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
async def model_storage(temp_data_dir):
    """Create a ModelStorageService instance."""
    from src.services.model_storage import ModelStorageService

    return ModelStorageService(data_dir=temp_data_dir)


@pytest.fixture
async def inference_manager(model_storage):
    """Create an InferenceManagerService instance."""
    from src.services.inference_manager import InferenceManagerService

    return InferenceManagerService(model_storage=model_storage)


class TestInferenceManagerServiceInit:
    """Tests for InferenceManagerService initialization."""

    @pytest.mark.asyncio
    async def test_init_creates_service(self, model_storage):
        """Test that service can be instantiated."""
        from src.services.inference_manager import InferenceManagerService

        service = InferenceManagerService(model_storage=model_storage)
        assert service is not None

    @pytest.mark.asyncio
    async def test_init_no_active_model(self, inference_manager):
        """Test that no model is active initially."""
        active = await inference_manager.get_active_model()
        assert active is None


class TestStartModel:
    """Tests for start_model method."""

    @pytest.mark.asyncio
    async def test_start_model_success(self, inference_manager, model_storage):
        """Test successful model start."""
        # Create a ready model
        model_id = await model_storage.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await model_storage.update_model_status(model_id, "ready")

        # Start the model
        result = await inference_manager.start_model(
            model_id=model_id,
            gpu_ids=[0],
            config=None,
        )

        assert result["status"] == "loading"
        assert result["model_id"] == model_id

    @pytest.mark.asyncio
    async def test_start_model_updates_status(self, inference_manager, model_storage):
        """Test that start_model updates model status."""
        model_id = await model_storage.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await model_storage.update_model_status(model_id, "ready")

        await inference_manager.start_model(
            model_id=model_id,
            gpu_ids=[0],
            config=None,
        )

        model = await model_storage.get_model(model_id)
        assert model.status in ("loading", "running")

    @pytest.mark.asyncio
    async def test_start_model_not_found(self, inference_manager):
        """Test start_model raises for non-existent model."""
        from src.services.inference_manager import ModelNotFoundError

        with pytest.raises(ModelNotFoundError):
            await inference_manager.start_model(
                model_id="model_nonexistent",
                gpu_ids=[0],
                config=None,
            )

    @pytest.mark.asyncio
    async def test_start_model_not_ready(self, inference_manager, model_storage):
        """Test start_model raises for model not in ready state."""
        from src.services.inference_manager import InvalidModelStateError

        model_id = await model_storage.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        # Model is still in 'pulling' state

        with pytest.raises(InvalidModelStateError):
            await inference_manager.start_model(
                model_id=model_id,
                gpu_ids=[0],
                config=None,
            )

    @pytest.mark.asyncio
    async def test_start_model_sets_active(self, inference_manager, model_storage):
        """Test that start_model sets the active model."""
        model_id = await model_storage.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await model_storage.update_model_status(model_id, "ready")

        await inference_manager.start_model(
            model_id=model_id,
            gpu_ids=[0],
            config=None,
        )

        active = await inference_manager.get_active_model()
        assert active == model_id

    @pytest.mark.asyncio
    async def test_start_model_with_config(self, inference_manager, model_storage):
        """Test start_model with custom configuration."""
        model_id = await model_storage.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await model_storage.update_model_status(model_id, "ready")

        config = {"tensor_parallel_size": 2, "max_model_len": 8192}
        result = await inference_manager.start_model(
            model_id=model_id,
            gpu_ids=[0, 1],
            config=config,
        )

        assert result["status"] == "loading"


class TestStopModel:
    """Tests for stop_model method."""

    @pytest.mark.asyncio
    async def test_stop_model_success(self, inference_manager, model_storage):
        """Test successful model stop."""
        model_id = await model_storage.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await model_storage.update_model_status(model_id, "ready")

        # Start then stop
        await inference_manager.start_model(model_id=model_id, gpu_ids=[0], config=None)
        result = await inference_manager.stop_model(model_id=model_id, force=False)

        assert result["status"] == "stopping"

    @pytest.mark.asyncio
    async def test_stop_model_updates_status(self, inference_manager, model_storage):
        """Test that stop_model updates model status."""
        model_id = await model_storage.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await model_storage.update_model_status(model_id, "ready")

        await inference_manager.start_model(model_id=model_id, gpu_ids=[0], config=None)
        await inference_manager.stop_model(model_id=model_id, force=False)

        model = await model_storage.get_model(model_id)
        assert model.status in ("stopping", "stopped", "ready")

    @pytest.mark.asyncio
    async def test_stop_model_clears_active(self, inference_manager, model_storage):
        """Test that stop_model clears active model."""
        model_id = await model_storage.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await model_storage.update_model_status(model_id, "ready")

        await inference_manager.start_model(model_id=model_id, gpu_ids=[0], config=None)
        await inference_manager.stop_model(model_id=model_id, force=False)

        active = await inference_manager.get_active_model()
        assert active is None

    @pytest.mark.asyncio
    async def test_stop_model_not_running(self, inference_manager, model_storage):
        """Test stop_model raises for model not running."""
        from src.services.inference_manager import InvalidModelStateError

        model_id = await model_storage.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await model_storage.update_model_status(model_id, "ready")

        with pytest.raises(InvalidModelStateError):
            await inference_manager.stop_model(model_id=model_id, force=False)

    @pytest.mark.asyncio
    async def test_stop_model_force(self, inference_manager, model_storage):
        """Test force stop."""
        model_id = await model_storage.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "upload"},
        )
        await model_storage.update_model_status(model_id, "ready")

        await inference_manager.start_model(model_id=model_id, gpu_ids=[0], config=None)
        result = await inference_manager.stop_model(model_id=model_id, force=True)

        assert result["status"] in ("stopping", "stopped", "ready")


class TestSwitchModel:
    """Tests for switch_model method."""

    @pytest.mark.asyncio
    async def test_switch_model_success(self, inference_manager, model_storage):
        """Test successful model switch."""
        # Create two ready models
        model1_id = await model_storage.create_model_entry(
            name="model1", model_type="llm", source={"type": "upload"}
        )
        await model_storage.update_model_status(model1_id, "ready")

        model2_id = await model_storage.create_model_entry(
            name="model2", model_type="llm", source={"type": "upload"}
        )
        await model_storage.update_model_status(model2_id, "ready")

        # Start first model
        await inference_manager.start_model(
            model_id=model1_id, gpu_ids=[0], config=None
        )

        # Switch to second model
        result = await inference_manager.switch_model(
            target_model_id=model2_id,
            gpu_ids=[0],
            graceful_timeout_seconds=30,
            config=None,
        )

        assert result["current_model_id"] == model2_id
        assert result["previous_model_id"] == model1_id

    @pytest.mark.asyncio
    async def test_switch_model_updates_active(self, inference_manager, model_storage):
        """Test that switch_model updates active model."""
        model1_id = await model_storage.create_model_entry(
            name="model1", model_type="llm", source={"type": "upload"}
        )
        await model_storage.update_model_status(model1_id, "ready")

        model2_id = await model_storage.create_model_entry(
            name="model2", model_type="llm", source={"type": "upload"}
        )
        await model_storage.update_model_status(model2_id, "ready")

        await inference_manager.start_model(
            model_id=model1_id, gpu_ids=[0], config=None
        )
        await inference_manager.switch_model(
            target_model_id=model2_id,
            gpu_ids=[0],
            graceful_timeout_seconds=30,
            config=None,
        )

        active = await inference_manager.get_active_model()
        assert active == model2_id

    @pytest.mark.asyncio
    async def test_switch_model_target_not_found(self, inference_manager, model_storage):
        """Test switch_model raises for non-existent target."""
        from src.services.inference_manager import ModelNotFoundError

        model_id = await model_storage.create_model_entry(
            name="model1", model_type="llm", source={"type": "upload"}
        )
        await model_storage.update_model_status(model_id, "ready")
        await inference_manager.start_model(model_id=model_id, gpu_ids=[0], config=None)

        with pytest.raises(ModelNotFoundError):
            await inference_manager.switch_model(
                target_model_id="model_nonexistent",
                gpu_ids=[0],
                graceful_timeout_seconds=30,
                config=None,
            )

    @pytest.mark.asyncio
    async def test_switch_model_no_active(self, inference_manager, model_storage):
        """Test switch_model when no model is currently active."""
        model_id = await model_storage.create_model_entry(
            name="model1", model_type="llm", source={"type": "upload"}
        )
        await model_storage.update_model_status(model_id, "ready")

        # Switch without any model running (essentially a start)
        result = await inference_manager.switch_model(
            target_model_id=model_id,
            gpu_ids=[0],
            graceful_timeout_seconds=30,
            config=None,
        )

        assert result["current_model_id"] == model_id
        assert result["previous_model_id"] is None


class TestGetModelInfo:
    """Tests for get_model_info method."""

    @pytest.mark.asyncio
    async def test_get_model_info_not_serving(self, inference_manager):
        """Test get_model_info when no model is serving."""
        info = await inference_manager.get_model_info()

        assert info["serving"] is False
        assert info["message"] == "No model currently serving"

    @pytest.mark.asyncio
    async def test_get_model_info_serving(self, inference_manager, model_storage):
        """Test get_model_info when model is serving."""
        model_id = await model_storage.create_model_entry(
            name="test-model",
            model_type="llm",
            source={"type": "huggingface", "repo": "test/repo"},
        )
        await model_storage.update_model_status(model_id, "ready")

        await inference_manager.start_model(model_id=model_id, gpu_ids=[0], config=None)

        info = await inference_manager.get_model_info()

        assert info["serving"] is True
        assert info["model"]["model_id"] == model_id
        assert info["model"]["name"] == "test-model"
