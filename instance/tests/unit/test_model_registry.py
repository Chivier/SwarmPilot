"""
Unit tests for src/model_registry.py
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, mock_open, patch
from src.model_registry import ModelRegistry, get_registry
from src.models import ModelRegistryEntry


@pytest.mark.unit
class TestModelRegistry:
    """Test suite for ModelRegistry class"""

    def test_registry_load_valid(self, temp_registry_file, mock_config):
        """Test loading a valid registry YAML file"""
        with patch("src.model_registry.config", mock_config):
            registry = ModelRegistry(registry_path=temp_registry_file)

            # Check that models were loaded
            assert len(registry.models) == 2
            assert "test-model" in registry.models
            assert "another-model" in registry.models

            # Check model data
            test_model = registry.models["test-model"]
            assert test_model.model_id == "test-model"
            assert test_model.name == "Test Model"
            assert test_model.directory == "test_model"

    def test_registry_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised when registry file doesn't exist"""
        non_existent_path = tmp_path / "non_existent.yaml"

        with pytest.raises(FileNotFoundError) as exc_info:
            ModelRegistry(registry_path=non_existent_path)

        assert "Model registry not found" in str(exc_info.value)
        assert str(non_existent_path) in str(exc_info.value)

    def test_registry_invalid_yaml(self, tmp_path):
        """Test that ValueError is raised for invalid YAML"""
        invalid_yaml_file = tmp_path / "invalid.yaml"
        invalid_yaml_file.write_text("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            ModelRegistry(registry_path=invalid_yaml_file)

    def test_registry_missing_models_key(self, tmp_path):
        """Test that ValueError is raised when 'models' key is missing"""
        invalid_registry = tmp_path / "invalid_registry.yaml"
        invalid_registry.write_text("""
invalid_key:
  - model_id: test-model
    name: Test Model
    directory: test_model
    resource_requirements: {}
""")

        with pytest.raises(ValueError) as exc_info:
            ModelRegistry(registry_path=invalid_registry)

        assert "Invalid registry format" in str(exc_info.value)
        assert "models" in str(exc_info.value)

    def test_registry_empty_file(self, tmp_path):
        """Test that ValueError is raised for empty registry file"""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")

        with pytest.raises(ValueError) as exc_info:
            ModelRegistry(registry_path=empty_file)

        assert "Invalid registry format" in str(exc_info.value)

    def test_get_model_exists(self, temp_registry_file):
        """Test retrieving an existing model"""
        registry = ModelRegistry(registry_path=temp_registry_file)

        model = registry.get_model("test-model")

        assert model is not None
        assert model.model_id == "test-model"
        assert model.name == "Test Model"
        assert model.directory == "test_model"

    def test_get_model_not_exists(self, temp_registry_file):
        """Test retrieving a non-existent model returns None"""
        registry = ModelRegistry(registry_path=temp_registry_file)

        model = registry.get_model("non-existent-model")

        assert model is None

    def test_list_models(self, temp_registry_file):
        """Test listing all models"""
        registry = ModelRegistry(registry_path=temp_registry_file)

        models = registry.list_models()

        # Check that we get a dict with all models
        assert isinstance(models, dict)
        assert len(models) == 2
        assert "test-model" in models
        assert "another-model" in models

        # Check that it returns a copy (not the original dict)
        models["new-model"] = Mock()
        assert "new-model" not in registry.models

    def test_model_exists_true(self, temp_registry_file):
        """Test model_exists returns True for existing model"""
        registry = ModelRegistry(registry_path=temp_registry_file)

        assert registry.model_exists("test-model") is True
        assert registry.model_exists("another-model") is True

    def test_model_exists_false(self, temp_registry_file):
        """Test model_exists returns False for non-existent model"""
        registry = ModelRegistry(registry_path=temp_registry_file)

        assert registry.model_exists("non-existent-model") is False
        assert registry.model_exists("") is False

    def test_get_model_directory(self, temp_registry_file, mock_config):
        """Test getting model directory path"""
        with patch("src.model_registry.config", mock_config):
            registry = ModelRegistry(registry_path=temp_registry_file)

            model_dir = registry.get_model_directory("test-model")

            assert model_dir is not None
            assert isinstance(model_dir, Path)
            # Verify that config.get_model_directory was called
            mock_config.get_model_directory.assert_called_with("test_model")

    def test_get_model_directory_not_exists(self, temp_registry_file):
        """Test getting directory for non-existent model returns None"""
        registry = ModelRegistry(registry_path=temp_registry_file)

        model_dir = registry.get_model_directory("non-existent-model")

        assert model_dir is None

    def test_reload_registry(self, temp_registry_file):
        """Test reloading the registry from file"""
        registry = ModelRegistry(registry_path=temp_registry_file)

        # Initial state
        assert len(registry.models) == 2

        # Manually add a model to the in-memory registry
        registry.models["manual-model"] = ModelRegistryEntry(
            model_id="manual-model",
            name="Manual Model",
            directory="manual_model",
            resource_requirements={}
        )
        assert len(registry.models) == 3

        # Reload should reset to file contents
        registry.reload()

        assert len(registry.models) == 2
        assert "manual-model" not in registry.models
        assert "test-model" in registry.models

    def test_reload_registry_updated_file(self, tmp_path):
        """Test that reload picks up changes to the registry file"""
        registry_file = tmp_path / "registry.yaml"

        # Initial registry with one model
        registry_file.write_text("""
models:
  - model_id: model-1
    name: Model 1
    directory: model_1
    resource_requirements: {}
""")

        registry = ModelRegistry(registry_path=registry_file)
        assert len(registry.models) == 1
        assert "model-1" in registry.models

        # Update file with two models
        registry_file.write_text("""
models:
  - model_id: model-1
    name: Model 1
    directory: model_1
    resource_requirements: {}
  - model_id: model-2
    name: Model 2
    directory: model_2
    resource_requirements: {}
""")

        # Reload
        registry.reload()

        # Should now have both models
        assert len(registry.models) == 2
        assert "model-1" in registry.models
        assert "model-2" in registry.models

    def test_registry_with_multiple_models(self, tmp_path):
        """Test registry with multiple models"""
        registry_file = tmp_path / "multi_model_registry.yaml"
        registry_file.write_text("""
models:
  - model_id: model-1
    name: Model 1
    directory: model_1
    resource_requirements: {memory: "1Gi"}
  - model_id: model-2
    name: Model 2
    directory: model_2
    resource_requirements: {memory: "2Gi", cpu: "2"}
  - model_id: model-3
    name: Model 3
    directory: model_3
    resource_requirements: {}
""")

        registry = ModelRegistry(registry_path=registry_file)

        assert len(registry.models) == 3
        assert registry.model_exists("model-1")
        assert registry.model_exists("model-2")
        assert registry.model_exists("model-3")

        # Check resource requirements
        model1 = registry.get_model("model-1")
        assert model1.resource_requirements == {"memory": "1Gi"}

        model2 = registry.get_model("model-2")
        assert model2.resource_requirements == {"memory": "2Gi", "cpu": "2"}

    def test_registry_default_path(self, mock_config):
        """Test that registry uses config.registry_path by default"""
        mock_config.registry_path = Path("/tmp/default_registry.yaml")

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open(read_data="""
models:
  - model_id: test-model
    name: Test Model
    directory: test_model
    resource_requirements: {}
""")):
                with patch("src.model_registry.config", mock_config):
                    registry = ModelRegistry()

                    # Verify it used the config path
                    assert registry.registry_path == mock_config.registry_path


@pytest.mark.unit
class TestGetRegistry:
    """Test suite for get_registry function"""

    def test_get_registry_singleton(self, temp_registry_file, mock_config, monkeypatch):
        """Test that get_registry returns a singleton instance"""
        # Mock config to use temp registry file
        mock_config.registry_path = temp_registry_file

        with patch("src.model_registry.config", mock_config):
            # Reset the global registry
            import src.model_registry
            src.model_registry._registry = None

            # First call creates the instance
            registry1 = get_registry()
            assert registry1 is not None

            # Second call returns the same instance
            registry2 = get_registry()
            assert registry2 is registry1

    def test_get_registry_creates_instance(self, temp_registry_file, mock_config):
        """Test that get_registry creates registry on first call"""
        mock_config.registry_path = temp_registry_file

        with patch("src.model_registry.config", mock_config):
            # Reset the global registry
            import src.model_registry
            src.model_registry._registry = None

            # Get registry
            registry = get_registry()

            assert registry is not None
            assert isinstance(registry, ModelRegistry)
            assert len(registry.models) > 0

    def test_get_registry_preserves_state(self, temp_registry_file, mock_config):
        """Test that get_registry preserves registry state across calls"""
        mock_config.registry_path = temp_registry_file

        with patch("src.model_registry.config", mock_config):
            # Reset the global registry
            import src.model_registry
            src.model_registry._registry = None

            # Get registry and add a manual model
            registry1 = get_registry()
            registry1.models["manual-model"] = ModelRegistryEntry(
                model_id="manual-model",
                name="Manual Model",
                directory="manual_model",
                resource_requirements={}
            )

            # Get registry again
            registry2 = get_registry()

            # Should have the manual model we added
            assert "manual-model" in registry2.models
