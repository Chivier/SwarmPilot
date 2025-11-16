"""
Unit tests for src/config.py
"""

import pytest
from pathlib import Path
from src.config import Config


@pytest.mark.unit
class TestConfig:
    """Test suite for Config class"""

    def test_config_default_values(self, monkeypatch):
        """Test that Config initializes with correct default values"""
        # Clear any existing environment variables
        for key in ["INSTANCE_ID", "INSTANCE_PORT", "DOCKER_NETWORK", "LOG_LEVEL",
                    "MAX_QUEUE_SIZE", "HEALTH_CHECK_INTERVAL", "HEALTH_CHECK_TIMEOUT"]:
            monkeypatch.delenv(key, raising=False)

        config = Config()

        # Check defaults
        assert config.instance_id == "instance-default"
        assert config.instance_port == 8000
        assert config.model_port == 9000  # instance_port + 1000
        assert config.docker_network == "instance_network"
        assert config.log_level == "INFO"
        assert config.max_queue_size == 100
        assert config.health_check_interval == 10
        assert config.health_check_timeout == 30

    def test_config_custom_env_vars(self, monkeypatch):
        """Test that Config correctly reads custom environment variables"""
        # Set custom environment variables
        monkeypatch.setenv("INSTANCE_ID", "custom-instance")
        monkeypatch.setenv("INSTANCE_PORT", "7000")
        monkeypatch.setenv("DOCKER_NETWORK", "custom_network")
        monkeypatch.setenv("INSTANCE_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("INSTANCE_LOG_DIR", "/custom/logs")
        monkeypatch.setenv("INSTANCE_ENABLE_JSON_LOGS", "true")
        monkeypatch.setenv("MAX_QUEUE_SIZE", "200")
        monkeypatch.setenv("HEALTH_CHECK_INTERVAL", "15")
        monkeypatch.setenv("HEALTH_CHECK_TIMEOUT", "60")

        config = Config()

        assert config.instance_id == "custom-instance"
        assert config.instance_port == 7000
        assert config.docker_network == "custom_network"
        assert config.log_level == "DEBUG"
        assert config.log_dir == "/custom/logs"
        assert config.enable_json_logs == True
        assert config.max_queue_size == 200
        assert config.health_check_interval == 15
        assert config.health_check_timeout == 60

    def test_model_port_calculation(self, monkeypatch):
        """Test that model_port is correctly calculated as instance_port + 1000"""
        test_cases = [
            (5000, 6000),
            (8000, 9000),
            (3000, 4000),
            (10000, 11000),
        ]

        for instance_port, expected_model_port in test_cases:
            monkeypatch.setenv("INSTANCE_PORT", str(instance_port))
            config = Config()
            assert config.model_port == expected_model_port, \
                f"Expected model_port {expected_model_port} for instance_port {instance_port}"

    def test_path_resolution(self, monkeypatch):
        """Test that paths are correctly resolved"""
        config = Config()

        # base_dir should be the parent of the src directory
        assert config.base_dir.name == "instance"
        assert config.base_dir.is_absolute()

        # dockers_dir should be base_dir/dockers
        assert config.dockers_dir == config.base_dir / "dockers"

        # registry_path should be dockers_dir/model_registry.yaml
        assert config.registry_path == config.dockers_dir / "model_registry.yaml"
        assert config.registry_path.name == "model_registry.yaml"

    def test_get_model_directory(self, monkeypatch):
        """Test get_model_directory method"""
        config = Config()

        # Test getting model directory
        model_dir = config.get_model_directory("test_model")
        assert model_dir == config.dockers_dir / "test_model"
        assert isinstance(model_dir, Path)

        # Test with different directory names
        model_dir2 = config.get_model_directory("another_model")
        assert model_dir2 == config.dockers_dir / "another_model"

        # Test with nested path
        model_dir3 = config.get_model_directory("subfolder/model")
        assert model_dir3 == config.dockers_dir / "subfolder/model"

    def test_get_model_container_name(self, monkeypatch):
        """Test get_model_container_name method"""
        monkeypatch.setenv("INSTANCE_ID", "test-instance")
        config = Config()

        # Test basic model ID
        container_name = config.get_model_container_name("test-model")
        assert container_name == "model_test-instance_test-model"

        # Test with another model ID
        container_name2 = config.get_model_container_name("gpt-3.5")
        assert container_name2 == "model_test-instance_gpt-3.5"

    def test_container_name_special_chars(self, monkeypatch):
        """Test that special characters in model_id are replaced with underscores"""
        monkeypatch.setenv("INSTANCE_ID", "test-instance")
        config = Config()

        # Test forward slash replacement
        container_name = config.get_model_container_name("org/model")
        assert "/" not in container_name
        assert container_name == "model_test-instance_org_model"

        # Test colon replacement
        container_name2 = config.get_model_container_name("model:v1.0")
        assert ":" not in container_name2
        assert container_name2 == "model_test-instance_model_v1.0"

        # Test multiple special characters
        container_name3 = config.get_model_container_name("org/model:v1.0")
        assert "/" not in container_name3
        assert ":" not in container_name3
        assert container_name3 == "model_test-instance_org_model_v1.0"

    def test_container_name_prefix(self, monkeypatch):
        """Test that container_name_prefix is correctly formed"""
        monkeypatch.setenv("INSTANCE_ID", "prod-instance-1")
        config = Config()

        assert config.container_name_prefix == "model_prod-instance-1"

        # Verify it's used in get_model_container_name
        container_name = config.get_model_container_name("test-model")
        assert container_name.startswith("model_prod-instance-1_")

    def test_integer_env_var_parsing(self, monkeypatch):
        """Test that integer environment variables are correctly parsed"""
        monkeypatch.setenv("INSTANCE_PORT", "9000")
        monkeypatch.setenv("MAX_QUEUE_SIZE", "500")
        monkeypatch.setenv("HEALTH_CHECK_INTERVAL", "20")
        monkeypatch.setenv("HEALTH_CHECK_TIMEOUT", "90")

        config = Config()

        # Verify types
        assert isinstance(config.instance_port, int)
        assert isinstance(config.model_port, int)
        assert isinstance(config.max_queue_size, int)
        assert isinstance(config.health_check_interval, int)
        assert isinstance(config.health_check_timeout, int)

        # Verify values
        assert config.instance_port == 9000
        assert config.max_queue_size == 500
        assert config.health_check_interval == 20
        assert config.health_check_timeout == 90

    def test_config_immutability_after_init(self, monkeypatch):
        """Test that changing environment variables after init doesn't affect existing config"""
        monkeypatch.setenv("INSTANCE_ID", "original-instance")
        monkeypatch.setenv("INSTANCE_PORT", "5000")

        config = Config()
        original_id = config.instance_id
        original_port = config.instance_port

        # Change environment variables
        monkeypatch.setenv("INSTANCE_ID", "new-instance")
        monkeypatch.setenv("INSTANCE_PORT", "6000")

        # Config should retain original values
        assert config.instance_id == original_id
        assert config.instance_port == original_port

    def test_paths_are_pathlib_objects(self):
        """Test that all path attributes are pathlib.Path objects"""
        config = Config()

        assert isinstance(config.base_dir, Path)
        assert isinstance(config.dockers_dir, Path)
        assert isinstance(config.registry_path, Path)

        # Test get_model_directory returns Path
        model_dir = config.get_model_directory("test_model")
        assert isinstance(model_dir, Path)
