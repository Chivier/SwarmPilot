"""
Tests for configuration management.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from swarmpilot.predictor.config import (
    PredictorConfig,
    get_config,
    set_config,
    reset_config,
)


class TestPredictorConfigDefaults:
    """Tests for PredictorConfig default values."""

    def test_default_host(self):
        """Should have default host of 0.0.0.0."""
        config = PredictorConfig()
        assert config.host == "0.0.0.0"

    def test_default_port(self):
        """Should have default port of 8000."""
        config = PredictorConfig()
        assert config.port == 8000

    def test_default_reload(self):
        """Should have reload disabled by default."""
        config = PredictorConfig()
        assert config.reload is False

    def test_default_workers(self):
        """Should have 1 worker by default."""
        config = PredictorConfig()
        assert config.workers == 1

    def test_default_storage_dir(self):
        """Should have 'models' as default storage directory."""
        config = PredictorConfig()
        assert config.storage_dir == "models"

    def test_default_log_level(self):
        """Should have 'info' as default log level."""
        config = PredictorConfig()
        assert config.log_level == "info"

    def test_default_log_dir(self):
        """Should have 'logs' as default log directory."""
        config = PredictorConfig()
        assert config.log_dir == "logs"


class TestPredictorConfigFromEnv:
    """Tests for configuration from environment variables."""

    def test_config_from_env_port(self):
        """Should read port from PREDICTOR_PORT env var."""
        with patch.dict(os.environ, {"PREDICTOR_PORT": "9000"}):
            config = PredictorConfig()
            assert config.port == 9000

    def test_config_from_env_host(self):
        """Should read host from PREDICTOR_HOST env var."""
        with patch.dict(os.environ, {"PREDICTOR_HOST": "127.0.0.1"}):
            config = PredictorConfig()
            assert config.host == "127.0.0.1"

    def test_config_from_env_log_level(self):
        """Should read log level from PREDICTOR_LOG_LEVEL env var."""
        with patch.dict(os.environ, {"PREDICTOR_LOG_LEVEL": "debug"}):
            config = PredictorConfig()
            assert config.log_level == "debug"

    def test_config_from_env_storage_dir(self):
        """Should read storage dir from PREDICTOR_STORAGE_DIR env var."""
        with patch.dict(os.environ, {"PREDICTOR_STORAGE_DIR": "/custom/path"}):
            config = PredictorConfig()
            assert config.storage_dir == "/custom/path"


class TestPredictorConfigFromToml:
    """Tests for configuration from TOML files."""

    def test_from_toml_with_file(self):
        """Should load configuration from TOML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "predictor.toml"
            config_file.write_text("""
[predictor]
host = "192.168.1.1"
port = 8888
workers = 4
storage_dir = "custom_models"
""")
            config = PredictorConfig.from_toml(config_file)

            assert config.host == "192.168.1.1"
            assert config.port == 8888
            assert config.workers == 4
            assert config.storage_dir == "custom_models"

    def test_from_toml_without_file(self):
        """Should return default config when no file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to temp dir where no config file exists
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                config = PredictorConfig.from_toml()

                # Should use defaults
                assert config.host == "0.0.0.0"
                assert config.port == 8000
            finally:
                os.chdir(original_cwd)

    def test_from_toml_explicit_none(self):
        """Should handle explicit None path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                config = PredictorConfig.from_toml(None)

                # Should use defaults
                assert config.host == "0.0.0.0"
            finally:
                os.chdir(original_cwd)


class TestPredictorConfigMethods:
    """Tests for PredictorConfig methods."""

    def test_to_dict(self):
        """Should convert config to dictionary."""
        config = PredictorConfig()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert "host" in result
        assert "port" in result
        assert "reload" in result
        assert "workers" in result
        assert "storage_dir" in result
        assert "log_level" in result
        assert "log_dir" in result
        assert "app_name" in result
        assert "app_version" in result

    def test_to_dict_values(self):
        """Should have correct values in dict."""
        config = PredictorConfig()
        result = config.to_dict()

        assert result["host"] == config.host
        assert result["port"] == config.port
        assert result["reload"] == config.reload

    def test_get_storage_path(self):
        """Should return Path object for storage directory."""
        config = PredictorConfig()
        path = config.get_storage_path()

        assert isinstance(path, Path)
        assert str(path) == config.storage_dir

    def test_ensure_storage_dir_creates_directory(self):
        """Should create storage directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "new_storage")
            config = PredictorConfig(storage_dir=storage_path)

            # Directory should not exist yet
            assert not os.path.exists(storage_path)

            result = config.ensure_storage_dir()

            # Directory should now exist
            assert os.path.exists(storage_path)
            assert result == Path(storage_path)


class TestGlobalConfig:
    """Tests for global configuration management."""

    def setup_method(self):
        """Reset global config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset global config after each test."""
        reset_config()

    def test_get_config_returns_instance(self):
        """Should return a PredictorConfig instance."""
        config = get_config()
        assert isinstance(config, PredictorConfig)

    def test_get_config_singleton(self):
        """Should return the same instance on multiple calls."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_set_config(self):
        """Should set the global config instance."""
        custom_config = PredictorConfig(port=9999)
        set_config(custom_config)

        config = get_config()
        assert config.port == 9999
        assert config is custom_config

    def test_reset_config(self):
        """Should reset global config to None."""
        # Set a custom config
        custom_config = PredictorConfig(port=9999)
        set_config(custom_config)

        # Reset
        reset_config()

        # Get should return a new default instance
        config = get_config()
        assert config.port == 8000  # Default
        assert config is not custom_config


class TestPredictorConfigValidation:
    """Tests for config validation and edge cases."""

    def test_config_ignores_extra_fields(self):
        """Should ignore unknown configuration fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "predictor.toml"
            config_file.write_text("""
[predictor]
host = "localhost"
unknown_field = "should be ignored"
another_unknown = 123
""")
            # Should not raise
            config = PredictorConfig.from_toml(config_file)
            assert config.host == "localhost"

    def test_config_case_insensitive_env_prefix(self):
        """Should handle case-insensitive environment variable names."""
        # The env_prefix should work regardless of case
        with patch.dict(os.environ, {"PREDICTOR_PORT": "7777"}):
            config = PredictorConfig()
            assert config.port == 7777
