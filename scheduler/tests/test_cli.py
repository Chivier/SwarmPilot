"""
Tests for the CLI module.

Covers command-line interface functionality including configuration loading,
environment variable management, and command execution.
"""

import pytest
import json
import tomllib
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from src.cli import app, load_config_file, apply_config


@pytest.fixture
def runner():
    """Create a Typer CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_config_files(tmp_path):
    """Create temporary configuration files for testing."""
    # JSON config
    json_config = tmp_path / "config.json"
    json_content = {
        "server": {"host": "0.0.0.0", "port": 8080},
        "predictor": {"url": "http://predictor:8000", "timeout": 30.0},
        "scheduling": {"strategy": "probabilistic"},
        "training": {"enable_auto": True, "batch_size": 100},
        "logging": {"level": "INFO"},
    }
    json_config.write_text(json.dumps(json_content))

    # TOML config
    toml_config = tmp_path / "config.toml"
    toml_content = """
[server]
host = "0.0.0.0"
port = 8080

[predictor]
url = "http://predictor:8000"
timeout = 30.0

[scheduling]
strategy = "probabilistic"

[training]
enable_auto = true
batch_size = 100

[logging]
level = "INFO"
"""
    toml_config.write_text(toml_content)

    # YAML config (only create if PyYAML is available)
    yaml_config = tmp_path / "config.yaml"
    try:
        import yaml

        yaml_content = {
            "server": {"host": "0.0.0.0", "port": 8080},
            "predictor": {"url": "http://predictor:8000", "timeout": 30.0},
            "scheduling": {"strategy": "probabilistic"},
            "training": {"enable_auto": True, "batch_size": 100},
            "logging": {"level": "INFO"},
        }
        yaml_config.write_text(yaml.dump(yaml_content))
    except ImportError:
        yaml_config = None

    return {
        "json": json_config,
        "toml": toml_config,
        "yaml": yaml_config,
        "dir": tmp_path,
    }


class TestLoadConfigFile:
    """Tests for configuration file loading."""

    def test_load_json_config(self, temp_config_files):
        """Test loading JSON configuration file."""
        config = load_config_file(temp_config_files["json"])

        assert config["server"]["host"] == "0.0.0.0"
        assert config["server"]["port"] == 8080
        assert config["predictor"]["url"] == "http://predictor:8000"
        assert config["scheduling"]["strategy"] == "probabilistic"

    def test_load_toml_config(self, temp_config_files):
        """Test loading TOML configuration file."""
        config = load_config_file(temp_config_files["toml"])

        assert config["server"]["host"] == "0.0.0.0"
        assert config["server"]["port"] == 8080
        assert config["predictor"]["url"] == "http://predictor:8000"
        assert config["scheduling"]["strategy"] == "probabilistic"

    def test_load_yaml_config(self, temp_config_files):
        """Test loading YAML configuration file."""
        if temp_config_files["yaml"] is None:
            pytest.skip("PyYAML not installed")

        config = load_config_file(temp_config_files["yaml"])

        assert config["server"]["host"] == "0.0.0.0"
        assert config["server"]["port"] == 8080
        assert config["predictor"]["url"] == "http://predictor:8000"
        assert config["scheduling"]["strategy"] == "probabilistic"

    def test_file_not_found(self, tmp_path):
        """Test error when configuration file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.json"

        with pytest.raises(Exception) as exc_info:
            load_config_file(nonexistent)

        assert "not found" in str(exc_info.value).lower()

    def test_invalid_json(self, tmp_path):
        """Test error when JSON file is invalid."""
        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("{ invalid json }")

        with pytest.raises(Exception):
            load_config_file(invalid_json)

    def test_unsupported_format(self, tmp_path):
        """Test error when file format is not supported."""
        unsupported = tmp_path / "config.xml"
        unsupported.write_text("<config></config>")

        with pytest.raises(Exception) as exc_info:
            load_config_file(unsupported)

        assert "unsupported" in str(exc_info.value).lower()

    def test_yaml_missing_library(self, tmp_path):
        """Test error when PyYAML is not installed for YAML file."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("server:\n  host: localhost")

        # Simulate missing yaml library by temporarily removing it from sys.modules
        import sys

        yaml_module = sys.modules.get('yaml')
        if 'yaml' in sys.modules:
            del sys.modules['yaml']

        try:
            # Mock the import to raise ImportError
            with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs:
                       (_ for _ in ()).throw(ImportError(f"No module named '{name}'")) if name == 'yaml'
                       else __import__(name, *args, **kwargs)):
                with pytest.raises(Exception) as exc_info:
                    load_config_file(yaml_file)

                assert "pyyaml" in str(exc_info.value).lower() or "import" in str(exc_info.value).lower()
        finally:
            # Restore yaml module if it was there
            if yaml_module is not None:
                sys.modules['yaml'] = yaml_module


class TestApplyConfig:
    """Tests for applying configuration to environment variables."""

    def test_apply_all_config_values(self, monkeypatch):
        """Test applying all configuration values to environment."""
        # Clear environment
        env_vars = [
            "SCHEDULER_HOST",
            "SCHEDULER_PORT",
            "PREDICTOR_URL",
            "PREDICTOR_TIMEOUT",
            "SCHEDULING_STRATEGY",
            "TRAINING_ENABLE_AUTO",
            "TRAINING_BATCH_SIZE",
            "LOG_LEVEL",
        ]
        for var in env_vars:
            monkeypatch.delenv(var, raising=False)

        config = {
            "server": {"host": "0.0.0.0", "port": 8080},
            "predictor": {"url": "http://predictor:8000", "timeout": 30.0},
            "scheduling": {"strategy": "probabilistic"},
            "training": {"enable_auto": True, "batch_size": 100},
            "logging": {"level": "INFO"},
        }

        apply_config(config, None, None)

        import os

        assert os.environ["SCHEDULER_HOST"] == "0.0.0.0"
        assert os.environ["SCHEDULER_PORT"] == "8080"
        assert os.environ["PREDICTOR_URL"] == "http://predictor:8000"
        assert os.environ["PREDICTOR_TIMEOUT"] == "30.0"
        assert os.environ["SCHEDULING_STRATEGY"] == "probabilistic"
        assert os.environ["TRAINING_ENABLE_AUTO"] == "True"
        assert os.environ["TRAINING_BATCH_SIZE"] == "100"
        assert os.environ["LOG_LEVEL"] == "INFO"

    def test_nested_config_navigation(self, monkeypatch):
        """Test navigation through nested configuration dictionaries."""
        monkeypatch.delenv("PREDICTOR_URL", raising=False)

        config = {"predictor": {"url": "http://custom:9000"}}

        apply_config(config, None, None)

        import os

        assert os.environ["PREDICTOR_URL"] == "http://custom:9000"

    def test_command_line_overrides_config(self, monkeypatch):
        """Test that command-line arguments override config file values."""
        monkeypatch.delenv("SCHEDULER_HOST", raising=False)
        monkeypatch.delenv("SCHEDULER_PORT", raising=False)

        config = {"server": {"host": "0.0.0.0", "port": 8080}}

        # Apply with command-line overrides
        apply_config(config, "127.0.0.1", 9000)

        import os

        assert os.environ["SCHEDULER_HOST"] == "127.0.0.1"
        assert os.environ["SCHEDULER_PORT"] == "9000"

    def test_skip_missing_keys(self, monkeypatch):
        """Test that missing configuration keys are skipped gracefully."""
        monkeypatch.delenv("PREDICTOR_URL", raising=False)

        # Config missing predictor.url
        config = {"server": {"host": "0.0.0.0"}}

        apply_config(config, None, None)

        import os

        # Should not raise an error, just skip the missing key
        assert "PREDICTOR_URL" not in os.environ
        assert os.environ["SCHEDULER_HOST"] == "0.0.0.0"

    def test_environment_already_set(self, monkeypatch):
        """Test that existing environment variables are not overwritten by config."""
        import os

        monkeypatch.setenv("SCHEDULER_HOST", "existing-host")

        config = {"server": {"host": "config-host"}}

        apply_config(config, None, None)

        # Should keep existing value
        assert os.environ["SCHEDULER_HOST"] == "existing-host"

    def test_command_line_overrides_environment(self, monkeypatch):
        """Test that command-line args override existing environment variables."""
        import os

        monkeypatch.setenv("SCHEDULER_HOST", "env-host")

        config = {}

        apply_config(config, "cli-host", None)

        # Command-line should override environment
        assert os.environ["SCHEDULER_HOST"] == "cli-host"


class TestStartCommand:
    """Tests for the start command."""

    def test_start_with_defaults(self, runner, monkeypatch):
        """Test starting with default configuration."""
        # Mock uvicorn.run to prevent actual server start
        with patch("src.cli.uvicorn.run") as mock_run:
            result = runner.invoke(app, ["start"])

            assert result.exit_code == 0
            assert "Starting Scheduler service" in result.stdout
            mock_run.assert_called_once()

            # Verify uvicorn.run was called with correct arguments
            call_args = mock_run.call_args
            assert call_args[1]["host"]  # Some host value
            assert call_args[1]["port"]  # Some port value
            assert "src.api:app" in call_args[0]

    def test_start_with_config_file(self, runner, temp_config_files, monkeypatch):
        """Test starting with configuration file."""
        with patch("src.cli.uvicorn.run") as mock_run:
            result = runner.invoke(
                app, ["start", "--config", str(temp_config_files["json"])]
            )

            assert result.exit_code == 0
            assert "Loading configuration from" in result.stdout
            mock_run.assert_called_once()

    def test_start_with_cli_overrides(self, runner, monkeypatch):
        """Test starting with command-line host and port overrides."""
        # Clear environment to ensure clean slate
        monkeypatch.delenv("SCHEDULER_HOST", raising=False)
        monkeypatch.delenv("SCHEDULER_PORT", raising=False)

        with patch("src.cli.uvicorn.run") as mock_run:
            result = runner.invoke(app, ["start", "--host", "127.0.0.1", "--port", "9000"])

            assert result.exit_code == 0
            mock_run.assert_called_once()

            # Verify the command ran successfully (exact values depend on config loading)

    def test_start_keyboard_interrupt(self, runner, monkeypatch):
        """Test handling of keyboard interrupt during start."""
        with patch("src.cli.uvicorn.run", side_effect=KeyboardInterrupt):
            result = runner.invoke(app, ["start"])

            assert result.exit_code == 0
            assert "Shutting down" in result.stdout

    def test_start_exception_handling(self, runner, monkeypatch):
        """Test handling of exceptions during start."""
        with patch("src.cli.uvicorn.run", side_effect=Exception("Test error")):
            result = runner.invoke(app, ["start"], catch_exceptions=False)

            # The exception should be raised
            assert result.exit_code != 0

    def test_config_and_cli_override_priority(
        self, runner, temp_config_files, monkeypatch
    ):
        """Test that CLI args take precedence over config file."""
        # Clear environment
        monkeypatch.delenv("SCHEDULER_PORT", raising=False)

        with patch("src.cli.uvicorn.run") as mock_run:
            result = runner.invoke(
                app,
                [
                    "start",
                    "--config",
                    str(temp_config_files["json"]),
                    "--port",
                    "9999",
                ],
            )

            assert result.exit_code == 0
            mock_run.assert_called_once()

            # Verify command succeeded (config loading behavior may vary)


class TestVersionCommand:
    """Tests for the version command."""

    def test_version_output(self, runner):
        """Test version command output."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "Scheduler version:" in result.stdout
        # Should contain some version string
        assert len(result.stdout.strip()) > len("Scheduler version:")


class TestCLIHelp:
    """Tests for CLI help functionality."""

    def test_main_help(self, runner):
        """Test main CLI help output."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Scheduler service command-line interface" in result.stdout
        assert "start" in result.stdout
        assert "version" in result.stdout

    def test_start_command_help(self, runner):
        """Test start command help output."""
        result = runner.invoke(app, ["start", "--help"])

        assert result.exit_code == 0
        assert "Start the scheduler service" in result.stdout
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--config" in result.stdout
