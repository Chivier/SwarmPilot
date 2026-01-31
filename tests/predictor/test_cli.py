"""
Tests for CLI commands.

Note: The `start` command is not tested as it actually starts a uvicorn server.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from swarmpilot.predictor.cli import app
from swarmpilot.predictor.config import PredictorConfig, reset_config


runner = CliRunner()


class TestVersionCommand:
    """Tests for the version command."""

    def test_version_command_runs(self):
        """Should run version command successfully."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0

    def test_version_command_shows_version(self):
        """Should show version information."""
        result = runner.invoke(app, ["version"])
        assert "Version:" in result.output

    def test_version_command_shows_dependencies(self):
        """Should show dependency versions."""
        result = runner.invoke(app, ["version"])
        assert "Dependencies:" in result.output
        assert "Python:" in result.output
        assert "FastAPI:" in result.output
        assert "PyTorch:" in result.output


class TestListModelsCommand:
    """Tests for the list command."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def test_list_models_empty(self):
        """Should handle empty storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, ["list", "--storage-dir", tmpdir])
            assert result.exit_code == 0
            assert "Total models: 0" in result.output
            assert "No models found" in result.output

    def test_list_models_shows_storage_dir(self):
        """Should show storage directory path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, ["list", "--storage-dir", tmpdir])
            assert "Storage directory:" in result.output


class TestConfigShowCommand:
    """Tests for the config show command."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def test_config_show_default(self):
        """Should show default configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = runner.invoke(app, ["config", "show"])
                assert result.exit_code == 0
                assert "Configuration" in result.output
            finally:
                os.chdir(original_cwd)

    def test_config_show_displays_values(self):
        """Should display configuration values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = runner.invoke(app, ["config", "show"])
                # Should show key fields
                assert "Host" in result.output or "host" in result.output.lower()
                assert "Port" in result.output or "port" in result.output.lower()
            finally:
                os.chdir(original_cwd)

    def test_config_show_with_file(self):
        """Should load and show config from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "predictor.toml"
            config_file.write_text("""
[predictor]
host = "custom.host.com"
port = 9999
""")
            result = runner.invoke(app, ["config", "show", "--config", str(config_file)])
            assert result.exit_code == 0
            assert "custom.host.com" in result.output or "9999" in result.output


class TestConfigInitCommand:
    """Tests for the config init command."""

    def test_config_init_creates_file(self):
        """Should create a configuration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_config.toml"

            result = runner.invoke(app, ["config", "init", "--output", str(output_file)])

            assert result.exit_code == 0
            assert output_file.exists()
            assert "Created configuration file" in result.output

    def test_config_init_file_contents(self):
        """Should create valid TOML content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_config.toml"

            runner.invoke(app, ["config", "init", "--output", str(output_file)])

            content = output_file.read_text()
            assert "[predictor]" in content
            assert "host" in content
            assert "port" in content

    def test_config_init_no_overwrite(self):
        """Should not overwrite existing file without --force."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_config.toml"
            output_file.write_text("existing content")

            result = runner.invoke(app, ["config", "init", "--output", str(output_file)])

            assert result.exit_code == 1
            assert "already exists" in result.output
            # Original content should remain
            assert output_file.read_text() == "existing content"

    def test_config_init_force_overwrite(self):
        """Should overwrite existing file with --force."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_config.toml"
            output_file.write_text("existing content")

            result = runner.invoke(app, ["config", "init", "--output", str(output_file), "--force"])

            assert result.exit_code == 0
            assert "Created configuration file" in result.output
            # Content should be replaced
            assert output_file.read_text() != "existing content"
            assert "[predictor]" in output_file.read_text()


class TestHealthCommand:
    """Tests for the health command."""

    def test_health_command_connection_error(self):
        """Should handle connection error gracefully."""
        # Use a port that's unlikely to have a service
        result = runner.invoke(app, ["health", "--host", "localhost", "--port", "59999"])

        assert result.exit_code == 1
        assert "Cannot connect" in result.output or "Error" in result.output

    def test_health_command_with_mock_success(self):
        """Should show healthy status on successful response."""
        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "healthy",
                "message": "Service is running",
                "timestamp": "2024-01-01T00:00:00Z"
            }
            mock_get.return_value = mock_response

            result = runner.invoke(app, ["health"])

            assert result.exit_code == 0
            assert "healthy" in result.output.lower()

    def test_health_command_with_mock_unhealthy(self):
        """Should show error on non-200 response."""
        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response

            result = runner.invoke(app, ["health"])

            assert result.exit_code == 1
            assert "500" in result.output

    def test_health_command_timeout(self):
        """Should handle timeout gracefully."""
        import httpx

        with patch("httpx.get") as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Timeout")

            result = runner.invoke(app, ["health"])

            assert result.exit_code == 1
            assert "timed out" in result.output.lower() or "timeout" in result.output.lower()


class TestAppMetadata:
    """Tests for CLI app metadata."""

    def test_app_name(self):
        """Should have correct app name."""
        assert app.info.name == "spredictor"

    def test_app_has_help(self):
        """Should have help text."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Runtime Predictor Service" in result.output or "predictor" in result.output.lower()

    def test_subcommand_help(self):
        """Should show help for subcommands."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "Configuration" in result.output or "config" in result.output.lower()
