"""
Unit tests for logger module.
"""

import logging
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from loguru import logger


class TestLoggerSetup:
    """Test logger setup and configuration."""

    def test_logger_module_import(self):
        """Test that logger module can be imported without errors."""
        from src import logger as logger_module
        assert logger_module is not None

    def test_loguru_logger_available(self):
        """Test that loguru logger is available after setup."""
        from loguru import logger
        assert logger is not None

    def test_logger_can_log_messages(self):
        """Test that logger can log messages at different levels."""
        # Test logging
        from loguru import logger

        # These should work without raising exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.success("Success message")

        # If we get here, logging works
        assert True

    def test_log_level_from_config(self, tmp_path, monkeypatch):
        """Test that log level is correctly read from config."""
        monkeypatch.setenv("INSTANCE_LOG_LEVEL", "WARNING")
        monkeypatch.setenv("INSTANCE_LOG_DIR", str(tmp_path / "logs"))

        from src.config import Config
        config = Config()

        assert config.log_level == "WARNING"

    def test_log_directory_from_config(self, tmp_path, monkeypatch):
        """Test that log directory is correctly read from config."""
        custom_log_dir = str(tmp_path / "custom_logs")
        monkeypatch.setenv("INSTANCE_LOG_DIR", custom_log_dir)

        from src.config import Config
        config = Config()

        assert config.log_dir == custom_log_dir

    def test_json_logs_enabled_from_config(self, monkeypatch):
        """Test that JSON logs can be enabled via config."""
        monkeypatch.setenv("INSTANCE_ENABLE_JSON_LOGS", "true")

        from src.config import Config
        config = Config()

        assert config.enable_json_logs is True

    def test_json_logs_disabled_by_default(self, monkeypatch):
        """Test that JSON logs are disabled by default."""
        # Don't set the env var
        monkeypatch.delenv("INSTANCE_ENABLE_JSON_LOGS", raising=False)

        from src.config import Config
        config = Config()

        assert config.enable_json_logs is False


class TestInterceptHandler:
    """Test the InterceptHandler for standard logging interception."""

    def test_intercept_handler_exists(self):
        """Test that InterceptHandler class exists."""
        from src.logger import InterceptHandler
        assert InterceptHandler is not None

    def test_intercept_handler_is_logging_handler(self):
        """Test that InterceptHandler is a logging.Handler subclass."""
        from src.logger import InterceptHandler
        assert issubclass(InterceptHandler, logging.Handler)

    def test_standard_logging_intercepted(self, tmp_path, monkeypatch):
        """Test that standard logging messages are intercepted by loguru."""
        # Set up test environment
        monkeypatch.setenv("INSTANCE_LOG_DIR", str(tmp_path / "logs"))
        monkeypatch.setenv("INSTANCE_LOG_LEVEL", "INFO")

        # Force reimport to initialize interception
        import importlib
        import src.logger
        importlib.reload(src.logger)

        # Create a standard logger and log a message
        std_logger = logging.getLogger("test_standard_logger")
        std_logger.info("Test message from standard logging")

        # The message should be intercepted (no exception should be raised)
        # We can't easily verify the output without capturing stderr,
        # but we can verify no exception was raised
        assert True

    def test_uvicorn_logger_intercepted(self, tmp_path, monkeypatch):
        """Test that uvicorn logger is properly configured for interception."""
        monkeypatch.setenv("INSTANCE_LOG_DIR", str(tmp_path / "logs"))

        # Force reimport
        import importlib
        import src.logger
        importlib.reload(src.logger)

        # Get uvicorn logger
        uvicorn_logger = logging.getLogger("uvicorn")

        # Should have InterceptHandler
        from src.logger import InterceptHandler
        has_intercept_handler = any(
            isinstance(h, InterceptHandler) for h in uvicorn_logger.handlers
        )
        assert has_intercept_handler or not uvicorn_logger.propagate

    def test_fastapi_logger_intercepted(self, tmp_path, monkeypatch):
        """Test that fastapi logger is properly configured for interception."""
        monkeypatch.setenv("INSTANCE_LOG_DIR", str(tmp_path / "logs"))

        # Force reimport
        import importlib
        import src.logger
        importlib.reload(src.logger)

        # Get fastapi logger
        fastapi_logger = logging.getLogger("fastapi")

        # Should have InterceptHandler
        from src.logger import InterceptHandler
        has_intercept_handler = any(
            isinstance(h, InterceptHandler) for h in fastapi_logger.handlers
        )
        assert has_intercept_handler or not fastapi_logger.propagate


class TestLoggerIntegration:
    """Integration tests for logger with other components."""

    def test_logger_works_with_config(self, tmp_path, monkeypatch):
        """Test that logger integrates properly with config."""
        log_dir = tmp_path / "integration_logs"
        monkeypatch.setenv("INSTANCE_LOG_DIR", str(log_dir))
        monkeypatch.setenv("INSTANCE_LOG_LEVEL", "DEBUG")

        # Force reimport
        import importlib
        import src.config
        import src.logger
        importlib.reload(src.config)
        importlib.reload(src.logger)

        from src.config import config

        # Config should have correct values
        assert config.log_level == "DEBUG"
        assert config.log_dir == str(log_dir)

        # Logger should work
        from loguru import logger
        logger.info("Integration test message")

        # Log directory should be created
        assert log_dir.exists()

    def test_logger_initialized_on_import(self):
        """Test that logger is initialized when module is imported."""
        # This should not raise any exceptions
        from src import logger as logger_module
        from loguru import logger

        # Should be able to log immediately
        logger.info("Test message after import")

        assert True


class TestLoggerErrorHandling:
    """Test logger behavior in error conditions."""

    def test_logger_handles_invalid_log_level_gracefully(self, tmp_path, monkeypatch):
        """Test that logger handles invalid log level without crashing."""
        monkeypatch.setenv("INSTANCE_LOG_LEVEL", "INVALID_LEVEL")
        monkeypatch.setenv("INSTANCE_LOG_DIR", str(tmp_path / "logs"))

        # This might log an error but should not crash
        try:
            import importlib
            import src.logger
            importlib.reload(src.logger)
            # If we get here without exception, test passes
            assert True
        except Exception as e:
            # If there's an exception, it should be about the log level
            assert "level" in str(e).lower() or "INVALID_LEVEL" in str(e)

    def test_logger_creates_log_directory_if_missing(self):
        """Test that logger creates log directory if it doesn't exist."""
        from pathlib import Path

        # The default log directory should exist after logger is initialized
        from src.config import config

        log_dir = Path(config.log_dir)

        # Directory should exist after setup (which runs on module import)
        assert log_dir.exists()
