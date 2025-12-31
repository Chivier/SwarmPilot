"""Unit tests for logger module.

Tests logger configuration and InterceptHandler.
"""

import logging
from unittest.mock import MagicMock, patch


class TestInterceptHandler:
    """Tests for InterceptHandler."""

    def test_emit_with_invalid_level_name(self):
        """Test emit() when level name is invalid (lines 34-35)."""
        from src.logger import InterceptHandler

        handler = InterceptHandler()

        # Create a log record with an invalid level name
        record = logging.LogRecord(
            name="test",
            level=99,  # Custom level number
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.levelname = "CUSTOM_LEVEL"  # Invalid level name

        # Should not raise an error, should fallback to levelno
        handler.emit(record)

    def test_emit_with_deep_call_stack(self):
        """Test emit() with deep call stack (lines 40-41)."""
        import sys

        from src.logger import InterceptHandler

        handler = InterceptHandler()

        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=logging.__file__,  # Set to logging module file
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Mock the frame to test the while loop
        with patch.object(sys, "_getframe") as mock_getframe:
            # Create mock frames
            mock_frame_1 = MagicMock()
            mock_frame_1.f_code.co_filename = logging.__file__
            mock_frame_1.f_back = None  # End of stack

            mock_getframe.return_value = mock_frame_1

            # Should handle frame traversal without error
            handler.emit(record)


class TestSetupLogger:
    """Tests for setup_logger."""

    def test_setup_logger_with_json_logs_enabled(self, tmp_path, monkeypatch):
        """Test setup_logger with JSON logs enabled (line 86)."""
        from loguru import logger


        # Create a temporary log directory
        log_dir = tmp_path / "logs"

        # Mock config to enable JSON logs
        mock_config = MagicMock()
        mock_config.logging.level = "INFO"
        mock_config.logging.log_dir = str(log_dir)
        mock_config.logging.enable_json_logs = True

        # Patch the config in the logger module
        monkeypatch.setattr("src.utils.logger.config", mock_config)

        # Remove existing handlers
        logger.remove()

        # Import and call setup_logger
        from src.logger import setup_logger

        # This should add JSON file handler (line 86)
        setup_logger()

        # Verify log directory was created
        assert log_dir.exists()

        # Clean up
        logger.remove()

    def test_setup_logger_without_json_logs(self, tmp_path, monkeypatch):
        """Test setup_logger without JSON logs enabled."""
        from loguru import logger

        # Create a temporary log directory
        log_dir = tmp_path / "logs"

        # Mock config to disable JSON logs
        mock_config = MagicMock()
        mock_config.logging.level = "INFO"
        mock_config.logging.log_dir = str(log_dir)
        mock_config.logging.enable_json_logs = False

        # Patch the config in the logger module
        monkeypatch.setattr("src.utils.logger.config", mock_config)

        # Remove existing handlers
        logger.remove()

        # Import and call setup_logger
        from src.logger import setup_logger

        # This should NOT add JSON file handler
        setup_logger()

        # Verify log directory was created
        assert log_dir.exists()

        # Clean up
        logger.remove()
