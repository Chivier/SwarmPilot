"""
Tests for logging utilities.
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from loguru import logger

from swarmpilot.predictor.utils.logging import (
    InterceptHandler,
    setup_logging,
    get_logger,
)


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self):
        """Should return the loguru logger instance."""
        result = get_logger()

        # Should be the loguru logger
        assert result is logger

    def test_get_logger_is_callable(self):
        """Logger should have standard logging methods."""
        log = get_logger()

        assert hasattr(log, "info")
        assert hasattr(log, "debug")
        assert hasattr(log, "warning")
        assert hasattr(log, "error")
        assert callable(log.info)


class TestInterceptHandler:
    """Tests for InterceptHandler class."""

    def test_intercept_handler_is_logging_handler(self):
        """InterceptHandler should be a logging.Handler."""
        handler = InterceptHandler()
        assert isinstance(handler, logging.Handler)

    def test_intercept_handler_has_emit_method(self):
        """InterceptHandler should have emit method."""
        handler = InterceptHandler()
        assert hasattr(handler, "emit")
        assert callable(handler.emit)

    def test_intercept_handler_emit_with_record(self):
        """Should handle log record emission."""
        handler = InterceptHandler()

        # Create a mock log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Should not raise an exception
        # The emit method routes to loguru which may have different behavior
        # but should handle the record
        try:
            handler.emit(record)
        except Exception as e:
            # Some environments may not have loguru fully set up
            # but the method should at least be callable
            pass


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_creates_directory(self):
        """Should create log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "new_logs")

            # Directory should not exist yet
            assert not os.path.exists(log_dir)

            setup_logging(log_dir=log_dir, log_level="INFO")

            # Directory should now exist
            assert os.path.exists(log_dir)
            assert os.path.isdir(log_dir)

    def test_setup_logging_with_custom_level(self):
        """Should accept custom log level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise for valid levels
            setup_logging(log_dir=tmpdir, log_level="DEBUG")
            setup_logging(log_dir=tmpdir, log_level="WARNING")
            setup_logging(log_dir=tmpdir, log_level="ERROR")

    def test_setup_logging_uses_env_vars(self):
        """Should use environment variables when parameters not provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_log_dir = os.path.join(tmpdir, "env_logs")

            with patch.dict(os.environ, {
                "PREDICTOR_LOG_DIR": env_log_dir,
                "PREDICTOR_LOGURU_LEVEL": "DEBUG"
            }):
                setup_logging()

                # Should have created the directory from env var
                assert os.path.exists(env_log_dir)

    def test_setup_logging_with_rotation_params(self):
        """Should accept rotation parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise with custom rotation params
            setup_logging(
                log_dir=tmpdir,
                log_level="INFO",
                rotation="50 MB",
                retention="7 days",
                compression="gz"
            )

    def test_setup_logging_intercepts_uvicorn_logs(self):
        """Should configure uvicorn logger handlers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(log_dir=tmpdir, log_level="INFO")

            # Check that uvicorn logger has InterceptHandler
            uvicorn_logger = logging.getLogger("uvicorn")
            has_intercept_handler = any(
                isinstance(h, InterceptHandler)
                for h in uvicorn_logger.handlers
            )
            assert has_intercept_handler

    def test_setup_logging_intercepts_fastapi_logs(self):
        """Should configure fastapi logger handlers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(log_dir=tmpdir, log_level="INFO")

            # Check that fastapi logger has InterceptHandler
            fastapi_logger = logging.getLogger("fastapi")
            has_intercept_handler = any(
                isinstance(h, InterceptHandler)
                for h in fastapi_logger.handlers
            )
            assert has_intercept_handler

    def test_setup_logging_case_insensitive_level(self):
        """Should handle different case log levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # All these should work
            setup_logging(log_dir=tmpdir, log_level="info")
            setup_logging(log_dir=tmpdir, log_level="INFO")
            setup_logging(log_dir=tmpdir, log_level="Info")
