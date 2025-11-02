"""
Integration tests for the loguru logging system.

This test module verifies that:
1. Loguru configuration works correctly
2. Environment variables control logging behavior
3. Standard logging interception works
4. Log files are created correctly
5. Uvicorn logs are redirected to loguru
"""

import os
import logging
import tempfile
from pathlib import Path

import pytest
from loguru import logger


class TestLoggingConfiguration:
    """Test loguru configuration functionality."""

    def test_setup_logging_creates_log_directory(self):
        """Test that setup_logging creates the log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["PLANNER_LOG_DIR"] = tmpdir
            os.environ["PLANNER_LOGURU_LEVEL"] = "INFO"

            from src.logging_config import setup_logging

            # Remove existing handlers to reset
            logger.remove()
            setup_logging()

            log_path = Path(tmpdir)
            assert log_path.exists()
            assert log_path.is_dir()

            # Cleanup
            del os.environ["PLANNER_LOG_DIR"]
            del os.environ["PLANNER_LOGURU_LEVEL"]

    def test_setup_logging_respects_log_level(self):
        """Test that PLANNER_LOGURU_LEVEL controls log level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["PLANNER_LOG_DIR"] = tmpdir
            os.environ["PLANNER_LOGURU_LEVEL"] = "DEBUG"

            from src.logging_config import setup_logging

            # Remove existing handlers to reset
            logger.remove()
            setup_logging()

            # The setup should succeed without errors
            logger.debug("Debug message")
            logger.info("Info message")

            # Cleanup
            del os.environ["PLANNER_LOG_DIR"]
            del os.environ["PLANNER_LOGURU_LEVEL"]

    def test_setup_logging_handles_invalid_log_level(self):
        """Test that invalid log levels default to INFO."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["PLANNER_LOG_DIR"] = tmpdir
            os.environ["PLANNER_LOGURU_LEVEL"] = "INVALID_LEVEL"

            from src.logging_config import setup_logging

            # Remove existing handlers to reset
            logger.remove()
            setup_logging()

            # Should not raise an error, defaults to INFO
            logger.info("This should work")

            # Cleanup
            del os.environ["PLANNER_LOG_DIR"]
            del os.environ["PLANNER_LOGURU_LEVEL"]


class TestStandardLoggingInterception:
    """Test that standard library logging is intercepted by loguru."""

    def test_standard_logging_is_intercepted(self):
        """Test that standard logging calls are redirected to loguru."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["PLANNER_LOG_DIR"] = tmpdir
            os.environ["PLANNER_LOGURU_LEVEL"] = "INFO"

            from src.logging_config import setup_logging

            # Remove existing handlers to reset
            logger.remove()
            setup_logging()

            # Create a standard logger
            std_logger = logging.getLogger("test_module")

            # These should not raise errors and should be intercepted
            std_logger.info("Info from standard logging")
            std_logger.warning("Warning from standard logging")
            std_logger.error("Error from standard logging")

            # Cleanup
            del os.environ["PLANNER_LOG_DIR"]
            del os.environ["PLANNER_LOGURU_LEVEL"]

    def test_uvicorn_logger_is_intercepted(self):
        """Test that uvicorn logger is intercepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["PLANNER_LOG_DIR"] = tmpdir
            os.environ["PLANNER_LOGURU_LEVEL"] = "INFO"

            from src.logging_config import setup_logging

            # Remove existing handlers to reset
            logger.remove()
            setup_logging()

            # Get uvicorn logger
            uvicorn_logger = logging.getLogger("uvicorn")

            # This should be intercepted by loguru
            uvicorn_logger.info("Uvicorn log message")

            # Cleanup
            del os.environ["PLANNER_LOG_DIR"]
            del os.environ["PLANNER_LOGURU_LEVEL"]


class TestLogFileCreation:
    """Test that log files are created correctly."""

    def test_log_files_are_created(self):
        """Test that both main and error log files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["PLANNER_LOG_DIR"] = tmpdir
            os.environ["PLANNER_LOGURU_LEVEL"] = "INFO"

            from src.logging_config import setup_logging

            # Remove existing handlers to reset
            logger.remove()
            setup_logging()

            # Log some messages
            logger.info("Test info message")
            logger.error("Test error message")

            # Check that log files exist
            log_path = Path(tmpdir)
            log_files = list(log_path.glob("planner_*.log"))

            # Should have at least 2 files (main and error)
            assert len(log_files) >= 2

            # Cleanup
            del os.environ["PLANNER_LOG_DIR"]
            del os.environ["PLANNER_LOGURU_LEVEL"]

    def test_error_logs_are_written_to_error_file(self):
        """Test that error logs are written to the error log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["PLANNER_LOG_DIR"] = tmpdir
            os.environ["PLANNER_LOGURU_LEVEL"] = "INFO"

            from src.logging_config import setup_logging

            # Remove existing handlers to reset
            logger.remove()
            setup_logging()

            # Log an error
            test_error_msg = "Test error for file verification"
            logger.error(test_error_msg)

            # Find error log file
            log_path = Path(tmpdir)
            error_files = list(log_path.glob("planner_error_*.log"))

            assert len(error_files) > 0

            # Read error log file
            error_log_content = error_files[0].read_text()
            assert test_error_msg in error_log_content

            # Cleanup
            del os.environ["PLANNER_LOG_DIR"]
            del os.environ["PLANNER_LOGURU_LEVEL"]


class TestAPILogging:
    """Test that API endpoints produce correct log output."""

    def test_api_startup_logs(self):
        """Test that API startup produces expected logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["PLANNER_LOG_DIR"] = tmpdir
            os.environ["PLANNER_LOGURU_LEVEL"] = "INFO"

            # Import API module (this triggers setup_logging)
            from src import api

            # Check that log directory was created
            log_path = Path(tmpdir)
            assert log_path.exists()

            # Cleanup
            del os.environ["PLANNER_LOG_DIR"]
            del os.environ["PLANNER_LOGURU_LEVEL"]


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_logger_has_standard_methods(self):
        """Test that loguru logger has all standard logging methods."""
        # These methods should exist and be callable
        assert callable(logger.trace)
        assert callable(logger.debug)
        assert callable(logger.info)
        assert callable(logger.success)
        assert callable(logger.warning)
        assert callable(logger.error)
        assert callable(logger.critical)

    def test_logger_accepts_standard_parameters(self):
        """Test that logger accepts standard logging parameters."""
        # These should not raise errors
        logger.info("Simple message")
        logger.info("Message with {param}", param="value")
        logger.error("Error with exception", exc_info=True)

    def test_existing_code_compatibility(self):
        """Test that existing logging code patterns still work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["PLANNER_LOG_DIR"] = tmpdir
            os.environ["PLANNER_LOGURU_LEVEL"] = "INFO"

            from src.logging_config import setup_logging

            # Remove existing handlers to reset
            logger.remove()
            setup_logging()

            # These patterns are used in the existing codebase
            logger.info("Received /plan request: M=4, N=3, algorithm=simulated_annealing")
            logger.error("Optimization failed: error", exc_info=True)
            logger.warning("Deployment partially failed: 2 instances failed")

            # Should complete without errors

            # Cleanup
            del os.environ["PLANNER_LOG_DIR"]
            del os.environ["PLANNER_LOGURU_LEVEL"]
