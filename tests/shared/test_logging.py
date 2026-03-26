"""Tests for shared logging factory."""

from __future__ import annotations

import logging

from loguru import logger

from swarmpilot.shared.logging import InterceptHandler, configure_loguru


def test_configure_loguru_creates_log_directory(tmp_path):
    """Log directory is created if it doesn't exist."""
    log_dir = tmp_path / "service-logs"
    assert not log_dir.exists()

    try:
        configure_loguru(
            log_dir=str(log_dir),
            log_level="INFO",
            file_pattern="service_{time:YYYY-MM-DD}.log",
        )
        assert log_dir.exists()
        assert log_dir.is_dir()
    finally:
        logger.remove()


def test_configure_loguru_intercepts_stdlib_logging(tmp_path):
    """Standard library loggers are intercepted after configuration."""
    try:
        configure_loguru(
            log_dir=str(tmp_path),
            log_level="INFO",
            file_pattern="service_{time:YYYY-MM-DD}.log",
            logger_names=("uvicorn",),
        )
        uvicorn_logger = logging.getLogger("uvicorn")
        assert any(
            isinstance(handler, InterceptHandler)
            for handler in uvicorn_logger.handlers
        )
    finally:
        logger.remove()


def test_configure_loguru_adds_console_and_file_sinks(tmp_path):
    """Both console and file sinks are configured."""
    try:
        configure_loguru(
            log_dir=str(tmp_path),
            log_level="INFO",
            file_pattern="service_{time:YYYY-MM-DD}.log",
        )
        logger.info("hello from shared factory")

        assert len(logger._core.handlers) >= 2
        assert list(tmp_path.glob("service_*.log"))
    finally:
        logger.remove()


def test_configure_loguru_error_file_sink(tmp_path):
    """Error-only file sink is added when error_log_file_pattern is set."""
    try:
        configure_loguru(
            log_dir=str(tmp_path),
            log_level="INFO",
            file_pattern="service_{time:YYYY-MM-DD}.log",
            error_log_file_pattern="service_error_{time:YYYY-MM-DD}.log",
        )
        logger.error("shared-error-message")

        error_files = list(tmp_path.glob("service_error_*.log"))
        assert error_files
        assert "shared-error-message" in error_files[0].read_text()
    finally:
        logger.remove()


def test_configure_loguru_json_file_sink(tmp_path):
    """JSON file sink is added when json_log_file_pattern is set."""
    try:
        configure_loguru(
            log_dir=str(tmp_path),
            log_level="INFO",
            file_pattern="service_{time:YYYY-MM-DD}.log",
            json_log_file_pattern="service_{time:YYYY-MM-DD}.json",
        )
        logger.info("shared-json-message")

        json_files = list(tmp_path.glob("service_*.json"))
        assert json_files
        assert "shared-json-message" in json_files[0].read_text()
    finally:
        logger.remove()
