"""Logging configuration using loguru.

This module configures loguru as the unified logging system for the entire
application, including FastAPI and uvicorn logs. It supports environment
variables:

- PREDICTOR_LOG_DIR: Directory to store log files (default: logs/)
- PREDICTOR_LOGURU_LEVEL: Logging level (default: INFO)
"""

from __future__ import annotations

import os

from loguru import logger

from swarmpilot.shared.logging import (
    InterceptHandler,  # noqa: F401
    configure_loguru,
)


def setup_logging(
    log_dir: str | None = None,
    log_level: str | None = None,
    rotation: str = "100 MB",
    retention: str = "10 days",
    compression: str = "zip",
) -> None:
    """Setup loguru logging with file output and console output.

    This function:
    1. Removes default loguru handlers
    2. Adds console handler (stderr) with colored output
    3. Adds file handler with rotation/retention
    4. Intercepts standard library logging (uvicorn, fastapi, etc.)

    Args:
        log_dir: Directory to store log files. If None, uses PREDICTOR_LOG_DIR
            environment variable or defaults to 'logs/'.
        log_level: Logging level. If None, uses PREDICTOR_LOGURU_LEVEL
            environment variable or defaults to 'INFO'.
        rotation: When to rotate log files (e.g., "100 MB", "1 day").
        retention: How long to keep old log files (e.g., "10 days").
        compression: Compression format for rotated logs (e.g., "zip", "gz").
    """
    # Get configuration from environment variables or use defaults
    log_dir = log_dir or os.getenv("PREDICTOR_LOG_DIR", "logs")
    log_level = (
        log_level or os.getenv("PREDICTOR_LOGURU_LEVEL", "INFO")
    ).upper()

    configure_loguru(
        log_dir=log_dir,
        log_level=log_level,
        file_pattern="predictor_{time}.log",
        rotation=rotation,
        retention=retention,
        compression=compression,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )


def get_logger():
    """Get the configured loguru logger instance.

    Returns:
        The loguru logger instance.
    """
    return logger
