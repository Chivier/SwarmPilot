"""Unified logging configuration using loguru.

This module provides a centralized logging configuration for the Planner service.
It uses loguru as the logging backend and provides functionality to intercept
standard library logging calls and redirect them to loguru.

Environment Variables:
    PLANNER_LOG_DIR: Directory for log files (default: ./logs)
    PLANNER_LOGURU_LEVEL: Log level (default: INFO)
                         Valid values: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL
"""

import os
import sys
from pathlib import Path

from loguru import logger

from swarmpilot.shared.logging import intercept_standard_logging


def setup_logging() -> None:
    """Configure loguru with environment variables and intercept standard logging.

    This function:
    1. Reads PLANNER_LOG_DIR and PLANNER_LOGURU_LEVEL environment variables
    2. Configures loguru to output to console and files
    3. Sets up interception of standard library logging
    """
    # Get configuration from environment
    log_dir = os.getenv("PLANNER_LOG_DIR", "./logs")
    log_level = os.getenv("PLANNER_LOGURU_LEVEL", "INFO").upper()

    # Validate log level
    valid_levels = {
        "TRACE",
        "DEBUG",
        "INFO",
        "SUCCESS",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }
    if log_level not in valid_levels:
        log_level = "INFO"

    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default logger
    logger.remove()

    # Add console handler without colorization
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} - "
        "{message}",
        level=log_level,
        colorize=False,
    )

    # Add file handler with rotation
    logger.add(
        log_path / "planner_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="00:00",  # Rotate at midnight
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress old logs
        encoding="utf-8",
    )

    # Add error-only file handler
    logger.add(
        log_path / "planner_error_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="00:00",
        retention="90 days",  # Keep error logs longer
        compression="zip",
        encoding="utf-8",
    )

    # Intercept standard library logging
    intercept_standard_logging()

    logger.info(f"Logging configured: level={log_level}, log_dir={log_dir}")
