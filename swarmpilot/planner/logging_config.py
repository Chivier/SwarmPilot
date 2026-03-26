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

from swarmpilot.shared.logging import configure_loguru


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

    configure_loguru(
        log_dir=log_dir,
        log_level=log_level,
        file_pattern="planner_{time:YYYY-MM-DD}.log",
        error_log_file_pattern="planner_error_{time:YYYY-MM-DD}.log",
        error_retention="90 days",
    )
