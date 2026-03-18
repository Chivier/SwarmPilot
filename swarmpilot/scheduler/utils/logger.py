"""Centralized logging configuration using loguru.

This module configures loguru for the entire scheduler application,
providing structured logging with rotation, retention, and customizable formats.
"""

import sys
from pathlib import Path

from loguru import logger

from swarmpilot.scheduler.config import config
from swarmpilot.shared.logging import (
    InterceptHandler,  # noqa: F401
    intercept_standard_logging,
)


def setup_logger():
    """Configure loguru logger with application-specific settings.

    This function:
    - Removes default handler
    - Adds console handler with custom format
    - Adds file handler with rotation and retention
    - Sets log level from configuration
    """
    # Remove default handler
    logger.remove()

    # Console handler without colored output
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level=config.logging.level,
        colorize=False,
    )

    # File handler with rotation (no colors for clean file output)
    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "scheduler_{time}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level=config.logging.level,
        rotation="00:00",  # Rotate at midnight
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress rotated logs
        encoding="utf-8",
        colorize=False,  # Disable ANSI color codes in file
    )

    # Add JSON file handler for structured logging (optional, can be enabled via config)
    if config.logging.enable_json_logs:
        logger.add(
            log_dir / "scheduler_{time:YYYY-MM-DD}.json",
            format="{message}",
            level=config.logging.level,
            rotation="00:00",
            retention="30 days",
            compression="zip",
            encoding="utf-8",
            serialize=True,  # Output as JSON
        )

    # Intercept standard logging and redirect to loguru
    intercept_standard_logging(
        logger_names=(
            "uvicorn",
            "uvicorn.error",
            "uvicorn.access",
            "fastapi",
            "httpx",
        )
    )

    logger.info("Logger initialized successfully")


# Initialize logger on module import
setup_logger()
