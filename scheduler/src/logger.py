"""
Centralized logging configuration using loguru.

This module configures loguru for the entire scheduler application,
providing structured logging with rotation, retention, and customizable formats.
"""

import sys
from pathlib import Path
from loguru import logger
from .config import config


def setup_logger():
    """
    Configure loguru logger with application-specific settings.

    This function:
    - Removes default handler
    - Adds console handler with custom format
    - Adds file handler with rotation and retention
    - Sets log level from configuration
    """
    # Remove default handler
    logger.remove()

    # Console handler with colored output
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=config.logging.level,
        colorize=True,
    )

    # File handler with rotation
    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "scheduler_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level=config.logging.level,
        rotation="00:00",  # Rotate at midnight
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress rotated logs
        encoding="utf-8",
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

    logger.info("Logger initialized successfully")


# Initialize logger on module import
setup_logger()
