"""Centralized logging configuration using loguru.

This module configures loguru for the entire scheduler application,
providing structured logging with rotation, retention, and customizable formats.
"""

from swarmpilot.scheduler.config import config
from swarmpilot.shared.logging import (
    InterceptHandler,  # noqa: F401
    configure_loguru,
)


def setup_logger():
    """Configure loguru logger with application-specific settings.

    This function:
    - Removes default handler
    - Adds console handler with custom format
    - Adds file handler with rotation and retention
    - Sets log level from configuration
    """
    json_pattern = (
        "scheduler_{time:YYYY-MM-DD}.json"
        if config.logging.enable_json_logs
        else None
    )
    configure_loguru(
        log_dir=config.logging.log_dir,
        log_level=config.logging.level,
        file_pattern="scheduler_{time}.log",
        logger_names=(
            "uvicorn",
            "uvicorn.error",
            "uvicorn.access",
            "fastapi",
            "httpx",
        ),
        json_log_file_pattern=json_pattern,
    )


# Initialize logger on module import
setup_logger()
