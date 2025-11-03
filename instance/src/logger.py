"""
Centralized logging configuration using loguru.

This module configures loguru for the entire instance application,
providing structured logging with rotation, retention, and customizable formats.
It intercepts uvicorn and FastAPI logs to provide consistent logging.
"""

import logging
import sys
from pathlib import Path
from loguru import logger
from .config import config


class InterceptHandler(logging.Handler):
    """
    Intercept standard logging messages and redirect them to loguru.

    This handler is used to capture logs from libraries that use the standard
    logging module (like uvicorn, fastapi, httpx) and route them through loguru
    for consistent formatting and handling.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record by forwarding it to loguru.

        Args:
            record: The log record from standard logging
        """
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logger():
    """
    Configure loguru logger with application-specific settings.

    This function:
    - Removes default handler
    - Adds console handler with custom format
    - Adds file handler with rotation and retention
    - Sets log level from configuration
    - Intercepts uvicorn and FastAPI logs
    """
    # Remove default handler
    logger.remove()

    # Console handler with colored output
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=config.log_level,
        colorize=True,
    )

    # File handler with rotation (no colors for clean file output)
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "instance_{time}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level=config.log_level,
        rotation="00:00",  # Rotate at midnight
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress rotated logs
        encoding="utf-8",
        colorize=False,  # Disable ANSI color codes in file
    )

    # Add JSON file handler for structured logging (optional, can be enabled via config)
    if config.enable_json_logs:
        logger.add(
            log_dir / "instance_{time}.json",
            format="{message}",
            level=config.log_level,
            rotation="00:00",
            retention="30 days",
            compression="zip",
            encoding="utf-8",
            serialize=True,  # Output as JSON
        )

    # Intercept standard logging and redirect to loguru
    # This ensures all logs from uvicorn, fastapi, httpx, etc. go through loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Configure specific loggers to use our interceptor
    for logger_name in [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
        "httpx",
    ]:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False

    logger.info("Logger initialized successfully")


# Initialize logger on module import
setup_logger()
