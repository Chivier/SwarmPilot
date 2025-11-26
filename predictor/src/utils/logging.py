"""Logging configuration using loguru.

This module configures loguru as the unified logging system for the entire application,
including FastAPI and uvicorn logs. It supports environment variables:

- PREDICTOR_LOG_DIR: Directory to store log files (default: logs/)
- PREDICTOR_LOGURU_LEVEL: Logging level (default: INFO)
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


class InterceptHandler(logging.Handler):
    """
    Intercept standard library logging and redirect to loguru.

    This handler captures logs from the standard logging module
    (used by uvicorn, fastapi, and other libraries) and routes them
    through loguru for unified logging.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to loguru."""
        # Get corresponding loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: Optional[str] = None,
    rotation: str = "100 MB",
    retention: str = "10 days",
    compression: str = "zip",
) -> None:
    """
    Setup loguru logging with file output and console output.

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
    log_level = (log_level or os.getenv("PREDICTOR_LOGURU_LEVEL", "INFO")).upper()

    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default loguru handler
    logger.remove()

    # Add console handler (stderr) without colors
    logger.add(
        sys.stderr,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        level=log_level,
        colorize=False,
        backtrace=True,
        diagnose=True,
    )

    # Add file handler with rotation and compression
    log_file = log_path / "predictor_{time}.log"
    logger.add(
        str(log_file),
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        level=log_level,
        rotation=rotation,
        retention=retention,
        compression=compression,
        backtrace=True,
        diagnose=True,
        enqueue=True,  # Thread-safe
    )

    # Intercept standard library logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Configure specific loggers
    loggers_to_intercept = [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
    ]

    for logger_name in loggers_to_intercept:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False

    logger.info(f"Logging initialized: level={log_level}, dir={log_path.absolute()}")


def get_logger():
    """
    Get the configured loguru logger instance.

    Returns:
        The loguru logger instance.
    """
    return logger
