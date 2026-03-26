"""Shared logging utilities for bridging stdlib logging to loguru.

This module provides :class:`InterceptHandler` and
:func:`intercept_standard_logging`, which are used by all three
SwarmPilot services (Scheduler, Predictor, Planner) to route
standard-library log records through loguru.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from loguru import logger


class InterceptHandler(logging.Handler):
    """Intercept standard logging calls and redirect them to loguru.

    Attach this handler to any stdlib logger to forward its
    records through loguru for consistent formatting and output.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record by forwarding it to loguru.

        Args:
            record: The log record from standard logging.
        """
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


DEFAULT_INTERCEPTED_LOGGERS = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "fastapi",
)


def intercept_standard_logging(
    logger_names: tuple[str, ...] = DEFAULT_INTERCEPTED_LOGGERS,
) -> None:
    """Replace stdlib handlers with :class:`InterceptHandler`.

    This sets up ``logging.basicConfig`` with an
    :class:`InterceptHandler` and replaces handlers on the
    specified loggers so all output goes through loguru.

    Args:
        logger_names: Names of stdlib loggers to intercept.
            Defaults to uvicorn and fastapi loggers.
    """
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in logger_names:
        stdlib_logger = logging.getLogger(name)
        stdlib_logger.handlers = [InterceptHandler()]
        stdlib_logger.propagate = False


def configure_loguru(
    *,
    log_dir: str,
    log_level: str,
    file_pattern: str,
    logger_names: tuple[str, ...] = DEFAULT_INTERCEPTED_LOGGERS,
    rotation: str = "00:00",
    retention: str = "30 days",
    compression: str = "zip",
    colorize_console: bool = False,
    json_log_file_pattern: str | None = None,
    error_log_file_pattern: str | None = None,
    error_retention: str = "90 days",
    enqueue: bool = False,
    backtrace: bool = False,
    diagnose: bool = False,
) -> None:
    """Configure shared loguru sinks and stdlib interception.

    Args:
        log_dir: Directory where log files are written.
        log_level: Minimum log level for configured sinks.
        file_pattern: Main log file naming pattern accepted by loguru.
        logger_names: Stdlib logger names to intercept.
        rotation: Rotation policy for main and optional sinks.
        retention: Retention policy for main and optional JSON sinks.
        compression: Compression algorithm for rotated files.
        colorize_console: Whether to colorize console output.
        json_log_file_pattern: Optional JSON file naming pattern.
        error_log_file_pattern: Optional error-only file naming pattern.
        error_retention: Retention policy for error-only files.
        enqueue: Whether file sinks should enqueue logs asynchronously.
        backtrace: Whether to enable enriched tracebacks in sinks.
        diagnose: Whether to enable variable diagnosis in tracebacks.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} - "
        "{message}"
    )

    logger.remove()

    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        colorize=colorize_console,
        backtrace=backtrace,
        diagnose=diagnose,
    )

    logger.add(
        log_path / file_pattern,
        format=log_format,
        level=log_level,
        rotation=rotation,
        retention=retention,
        compression=compression,
        encoding="utf-8",
        enqueue=enqueue,
        backtrace=backtrace,
        diagnose=diagnose,
    )

    if error_log_file_pattern is not None:
        logger.add(
            log_path / error_log_file_pattern,
            format=log_format,
            level="ERROR",
            rotation=rotation,
            retention=error_retention,
            compression=compression,
            encoding="utf-8",
            enqueue=enqueue,
            backtrace=backtrace,
            diagnose=diagnose,
        )

    if json_log_file_pattern is not None:
        logger.add(
            log_path / json_log_file_pattern,
            format="{message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            encoding="utf-8",
            serialize=True,
            enqueue=enqueue,
            backtrace=backtrace,
            diagnose=diagnose,
        )

    intercept_standard_logging(logger_names)
    logger.info(
        "Logging configured: level={}, log_dir={}",
        log_level,
        log_path.absolute(),
    )
