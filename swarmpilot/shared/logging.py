"""Shared logging utilities for bridging stdlib logging to loguru.

This module provides :class:`InterceptHandler` and
:func:`intercept_standard_logging`, which are used by all three
SwarmPilot services (Scheduler, Predictor, Planner) to route
standard-library log records through loguru.
"""

from __future__ import annotations

import logging
import sys

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
    logging.basicConfig(
        handlers=[InterceptHandler()], level=0, force=True
    )
    for name in logger_names:
        stdlib_logger = logging.getLogger(name)
        stdlib_logger.handlers = [InterceptHandler()]
        stdlib_logger.propagate = False
