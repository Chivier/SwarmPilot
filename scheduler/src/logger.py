"""Centralized logging configuration.

This module provides backward compatibility by re-exporting
from src.utils.logger.
"""

from src.utils.logger import InterceptHandler, setup_logger

__all__ = ["InterceptHandler", "setup_logger"]
