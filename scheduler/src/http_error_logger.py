"""HTTP error logging utilities.

This module provides backward compatibility by re-exporting
from src.utils.http_error_logger.
"""

from loguru import logger

from src.utils.http_error_logger import (
    MAX_BODY_LENGTH,
    _sanitize_headers,
    _truncate_body,
    log_http_error,
)

__all__ = ["MAX_BODY_LENGTH", "_sanitize_headers", "_truncate_body", "log_http_error", "logger"]
