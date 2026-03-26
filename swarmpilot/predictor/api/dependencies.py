"""Shared dependencies and utilities for API endpoints."""

from __future__ import annotations

import traceback
from typing import Any

from swarmpilot.predictor.api.cache import ModelCache
from swarmpilot.predictor.config import get_config
from swarmpilot.predictor.preprocessor.preprocessors_registry import (
    PreprocessorsRegistry,
)
from swarmpilot.predictor.storage.model_storage import ModelStorage
from swarmpilot.predictor.utils.logging import get_logger

logger = get_logger()


def _log_error(
    error_context: str,
    error_detail: dict[str, Any],
    exception: Exception | None = None,
    include_traceback: bool = True,
) -> None:
    """Log error with standardized format including context and traceback.

    Args:
        error_context: Description of where/what the error occurred.
        error_detail: The error detail dict that will be returned to client.
        exception: The exception object (if any).
        include_traceback: Whether to include full traceback in log.
    """
    log_lines = [
        f"Error occurred: {error_context}",
        f"Client response: {error_detail}",
    ]

    if exception:
        log_lines.append(f"Exception type: {type(exception).__name__}")
        log_lines.append(f"Exception message: {exception!s}")

    if include_traceback:
        log_lines.append(f"Traceback:\n{traceback.format_exc()}")

    logger.error("\n".join(log_lines))


def get_storage() -> ModelStorage:
    """Get ModelStorage instance using current configuration.

    Returns:
        ModelStorage instance configured with storage directory from config.
    """
    config = get_config()
    return ModelStorage(storage_dir=config.storage_dir)


# Global instances
storage = get_storage()
preprocessors_registry = PreprocessorsRegistry()
model_cache = ModelCache(max_size=100)
