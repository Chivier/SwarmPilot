"""Shared dependencies and utilities for API endpoints.

This module provides shared instances for the HTTP API layer:
- predictor_core: High-level API with accumulator pattern
- predictor_api: Low-level API for direct model control
- storage, model_cache: Legacy references for backwards compatibility
"""

from __future__ import annotations

import traceback
from typing import Any

from src.api.cache import ModelCache
from src.api.core import PredictorCore, PredictorLowLevel
from src.config import get_config
from src.preprocessor.preprocessors_registry import PreprocessorsRegistry
from src.storage.model_storage import ModelStorage
from src.utils.logging import get_logger

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
        log_lines.append(f"Exception message: {str(exception)}")

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


# =============================================================================
# Library API Instances (Recommended)
# =============================================================================

# High-level API with accumulator pattern (collect -> train -> predict)
predictor_core = PredictorCore()

# Low-level API for direct model control (train_predictor, save_model, etc.)
predictor_api = predictor_core._low_level


# =============================================================================
# Legacy Instances (Backwards Compatibility)
# =============================================================================

# These are kept for backwards compatibility with existing code.
# New code should use predictor_core or predictor_api instead.
storage = predictor_api._storage
preprocessors_registry = predictor_api._preprocessors_registry
model_cache = predictor_api._cache
