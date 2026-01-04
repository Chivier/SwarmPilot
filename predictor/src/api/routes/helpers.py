"""Common utilities for HTTP route handlers.

This module provides shared helper functions for HTTP routes, centralizing
common patterns like exception mapping and response formatting.
"""

from __future__ import annotations

from fastapi import HTTPException, status

from src.api.core import (
    ModelNotFoundError,
    PredictionError,
    PredictorError,
    TrainingError,
    ValidationError,
)
from src.utils.logging import get_logger

logger = get_logger()


def handle_library_exception(e: Exception, context: str = "") -> HTTPException:
    """Map library exceptions to HTTPException.

    Converts library-level exceptions into appropriate HTTP responses
    with consistent error format.

    Args:
        e: Exception from library API.
        context: Error context for logging (e.g., "train model_id=test").

    Returns:
        HTTPException with appropriate status code and detail.

    Example:
        >>> try:
        ...     result = predictor_api.load_model(...)
        ... except (ModelNotFoundError, ValidationError) as e:
        ...     raise handle_library_exception(e, "load model_id=test")
    """
    log_context = f" [{context}]" if context else ""

    if isinstance(e, ValidationError):
        logger.warning(f"Validation error{log_context}: {e}")
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Validation error",
                "message": str(e),
            },
        )

    if isinstance(e, ModelNotFoundError):
        logger.warning(f"Model not found{log_context}: {e}")
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "Model not found",
                "message": str(e),
            },
        )

    if isinstance(e, TrainingError):
        logger.error(f"Training failed{log_context}: {e}")
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Training failed",
                "message": str(e),
            },
        )

    if isinstance(e, PredictionError):
        logger.error(f"Prediction failed{log_context}: {e}")
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Prediction failed",
                "message": str(e),
            },
        )

    if isinstance(e, PredictorError):
        logger.error(f"Predictor error{log_context}: {e}")
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Predictor error",
                "message": str(e),
            },
        )

    # Fallback for unexpected exceptions
    logger.exception(f"Unexpected error{log_context}: {e}")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={
            "error": "Unexpected error",
            "message": str(e),
        },
    )
