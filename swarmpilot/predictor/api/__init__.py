"""FastAPI application for runtime prediction service.

This module provides backwards-compatible imports for the API layer.
The implementation has been split into separate modules for better organization:

- cache.py: ModelCache class for caching predictor instances
- dependencies.py: Shared dependencies and utilities
- app.py: FastAPI application factory and lifespan management
- routes/: API endpoint handlers organized by domain
"""

from swarmpilot.predictor.api.app import app, create_app, lifespan
from swarmpilot.predictor.api.cache import ModelCache
from swarmpilot.predictor.api.dependencies import (
    _log_error,
    get_storage,
    model_cache,
    preprocessors_registry,
    storage,
)

# Re-export ModelStorage for backwards compatibility with test fixtures
from swarmpilot.predictor.storage.model_storage import ModelStorage

__all__ = [
    "ModelCache",
    "ModelStorage",
    "_log_error",
    "app",
    "create_app",
    "get_storage",
    "lifespan",
    "model_cache",
    "preprocessors_registry",
    "storage",
]
