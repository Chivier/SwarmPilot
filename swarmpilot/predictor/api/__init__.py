"""FastAPI application for runtime prediction service.

This module provides backwards-compatible imports for the API layer.
The implementation has been split into separate modules for better organization:

- cache.py: ModelCache class for caching predictor instances
- dependencies.py: Shared dependencies and utilities
- app.py: FastAPI application factory and lifespan management
- routes/: API endpoint handlers organized by domain
"""

from swarmpilot.predictor.api.app import app
from swarmpilot.predictor.api.app import create_app
from swarmpilot.predictor.api.app import lifespan
from swarmpilot.predictor.api.cache import ModelCache
from swarmpilot.predictor.api.dependencies import _log_error
from swarmpilot.predictor.api.dependencies import get_storage
from swarmpilot.predictor.api.dependencies import model_cache
from swarmpilot.predictor.api.dependencies import preprocessors_registry
from swarmpilot.predictor.api.dependencies import storage

# Re-export ModelStorage for backwards compatibility with test fixtures
from swarmpilot.predictor.storage.model_storage import ModelStorage

__all__ = [
    # App
    "app",
    "create_app",
    "lifespan",
    # Cache
    "ModelCache",
    "model_cache",
    # Dependencies
    "ModelStorage",
    "_log_error",
    "get_storage",
    "preprocessors_registry",
    "storage",
]
