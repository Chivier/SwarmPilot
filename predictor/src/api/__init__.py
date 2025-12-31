"""FastAPI application for runtime prediction service.

This module provides backwards-compatible imports for the API layer.
The implementation has been split into separate modules for better organization:

- cache.py: ModelCache class for caching predictor instances
- dependencies.py: Shared dependencies and utilities
- app.py: FastAPI application factory and lifespan management
- routes/: API endpoint handlers organized by domain
"""

from src.api.app import app
from src.api.app import create_app
from src.api.app import lifespan
from src.api.cache import ModelCache
from src.api.dependencies import _log_error
from src.api.dependencies import get_storage
from src.api.dependencies import model_cache
from src.api.dependencies import preprocessors_registry
from src.api.dependencies import storage

# Re-export ModelStorage for backwards compatibility with test fixtures
from src.storage.model_storage import ModelStorage

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
