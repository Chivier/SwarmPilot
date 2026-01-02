"""FastAPI application for runtime prediction service.

This module provides backwards-compatible imports for the API layer.
The implementation has been split into separate modules for better organization:

- cache.py: ModelCache class for caching predictor instances
- core.py: Library API (PredictorCore, PredictorLowLevel)
- dependencies.py: Shared dependencies and utilities
- app.py: FastAPI application factory and lifespan management
- routes/: API endpoint handlers organized by domain

Library API Usage:
    from src.api import PredictorCore, PredictorLowLevel

    # High-level API (recommended for most use cases)
    core = PredictorCore()
    core.collect(model_id, platform_info, prediction_type, features, runtime_ms)
    result = core.train(model_id, platform_info, prediction_type)
    prediction = core.predict(model_id, platform_info, prediction_type, features)

    # Low-level API (for advanced control)
    low = PredictorLowLevel()
    predictor = low.train_predictor(features_list, prediction_type)
    low.save_model(model_id, platform_info, prediction_type, predictor)
"""

from src.api.app import app
from src.api.app import create_app
from src.api.app import lifespan
from src.api.cache import ModelCache
from src.api.core import ModelNotFoundError
from src.api.core import PredictionError
from src.api.core import PredictorCore
from src.api.core import PredictorError
from src.api.core import PredictorLowLevel
from src.api.core import TrainingError
from src.api.core import ValidationError
from src.api.dependencies import _log_error
from src.api.dependencies import get_storage
from src.api.dependencies import model_cache
from src.api.dependencies import preprocessors_registry
from src.api.dependencies import storage

# Re-export ModelStorage for backwards compatibility with test fixtures
from src.storage.model_storage import ModelStorage

# Re-export library API result types from models
from src.models import CollectedSample
from src.models import ModelInfo
from src.models import PredictionResult
from src.models import PredictionType
from src.models import TrainingResult

__all__ = [
    # App
    "app",
    "create_app",
    "lifespan",
    # Cache
    "ModelCache",
    "model_cache",
    # Library API - Core Classes
    "PredictorCore",
    "PredictorLowLevel",
    # Library API - Exceptions
    "PredictorError",
    "ModelNotFoundError",
    "ValidationError",
    "TrainingError",
    "PredictionError",
    # Library API - Result Types
    "TrainingResult",
    "PredictionResult",
    "ModelInfo",
    "CollectedSample",
    "PredictionType",
    # Dependencies
    "ModelStorage",
    "_log_error",
    "get_storage",
    "preprocessors_registry",
    "storage",
]
