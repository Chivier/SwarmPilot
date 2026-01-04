"""API route modules."""

from src.api.routes import cache
from src.api.routes import health
from src.api.routes import models
from src.api.routes import prediction
from src.api.routes import prediction_v2
from src.api.routes import training
from src.api.routes import training_v2
from src.api.routes import websocket

__all__ = [
    "cache",
    "health",
    "models",
    "prediction",
    "prediction_v2",
    "training",
    "training_v2",
    "websocket",
]
