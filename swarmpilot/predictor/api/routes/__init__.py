"""API route modules."""

from swarmpilot.predictor.api.routes import cache
from swarmpilot.predictor.api.routes import health
from swarmpilot.predictor.api.routes import models
from swarmpilot.predictor.api.routes import prediction
from swarmpilot.predictor.api.routes import training
from swarmpilot.predictor.api.routes import websocket

__all__ = [
    "cache",
    "health",
    "models",
    "prediction",
    "training",
    "websocket",
]
