"""API route modules."""

from swarmpilot.predictor.api.routes import (
    cache,
    health,
    models,
    prediction,
    training,
    websocket,
)

__all__ = [
    "cache",
    "health",
    "models",
    "prediction",
    "training",
    "websocket",
]
