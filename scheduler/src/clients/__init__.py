"""External service clients package.

This package contains clients for external services like
the predictor and training data collection.
"""

from src.clients.predictor_client import Prediction, PredictorClient
from src.clients.training_client import TrainingClient, TrainingSample

__all__ = [
    "PredictorClient",
    "Prediction",
    "TrainingClient",
    "TrainingSample",
]
