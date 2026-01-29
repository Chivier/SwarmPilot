"""External service clients package.

This package contains clients for the predictor and training
data collection, using direct library imports (no HTTP).
"""

from src.clients.models import Prediction, TrainingSample
from src.clients.predictor_library_client import PredictorClient
from src.clients.training_library_client import TrainingClient

__all__ = [
    "Prediction",
    "PredictorClient",
    "TrainingClient",
    "TrainingSample",
]
