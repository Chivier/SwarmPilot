"""External service clients package.

This package contains clients for the predictor and training
data collection, using direct library imports (no HTTP).
"""

from swarmpilot.scheduler.clients.models import Prediction, TrainingSample
from swarmpilot.scheduler.clients.predictor_library_client import PredictorClient
from swarmpilot.scheduler.clients.training_library_client import TrainingClient

__all__ = [
    "Prediction",
    "PredictorClient",
    "TrainingClient",
    "TrainingSample",
]
