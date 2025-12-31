"""Client for the predictor service.

This module provides backward compatibility by re-exporting
from src.clients.predictor_client.
"""

from src.clients.predictor_client import Prediction, PredictorClient

__all__ = ["PredictorClient", "Prediction"]
