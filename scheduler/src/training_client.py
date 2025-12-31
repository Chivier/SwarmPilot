"""Client for training data collection.

This module provides backward compatibility by re-exporting
from src.clients.training_client.
"""

from src.clients.training_client import TrainingClient, TrainingSample

__all__ = ["TrainingClient", "TrainingSample"]
