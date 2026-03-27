"""Shared data models for predictor and training clients.

This module contains dataclasses shared across client implementations,
extracted to avoid coupling to any specific transport mechanism.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Prediction:
    """Prediction result for a task on a specific instance."""

    instance_id: str
    predicted_time_ms: float
    confidence: float | None = None
    quantiles: dict[float, float] | None = (
        None  # e.g., {0.5: 120.5, 0.9: 250.3}
    )
    error_margin_ms: float | None = None


@dataclass
class TrainingSample:
    """A single training sample for the predictor."""

    model_id: str
    platform_info: dict[str, str]
    features: dict[str, Any]
    actual_runtime_ms: float
    timestamp: str
