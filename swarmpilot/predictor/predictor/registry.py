"""Predictor type registry and factory."""

from __future__ import annotations

from swarmpilot.predictor.predictor.base import BasePredictor
from swarmpilot.predictor.predictor.expect_error import ExpectErrorPredictor
from swarmpilot.predictor.predictor.quantile import QuantilePredictor

PREDICTOR_CLASSES: dict[str, type[BasePredictor]] = {
    "expect_error": ExpectErrorPredictor,
    "quantile": QuantilePredictor,
}


def create_predictor(prediction_type: str) -> BasePredictor:
    """Create a predictor instance by type name.

    Args:
        prediction_type: One of the keys in PREDICTOR_CLASSES.

    Returns:
        A new predictor instance.

    Raises:
        ValueError: If prediction_type is not recognized.
    """
    cls = PREDICTOR_CLASSES.get(prediction_type)
    if cls is None:
        raise ValueError(
            f"prediction_type must be one of {list(PREDICTOR_CLASSES.keys())}, "
            f"got '{prediction_type}'"
        )
    return cls()
