"""Tests for preprocessor registry error handling."""

import pytest

from swarmpilot.predictor.preprocessor.preprocessors_registry import (
    PreprocessorsRegistry,
)


def test_get_preprocessor_not_found_raises_key_error() -> None:
    """Ensure unknown preprocessors raise KeyError."""
    registry = PreprocessorsRegistry()

    with pytest.raises(
        KeyError, match="Preprocessor nonexistent_preprocessor not found"
    ):
        registry.get_preprocessor("nonexistent_preprocessor")
