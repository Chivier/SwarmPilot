"""Tests for preprocessor functionality.

These tests require the semantic model checkpoint which may not be
available in CI environments. Tests are skipped if the model is missing.
"""

import time
from pathlib import Path

import pytest

from src.preprocessor.preprocessors_registry import PreprocessorsRegistry


# Get predictor directory (parent of tests directory)
PREDICTOR_DIR = Path(__file__).parent.parent
MODEL_PATH = PREDICTOR_DIR / "preprocessors" / "sematic_35M" / "model_35M.pt"
CONFIG_PATH = PREDICTOR_DIR / "preprocessors" / "sematic_35M" / "model_35M.yaml"

# Skip condition for tests requiring the semantic model
requires_semantic_model = pytest.mark.skipif(
    not (MODEL_PATH.exists() and CONFIG_PATH.exists()),
    reason="Semantic model checkpoint not available (CI environment)",
)


@requires_semantic_model
def test_preprocessor_registry():
    """Test that the preprocessor registry loads the semantic preprocessor."""
    from src.preprocessor.semantic_predictor import SemanticPredictor

    registry = PreprocessorsRegistry()
    assert registry is not None
    assert registry.get_preprocessor("semantic") is not None
    assert isinstance(registry.get_preprocessor("semantic"), SemanticPredictor)


@requires_semantic_model
def test_semantic_predictor():
    """Test SemanticPredictor initialization and prediction."""
    from src.preprocessor.semantic_predictor import SemanticPredictor

    start_time = time.time()
    predictor = SemanticPredictor(
        model_path=str(MODEL_PATH), model_config_path=str(CONFIG_PATH)
    )
    load_end_time = time.time()

    assert predictor is not None

    predict_start_time = time.time()
    output, remove_origin = predictor.predict(["Hello, world!"])
    predict_end_time = time.time()

    assert isinstance(output, dict)
    assert "output_length" in output
    assert isinstance(output["output_length"], int)
    assert isinstance(remove_origin, bool)
    assert remove_origin is True

    print(f"Load time: {load_end_time - start_time} seconds")
    print(f"Predict time: {predict_end_time - predict_start_time} seconds")
    print(f"Predict output length of 'Hello, world!': {output['output_length']}")


@requires_semantic_model
def test_semantic_predictor_multiple_inputs():
    """Test that SemanticPredictor rejects multiple inputs."""
    from src.preprocessor.semantic_predictor import SemanticPredictor

    predictor = SemanticPredictor(
        model_path=str(MODEL_PATH), model_config_path=str(CONFIG_PATH)
    )
    try:
        predictor.predict(["Hello, world!", "Hello, world!"])
        assert False, "Expected AssertionError for multiple inputs"
    except AssertionError as e:
        assert "SemanticPredictor only supports one input text" in str(e)
        print("Correctly raised AssertionError for multiple inputs")


@requires_semantic_model
def test_semantic_predictor_with_preprocessor_registry():
    """Test SemanticPredictor accessed through the registry."""
    from src.preprocessor.semantic_predictor import SemanticPredictor

    registry = PreprocessorsRegistry()
    predictor = registry.get_preprocessor("semantic")
    assert predictor is not None
    assert isinstance(predictor, SemanticPredictor)
    output, remove_origin = predictor.predict(["Hello, world!"])
    assert isinstance(output, dict)
    assert "output_length" in output
    assert isinstance(output["output_length"], int)
    assert isinstance(remove_origin, bool)
    assert remove_origin is True
