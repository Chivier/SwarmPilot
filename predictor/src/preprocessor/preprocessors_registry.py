"""Registry for available preprocessors.

Manages the collection of preprocessor instances available for feature
transformation during training and prediction.
"""

from __future__ import annotations

from pathlib import Path

from src.preprocessor.base_preprocessor import BasePreprocessor
from src.preprocessor.semantic_predictor import SemanticPredictor


class PreprocessorsRegistry:
    """Registry for managing preprocessor instances.

    Attributes:
        preprocessors: Dictionary mapping names to preprocessor instances.
    """

    def __init__(self) -> None:
        """Initialize the registry with available preprocessors."""
        # Get the predictor directory path (parent of src)
        predictor_dir = Path(__file__).parent.parent.parent
        model_dir = predictor_dir / "preprocessors" / "sematic_35M"

        self.preprocessors: dict[str, BasePreprocessor] = {
            "semantic": SemanticPredictor(
                str(model_dir / "model_35M.pt"),
                str(model_dir / "model_35M.yaml"),
            )
        }

    def get_preprocessor(self, preprocessor_name: str) -> BasePreprocessor:
        """Get a preprocessor by name.

        Args:
            preprocessor_name: Name of the preprocessor to retrieve.

        Returns:
            The preprocessor instance.

        Raises:
            AssertionError: If preprocessor name not found.
        """
        assert preprocessor_name in self.preprocessors, (
            f"Preprocessor {preprocessor_name} not found"
        )
        return self.preprocessors[preprocessor_name]
