"""Base preprocessor interface for feature transformation.

Defines the abstract interface that all preprocessors must implement.
"""

from __future__ import annotations

from typing import Any


class BasePreprocessor:
    """Base class for all preprocessors.

    Preprocessors transform input features into derived features
    that can be used for model training and prediction.
    """

    def __init__(self) -> None:
        """Initialize the preprocessor."""
        pass

    def __call__(
        self,
        input_text: list[str],
    ) -> tuple[dict[str, Any], bool]:
        """Transform input features.

        Args:
            input_text: List of input strings to process.

        Returns:
            Tuple of (output_dict, remove_origin) where output_dict contains
            the transformed features and remove_origin indicates whether
            to remove the original input features.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
