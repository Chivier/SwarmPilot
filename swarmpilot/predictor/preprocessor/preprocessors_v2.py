"""Concrete V2 preprocessor implementations.

This module provides ready-to-use preprocessors for common operations:
- MultiplyPreprocessor: Multiply two features to create a new one
- RemoveFeaturePreprocessor: Remove specified features
- TokenLengthPreprocessor: Calculate token length from text
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from swarmpilot.predictor.preprocessor.base_preprocessor_v2 import BasePreprocessorV2
from swarmpilot.predictor.preprocessor.base_preprocessor_v2 import FeatureContext
from swarmpilot.predictor.preprocessor.base_preprocessor_v2 import OperationType


class MultiplyPreprocessor(BasePreprocessorV2):
    """Multiply two features to create a new feature.

    This preprocessor multiplies two input features and stores the result
    in a new output feature. Optionally, the input features can be removed
    after the multiplication.

    Example:
        >>> prep = MultiplyPreprocessor("width", "height", "pixel_num")
        >>> context = FeatureContext(features={"width": 100, "height": 200})
        >>> prep(context)
        >>> context.features
        {'width': 100, 'height': 200, 'pixel_num': 20000}

    Attributes:
        feature_a: First feature name to multiply.
        feature_b: Second feature name to multiply.
        output_feature: Name for the result feature.
        remove_inputs: Whether to remove input features after multiplication.
    """

    def __init__(
        self,
        feature_a: str,
        feature_b: str,
        output_feature: str,
        remove_inputs: bool = False,
    ) -> None:
        """Initialize multiply preprocessor.

        Args:
            feature_a: First feature name to multiply.
            feature_b: Second feature name to multiply.
            output_feature: Name for the result feature.
            remove_inputs: Whether to remove input features after.
        """
        super().__init__(
            name=f"multiply_{feature_a}_{feature_b}",
            operation_type=OperationType.TRANSFORM,
        )
        self.feature_a = feature_a
        self.feature_b = feature_b
        self.output_feature = output_feature
        self.remove_inputs = remove_inputs

    @property
    def input_features(self) -> list[str]:
        """Feature names this preprocessor requires as input."""
        return [self.feature_a, self.feature_b]

    @property
    def output_features(self) -> list[str]:
        """Feature names this preprocessor produces."""
        return [self.output_feature]

    @property
    def removes_features(self) -> list[str]:
        """Feature names this preprocessor removes."""
        if self.remove_inputs:
            return [self.feature_a, self.feature_b]
        return []

    def transform(self, context: FeatureContext) -> None:
        """Multiply the two features.

        Args:
            context: The feature context to transform.
        """
        val_a = context.get(self.feature_a)
        val_b = context.get(self.feature_b)

        result = val_a * val_b
        context.set(self.output_feature, result)

        if self.remove_inputs:
            context.remove(self.feature_a)
            context.remove(self.feature_b)


class RemoveFeaturePreprocessor(BasePreprocessorV2):
    """Remove specified features from the context.

    This preprocessor removes one or more features from the context.
    It silently ignores features that don't exist.

    Example:
        >>> prep = RemoveFeaturePreprocessor(["temp", "debug"])
        >>> context = FeatureContext(features={"temp": 1, "keep": 2})
        >>> prep(context)
        >>> context.features
        {'keep': 2}

    Attributes:
        features_to_remove: List of feature names to remove.
    """

    def __init__(self, features_to_remove: list[str]) -> None:
        """Initialize remove preprocessor.

        Args:
            features_to_remove: List of feature names to remove.
        """
        super().__init__(
            name=f"remove_{','.join(features_to_remove)}",
            operation_type=OperationType.REMOVE,
        )
        self._features_to_remove = features_to_remove

    @property
    def input_features(self) -> list[str]:
        """Feature names this preprocessor requires as input."""
        return self._features_to_remove

    @property
    def output_features(self) -> list[str]:
        """Feature names this preprocessor produces."""
        return []

    @property
    def removes_features(self) -> list[str]:
        """Feature names this preprocessor removes."""
        return self._features_to_remove

    def validate_inputs(self, context: FeatureContext) -> list[str]:
        """Override to not require features to exist for removal.

        Args:
            context: The feature context to validate.

        Returns:
            Always returns empty list - features don't need to exist.
        """
        # Don't require features to exist for removal
        return []

    def transform(self, context: FeatureContext) -> None:
        """Remove the specified features.

        Args:
            context: The feature context to transform.
        """
        for feature in self._features_to_remove:
            context.remove(feature)


class TokenLengthPreprocessor(BasePreprocessorV2):
    """Calculate token length from text feature.

    This preprocessor tokenizes a text feature and outputs the token count.
    Uses whitespace tokenization by default, but supports custom tokenizers.

    Example:
        >>> prep = TokenLengthPreprocessor("prompt", "input_length")
        >>> context = FeatureContext(features={"prompt": "Hello world"})
        >>> prep(context)
        >>> context.features
        {'prompt': 'Hello world', 'input_length': 2}

    Attributes:
        input_feature: Name of text feature to tokenize.
        output_feature: Name for the token count feature.
        tokenizer: Function to tokenize text.
        remove_input: Whether to remove the input feature.
    """

    def __init__(
        self,
        input_feature: str,
        output_feature: str = "input_length",
        tokenizer: Callable[[str], list[Any]] | None = None,
        remove_input: bool = False,
    ) -> None:
        """Initialize token length preprocessor.

        Args:
            input_feature: Name of text feature to tokenize.
            output_feature: Name for the token count feature.
            tokenizer: Optional tokenizer function. Defaults to whitespace split.
            remove_input: Whether to remove the input feature.
        """
        super().__init__(
            name=f"token_length_{input_feature}",
            operation_type=OperationType.TRANSFORM,
        )
        self._input_feature = input_feature
        self._output_feature = output_feature
        self._tokenizer = tokenizer or (lambda x: x.split())
        self._remove_input = remove_input

    @property
    def input_features(self) -> list[str]:
        """Feature names this preprocessor requires as input."""
        return [self._input_feature]

    @property
    def output_features(self) -> list[str]:
        """Feature names this preprocessor produces."""
        return [self._output_feature]

    @property
    def removes_features(self) -> list[str]:
        """Feature names this preprocessor removes."""
        if self._remove_input:
            return [self._input_feature]
        return []

    def transform(self, context: FeatureContext) -> None:
        """Calculate token length.

        Args:
            context: The feature context to transform.
        """
        text = context.get(self._input_feature)
        tokens = self._tokenizer(text)
        context.set(self._output_feature, len(tokens))

        if self._remove_input:
            context.remove(self._input_feature)
