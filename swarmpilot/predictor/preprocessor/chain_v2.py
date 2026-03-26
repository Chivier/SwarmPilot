"""Preprocessor chain for V2 preprocessing system.

This module provides the PreprocessorChainV2 class that orchestrates
the execution of multiple preprocessors in sequence.
"""

from __future__ import annotations

from typing import Any, Self

from swarmpilot.predictor.preprocessor.base_preprocessor_v2 import (
    BasePreprocessorV2,
    FeatureContext,
)


class PreprocessorChainV2:
    """Orchestrates sequential execution of V2 preprocessors.

    A chain manages an ordered list of preprocessors and executes them
    in sequence, passing the modified FeatureContext through each step.
    Supports fluent interface for chain building.

    Example:
        >>> chain = (
        ...     PreprocessorChainV2(name="image_pipeline")
        ...     .add(MultiplyPreprocessor("width", "height", "pixels"))
        ...     .add(RemoveFeaturePreprocessor(["width", "height"]))
        ... )
        >>> result = chain.transform({"width": 100, "height": 200})
        >>> result
        {'pixels': 20000}

    Attributes:
        name: Identifier for this chain.
        preprocessors: Ordered list of preprocessors in the chain.
    """

    def __init__(self, name: str) -> None:
        """Initialize an empty chain.

        Args:
            name: Identifier for this chain.
        """
        self._name = name
        self._preprocessors: list[BasePreprocessorV2] = []

    @property
    def name(self) -> str:
        """Name identifier for this chain."""
        return self._name

    @property
    def preprocessors(self) -> list[BasePreprocessorV2]:
        """Ordered list of preprocessors in the chain."""
        return self._preprocessors

    def add(self, preprocessor: BasePreprocessorV2) -> Self:
        """Append a preprocessor to the end of the chain.

        Args:
            preprocessor: The preprocessor to add.

        Returns:
            Self for fluent interface chaining.
        """
        self._preprocessors.append(preprocessor)
        return self

    def insert(self, index: int, preprocessor: BasePreprocessorV2) -> Self:
        """Insert a preprocessor at a specific position.

        Args:
            index: Position to insert at (0-based).
            preprocessor: The preprocessor to insert.

        Returns:
            Self for fluent interface chaining.
        """
        self._preprocessors.insert(index, preprocessor)
        return self

    def get_required_inputs(self) -> set[str]:
        """Calculate the initial features required by this chain.

        Analyzes the chain to determine which features must be present
        in the initial input. Features produced by earlier preprocessors
        in the chain are not considered required from the initial input.

        Returns:
            Set of feature names required from initial input.
        """
        required: set[str] = set()
        available: set[str] = set()

        for prep in self._preprocessors:
            for input_feature in prep.input_features:
                if input_feature not in available:
                    required.add(input_feature)

            for output_feature in prep.output_features:
                available.add(output_feature)

        return required

    def get_final_outputs(self) -> set[str]:
        """Get features that will be produced by this chain.

        Returns:
            Set of feature names produced by the chain.
        """
        outputs: set[str] = set()

        for prep in self._preprocessors:
            for output_feature in prep.output_features:
                outputs.add(output_feature)

        return outputs

    def validate(self, initial_features: set[str]) -> list[str]:
        """Validate that the chain can run with given initial features.

        Checks each preprocessor in order, tracking which features are
        available at each step (initial + outputs from previous steps).

        Args:
            initial_features: Set of features available at chain start.

        Returns:
            List of error messages. Empty list if valid.
        """
        errors: list[str] = []
        available = set(initial_features)

        for prep in self._preprocessors:
            for input_feature in prep.input_features:
                if input_feature not in available:
                    errors.append(
                        f"Preprocessor '{prep.name}' requires feature "
                        f"'{input_feature}' which is not available"
                    )

            for output_feature in prep.output_features:
                available.add(output_feature)

            for removed in prep.removes_features:
                available.discard(removed)

        return errors

    def transform(self, features: dict[str, Any]) -> dict[str, Any]:
        """Apply the chain to a features dictionary.

        Creates a FeatureContext from the input, runs all preprocessors
        in sequence, and returns the final features dictionary.

        Args:
            features: Input features dictionary. Not modified.

        Returns:
            New dictionary with transformed features.

        Raises:
            ValueError: If any preprocessor fails validation.
        """
        context = FeatureContext(features=features.copy())

        for prep in self._preprocessors:
            prep(context)

        return dict(context.features)

    def __call__(self, features: dict[str, Any]) -> dict[str, Any]:
        """Apply the chain to features (callable interface).

        Args:
            features: Input features dictionary.

        Returns:
            New dictionary with transformed features.
        """
        return self.transform(features)
