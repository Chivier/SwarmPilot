"""Adapter classes for bridging V1 and V2 preprocessor interfaces.

This module provides adapters that allow V1 preprocessors to be used
within the V2 preprocessing chain system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from swarmpilot.predictor.preprocessor.base_preprocessor_v2 import (
    BasePreprocessorV2,
    FeatureContext,
    OperationType,
)

if TYPE_CHECKING:
    from swarmpilot.predictor.preprocessor.base_preprocessor import (
        BasePreprocessor,
    )


class V1PreprocessorAdapter(BasePreprocessorV2):
    """Adapter to use V1 preprocessors in V2 chains.

    This adapter wraps a V1 BasePreprocessor and allows it to work with
    the V2 FeatureContext-based preprocessing system.

    V1 Interface:
        Input: list[str] (typically single text string)
        Output: tuple[dict[str, Any], bool] where dict contains derived
                features and bool indicates whether to remove input.

    V2 Interface:
        Input: FeatureContext with features dict
        Output: Modifies FeatureContext in place

    Example:
        >>> from swarmpilot.predictor.preprocessor.semantic_predictor import (
        ...     SemanticPredictor,
        ... )
        >>> v1_prep = SemanticPredictor(model_path, config_path)
        >>> adapter = V1PreprocessorAdapter(v1_prep, input_feature="prompt")
        >>> context = FeatureContext(features={"prompt": "Hello world"})
        >>> adapter(context)
        >>> context.features
        {'prompt': 'Hello world', 'output_length': 42}

    Attributes:
        v1_preprocessor: The wrapped V1 preprocessor instance.
        input_feature: Name of the feature to extract text from.
    """

    def __init__(
        self,
        v1_preprocessor: BasePreprocessor,
        input_feature: str,
        name: str | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            v1_preprocessor: The V1 preprocessor to wrap.
            input_feature: Name of the feature containing text to process.
            name: Optional custom name. Defaults to class name of V1 preprocessor.
        """
        adapter_name = (
            name or f"v1_adapter_{v1_preprocessor.__class__.__name__}"
        )
        super().__init__(
            name=adapter_name,
            operation_type=OperationType.TRANSFORM,
        )
        self.v1_preprocessor = v1_preprocessor
        self._input_feature = input_feature

    @property
    def input_features(self) -> list[str]:
        """Feature names this preprocessor requires as input."""
        return [self._input_feature]

    @property
    def output_features(self) -> list[str]:
        """Feature names this preprocessor produces.

        V1 preprocessors have dynamic output, so we can't declare
        output features statically. Returns empty list.
        """
        return []

    @property
    def removes_features(self) -> list[str]:
        """Feature names this preprocessor removes.

        Removal is decided at runtime by the V1 preprocessor's return
        value, so we can't declare it statically. Returns empty list.
        """
        return []

    def transform(self, context: FeatureContext) -> None:
        """Apply the V1 preprocessor to the context.

        Extracts the input feature value, passes it to the V1 preprocessor,
        and updates the context with the results.

        Args:
            context: The feature context to transform.
        """
        input_value = context.get(self._input_feature)

        output_dict, remove_origin = self.v1_preprocessor([input_value])

        for key, value in output_dict.items():
            context.set(key, value)

        if remove_origin:
            context.remove(self._input_feature)
