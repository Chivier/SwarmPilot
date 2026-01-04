"""V2 preprocessor base classes with flexible feature operations.

This module provides the foundation for the V2 preprocessing system that allows:
- Modifying feature values in place
- Adding new computed features
- Removing specific features
- Working with any feature type (not just text)
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Iterator


class OperationType(Enum):
    """Types of operations a preprocessor can perform."""

    MODIFY = "modify"
    ADD = "add"
    REMOVE = "remove"
    TRANSFORM = "transform"


@dataclass
class FeatureContext:
    """Context object for tracking feature state during preprocessing.

    FeatureContext provides a mutable container for features that tracks
    all modifications (additions, modifications, removals) during the
    preprocessing pipeline. This enables auditability and debugging.

    Attributes:
        features: Current mutable feature dictionary.
        original_features: Immutable copy of original features at creation.
        removed_features: Set of feature names that were removed.
        added_features: Set of feature names that were added.
        modified_features: Set of feature names that were modified.

    Example:
        >>> context = FeatureContext(features={"width": 100, "height": 200})
        >>> context.set("pixel_num", context.get("width") * context.get("height"))
        >>> context.remove("width")
        >>> context.remove("height")
        >>> context.features
        {'pixel_num': 20000}
    """

    features: dict[str, Any]
    original_features: dict[str, Any] = field(default_factory=dict)
    removed_features: set[str] = field(default_factory=set)
    added_features: set[str] = field(default_factory=set)
    modified_features: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Store original features if not provided."""
        if not self.original_features:
            self.original_features = self.features.copy()

    def get(self, name: str, default: Any = None) -> Any:
        """Get a feature value.

        Args:
            name: Feature name to retrieve.
            default: Value to return if feature doesn't exist.

        Returns:
            The feature value or default if not found.
        """
        return self.features.get(name, default)

    def set(self, name: str, value: Any) -> None:
        """Set a feature value, tracking if it's new or modified.

        Args:
            name: Feature name to set.
            value: Value to assign to the feature.
        """
        if name in self.features:
            self.modified_features.add(name)
        else:
            self.added_features.add(name)
        self.features[name] = value

    def remove(self, name: str) -> Any | None:
        """Remove a feature, returning its value.

        Args:
            name: Feature name to remove.

        Returns:
            The removed feature's value, or None if it didn't exist.
        """
        if name in self.features:
            self.removed_features.add(name)
            return self.features.pop(name)
        return None

    def has(self, name: str) -> bool:
        """Check if a feature exists.

        Args:
            name: Feature name to check.

        Returns:
            True if the feature exists, False otherwise.
        """
        return name in self.features

    def keys(self) -> Iterator[str]:
        """Get current feature names.

        Returns:
            Iterator of feature names.
        """
        return iter(self.features.keys())


class BasePreprocessorV2(ABC):
    """Base class for V2 preprocessors with flexible feature operations.

    V2 preprocessors work with FeatureContext to allow:
    - Modifying feature values in place
    - Adding new features
    - Removing specific features
    - Working with any feature type

    Subclasses must implement:
    - input_features property: Features required as input
    - output_features property: Features produced as output
    - transform method: The actual transformation logic

    Attributes:
        name: Unique name for the preprocessor.
        operation_type: Type of operation this preprocessor performs.

    Example:
        >>> class DoublePreprocessor(BasePreprocessorV2):
        ...     @property
        ...     def input_features(self) -> list[str]:
        ...         return ["value"]
        ...     @property
        ...     def output_features(self) -> list[str]:
        ...         return ["doubled"]
        ...     def transform(self, context: FeatureContext) -> None:
        ...         context.set("doubled", context.get("value") * 2)
    """

    def __init__(
        self,
        name: str,
        operation_type: OperationType = OperationType.TRANSFORM,
    ) -> None:
        """Initialize the preprocessor.

        Args:
            name: Unique name for this preprocessor instance.
            operation_type: Type of operation performed.
        """
        self.name = name
        self.operation_type = operation_type

    @property
    @abstractmethod
    def input_features(self) -> list[str]:
        """Feature names this preprocessor requires as input."""
        pass

    @property
    @abstractmethod
    def output_features(self) -> list[str]:
        """Feature names this preprocessor produces."""
        pass

    @property
    def removes_features(self) -> list[str]:
        """Feature names this preprocessor removes. Override if needed."""
        return []

    @abstractmethod
    def transform(self, context: FeatureContext) -> None:
        """Apply transformation to the feature context.

        This method should modify the context in place:
        - Use context.set() to add/modify features
        - Use context.remove() to remove features
        - Use context.get() to read features

        Args:
            context: The feature context to transform.
        """
        pass

    def validate_inputs(self, context: FeatureContext) -> list[str]:
        """Validate that required input features are present.

        Args:
            context: The feature context to validate.

        Returns:
            List of missing feature names (empty if all present).
        """
        return [f for f in self.input_features if not context.has(f)]

    def __call__(self, context: FeatureContext) -> None:
        """Callable interface for the preprocessor.

        Args:
            context: The feature context to transform.

        Raises:
            ValueError: If required input features are missing.
        """
        missing = self.validate_inputs(context)
        if missing:
            available = list(context.keys())
            raise ValueError(
                f"Preprocessor '{self.name}' missing required features: {missing}. "
                f"Available features: {available}"
            )
        self.transform(context)
