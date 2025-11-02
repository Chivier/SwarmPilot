"""
Base predictor interface.

Defines the abstract base class for all predictor implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BasePredictor(ABC):
    """Abstract base class for runtime predictors."""

    @abstractmethod
    def train(self, features_list: List[Dict[str, Any]], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train the predictor on the given feature data.

        Args:
            features_list: List of feature dictionaries with runtime_ms field
            config: Optional training configuration

        Returns:
            Dict containing training metadata (feature_names, samples_count, etc.)

        Raises:
            ValueError: If training data is invalid
        """
        pass

    @abstractmethod
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for the given features.

        Args:
            features: Dictionary of feature values

        Returns:
            Dict containing prediction results (format depends on predictor type)

        Raises:
            ValueError: If features are invalid or model not trained
        """
        pass

    @abstractmethod
    def get_model_state(self) -> Dict[str, Any]:
        """
        Get the complete model state for serialization.

        Returns:
            Dict containing all model parameters and metadata
        """
        pass

    @abstractmethod
    def load_model_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved model state.

        Args:
            state: Model state dict from get_model_state()
        """
        pass

    def validate_features(self, features: Dict[str, Any], expected_features: List[str]) -> None:
        """
        Validate that features match expected feature names.

        Args:
            features: Feature dictionary to validate
            expected_features: List of expected feature names

        Raises:
            ValueError: If features don't match expectations
        """
        provided_features = set(features.keys())
        expected_features_set = set(expected_features)

        # Check for missing features
        missing = expected_features_set - provided_features
        if missing:
            raise ValueError(
                f"Missing required features: {sorted(missing)}. "
                f"Expected: {sorted(expected_features)}, "
                f"Got: {sorted(provided_features)}"
            )

        # Check for extra features (warning via extra key, but not failing)
        extra = provided_features - expected_features_set
        if extra:
            # Store extra features for potential logging, but don't fail
            pass

    def extract_features_and_labels(self, features_list: List[Dict[str, Any]]) -> tuple:
        """
        Extract feature matrix X and label vector y from features_list.

        Args:
            features_list: List of feature dictionaries with runtime_ms

        Returns:
            Tuple of (X, y, feature_names) where:
                X: List of feature value lists
                y: List of runtime_ms values
                feature_names: Ordered list of feature names (excluding runtime_ms)

        Raises:
            ValueError: If samples have inconsistent features
        """
        if not features_list:
            raise ValueError("features_list is empty")

        # Extract runtime_ms labels
        y = [sample['runtime_ms'] for sample in features_list]

        # Get feature names (all keys except runtime_ms) from first sample
        first_sample = features_list[0]
        feature_names = sorted([k for k in first_sample.keys() if k != 'runtime_ms'])

        if not feature_names:
            raise ValueError("No features found (only runtime_ms present)")

        # Extract feature values in consistent order
        X = []
        for idx, sample in enumerate(features_list):
            sample_features = sorted([k for k in sample.keys() if k != 'runtime_ms'])

            # Validate all samples have same features
            if sample_features != feature_names:
                raise ValueError(
                    f"Sample at index {idx} has different features. "
                    f"Expected: {feature_names}, Got: {sample_features}"
                )

            # Extract values in the same order as feature_names
            feature_values = [sample[fname] for fname in feature_names]
            X.append(feature_values)

        return X, y, feature_names
