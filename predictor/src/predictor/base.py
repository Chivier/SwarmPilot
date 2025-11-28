"""
Base predictor interface.

Defines the abstract base class for all predictor implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..utils.logging import get_logger

logger = get_logger()


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
            error_msg = (
                f"Missing required features: {sorted(missing)}. "
                f"Expected: {sorted(expected_features)}, "
                f"Got: {sorted(provided_features)}"
            )
            logger.error(
                f"Feature validation failed\n"
                f"Error: {error_msg}\n"
                f"Missing features: {sorted(missing)}\n"
                f"Provided features: {sorted(provided_features)}\n"
                f"Expected features: {sorted(expected_features)}"
            )
            raise ValueError(error_msg)

        # Check for extra features (warning via extra key, but not failing)
        extra = provided_features - expected_features_set
        if extra:
            logger.debug(f"Extra features provided (ignored): {sorted(extra)}")

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
            error_msg = "features_list is empty"
            logger.error(f"Feature extraction failed: {error_msg}")
            raise ValueError(error_msg)

        # Extract runtime_ms labels
        y = [sample['runtime_ms'] for sample in features_list]

        # Get feature names (all keys except runtime_ms) from first sample
        first_sample = features_list[0]
        feature_names = sorted([k for k in first_sample.keys() if k != 'runtime_ms'])

        if not feature_names:
            error_msg = "No features found (only runtime_ms present)"
            logger.error(
                f"Feature extraction failed\n"
                f"Error: {error_msg}\n"
                f"First sample keys: {list(first_sample.keys())}"
            )
            raise ValueError(error_msg)

        # Extract feature values in consistent order
        X = []
        for idx, sample in enumerate(features_list):
            sample_features = sorted([k for k in sample.keys() if k != 'runtime_ms'])

            # Validate all samples have same features
            if sample_features != feature_names:
                error_msg = (
                    f"Sample at index {idx} has different features. "
                    f"Expected: {feature_names}, Got: {sample_features}"
                )
                logger.error(
                    f"Feature extraction failed - inconsistent features\n"
                    f"Error: {error_msg}\n"
                    f"Sample index: {idx}\n"
                    f"Expected features: {feature_names}\n"
                    f"Got features: {sample_features}"
                )
                raise ValueError(error_msg)

            # Extract values in the same order as feature_names
            feature_values = [sample[fname] for fname in feature_names]
            X.append(feature_values)

        return X, y, feature_names

    def filter_constant_features(
        self, X: List[List[Any]], feature_names: List[str]
    ) -> tuple:
        """
        Filter out constant features (features with zero variance).

        Constant features provide no predictive value and can cause numerical
        issues during normalization (division by zero or near-zero std).

        Args:
            X: Feature matrix as list of lists
            feature_names: Ordered list of feature names

        Returns:
            Tuple of (filtered_X, filtered_feature_names, removed_features) where:
                filtered_X: Feature matrix with constant features removed
                filtered_feature_names: Feature names after filtering
                removed_features: List of feature names that were removed
        """
        if not X or not feature_names:
            return X, feature_names, []

        num_features = len(feature_names)
        removed_features = []
        keep_indices = []

        # Check each feature for variance
        for i in range(num_features):
            values = [row[i] for row in X]
            unique_values = set(values)

            if len(unique_values) <= 1:
                # Constant feature (0 or 1 unique value)
                removed_features.append(feature_names[i])
            else:
                keep_indices.append(i)

        # Filter X and feature_names
        filtered_feature_names = [feature_names[i] for i in keep_indices]
        filtered_X = [[row[i] for i in keep_indices] for row in X]

        return filtered_X, filtered_feature_names, removed_features

    def filter_features_for_prediction(
        self, features: Dict[str, Any], valid_feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Filter input features to only include valid (non-constant) features.

        Used during prediction to automatically remove features that were
        identified as constant during training.

        Args:
            features: Full feature dictionary from user
            valid_feature_names: List of feature names that should be kept

        Returns:
            Filtered feature dictionary containing only valid features
        """
        return {k: v for k, v in features.items() if k in valid_feature_names}
