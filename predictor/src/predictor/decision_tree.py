"""
Decision Tree predictor.

Provides expected runtime and error margin predictions using Scikit-Learn's DecisionTreeRegressor.
"""

import numpy as np
import pickle
import base64
from typing import Any, Dict, List
from sklearn.tree import DecisionTreeRegressor

from .base import BasePredictor


class DecisionTreePredictor(BasePredictor):
    """
    Predictor for expect/error prediction type using Decision Tree Regression.

    Trains a Decision Tree model to predict expected runtime and computes error margin from residuals.
    """

    def __init__(self):
        """Initialize the predictor."""
        self.model = None
        self.feature_names = None
        self.mean_error = None

    def train(self, features_list: List[Dict[str, Any]], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train the predictor on the given data.

        Args:
            features_list: List of training samples with features and runtime_ms
            config: Optional training configuration. Supported keys:
                - max_depth: Maximum depth of the tree (default: None)
                - min_samples_split: Minimum samples required to split (default: 2)

        Returns:
            Training metadata dictionary

        Raises:
            ValueError: If training data is insufficient or invalid
        """
        # Validate minimum samples
        if len(features_list) < 10:
            raise ValueError(
                f"Insufficient training data: need at least 10 samples, got {len(features_list)}"
            )

        # Extract features and labels
        X, y, feature_names = self.extract_features_and_labels(features_list)
        self.feature_names = feature_names

        # Convert to numpy arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Get hyperparameters from config
        if config is None:
            config = {}
        max_depth = config.get('max_depth', None)
        min_samples_split = config.get('min_samples_split', 2)

        # Create and train model
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        self.model.fit(X, y)

        # Compute error margin from residuals
        predictions = self.model.predict(X)
        residuals = np.abs(y - predictions)
        self.mean_error = float(np.mean(residuals))

        return {
            'feature_names': self.feature_names,
            'samples_count': len(features_list),
            'mean_error': self.mean_error,
            'r2_score': float(self.model.score(X, y))
        }

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for the given features.

        Args:
            features: Feature dictionary

        Returns:
            Dict with expected_runtime_ms and error_margin_ms

        Raises:
            ValueError: If model not trained or features invalid
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Validate features
        self.validate_features(features, self.feature_names)

        # Extract feature values in correct order
        feature_values = [features[fname] for fname in self.feature_names]
        X = np.array([feature_values], dtype=np.float32)

        # Predict
        prediction = self.model.predict(X)[0]

        return {
            'expected_runtime_ms': float(prediction),
            'error_margin_ms': float(self.mean_error)
        }

    def get_model_state(self) -> Dict[str, Any]:
        """
        Get complete model state for serialization.

        Returns:
            Dict containing all model parameters and metadata
        """
        if self.model is None:
            raise ValueError("No model to serialize")

        # Serialize model using pickle and base64
        model_bytes = pickle.dumps(self.model)
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')

        return {
            'model_b64': model_b64,
            'feature_names': self.feature_names,
            'mean_error': self.mean_error
        }

    def load_model_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved model state.

        Args:
            state: Model state dict from get_model_state()
        """
        # Deserialize model
        model_bytes = base64.b64decode(state['model_b64'])
        self.model = pickle.loads(model_bytes)
        
        # Restore metadata
        self.feature_names = state['feature_names']
        self.mean_error = state['mean_error']
