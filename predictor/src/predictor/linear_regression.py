"""
Linear Regression predictor.

Provides expected runtime and error margin predictions using Scikit-Learn.
"""

import traceback
import numpy as np
from typing import Any, Dict, List
from sklearn.linear_model import LinearRegression

from .base import BasePredictor
from ..utils.logging import get_logger

logger = get_logger()


class LinearRegressionPredictor(BasePredictor):
    """
    Predictor for expect/error prediction type using Linear Regression.

    Trains a Linear Regression model to predict expected runtime and computes error margin from residuals.
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
            config: Optional training configuration (not used for LinearRegression but kept for interface)

        Returns:
            Training metadata dictionary

        Raises:
            ValueError: If training data is insufficient or invalid
        """
        # Validate minimum samples
        if len(features_list) < 10:
            error_msg = f"Insufficient training data: need at least 10 samples, got {len(features_list)}"
            logger.error(
                f"LinearRegressionPredictor training failed\n"
                f"Error: {error_msg}\n"
                f"Samples provided: {len(features_list)}\n"
                f"Minimum required: 10"
            )
            raise ValueError(error_msg)

        # Extract features and labels
        X, y, feature_names = self.extract_features_and_labels(features_list)
        self.feature_names = feature_names

        # Log selected features for training
        print(f"Training with {len(feature_names)} features: {feature_names}")

        # Convert to numpy arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Create and train model
        self.model = LinearRegression()
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
            error_msg = "Model not trained. Call train() first."
            logger.error(f"LinearRegressionPredictor prediction failed: {error_msg}")
            raise ValueError(error_msg)

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
            error_msg = "No model to serialize"
            logger.error(f"LinearRegressionPredictor get_model_state failed: {error_msg}")
            raise ValueError(error_msg)

        return {
            'coef': self.model.coef_.tolist(),
            'intercept': float(self.model.intercept_),
            'feature_names': self.feature_names,
            'mean_error': self.mean_error
        }

    def load_model_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved model state.

        Args:
            state: Model state dict from get_model_state()
        """
        # Recreate model
        self.model = LinearRegression()
        
        # Manually set parameters
        self.model.coef_ = np.array(state['coef'])
        self.model.intercept_ = state['intercept']
        
        # Restore metadata
        self.feature_names = state['feature_names']
        self.mean_error = state['mean_error']
