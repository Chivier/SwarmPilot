"""Expect/Error predictor using MLP with MSE loss.

Provides expected runtime and error margin predictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch import optim

from src.predictor.base import BasePredictor
from src.predictor.mlp import MLP
from src.utils.logging import get_logger


logger = get_logger()


def _set_model_inference_mode(model: nn.Module) -> None:
    """Set model to inference mode (disables dropout, batchnorm training)."""
    model.train(False)


class ExpectErrorPredictor(BasePredictor):
    """Predictor for expect/error prediction type.

    Trains MLP to predict expected runtime and computes error margin
    from residuals.

    Attributes:
        model: The trained MLP model.
        feature_names: List of feature names used for training.
        mean_error: Mean absolute error computed from training residuals.
        feature_mean: Mean values for feature normalization.
        feature_std: Standard deviation values for feature normalization.
    """

    def __init__(self) -> None:
        """Initialize the predictor."""
        self.model: MLP | None = None
        self.feature_names: list[str] | None = None
        self.mean_error: float | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    def train(
        self,
        features_list: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Train the predictor on the given data.

        Args:
            features_list: List of training samples with features and
                runtime_ms.
            config: Optional training configuration (epochs, learning_rate).

        Returns:
            Training metadata dictionary.

        Raises:
            ValueError: If training data is insufficient or invalid.
        """
        # Validate minimum samples
        if len(features_list) < 10:
            error_msg = (
                f"Insufficient training data: need at least 10 samples, "
                f"got {len(features_list)}"
            )
            logger.error(
                f"ExpectErrorPredictor training failed\n"
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

        # Normalize features (z-score normalization)
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0) + 1e-8
        X_normalized = (X - self.feature_mean) / self.feature_std

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        # Get training hyperparameters from config
        if config is None:
            config = {}
        epochs = config.get('epochs', 500)
        learning_rate = config.get('learning_rate', 0.01)
        hidden_layers = config.get('hidden_layers', [64, 32])

        # Create and train model
        input_dim = len(feature_names)
        self.model = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_layers=hidden_layers,
        )

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.model(X_tensor)
            loss = criterion(predictions, y_tensor)
            loss.backward()
            optimizer.step()

        # Compute error margin from residuals
        _set_model_inference_mode(self.model)
        with torch.no_grad():
            final_predictions = self.model(X_tensor).numpy().flatten()
            residuals = np.abs(y - final_predictions)
            self.mean_error = float(np.mean(residuals))

        return {
            'feature_names': self.feature_names,
            'samples_count': len(features_list),
            'mean_error': self.mean_error,
            'final_loss': float(loss.item()),
        }

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Make a prediction for the given features.

        Args:
            features: Feature dictionary.

        Returns:
            Dict with expected_runtime_ms and error_margin_ms.

        Raises:
            ValueError: If model not trained or features invalid.
        """
        if self.model is None:
            error_msg = "Model not trained. Call train() first."
            logger.error(f"ExpectErrorPredictor prediction failed: {error_msg}")
            raise ValueError(error_msg)

        # Validate features
        self.validate_features(features, self.feature_names)

        # Extract feature values in correct order
        feature_values = [features[fname] for fname in self.feature_names]
        X = np.array([feature_values], dtype=np.float32)

        # Normalize using training statistics
        X_normalized = (X - self.feature_mean) / self.feature_std

        # Convert to tensor and predict
        X_tensor = torch.tensor(X_normalized, dtype=torch.float32)

        _set_model_inference_mode(self.model)
        with torch.no_grad():
            prediction = self.model(X_tensor).item()

        return {
            'expected_runtime_ms': float(prediction),
            'error_margin_ms': float(self.mean_error),
        }

    def get_model_state(self) -> dict[str, Any]:
        """Get complete model state for serialization.

        Returns:
            Dict containing all model parameters and metadata.

        Raises:
            ValueError: If no model to serialize.
        """
        if self.model is None:
            error_msg = "No model to serialize"
            logger.error(
                f"ExpectErrorPredictor get_model_state failed: {error_msg}"
            )
            raise ValueError(error_msg)

        return {
            'model_config': self.model.get_config(),
            'model_state_dict': self.model.state_dict(),
            'feature_names': self.feature_names,
            'mean_error': self.mean_error,
            'feature_mean': self.feature_mean.tolist(),
            'feature_std': self.feature_std.tolist(),
        }

    def load_model_state(self, state: dict[str, Any]) -> None:
        """Load a previously saved model state.

        Args:
            state: Model state dict from get_model_state().
        """
        # Recreate model architecture
        self.model = MLP.from_config(state['model_config'])
        self.model.load_state_dict(state['model_state_dict'])
        _set_model_inference_mode(self.model)

        # Restore metadata
        self.feature_names = state['feature_names']
        self.mean_error = state['mean_error']
        self.feature_mean = np.array(state['feature_mean'])
        self.feature_std = np.array(state['feature_std'])
