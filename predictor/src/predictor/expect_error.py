"""
Expect/Error predictor using MLP with MSE loss.

Provides expected runtime and error margin predictions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Any, Dict, List

from .base import BasePredictor
from .mlp import MLP


class ExpectErrorPredictor(BasePredictor):
    """
    Predictor for expect/error prediction type.

    Trains MLP to predict expected runtime and computes error margin from residuals.
    """

    def __init__(self):
        """Initialize the predictor."""
        self.model = None
        self.feature_names = None
        self.mean_error = None
        self.feature_mean = None
        self.feature_std = None

    def train(self, features_list: List[Dict[str, Any]], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train the predictor on the given data.

        Args:
            features_list: List of training samples with features and runtime_ms
            config: Optional training configuration (epochs, learning_rate, etc.)

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

        # Normalize features (z-score normalization)
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0) + 1e-8  # Add small value to avoid division by zero
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
        self.model = MLP(input_dim=input_dim, output_dim=1, hidden_layers=hidden_layers)

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
        self.model.eval()
        with torch.no_grad():
            final_predictions = self.model(X_tensor).numpy().flatten()
            residuals = np.abs(y - final_predictions)
            self.mean_error = float(np.mean(residuals))

        return {
            'feature_names': self.feature_names,
            'samples_count': len(features_list),
            'mean_error': self.mean_error,
            'final_loss': float(loss.item())
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

        # Normalize using training statistics
        X_normalized = (X - self.feature_mean) / self.feature_std

        # Convert to tensor and predict
        X_tensor = torch.tensor(X_normalized, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X_tensor).item()

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

        return {
            'model_config': self.model.get_config(),
            'model_state_dict': self.model.state_dict(),
            'feature_names': self.feature_names,
            'mean_error': self.mean_error,
            'feature_mean': self.feature_mean.tolist(),
            'feature_std': self.feature_std.tolist()
        }

    def load_model_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved model state.

        Args:
            state: Model state dict from get_model_state()
        """
        # Recreate model architecture
        self.model = MLP.from_config(state['model_config'])
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()

        # Restore metadata
        self.feature_names = state['feature_names']
        self.mean_error = state['mean_error']
        self.feature_mean = np.array(state['feature_mean'])
        self.feature_std = np.array(state['feature_std'])
