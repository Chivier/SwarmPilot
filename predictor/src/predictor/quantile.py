"""
Quantile predictor using MLP with pinball loss.

Provides quantile-based runtime predictions for SLA-aware scheduling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Any, Dict, List

from .base import BasePredictor
from .mlp import MLP


class PinballLoss(nn.Module):
    """
    Pinball loss (quantile loss) for quantile regression.

    Loss = max(q * (y - ŷ), (q - 1) * (y - ŷ))
    where q is the quantile level.
    """

    def __init__(self, quantiles: List[float]):
        """
        Initialize pinball loss.

        Args:
            quantiles: List of quantile levels (e.g., [0.5, 0.9, 0.95, 0.99])
        """
        super(PinballLoss, self).__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute pinball loss.

        Args:
            predictions: Predicted quantiles of shape (batch_size, num_quantiles)
            targets: True values of shape (batch_size, 1)

        Returns:
            Scalar loss value
        """
        # Expand targets to match predictions shape
        # targets: (batch_size, 1) -> (batch_size, num_quantiles)
        targets_expanded = targets.expand_as(predictions)

        # Compute errors
        errors = targets_expanded - predictions

        # Pinball loss: max(q * error, (q - 1) * error)
        # This is equivalent to: q * error if error >= 0, else (q - 1) * error
        quantiles_expanded = self.quantiles.unsqueeze(0).expand_as(predictions)

        loss = torch.where(
            errors >= 0,
            quantiles_expanded * errors,
            (quantiles_expanded - 1) * errors
        )

        return loss.mean()


class QuantilePredictor(BasePredictor):
    """
    Predictor for quantile-based runtime prediction.

    Trains MLP to predict multiple quantiles simultaneously using pinball loss.
    """

    def __init__(self):
        """Initialize the predictor."""
        self.model = None
        self.feature_names = None
        self.quantiles = None
        self.feature_mean = None
        self.feature_std = None

    def train(self, features_list: List[Dict[str, Any]], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train the predictor on the given data.

        Args:
            features_list: List of training samples with features and runtime_ms
            config: Optional training configuration (epochs, learning_rate, quantiles, etc.)

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
        self.feature_std = X.std(axis=0) + 1e-8
        X_normalized = (X - self.feature_mean) / self.feature_std

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        # Get training hyperparameters from config
        if config is None:
            config = {}

        # Get quantiles from config or use defaults
        self.quantiles = config.get('quantiles', [0.5, 0.9, 0.95, 0.99])

        # Validate quantiles
        for q in self.quantiles:
            if not (0 < q < 1):
                raise ValueError(f"Invalid quantile value {q}. Must be between 0 and 1.")

        epochs = config.get('epochs', 500)
        learning_rate = config.get('learning_rate', 0.01)
        hidden_layers = config.get('hidden_layers', [64, 32])

        # Create and train model
        input_dim = len(feature_names)
        output_dim = len(self.quantiles)  # One output per quantile

        self.model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers)

        criterion = PinballLoss(self.quantiles)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.model(X_tensor)
            loss = criterion(predictions, y_tensor)
            loss.backward()
            optimizer.step()

        return {
            'feature_names': self.feature_names,
            'samples_count': len(features_list),
            'quantiles': self.quantiles,
            'final_loss': float(loss.item())
        }

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for the given features.

        Args:
            features: Feature dictionary

        Returns:
            Dict with 'quantiles' key containing dict of quantile: value pairs

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
            predictions = self.model(X_tensor).numpy().flatten()

        # Format results as dict with quantile: value pairs
        quantile_results = {
            str(q): float(pred)
            for q, pred in zip(self.quantiles, predictions)
        }

        return {
            'quantiles': quantile_results
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
            'quantiles': self.quantiles,
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
        self.quantiles = state['quantiles']
        self.feature_mean = np.array(state['feature_mean'])
        self.feature_std = np.array(state['feature_std'])
