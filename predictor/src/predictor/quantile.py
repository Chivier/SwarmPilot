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

    Optionally includes a monotonicity penalty to ensure q_i <= q_{i+1}.
    """

    def __init__(self, quantiles: List[float], monotonicity_penalty: float = 0.0):
        """
        Initialize pinball loss.

        Args:
            quantiles: List of quantile levels (e.g., [0.5, 0.9, 0.95, 0.99])
            monotonicity_penalty: Weight for monotonicity penalty term (default: 0.0).
                Higher values enforce stricter monotonicity. Typical range: 0.1 - 10.0.
        """
        super(PinballLoss, self).__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)
        self.monotonicity_penalty = monotonicity_penalty

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute pinball loss with optional monotonicity penalty.

        Args:
            predictions: Predicted quantiles of shape (batch_size, num_quantiles)
            targets: True values of shape (batch_size, 1)

        Returns:
            Scalar loss value (pinball loss + monotonicity penalty)
        """
        # Expand targets to match predictions shape
        # targets: (batch_size, 1) -> (batch_size, num_quantiles)
        targets_expanded = targets.expand_as(predictions)

        # Compute errors
        errors = targets_expanded - predictions

        # Pinball loss: max(q * error, (q - 1) * error)
        # This is equivalent to: q * error if error >= 0, else (q - 1) * error
        quantiles_expanded = self.quantiles.unsqueeze(0).expand_as(predictions)

        pinball_loss = torch.where(
            errors >= 0,
            quantiles_expanded * errors,
            (quantiles_expanded - 1) * errors
        )

        # Compute mean pinball loss
        total_loss = pinball_loss.mean()

        # Add monotonicity penalty if enabled and there are multiple quantiles
        if self.monotonicity_penalty > 0 and predictions.shape[1] > 1:
            # Compute violations: max(0, q_i - q_{i+1}) for adjacent quantiles
            # predictions[:, :-1] = [q_0.5, q_0.9, q_0.95] (all but last)
            # predictions[:, 1:]  = [q_0.9, q_0.95, q_0.99] (all but first)
            # We want to penalize when q_i > q_{i+1}
            diffs = predictions[:, :-1] - predictions[:, 1:]

            # Only penalize positive violations (when lower quantile > higher quantile)
            violations = torch.relu(diffs)  # max(0, diffs)

            # Mean violation across all samples and quantile pairs
            monotonicity_loss = violations.mean()

            # Add weighted penalty to total loss
            total_loss = total_loss + self.monotonicity_penalty * monotonicity_loss

        return total_loss


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
        self.target_mean = None
        self.target_std = None

    def train(self, features_list: List[Dict[str, Any]], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train the predictor on the given data.

        Args:
            features_list: List of training samples with features and runtime_ms
            config: Optional training configuration with keys:
                - epochs (int): Number of training epochs (default: 500)
                - learning_rate (float): Adam optimizer learning rate (default: 0.01)
                - hidden_layers (list): Hidden layer sizes (default: [64, 32])
                - quantiles (list): Quantile levels to predict (default: [0.5, 0.9, 0.95, 0.99])
                - monotonicity_penalty (float): Weight for monotonicity constraint (default: 0.0)
                  Higher values enforce stricter quantile ordering. Typical range: 0.1 - 10.0

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

        # Normalize targets (z-score normalization)
        self.target_mean = y.mean()
        self.target_std = y.std() + 1e-8
        y_normalized = (y - self.target_mean) / self.target_std

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
        y_tensor = torch.tensor(y_normalized, dtype=torch.float32).unsqueeze(1)

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
        monotonicity_penalty = config.get('monotonicity_penalty', 0.0)

        # Create and train model
        input_dim = len(feature_names)
        output_dim = len(self.quantiles)  # One output per quantile

        self.model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers)

        criterion = PinballLoss(self.quantiles, monotonicity_penalty=monotonicity_penalty)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.model(X_tensor)
            loss = criterion(predictions, y_tensor)
            loss.backward()
            optimizer.step()

            # Print loss every 50 epochs and at the end
            if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

        return {
            'feature_names': self.feature_names,
            'samples_count': len(features_list),
            'quantiles': self.quantiles,
            'final_loss': float(loss.item())
        }

    @staticmethod
    def _enforce_monotonicity(predictions: np.ndarray) -> np.ndarray:
        """
        Enforce monotonicity constraint on quantile predictions.

        Uses a forward-pass algorithm that ensures strict monotonicity by
        propagating the maximum value forward when violations are detected.

        Args:
            predictions: Array of quantile predictions in ascending order

        Returns:
            Monotonically non-decreasing array of predictions
        """
        predictions = predictions.copy()  # Don't modify input

        # Forward pass: ensure each element is >= previous element
        for i in range(1, len(predictions)):
            if predictions[i] < predictions[i - 1]:
                predictions[i] = predictions[i - 1]

        return predictions

    def predict(self, features: Dict[str, Any], enforce_monotonicity: bool = False) -> Dict[str, Any]:
        """
        Make a prediction for the given features.

        Args:
            features: Feature dictionary
            enforce_monotonicity: If True, applies post-processing to strictly enforce
                monotonic ordering of quantiles (default: False). Use this when you need
                guaranteed monotonicity regardless of training configuration.

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
            predictions_normalized = self.model(X_tensor).numpy().flatten()

        # Denormalize predictions back to original scale
        predictions = predictions_normalized * self.target_std + self.target_mean

        # Apply monotonicity enforcement if requested
        if enforce_monotonicity:
            predictions = self._enforce_monotonicity(predictions)

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
            'feature_std': self.feature_std.tolist(),
            'target_mean': float(self.target_mean),
            'target_std': float(self.target_std)
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
        self.target_mean = state['target_mean']
        self.target_std = state['target_std']
