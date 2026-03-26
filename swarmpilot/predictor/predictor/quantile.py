"""Quantile predictor using MLP with pinball loss.

Provides quantile-based runtime predictions for SLA-aware scheduling.

Architecture: Base + Delta design for guaranteed monotonicity.
    - MLP outputs: [base, delta_1, delta_2, ..., delta_n-1]
    - Quantiles: q_0 = base, q_i = q_{i-1} + softplus(delta_i)
    - This ensures q_0 <= q_1 <= ... <= q_{n-1} by construction
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional

from swarmpilot.predictor.predictor.base import BasePredictor
from swarmpilot.predictor.predictor.mlp import MLP
from swarmpilot.predictor.utils.logging import get_logger

logger = get_logger()


class BaseDeltaTransform(nn.Module):
    """Transform base + delta outputs to monotonic quantile predictions.

    Takes raw MLP output [base, delta_1, delta_2, ...] and converts to
    quantile values [q_0, q_1, q_2, ...] where q_i = q_{i-1} + softplus(delta_i).

    This guarantees monotonicity by construction since softplus(x) > 0 for all x.

    Attributes:
        num_quantiles: Number of quantile levels to predict.
        delta_scale: Scale factor for deltas.
    """

    def __init__(self, num_quantiles: int, delta_scale: float = 1.0) -> None:
        """Initialize transform.

        Args:
            num_quantiles: Number of quantile levels to predict.
            delta_scale: Scale factor for deltas (default: 1.0).
                Higher values allow larger gaps between quantiles.
        """
        super().__init__()
        self.num_quantiles = num_quantiles
        self.delta_scale = delta_scale

    def forward(self, raw_output: torch.Tensor) -> torch.Tensor:
        """Convert base + delta representation to quantile values.

        Args:
            raw_output: Shape (batch_size, num_quantiles) where
                raw_output[:, 0] is the base value (lowest quantile) and
                raw_output[:, 1:] are deltas to be transformed via softplus.

        Returns:
            Quantile predictions of shape (batch_size, num_quantiles)
            guaranteed to be monotonically non-decreasing.
        """
        if self.num_quantiles == 1:
            # Single quantile: just return the base
            return raw_output

        # Split into base and deltas
        base = raw_output[:, 0:1]  # Shape: (batch_size, 1)
        deltas = raw_output[:, 1:]  # Shape: (batch_size, num_quantiles - 1)

        # Transform deltas to positive values using softplus
        # softplus(x) = log(1 + exp(x)), always positive
        positive_deltas = functional.softplus(deltas) * self.delta_scale

        # Build quantiles by cumulative sum of deltas
        # q[0] = base
        # q[i] = base + sum(positive_deltas[:i]) for i > 0
        cumulative_deltas = torch.cumsum(positive_deltas, dim=1)

        # Combine: [base, base + δ₁, base + δ₁ + δ₂, ...]
        quantiles = torch.cat([base, base + cumulative_deltas], dim=1)

        return quantiles


class PinballLoss(nn.Module):
    """Pinball loss (quantile loss) for quantile regression.

    Loss = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    where q is the quantile level.

    Optionally includes a monotonicity penalty to ensure q_i <= q_{i+1}.

    Attributes:
        quantiles: Tensor of quantile levels.
        monotonicity_penalty: Weight for monotonicity penalty term.
    """

    def __init__(
        self,
        quantiles: list[float],
        monotonicity_penalty: float = 0.0,
    ) -> None:
        """Initialize pinball loss.

        Args:
            quantiles: List of quantile levels (e.g., [0.5, 0.9, 0.95, 0.99])
            monotonicity_penalty: Weight for monotonicity penalty term (default: 0.0).
                Higher values enforce stricter monotonicity. Typical range: 0.1 - 10.0.
        """
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)
        self.monotonicity_penalty = monotonicity_penalty

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute pinball loss with optional monotonicity penalty.

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
            (quantiles_expanded - 1) * errors,
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
            total_loss = (
                total_loss + self.monotonicity_penalty * monotonicity_loss
            )

        return total_loss


class QuantilePredictor(BasePredictor):
    """Predictor for quantile-based runtime prediction.

    Trains MLP to predict multiple quantiles simultaneously using pinball loss.

    Internal Architecture (Base + Delta):
        - MLP outputs [base, δ₁, δ₂, ...] in normalized space
        - BaseDeltaTransform converts to monotonic quantiles via softplus
        - This guarantees monotonicity without post-processing or penalty terms
    """

    def __init__(self):
        """Initialize the predictor."""
        self.model = None
        self.transform = None  # BaseDeltaTransform for converting MLP output
        self.feature_names = (
            None  # Features actually used by model (after filtering)
        )
        self.removed_features = (
            None  # Constant features removed during training
        )
        self.quantiles = None
        self.feature_mean = None
        self.feature_std = None
        self.target_mean = None
        self.target_std = None
        self.delta_scale = None  # Scale factor for deltas

        # Log transform for runtime_ms (applied before normalization)
        # When enabled, training uses log(runtime_ms), prediction applies exp()
        self.log_transform_enabled = False

        # Residual calibration parameters (learned from training data)
        self.residual_calibration_enabled = False
        self.residual_mu = None  # Log-space mean of residuals
        self.residual_sigma = None  # Log-space std of residuals

    def train(
        self,
        features_list: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Train the predictor on the given data.

        Args:
            features_list: List of training samples with features and runtime_ms
            config: Optional training configuration with keys:
                - epochs (int): Number of training epochs (default: 500)
                - learning_rate (float): Adam optimizer learning rate (default: 0.01)
                - hidden_layers (list): Hidden layer sizes (default: [64, 32])
                - quantiles (list): Quantile levels to predict (default: [0.5, 0.9, 0.95, 0.99])
                - monotonicity_penalty (float): Weight for monotonicity constraint (default: 0.0)
                  Note: With base+delta architecture, monotonicity is guaranteed by design.
                  This parameter is kept for backward compatibility but is less critical.
                - delta_scale (float): Scale factor for delta outputs (default: 1.0)
                  Higher values allow larger gaps between quantiles.

                Data Augmentation (for single-sample-per-feature data):
                - data_augmentation (dict): Enable runtime augmentation to simulate variance
                    - enabled (bool): Enable data augmentation (default: True)
                      Set to False to disable augmentation entirely.
                    - cv (float): Coefficient of variation for augmentation.
                      If not specified, auto-calculated from training data:
                      * For multi-sample groups: weighted average of group CVs
                      * For single-sample data: heuristic based on overall distribution
                      * Bounded to [0.1, 1.5] range
                      User-specified value overrides auto-calculation.
                    - samples_per_point (int): Number of augmented samples per original (default: 5)
                    - distribution (str): "lognormal" or "normal" (default: "lognormal")

                Residual Calibration (post-training distribution adjustment):
                - residual_calibration (dict): Calibrate quantiles using residual analysis
                    - enabled (bool): Enable residual calibration (default: False)
                    - min_sigma (float): Minimum residual sigma (default: 0.1)

                Log Transform (for right-skewed runtime distributions):
                - log_transform (dict): Apply log transform to runtime_ms
                    - enabled (bool): Enable log transform (default: False)
                      When enabled, training uses log(runtime_ms) and prediction
                      applies exp() to restore original scale. This helps when
                      runtime distributions are right-skewed (common for latency data).

        Returns:
            Training metadata dictionary

        Raises:
            ValueError: If training data is insufficient or invalid
        """
        # Validate minimum samples
        if len(features_list) < 10:
            error_msg = f"Insufficient training data: need at least 10 samples, got {len(features_list)}"
            logger.error(
                f"QuantilePredictor training failed\n"
                f"Error: {error_msg}\n"
                f"Samples provided: {len(features_list)}\n"
                f"Minimum required: 10"
            )
            raise ValueError(error_msg)

        # Get config
        if config is None:
            config = {}

        # ============================================================
        # Data Augmentation: Generate synthetic samples to simulate variance
        # Default: ENABLED with auto-calculated parameters
        # ============================================================
        augmentation_config = config.get("data_augmentation", {})

        # Default to enabled unless explicitly disabled
        augmentation_enabled = augmentation_config.get("enabled", True)
        augmented_features_list = features_list

        if augmentation_enabled:
            # Auto-calculate CV from training data if not explicitly specified
            auto_cv = self._estimate_cv_from_data(features_list)

            # Get parameters with auto-calculated defaults
            # User-specified values override auto-calculated ones
            aug_cv = augmentation_config.get("cv", auto_cv)
            aug_samples = augmentation_config.get("samples_per_point", 5)
            aug_dist = augmentation_config.get("distribution", "lognormal")

            print(
                f"Data augmentation enabled: cv={aug_cv:.3f} (auto={auto_cv:.3f}), "
                f"samples_per_point={aug_samples}, distribution={aug_dist}"
            )

            augmented_features_list = self._augment_training_data(
                features_list,
                cv=aug_cv,
                samples_per_point=aug_samples,
                distribution=aug_dist,
            )
            print(
                f"Augmented {len(features_list)} samples to {len(augmented_features_list)} samples"
            )
        else:
            print("Data augmentation disabled by user configuration")

        # Extract features and labels
        X, y, all_feature_names = self.extract_features_and_labels(
            augmented_features_list
        )

        # Filter out constant features (zero variance)
        # Constant features provide no predictive value and cause numerical issues
        X, feature_names, self.removed_features = self.filter_constant_features(
            X, all_feature_names
        )

        if self.removed_features:
            print(
                f"Filtered {len(self.removed_features)} constant features: {self.removed_features}"
            )

        if not feature_names:
            error_msg = (
                "No valid features remaining after filtering constant features. "
                f"All features were constant: {all_feature_names}"
            )
            logger.error(
                f"QuantilePredictor training failed - no valid features\n"
                f"Error: {error_msg}\n"
                f"Removed features: {self.removed_features}"
            )
            raise ValueError(error_msg)

        self.feature_names = feature_names

        # Log selected features for training
        print(f"Training with {len(feature_names)} features: {feature_names}")

        # Convert to numpy arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # ============================================================
        # Log Transform: Apply log to runtime_ms before normalization
        # This helps with right-skewed distributions (common for latency)
        # ============================================================
        log_transform_config = config.get("log_transform", {})
        self.log_transform_enabled = log_transform_config.get("enabled", False)

        if self.log_transform_enabled:
            # Clip to avoid log(0) or log(negative)
            y = np.maximum(y, 1e-6)
            y = np.log(y)
            print(
                f"Log transform enabled: applied log() to {len(y)} runtime values"
            )

        # Normalize features (z-score normalization)
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0) + 1e-8
        X_normalized = (X - self.feature_mean) / self.feature_std

        # Normalize targets (z-score normalization)
        # Note: when log_transform is enabled, this normalizes log(runtime_ms)
        self.target_mean = y.mean()
        self.target_std = y.std() + 1e-8
        y_normalized = (y - self.target_mean) / self.target_std

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
        y_tensor = torch.tensor(y_normalized, dtype=torch.float32).unsqueeze(1)

        # Get quantiles from config or use defaults
        self.quantiles = config.get("quantiles", [0.5, 0.9, 0.95, 0.99])

        # Validate quantiles
        for q in self.quantiles:
            if not (0 < q < 1):
                error_msg = (
                    f"Invalid quantile value {q}. Must be between 0 and 1."
                )
                logger.error(
                    f"QuantilePredictor training failed - invalid quantile\n"
                    f"Error: {error_msg}\n"
                    f"Quantiles provided: {self.quantiles}"
                )
                raise ValueError(error_msg)

        # Sort quantiles to ensure proper ordering for base+delta transform
        self.quantiles = sorted(self.quantiles)

        epochs = config.get("epochs", 500)
        learning_rate = config.get("learning_rate", 0.01)
        hidden_layers = config.get("hidden_layers", [64, 32])
        monotonicity_penalty = config.get("monotonicity_penalty", 0.0)
        self.delta_scale = config.get("delta_scale", 1.0)

        # Create model and transform
        input_dim = len(feature_names)
        output_dim = len(
            self.quantiles
        )  # One output per quantile (base + deltas)

        # MLP outputs raw [base, δ₁, δ₂, ...] values
        self.model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
        )

        # Transform converts raw outputs to monotonic quantiles
        self.transform = BaseDeltaTransform(
            num_quantiles=output_dim, delta_scale=self.delta_scale
        )

        criterion = PinballLoss(
            self.quantiles, monotonicity_penalty=monotonicity_penalty
        )
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass: MLP → BaseDeltaTransform → quantile predictions
            raw_output = self.model(X_tensor)
            predictions = self.transform(raw_output)

            loss = criterion(predictions, y_tensor)
            loss.backward()
            optimizer.step()

            # Print loss every 50 epochs and at the end
            if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

        # ============================================================
        # Residual Calibration: Analyze prediction residuals for better quantile estimation
        # ============================================================
        residual_config = config.get("residual_calibration", {})
        self.residual_calibration_enabled = residual_config.get(
            "enabled", False
        )
        min_sigma = residual_config.get("min_sigma", 0.1)

        calibration_stats = {}
        if self.residual_calibration_enabled:
            calibration_stats = self._compute_residual_calibration(
                X_tensor, y, min_sigma=min_sigma
            )
            print("\nResidual calibration enabled:")
            print(f"  Residual μ (log-space): {self.residual_mu:.4f}")
            print(f"  Residual sigma (log-space): {self.residual_sigma:.4f}")
            print(
                f"  Estimated CV: {calibration_stats.get('estimated_cv', 0):.4f}"
            )

        return {
            "feature_names": self.feature_names,
            "removed_features": self.removed_features,
            "samples_count": len(features_list),
            "augmented_samples_count": len(augmented_features_list)
            if augmentation_enabled
            else len(features_list),
            "quantiles": self.quantiles,
            "final_loss": float(loss.item()),
            "data_augmentation": augmentation_config
            if augmentation_enabled
            else None,
            "residual_calibration": calibration_stats
            if self.residual_calibration_enabled
            else None,
            "log_transform_enabled": self.log_transform_enabled,
        }

    def _estimate_cv_from_data(
        self, features_list: list[dict[str, Any]]
    ) -> float:
        """Estimate coefficient of variation (CV) from training data.

        Strategy:
        1. Group samples by feature values (excluding runtime_ms)
        2. For groups with multiple samples, calculate CV directly
        3. For single-sample groups, use overall runtime distribution as fallback
        4. Return weighted average CV with reasonable bounds

        Args:
            features_list: Training samples with features and runtime_ms

        Returns:
            Estimated CV (bounded between 0.1 and 1.5)
        """
        from collections import defaultdict

        # Extract runtimes
        runtimes = np.array([s["runtime_ms"] for s in features_list])

        # Group samples by feature signature (excluding runtime_ms)
        groups = defaultdict(list)
        for sample in features_list:
            # Create hashable key from features (excluding runtime_ms and metadata)
            feature_key = tuple(
                (k, round(v, 6) if isinstance(v, float) else v)
                for k, v in sorted(sample.items())
                if k != "runtime_ms" and not k.startswith("_")
            )
            groups[feature_key].append(sample["runtime_ms"])

        # Calculate CV for groups with multiple samples
        group_cvs = []
        group_weights = []

        for _key, runtime_values in groups.items():
            if len(runtime_values) >= 2:
                arr = np.array(runtime_values)
                mean_val = np.mean(arr)
                std_val = np.std(arr, ddof=1)  # Sample std
                if mean_val > 0:
                    cv = std_val / mean_val
                    group_cvs.append(cv)
                    group_weights.append(len(runtime_values))

        if group_cvs:
            # Weighted average CV from groups with multiple samples
            weighted_cv = np.average(group_cvs, weights=group_weights)
            estimated_cv = float(weighted_cv)
        else:
            # Fallback: estimate from overall runtime distribution
            # For single-sample-per-feature data, use a heuristic based on
            # the ratio of runtime range to mean
            mean_runtime = np.mean(runtimes)
            std_runtime = np.std(runtimes)

            if mean_runtime > 0:
                # Use overall CV as a rough estimate, but scale it down
                # since feature variation contributes to this spread
                overall_cv = std_runtime / mean_runtime
                # Assume about 30-50% of the spread is due to actual runtime variance
                estimated_cv = overall_cv * 0.4
            else:
                estimated_cv = 0.3  # Default fallback

        # Bound CV to reasonable range
        # - Minimum 0.1: ensure some spread for quantile predictions
        # - Maximum 1.5: prevent extreme augmentation
        estimated_cv = max(0.1, min(1.5, estimated_cv))

        return estimated_cv

    def _augment_training_data(
        self,
        features_list: list[dict[str, Any]],
        cv: float = 0.3,
        samples_per_point: int = 5,
        distribution: str = "lognormal",
    ) -> list[dict[str, Any]]:
        """Augment training data by generating synthetic runtime variations.

        For each original sample, generates multiple samples with the same features
        but different runtime values sampled from a distribution around the original.

        This helps the MLP learn variance patterns when each feature combination
        has only one original runtime sample.

        Args:
            features_list: Original training samples
            cv: Coefficient of variation for noise (default: 0.3 = 30%)
            samples_per_point: Number of augmented samples per original (default: 5)
            distribution: "lognormal" or "normal" (default: "lognormal")

        Returns:
            Augmented features list with synthetic samples
        """
        np.random.seed(42)
        augmented = []

        for sample in features_list:
            original_runtime = sample["runtime_ms"]
            augmented.append(sample.copy())

            for _ in range(samples_per_point - 1):
                new_sample = sample.copy()

                if distribution == "lognormal":
                    sigma_ln = np.sqrt(np.log(1 + cv**2))
                    mu_ln = np.log(original_runtime) - sigma_ln**2 / 2
                    new_runtime = np.random.lognormal(mu_ln, sigma_ln)
                else:
                    sigma = original_runtime * cv
                    new_runtime = np.random.normal(original_runtime, sigma)
                    new_runtime = max(new_runtime, 1.0)

                new_sample["runtime_ms"] = float(new_runtime)
                augmented.append(new_sample)

        return augmented

    def _compute_residual_calibration(
        self,
        X_tensor: torch.Tensor,
        y_original: np.ndarray,
        min_sigma: float = 0.1,
    ) -> dict[str, Any]:
        """Compute residual distribution parameters for calibration.

        Analyzes the ratio between actual runtimes and model predictions
        to estimate the uncertainty distribution.

        Args:
            X_tensor: Normalized feature tensor
            y_original: Original (non-normalized) runtime values
            min_sigma: Minimum sigma to prevent degenerate distributions

        Returns:
            Dict with calibration statistics
        """
        self.model.eval()
        with torch.no_grad():
            raw_output = self.model(X_tensor)
            predictions_normalized = self.transform(raw_output).numpy()

        median_idx = 0
        for i, q in enumerate(self.quantiles):
            if q == 0.5:
                median_idx = i
                break

        predictions_median_normalized = predictions_normalized[:, median_idx]
        predictions_median = (
            predictions_median_normalized * self.target_std + self.target_mean
        )

        predictions_clipped = np.maximum(predictions_median, 1e-6)
        residuals = y_original / predictions_clipped

        log_residuals = np.log(np.maximum(residuals, 1e-6))

        self.residual_mu = float(np.mean(log_residuals))
        self.residual_sigma = float(max(np.std(log_residuals), min_sigma))

        estimated_cv = np.sqrt(np.exp(self.residual_sigma**2) - 1)

        return {
            "residual_mu": self.residual_mu,
            "residual_sigma": self.residual_sigma,
            "estimated_cv": float(estimated_cv),
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
        }

    def _get_residual_quantile_multiplier(self, quantile: float) -> float:
        """Get the multiplier for a given quantile based on residual distribution.

        For log-normal residuals with parameters (mu, sigma):
            quantile(p) = exp(mu + sigma * z_p)

        Args:
            quantile: Quantile level (0 < q < 1)

        Returns:
            Multiplier to apply to base prediction
        """
        from scipy import stats

        if self.residual_mu is None or self.residual_sigma is None:
            return 1.0

        z = stats.norm.ppf(quantile)
        return float(np.exp(self.residual_mu + self.residual_sigma * z))

    @staticmethod
    def _enforce_monotonicity(predictions: np.ndarray) -> np.ndarray:
        """Enforce monotonicity constraint on quantile predictions.

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

    def predict(
        self, features: dict[str, Any], enforce_monotonicity: bool = False
    ) -> dict[str, Any]:
        """Make a prediction for the given features.

        Automatically filters out constant features that were removed during training.
        Users can provide the full feature set - constant features will be ignored.

        Args:
            features: Feature dictionary (may include constant features that will be filtered)
            enforce_monotonicity: If True, applies post-processing to strictly enforce
                monotonic ordering of quantiles (default: False). Use this when you need
                guaranteed monotonicity regardless of training configuration.

        Returns:
            Dict with 'quantiles' key containing dict of quantile: value pairs

        Raises:
            ValueError: If model not trained or required features missing
        """
        if self.model is None:
            error_msg = "Model not trained. Call train() first."
            logger.error(f"QuantilePredictor prediction failed: {error_msg}")
            raise ValueError(error_msg)

        # Filter out constant features that were removed during training
        # This allows users to pass the full feature set without manual filtering
        filtered_features = self.filter_features_for_prediction(
            features, self.feature_names
        )

        # Validate that all required (non-constant) features are present
        self.validate_features(filtered_features, self.feature_names)

        # Extract feature values in correct order
        feature_values = [
            filtered_features[fname] for fname in self.feature_names
        ]
        X = np.array([feature_values], dtype=np.float32)

        # Normalize using training statistics
        X_normalized = (X - self.feature_mean) / self.feature_std

        # Convert to tensor and predict
        X_tensor = torch.tensor(X_normalized, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            # Forward pass: MLP -> BaseDeltaTransform -> monotonic quantiles
            raw_output = self.model(X_tensor)
            predictions_normalized = (
                self.transform(raw_output).numpy().flatten()
            )

        # Denormalize predictions back to original scale
        # Note: when log_transform is enabled, this gives log(runtime_ms)
        predictions = (
            predictions_normalized * self.target_std + self.target_mean
        )

        # Apply inverse log transform if enabled during training
        # This converts log(runtime_ms) back to runtime_ms using exponential
        if self.log_transform_enabled:
            predictions = np.power(np.e, predictions)

        # Apply additional monotonicity enforcement if requested
        # Note: base+delta architecture guarantees monotonicity, but this
        # provides extra safety for edge cases or numerical precision issues
        if enforce_monotonicity:
            predictions = self._enforce_monotonicity(predictions)

        # Format results as dict with quantile: value pairs
        quantile_results = {
            str(q): float(pred) for q, pred in zip(self.quantiles, predictions)
        }

        return {"quantiles": quantile_results}

    def get_model_state(self) -> dict[str, Any]:
        """Get complete model state for serialization.

        Returns:
            Dict containing all model parameters and metadata
        """
        if self.model is None:
            error_msg = "No model to serialize"
            logger.error(
                f"QuantilePredictor get_model_state failed: {error_msg}"
            )
            raise ValueError(error_msg)

        return {
            "model_config": self.model.get_config(),
            "model_state_dict": self.model.state_dict(),
            "feature_names": self.feature_names,
            "removed_features": self.removed_features
            or [],  # Constant features filtered during training
            "quantiles": self.quantiles,
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "target_mean": float(self.target_mean),
            "target_std": float(self.target_std),
            "delta_scale": float(self.delta_scale) if self.delta_scale else 1.0,
            # Log transform parameter
            "log_transform_enabled": self.log_transform_enabled,
            # Residual calibration parameters
            "residual_calibration_enabled": self.residual_calibration_enabled,
            "residual_mu": self.residual_mu,
            "residual_sigma": self.residual_sigma,
        }

    def load_model_state(self, state: dict[str, Any]) -> None:
        """Load a previously saved model state.

        Args:
            state: Model state dict from get_model_state()
        """
        # Recreate model architecture
        self.model = MLP.from_config(state["model_config"])
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

        # Restore metadata
        self.feature_names = state["feature_names"]
        self.quantiles = state["quantiles"]
        self.feature_mean = np.array(state["feature_mean"])
        self.feature_std = np.array(state["feature_std"])
        self.target_mean = state["target_mean"]
        self.target_std = state["target_std"]

        # Restore removed_features (backward compatible with old models)
        self.removed_features = state.get("removed_features", [])

        # Restore delta_scale (backward compatible with old models)
        self.delta_scale = state.get("delta_scale", 1.0)

        # Recreate transform with correct parameters
        self.transform = BaseDeltaTransform(
            num_quantiles=len(self.quantiles), delta_scale=self.delta_scale
        )

        # Restore log transform setting (backward compatible with old models)
        self.log_transform_enabled = state.get("log_transform_enabled", False)

        # Restore residual calibration parameters (backward compatible)
        self.residual_calibration_enabled = state.get(
            "residual_calibration_enabled", False
        )
        self.residual_mu = state.get("residual_mu")
        self.residual_sigma = state.get("residual_sigma")
