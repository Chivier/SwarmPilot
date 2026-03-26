"""Tests for predictor implementations."""

import pytest

from swarmpilot.predictor.predictor.expect_error import ExpectErrorPredictor
from swarmpilot.predictor.predictor.quantile import QuantilePredictor


def generate_training_data(n_samples=20):
    """Generate synthetic training data."""
    data = []
    for i in range(n_samples):
        batch_size = 16 + i
        data.append(
            {
                "batch_size": batch_size,
                "sequence_length": 128,
                "runtime_ms": 100 + batch_size * 2,  # Linear relationship
            }
        )
    return data


class TestExpectErrorPredictor:
    """Test ExpectErrorPredictor functionality."""

    def test_train_with_valid_data(self):
        """Should train successfully with valid data."""
        predictor = ExpectErrorPredictor()
        training_data = generate_training_data(20)

        metadata = predictor.train(training_data, config={"epochs": 100})

        assert "feature_names" in metadata
        assert "samples_count" in metadata
        assert "mean_error" in metadata
        assert metadata["samples_count"] == 20
        assert (
            len(metadata["feature_names"]) == 2
        )  # batch_size, sequence_length

    def test_train_with_insufficient_samples_raises_error(self):
        """Should raise error with less than 10 samples."""
        predictor = ExpectErrorPredictor()
        training_data = generate_training_data(5)

        with pytest.raises(ValueError, match="at least 10 samples"):
            predictor.train(training_data)

    def test_predict_after_training(self):
        """Should make predictions after training."""
        predictor = ExpectErrorPredictor()
        training_data = generate_training_data(20)
        predictor.train(training_data, config={"epochs": 100})

        result = predictor.predict({"batch_size": 25, "sequence_length": 128})

        assert "expected_runtime_ms" in result
        assert "error_margin_ms" in result
        assert isinstance(result["expected_runtime_ms"], float)
        assert isinstance(result["error_margin_ms"], float)
        assert result["error_margin_ms"] >= 0

    def test_predict_before_training_raises_error(self):
        """Should raise error if predicting before training."""
        predictor = ExpectErrorPredictor()

        with pytest.raises(ValueError, match="not trained"):
            predictor.predict({"batch_size": 25})

    def test_predict_with_missing_features_raises_error(self):
        """Should raise error for missing features."""
        predictor = ExpectErrorPredictor()
        training_data = generate_training_data(20)
        predictor.train(training_data, config={"epochs": 100})

        with pytest.raises(ValueError, match="Missing required features"):
            predictor.predict({"batch_size": 25})  # Missing sequence_length

    def test_model_serialization(self):
        """Should serialize and deserialize model state."""
        predictor1 = ExpectErrorPredictor()
        training_data = generate_training_data(20)
        predictor1.train(training_data, config={"epochs": 100})

        # Get predictions before serialization
        test_features = {"batch_size": 25, "sequence_length": 128}
        result1 = predictor1.predict(test_features)

        # Serialize and deserialize
        state = predictor1.get_model_state()
        predictor2 = ExpectErrorPredictor()
        predictor2.load_model_state(state)

        # Predictions should be identical
        result2 = predictor2.predict(test_features)
        assert result1["expected_runtime_ms"] == result2["expected_runtime_ms"]
        assert result1["error_margin_ms"] == result2["error_margin_ms"]


class TestQuantilePredictor:
    """Test QuantilePredictor functionality."""

    def test_train_with_default_quantiles(self):
        """Should train with default quantiles."""
        predictor = QuantilePredictor()
        training_data = generate_training_data(20)

        metadata = predictor.train(training_data, config={"epochs": 100})

        assert "feature_names" in metadata
        assert "samples_count" in metadata
        assert "quantiles" in metadata
        assert metadata["samples_count"] == 20
        assert metadata["quantiles"] == [0.5, 0.9, 0.95, 0.99]

    def test_train_with_custom_quantiles(self):
        """Should train with custom quantiles."""
        predictor = QuantilePredictor()
        training_data = generate_training_data(20)

        custom_quantiles = [0.1, 0.5, 0.9]
        metadata = predictor.train(
            training_data, config={"epochs": 100, "quantiles": custom_quantiles}
        )

        assert metadata["quantiles"] == custom_quantiles

    def test_train_with_invalid_quantile_raises_error(self):
        """Should raise error for invalid quantile values."""
        predictor = QuantilePredictor()
        training_data = generate_training_data(20)

        with pytest.raises(ValueError, match="Invalid quantile value"):
            predictor.train(training_data, config={"quantiles": [0.5, 1.5]})

    def test_predict_returns_quantiles(self):
        """Should return quantile predictions."""
        predictor = QuantilePredictor()
        training_data = generate_training_data(20)
        predictor.train(training_data, config={"epochs": 100})

        result = predictor.predict({"batch_size": 25, "sequence_length": 128})

        assert "quantiles" in result
        quantiles = result["quantiles"]

        # Should have all default quantiles
        assert "0.5" in quantiles
        assert "0.9" in quantiles
        assert "0.95" in quantiles
        assert "0.99" in quantiles

        # All values should be numeric
        for value in quantiles.values():
            assert isinstance(value, float)

    def test_quantile_predictions_are_monotonic(self):
        """Quantile predictions should be non-decreasing when enforce_monotonicity is used."""
        predictor = QuantilePredictor()
        training_data = generate_training_data(
            50
        )  # More samples for better training
        predictor.train(training_data, config={"epochs": 500})

        # Use enforce_monotonicity to ensure strict ordering
        result = predictor.predict(
            {"batch_size": 25, "sequence_length": 128},
            enforce_monotonicity=True,
        )
        quantiles = result["quantiles"]

        # Extract values in order
        values = [
            quantiles["0.5"],
            quantiles["0.9"],
            quantiles["0.95"],
            quantiles["0.99"],
        ]

        # Check strict monotonicity (no tolerance needed with enforcement)
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1], (
                f"Quantiles not monotonic: {values}"
            )

    def test_model_serialization(self):
        """Should serialize and deserialize model state."""
        predictor1 = QuantilePredictor()
        training_data = generate_training_data(20)
        predictor1.train(training_data, config={"epochs": 100})

        # Get predictions before serialization
        test_features = {"batch_size": 25, "sequence_length": 128}
        result1 = predictor1.predict(test_features)

        # Serialize and deserialize
        state = predictor1.get_model_state()
        predictor2 = QuantilePredictor()
        predictor2.load_model_state(state)

        # Predictions should be identical
        result2 = predictor2.predict(test_features)

        for q in ["0.5", "0.9", "0.95", "0.99"]:
            assert result1["quantiles"][q] == result2["quantiles"][q]

    def test_predict_with_custom_quantiles(self):
        """Should predict custom quantiles."""
        predictor = QuantilePredictor()
        training_data = generate_training_data(20)

        custom_quantiles = [0.25, 0.5, 0.75]
        predictor.train(
            training_data, config={"epochs": 100, "quantiles": custom_quantiles}
        )

        result = predictor.predict({"batch_size": 25, "sequence_length": 128})

        quantiles = result["quantiles"]
        assert "0.25" in quantiles
        assert "0.5" in quantiles
        assert "0.75" in quantiles
        assert len(quantiles) == 3

    def test_monotonicity_penalty_improves_ordering(self):
        """Monotonicity penalty should improve quantile ordering."""
        training_data = generate_training_data(50)

        # Train without monotonicity penalty
        predictor_no_penalty = QuantilePredictor()
        predictor_no_penalty.train(
            training_data, config={"epochs": 300, "monotonicity_penalty": 0.0}
        )

        # Train with monotonicity penalty
        predictor_with_penalty = QuantilePredictor()
        predictor_with_penalty.train(
            training_data, config={"epochs": 300, "monotonicity_penalty": 1.0}
        )

        # Test on multiple samples
        test_features = [
            {"batch_size": 20, "sequence_length": 128},
            {"batch_size": 30, "sequence_length": 128},
            {"batch_size": 40, "sequence_length": 128},
        ]

        violations_no_penalty = 0
        violations_with_penalty = 0

        for features in test_features:
            # Check predictions without penalty
            result_no_penalty = predictor_no_penalty.predict(features)
            values_no_penalty = [
                result_no_penalty["quantiles"]["0.5"],
                result_no_penalty["quantiles"]["0.9"],
                result_no_penalty["quantiles"]["0.95"],
                result_no_penalty["quantiles"]["0.99"],
            ]

            # Check predictions with penalty
            result_with_penalty = predictor_with_penalty.predict(features)
            values_with_penalty = [
                result_with_penalty["quantiles"]["0.5"],
                result_with_penalty["quantiles"]["0.9"],
                result_with_penalty["quantiles"]["0.95"],
                result_with_penalty["quantiles"]["0.99"],
            ]

            # Count violations
            for i in range(len(values_no_penalty) - 1):
                if values_no_penalty[i] > values_no_penalty[i + 1]:
                    violations_no_penalty += 1

            for i in range(len(values_with_penalty) - 1):
                if values_with_penalty[i] > values_with_penalty[i + 1]:
                    violations_with_penalty += 1

        # With penalty, there should be fewer or equal violations
        assert violations_with_penalty <= violations_no_penalty

    def test_enforce_monotonicity_parameter(self):
        """enforce_monotonicity parameter should strictly enforce ordering."""
        predictor = QuantilePredictor()
        training_data = generate_training_data(50)
        predictor.train(training_data, config={"epochs": 300})

        # Test multiple samples with enforce_monotonicity=True
        test_features = [
            {"batch_size": 20, "sequence_length": 128},
            {"batch_size": 30, "sequence_length": 128},
            {"batch_size": 40, "sequence_length": 128},
        ]

        for features in test_features:
            result = predictor.predict(features, enforce_monotonicity=True)
            values = [
                result["quantiles"]["0.5"],
                result["quantiles"]["0.9"],
                result["quantiles"]["0.95"],
                result["quantiles"]["0.99"],
            ]

            # With enforce_monotonicity=True, should have ZERO violations
            for i in range(len(values) - 1):
                assert values[i] <= values[i + 1], (
                    f"Monotonicity violated even with enforce_monotonicity=True: {values}"
                )


class TestConstantFeatureFiltering:
    """Test automatic constant feature filtering."""

    def test_constant_features_are_filtered(self):
        """Should automatically filter out constant features during training."""
        predictor = QuantilePredictor()

        # Create data with constant features (same value across all samples)
        training_data = []
        for i in range(20):
            training_data.append(
                {
                    "constant_feature": 100,  # Same value - should be filtered
                    "another_constant": 50,  # Same value - should be filtered
                    "varying_feature": 16 + i,  # Varies - should be kept
                    "runtime_ms": 100 + i * 2,
                }
            )

        metadata = predictor.train(training_data, config={"epochs": 100})

        # Verify constant features were removed
        assert "constant_feature" not in metadata["feature_names"]
        assert "another_constant" not in metadata["feature_names"]
        assert "varying_feature" in metadata["feature_names"]

        # Verify removed features are tracked
        assert "constant_feature" in metadata["removed_features"]
        assert "another_constant" in metadata["removed_features"]

    def test_predict_with_full_feature_set(self):
        """Should accept full feature set and automatically filter constants."""
        predictor = QuantilePredictor()

        training_data = []
        for i in range(20):
            training_data.append(
                {
                    "constant_feature": 100,
                    "varying_feature": 16 + i,
                    "runtime_ms": 100 + i * 2,
                }
            )

        predictor.train(training_data, config={"epochs": 100})

        # Predict with full feature set (including constant)
        result = predictor.predict(
            {
                "constant_feature": 100,  # Will be auto-filtered
                "varying_feature": 25,
            }
        )

        assert "quantiles" in result
        assert "0.5" in result["quantiles"]

    def test_serialization_preserves_removed_features(self):
        """Should preserve removed features info through serialization."""
        predictor1 = QuantilePredictor()

        training_data = []
        for i in range(20):
            training_data.append(
                {
                    "constant_feature": 100,
                    "varying_feature": 16 + i,
                    "runtime_ms": 100 + i * 2,
                }
            )

        predictor1.train(training_data, config={"epochs": 100})

        # Serialize and deserialize
        state = predictor1.get_model_state()
        predictor2 = QuantilePredictor()
        predictor2.load_model_state(state)

        # Verify removed_features preserved
        assert predictor2.removed_features == predictor1.removed_features
        assert "constant_feature" in predictor2.removed_features

        # Verify prediction still works with full feature set
        result = predictor2.predict(
            {"constant_feature": 100, "varying_feature": 25}
        )
        assert "quantiles" in result

    def test_all_constant_features_raises_error(self):
        """Should raise error if all features are constant."""
        predictor = QuantilePredictor()

        # All features are constant
        training_data = [
            {"constant1": 100, "constant2": 50, "runtime_ms": 100 + i}
            for i in range(20)
        ]

        with pytest.raises(ValueError, match="No valid features remaining"):
            predictor.train(training_data, config={"epochs": 100})


class TestFeatureValidation:
    """Test feature validation across predictors."""

    def test_inconsistent_features_in_training_data(self):
        """Should raise error for inconsistent features across samples."""
        predictor = ExpectErrorPredictor()

        # First sample has batch_size and sequence_length
        # Second sample has batch_size and model_size (different feature)
        inconsistent_data = [
            {"batch_size": 16, "sequence_length": 128, "runtime_ms": 100},
            {"batch_size": 32, "model_size": 256, "runtime_ms": 200},
        ] * 5  # Repeat to get enough samples

        with pytest.raises(ValueError, match="different features"):
            predictor.train(inconsistent_data)

    def test_empty_training_data(self):
        """Should handle empty training data gracefully."""
        predictor = ExpectErrorPredictor()

        with pytest.raises(ValueError):
            predictor.train([])


class TestLogTransform:
    """Test log transform functionality for QuantilePredictor."""

    def generate_skewed_training_data(self, n_samples=30):
        """Generate right-skewed runtime data (typical for latency)."""
        import numpy as np

        np.random.seed(42)
        data = []
        for i in range(n_samples):
            batch_size = 16 + i
            # Simulate right-skewed latency: base + exponential noise
            base_runtime = 50 + batch_size * 2
            skewed_runtime = base_runtime * np.random.lognormal(0, 0.3)
            data.append(
                {
                    "batch_size": batch_size,
                    "sequence_length": 128,
                    "runtime_ms": max(10.0, skewed_runtime),
                }
            )
        return data

    def test_log_transform_enabled_in_training(self):
        """Should enable log transform when configured."""
        predictor = QuantilePredictor()
        training_data = self.generate_skewed_training_data(30)

        config = {"epochs": 100, "log_transform": {"enabled": True}}
        metadata = predictor.train(training_data, config=config)

        assert metadata["log_transform_enabled"] is True
        assert predictor.log_transform_enabled is True

    def test_log_transform_disabled_by_default(self):
        """Log transform should be disabled by default."""
        predictor = QuantilePredictor()
        training_data = self.generate_skewed_training_data(30)

        metadata = predictor.train(training_data, config={"epochs": 100})

        assert metadata["log_transform_enabled"] is False
        assert predictor.log_transform_enabled is False

    def test_log_transform_predictions_are_positive(self):
        """Predictions with log transform should be positive (after exp)."""
        predictor = QuantilePredictor()
        training_data = self.generate_skewed_training_data(30)

        predictor.train(
            training_data,
            config={"epochs": 200, "log_transform": {"enabled": True}},
        )

        result = predictor.predict({"batch_size": 25, "sequence_length": 128})

        for q, value in result["quantiles"].items():
            assert value > 0, f"Quantile {q} should be positive, got {value}"

    def test_log_transform_predictions_in_reasonable_range(self):
        """Predictions with log transform should be in reasonable range."""
        predictor = QuantilePredictor()
        training_data = self.generate_skewed_training_data(30)

        # Get min/max runtime from training data
        runtimes = [d["runtime_ms"] for d in training_data]
        min_runtime = min(runtimes)
        max_runtime = max(runtimes)

        predictor.train(
            training_data,
            config={"epochs": 300, "log_transform": {"enabled": True}},
        )

        result = predictor.predict({"batch_size": 25, "sequence_length": 128})

        # Predictions should be within reasonable bounds
        for q, value in result["quantiles"].items():
            # Allow some margin for extrapolation
            assert value > min_runtime * 0.1, (
                f"Quantile {q}={value} too small (min training: {min_runtime})"
            )
            assert value < max_runtime * 10, (
                f"Quantile {q}={value} too large (max training: {max_runtime})"
            )

    def test_log_transform_serialization(self):
        """Log transform setting should be preserved through serialization."""
        predictor1 = QuantilePredictor()
        training_data = self.generate_skewed_training_data(30)

        predictor1.train(
            training_data,
            config={"epochs": 100, "log_transform": {"enabled": True}},
        )

        # Get predictions before serialization
        test_features = {"batch_size": 25, "sequence_length": 128}
        result1 = predictor1.predict(test_features)

        # Serialize and deserialize
        state = predictor1.get_model_state()
        predictor2 = QuantilePredictor()
        predictor2.load_model_state(state)

        # Verify log_transform_enabled is preserved
        assert predictor2.log_transform_enabled is True

        # Predictions should be identical
        result2 = predictor2.predict(test_features)
        for q in result1["quantiles"]:
            assert abs(result1["quantiles"][q] - result2["quantiles"][q]) < 1e-5

    def test_log_transform_backward_compatible(self):
        """Old models without log_transform should load with it disabled."""
        predictor1 = QuantilePredictor()
        training_data = self.generate_skewed_training_data(30)

        # Train without log_transform
        predictor1.train(training_data, config={"epochs": 100})

        # Serialize
        state = predictor1.get_model_state()

        # Simulate old model by removing log_transform_enabled key
        if "log_transform_enabled" in state:
            del state["log_transform_enabled"]

        # Load should still work with log_transform disabled
        predictor2 = QuantilePredictor()
        predictor2.load_model_state(state)

        assert predictor2.log_transform_enabled is False

    def test_log_transform_quantiles_monotonic(self):
        """Quantiles with log transform should remain monotonic."""
        predictor = QuantilePredictor()
        training_data = self.generate_skewed_training_data(50)

        predictor.train(
            training_data,
            config={"epochs": 300, "log_transform": {"enabled": True}},
        )

        result = predictor.predict(
            {"batch_size": 25, "sequence_length": 128},
            enforce_monotonicity=True,
        )

        values = [
            result["quantiles"]["0.5"],
            result["quantiles"]["0.9"],
            result["quantiles"]["0.95"],
            result["quantiles"]["0.99"],
        ]

        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1], (
                f"Quantiles not monotonic with log transform: {values}"
            )


class TestBasePredictorValidation:
    """Test base predictor validation and error handling."""

    def test_train_with_insufficient_samples_raises_error(self):
        """Should raise error when features_list has too few samples."""
        predictor = ExpectErrorPredictor()
        # Less than 10 samples
        data = [{"batch_size": 16, "runtime_ms": 100}] * 5

        with pytest.raises(ValueError, match="at least 10 samples"):
            predictor.train(data)

    def test_extract_features_no_features_raises_error(self):
        """Should raise error when samples have only runtime_ms."""
        predictor = ExpectErrorPredictor()
        # Samples with only runtime_ms and no features
        data = [{"runtime_ms": 100 + i} for i in range(15)]

        with pytest.raises(ValueError, match="No features"):
            predictor.train(data)

    def test_extract_features_inconsistent_features_raises_error(self):
        """Should raise error when samples have different features."""
        predictor = ExpectErrorPredictor()
        # Build samples with inconsistent features (need at least 10)
        data = []
        for i in range(5):
            data.append({"batch_size": 16 + i, "runtime_ms": 100 + i})
        for i in range(5):
            data.append({"sequence_length": 128 + i, "runtime_ms": 200 + i})

        with pytest.raises(ValueError, match="different features"):
            predictor.train(data)

    def test_filter_constant_features(self):
        """Should filter out constant features."""
        predictor = ExpectErrorPredictor()
        # All samples have same sequence_length
        data = []
        for i in range(20):
            data.append(
                {
                    "batch_size": 16 + i,
                    "sequence_length": 128,  # Constant
                    "runtime_ms": 100 + i * 2,
                }
            )

        metadata = predictor.train(data, config={"epochs": 100})
        # Should work despite constant feature
        assert "feature_names" in metadata

    def test_predict_before_train_raises_error(self):
        """Should raise error when predicting before training."""
        predictor = ExpectErrorPredictor()

        with pytest.raises(ValueError, match="train"):
            predictor.predict({"batch_size": 25, "sequence_length": 128})


class TestPredictorEdgeCases:
    """Test predictor edge cases and boundary conditions."""

    def test_train_with_single_feature(self):
        """Should handle training with single feature."""
        predictor = ExpectErrorPredictor()
        data = []
        for i in range(20):
            data.append({"batch_size": 16 + i, "runtime_ms": 100 + i * 2})

        metadata = predictor.train(data, config={"epochs": 100})
        assert len(metadata["feature_names"]) == 1
        assert "batch_size" in metadata["feature_names"]

    def test_train_with_many_features(self):
        """Should handle training with many features."""
        predictor = ExpectErrorPredictor()
        data = []
        for i in range(20):
            data.append(
                {
                    "feature1": 10 + i,
                    "feature2": 20 + i,
                    "feature3": 30 + i,
                    "feature4": 40 + i,
                    "feature5": 50 + i,
                    "runtime_ms": 100 + i * 2,
                }
            )

        metadata = predictor.train(data, config={"epochs": 100})
        assert len(metadata["feature_names"]) == 5

    def test_predict_with_extreme_values(self):
        """Should handle predictions with extreme input values."""
        predictor = ExpectErrorPredictor()
        data = generate_training_data(20)
        predictor.train(data, config={"epochs": 100})

        # Very large batch size
        result = predictor.predict(
            {"batch_size": 10000, "sequence_length": 128}
        )
        assert "expected_runtime_ms" in result
        assert result["expected_runtime_ms"] > 0

        # Very small batch size
        result = predictor.predict({"batch_size": 1, "sequence_length": 128})
        assert "expected_runtime_ms" in result
        assert result["expected_runtime_ms"] > 0

    def test_quantile_predict_before_train_raises_error(self):
        """Should raise error when predicting before training."""
        predictor = QuantilePredictor()

        with pytest.raises(ValueError, match="train"):
            predictor.predict({"batch_size": 25, "sequence_length": 128})


class TestQuantileResidualCalibration:
    """Test quantile predictor with residual calibration."""

    def test_train_with_residual_calibration(self):
        """Should train with residual calibration enabled."""
        predictor = QuantilePredictor()
        data = generate_training_data(30)

        metadata = predictor.train(
            data,
            config={"epochs": 100, "residual_calibration": {"enabled": True}},
        )

        assert "residual_calibration" in metadata
        # Calibration stats should be present
        if metadata["residual_calibration"]:
            assert "residual_mu" in metadata["residual_calibration"]
            assert "residual_sigma" in metadata["residual_calibration"]

    def test_predict_with_residual_calibration(self):
        """Should make predictions with residual calibration."""
        predictor = QuantilePredictor()
        data = generate_training_data(30)

        predictor.train(
            data,
            config={"epochs": 100, "residual_calibration": {"enabled": True}},
        )

        result = predictor.predict({"batch_size": 25, "sequence_length": 128})

        assert "quantiles" in result
        for q in ["0.5", "0.9", "0.95", "0.99"]:
            assert q in result["quantiles"]


class TestQuantileWithoutAugmentation:
    """Test quantile predictor without data augmentation."""

    def test_train_without_augmentation(self):
        """Should train without data augmentation."""
        predictor = QuantilePredictor()
        data = generate_training_data(30)

        metadata = predictor.train(
            data,
            config={"epochs": 100, "data_augmentation": {"enabled": False}},
        )

        # Augmented count should equal original count
        assert metadata["samples_count"] == 30
        assert metadata.get("data_augmentation") is None

    def test_predict_without_augmentation(self):
        """Should make predictions without data augmentation."""
        predictor = QuantilePredictor()
        data = generate_training_data(30)

        predictor.train(
            data,
            config={"epochs": 100, "data_augmentation": {"enabled": False}},
        )

        result = predictor.predict({"batch_size": 25, "sequence_length": 128})

        assert "quantiles" in result
