"""
Tests for predictor implementations.
"""

import pytest
from src.predictor.expect_error import ExpectErrorPredictor
from src.predictor.quantile import QuantilePredictor


def generate_training_data(n_samples=20):
    """Generate synthetic training data."""
    data = []
    for i in range(n_samples):
        batch_size = 16 + i
        data.append({
            'batch_size': batch_size,
            'sequence_length': 128,
            'runtime_ms': 100 + batch_size * 2  # Linear relationship
        })
    return data


class TestExpectErrorPredictor:
    """Test ExpectErrorPredictor functionality."""

    def test_train_with_valid_data(self):
        """Should train successfully with valid data."""
        predictor = ExpectErrorPredictor()
        training_data = generate_training_data(20)

        metadata = predictor.train(training_data, config={'epochs': 100})

        assert 'feature_names' in metadata
        assert 'samples_count' in metadata
        assert 'mean_error' in metadata
        assert metadata['samples_count'] == 20
        assert len(metadata['feature_names']) == 2  # batch_size, sequence_length

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
        predictor.train(training_data, config={'epochs': 100})

        result = predictor.predict({'batch_size': 25, 'sequence_length': 128})

        assert 'expected_runtime_ms' in result
        assert 'error_margin_ms' in result
        assert isinstance(result['expected_runtime_ms'], float)
        assert isinstance(result['error_margin_ms'], float)
        assert result['error_margin_ms'] >= 0

    def test_predict_before_training_raises_error(self):
        """Should raise error if predicting before training."""
        predictor = ExpectErrorPredictor()

        with pytest.raises(ValueError, match="not trained"):
            predictor.predict({'batch_size': 25})

    def test_predict_with_missing_features_raises_error(self):
        """Should raise error for missing features."""
        predictor = ExpectErrorPredictor()
        training_data = generate_training_data(20)
        predictor.train(training_data, config={'epochs': 100})

        with pytest.raises(ValueError, match="Missing required features"):
            predictor.predict({'batch_size': 25})  # Missing sequence_length

    def test_model_serialization(self):
        """Should serialize and deserialize model state."""
        predictor1 = ExpectErrorPredictor()
        training_data = generate_training_data(20)
        predictor1.train(training_data, config={'epochs': 100})

        # Get predictions before serialization
        test_features = {'batch_size': 25, 'sequence_length': 128}
        result1 = predictor1.predict(test_features)

        # Serialize and deserialize
        state = predictor1.get_model_state()
        predictor2 = ExpectErrorPredictor()
        predictor2.load_model_state(state)

        # Predictions should be identical
        result2 = predictor2.predict(test_features)
        assert result1['expected_runtime_ms'] == result2['expected_runtime_ms']
        assert result1['error_margin_ms'] == result2['error_margin_ms']


class TestQuantilePredictor:
    """Test QuantilePredictor functionality."""

    def test_train_with_default_quantiles(self):
        """Should train with default quantiles."""
        predictor = QuantilePredictor()
        training_data = generate_training_data(20)

        metadata = predictor.train(training_data, config={'epochs': 100})

        assert 'feature_names' in metadata
        assert 'samples_count' in metadata
        assert 'quantiles' in metadata
        assert metadata['samples_count'] == 20
        assert metadata['quantiles'] == [0.5, 0.9, 0.95, 0.99]

    def test_train_with_custom_quantiles(self):
        """Should train with custom quantiles."""
        predictor = QuantilePredictor()
        training_data = generate_training_data(20)

        custom_quantiles = [0.1, 0.5, 0.9]
        metadata = predictor.train(
            training_data,
            config={'epochs': 100, 'quantiles': custom_quantiles}
        )

        assert metadata['quantiles'] == custom_quantiles

    def test_train_with_invalid_quantile_raises_error(self):
        """Should raise error for invalid quantile values."""
        predictor = QuantilePredictor()
        training_data = generate_training_data(20)

        with pytest.raises(ValueError, match="Invalid quantile value"):
            predictor.train(training_data, config={'quantiles': [0.5, 1.5]})

    def test_predict_returns_quantiles(self):
        """Should return quantile predictions."""
        predictor = QuantilePredictor()
        training_data = generate_training_data(20)
        predictor.train(training_data, config={'epochs': 100})

        result = predictor.predict({'batch_size': 25, 'sequence_length': 128})

        assert 'quantiles' in result
        quantiles = result['quantiles']

        # Should have all default quantiles
        assert '0.5' in quantiles
        assert '0.9' in quantiles
        assert '0.95' in quantiles
        assert '0.99' in quantiles

        # All values should be numeric
        for value in quantiles.values():
            assert isinstance(value, float)

    def test_quantile_predictions_are_monotonic(self):
        """Quantile predictions should be non-decreasing when enforce_monotonicity is used."""
        predictor = QuantilePredictor()
        training_data = generate_training_data(50)  # More samples for better training
        predictor.train(training_data, config={'epochs': 500})

        # Use enforce_monotonicity to ensure strict ordering
        result = predictor.predict({'batch_size': 25, 'sequence_length': 128},
                                   enforce_monotonicity=True)
        quantiles = result['quantiles']

        # Extract values in order
        values = [
            quantiles['0.5'],
            quantiles['0.9'],
            quantiles['0.95'],
            quantiles['0.99']
        ]

        # Check strict monotonicity (no tolerance needed with enforcement)
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1], \
                f"Quantiles not monotonic: {values}"

    def test_model_serialization(self):
        """Should serialize and deserialize model state."""
        predictor1 = QuantilePredictor()
        training_data = generate_training_data(20)
        predictor1.train(training_data, config={'epochs': 100})

        # Get predictions before serialization
        test_features = {'batch_size': 25, 'sequence_length': 128}
        result1 = predictor1.predict(test_features)

        # Serialize and deserialize
        state = predictor1.get_model_state()
        predictor2 = QuantilePredictor()
        predictor2.load_model_state(state)

        # Predictions should be identical
        result2 = predictor2.predict(test_features)

        for q in ['0.5', '0.9', '0.95', '0.99']:
            assert result1['quantiles'][q] == result2['quantiles'][q]

    def test_predict_with_custom_quantiles(self):
        """Should predict custom quantiles."""
        predictor = QuantilePredictor()
        training_data = generate_training_data(20)

        custom_quantiles = [0.25, 0.5, 0.75]
        predictor.train(training_data, config={'epochs': 100, 'quantiles': custom_quantiles})

        result = predictor.predict({'batch_size': 25, 'sequence_length': 128})

        quantiles = result['quantiles']
        assert '0.25' in quantiles
        assert '0.5' in quantiles
        assert '0.75' in quantiles
        assert len(quantiles) == 3

    def test_monotonicity_penalty_improves_ordering(self):
        """Monotonicity penalty should improve quantile ordering."""
        training_data = generate_training_data(50)

        # Train without monotonicity penalty
        predictor_no_penalty = QuantilePredictor()
        predictor_no_penalty.train(training_data, config={
            'epochs': 300,
            'monotonicity_penalty': 0.0
        })

        # Train with monotonicity penalty
        predictor_with_penalty = QuantilePredictor()
        predictor_with_penalty.train(training_data, config={
            'epochs': 300,
            'monotonicity_penalty': 1.0
        })

        # Test on multiple samples
        test_features = [
            {'batch_size': 20, 'sequence_length': 128},
            {'batch_size': 30, 'sequence_length': 128},
            {'batch_size': 40, 'sequence_length': 128}
        ]

        violations_no_penalty = 0
        violations_with_penalty = 0

        for features in test_features:
            # Check predictions without penalty
            result_no_penalty = predictor_no_penalty.predict(features)
            values_no_penalty = [
                result_no_penalty['quantiles']['0.5'],
                result_no_penalty['quantiles']['0.9'],
                result_no_penalty['quantiles']['0.95'],
                result_no_penalty['quantiles']['0.99']
            ]

            # Check predictions with penalty
            result_with_penalty = predictor_with_penalty.predict(features)
            values_with_penalty = [
                result_with_penalty['quantiles']['0.5'],
                result_with_penalty['quantiles']['0.9'],
                result_with_penalty['quantiles']['0.95'],
                result_with_penalty['quantiles']['0.99']
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
        predictor.train(training_data, config={'epochs': 300})

        # Test multiple samples with enforce_monotonicity=True
        test_features = [
            {'batch_size': 20, 'sequence_length': 128},
            {'batch_size': 30, 'sequence_length': 128},
            {'batch_size': 40, 'sequence_length': 128}
        ]

        for features in test_features:
            result = predictor.predict(features, enforce_monotonicity=True)
            values = [
                result['quantiles']['0.5'],
                result['quantiles']['0.9'],
                result['quantiles']['0.95'],
                result['quantiles']['0.99']
            ]

            # With enforce_monotonicity=True, should have ZERO violations
            for i in range(len(values) - 1):
                assert values[i] <= values[i + 1], \
                    f"Monotonicity violated even with enforce_monotonicity=True: {values}"


class TestFeatureValidation:
    """Test feature validation across predictors."""

    def test_inconsistent_features_in_training_data(self):
        """Should raise error for inconsistent features across samples."""
        predictor = ExpectErrorPredictor()

        # First sample has batch_size and sequence_length
        # Second sample has batch_size and model_size (different feature)
        inconsistent_data = [
            {'batch_size': 16, 'sequence_length': 128, 'runtime_ms': 100},
            {'batch_size': 32, 'model_size': 256, 'runtime_ms': 200}
        ] * 5  # Repeat to get enough samples

        with pytest.raises(ValueError, match="different features"):
            predictor.train(inconsistent_data)

    def test_empty_training_data(self):
        """Should handle empty training data gracefully."""
        predictor = ExpectErrorPredictor()

        with pytest.raises(ValueError):
            predictor.train([])
