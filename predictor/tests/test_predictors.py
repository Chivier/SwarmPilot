"""
Tests for predictor implementations.
"""

import pytest
import numpy as np
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
        """Quantile predictions should be non-decreasing."""
        predictor = QuantilePredictor()
        training_data = generate_training_data(50)  # More samples for better training
        predictor.train(training_data, config={'epochs': 500})

        result = predictor.predict({'batch_size': 25, 'sequence_length': 128})
        quantiles = result['quantiles']

        # Extract values in order
        values = [
            quantiles['0.5'],
            quantiles['0.9'],
            quantiles['0.95'],
            quantiles['0.99']
        ]

        # Check monotonicity (allowing for small numerical errors)
        for i in range(len(values) - 1):
            # Allow 1% tolerance for numerical instability
            assert values[i] <= values[i + 1] * 1.01, \
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
