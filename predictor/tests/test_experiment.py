"""
Tests for experiment mode functionality.
"""

import pytest
from src.utils.experiment import (
    is_experiment_mode,
    get_exp_runtime,
    generate_expect_error_prediction,
    generate_quantile_prediction,
    generate_experiment_prediction
)


class TestExperimentModeDetection:
    """Test experiment mode detection logic."""

    def test_detect_exp_runtime_in_features(self):
        """Should detect experiment mode when exp_runtime in features."""
        features = {'exp_runtime': 100.0, 'batch_size': 32}
        platform_info = {
            'software_name': 'pytorch',
            'software_version': '2.0',
            'hardware_name': 'gpu'
        }

        assert is_experiment_mode(features, platform_info) is True

    def test_detect_all_exp_platform(self):
        """Should detect experiment mode when all platform fields are 'exp'."""
        features = {'batch_size': 32}
        platform_info = {
            'software_name': 'exp',
            'software_version': 'exp',
            'hardware_name': 'exp'
        }

        assert is_experiment_mode(features, platform_info) is True

    def test_not_experiment_mode(self):
        """Should not detect experiment mode for normal requests."""
        features = {'batch_size': 32}
        platform_info = {
            'software_name': 'pytorch',
            'software_version': '2.0',
            'hardware_name': 'gpu'
        }

        assert is_experiment_mode(features, platform_info) is False

    def test_partial_exp_platform_not_detected(self):
        """Should not detect experiment mode if only some platform fields are 'exp'."""
        features = {'batch_size': 32}
        platform_info = {
            'software_name': 'exp',
            'software_version': '2.0',
            'hardware_name': 'gpu'
        }

        assert is_experiment_mode(features, platform_info) is False


class TestExpRuntimeExtraction:
    """Test exp_runtime extraction."""

    def test_extract_valid_exp_runtime(self):
        """Should extract valid exp_runtime."""
        features = {'exp_runtime': 150.5, 'batch_size': 32}
        assert get_exp_runtime(features) == 150.5

    def test_extract_integer_exp_runtime(self):
        """Should handle integer exp_runtime."""
        features = {'exp_runtime': 100}
        assert get_exp_runtime(features) == 100.0

    def test_missing_exp_runtime_raises_error(self):
        """Should raise error if exp_runtime missing."""
        features = {'batch_size': 32}
        with pytest.raises(ValueError, match="requires 'exp_runtime'"):
            get_exp_runtime(features)

    def test_invalid_exp_runtime_type_raises_error(self):
        """Should raise error for non-numeric exp_runtime."""
        features = {'exp_runtime': 'fast'}
        with pytest.raises(ValueError, match="must be numeric"):
            get_exp_runtime(features)

    def test_negative_exp_runtime_raises_error(self):
        """Should raise error for negative exp_runtime."""
        features = {'exp_runtime': -10.0}
        with pytest.raises(ValueError, match="must be positive"):
            get_exp_runtime(features)


class TestExpectErrorPrediction:
    """Test expect/error synthetic prediction generation."""

    def test_generate_expect_error(self):
        """Should generate correct expect/error prediction."""
        result = generate_expect_error_prediction(100.0)

        assert 'expected_runtime_ms' in result
        assert 'error_margin_ms' in result
        assert result['expected_runtime_ms'] == 100.0
        assert result['error_margin_ms'] == 5.0  # 5% of 100

    def test_error_margin_scales_with_runtime(self):
        """Error margin should be 5% of runtime."""
        result = generate_expect_error_prediction(200.0)
        assert result['error_margin_ms'] == 10.0


class TestQuantilePrediction:
    """Test quantile synthetic prediction generation."""

    def test_generate_default_quantiles(self):
        """Should generate predictions for default quantiles."""
        result = generate_quantile_prediction(100.0)

        assert 'quantiles' in result
        quantiles = result['quantiles']

        assert '0.5' in quantiles
        assert '0.9' in quantiles
        assert '0.95' in quantiles
        assert '0.99' in quantiles

        # Check values match expected multipliers (using pytest.approx for floating point)
        assert quantiles['0.5'] == pytest.approx(100.0)  # 1.0x
        assert quantiles['0.9'] == pytest.approx(105.0)  # 1.05x
        assert quantiles['0.95'] == pytest.approx(107.5)  # 1.075x
        assert quantiles['0.99'] == pytest.approx(112.0)  # 1.12x

    def test_generate_custom_quantiles(self):
        """Should handle custom quantile list."""
        result = generate_quantile_prediction(100.0, quantiles=[0.5, 0.9])

        quantiles = result['quantiles']
        assert len(quantiles) == 2
        assert '0.5' in quantiles
        assert '0.9' in quantiles

    def test_quantiles_are_monotonic(self):
        """Quantile predictions should be non-decreasing."""
        result = generate_quantile_prediction(100.0)
        quantiles = result['quantiles']

        values = [quantiles['0.5'], quantiles['0.9'], quantiles['0.95'], quantiles['0.99']]
        assert values == sorted(values)


class TestExperimentPredictionGeneration:
    """Test complete experiment prediction generation."""

    def test_generate_expect_error_type(self):
        """Should generate expect_error prediction."""
        features = {'exp_runtime': 120.0, 'batch_size': 32}
        result = generate_experiment_prediction('expect_error', features)

        assert 'expected_runtime_ms' in result
        assert 'error_margin_ms' in result
        assert result['expected_runtime_ms'] == 120.0

    def test_generate_quantile_type(self):
        """Should generate quantile prediction."""
        features = {'exp_runtime': 120.0, 'batch_size': 32}
        result = generate_experiment_prediction('quantile', features)

        assert 'quantiles' in result
        assert '0.5' in result['quantiles']

    def test_invalid_prediction_type_raises_error(self):
        """Should raise error for invalid prediction type."""
        features = {'exp_runtime': 120.0}
        with pytest.raises(ValueError, match="Unknown prediction_type"):
            generate_experiment_prediction('invalid_type', features)

    def test_missing_exp_runtime_raises_error(self):
        """Should raise error if exp_runtime missing."""
        features = {'batch_size': 32}
        with pytest.raises(ValueError, match="requires 'exp_runtime'"):
            generate_experiment_prediction('expect_error', features)

    def test_quantile_with_custom_config(self):
        """Should use custom quantiles from config."""
        features = {'exp_runtime': 100.0}
        config = {'quantiles': [0.5, 0.9]}

        result = generate_experiment_prediction('quantile', features, config)

        quantiles = result['quantiles']
        assert len(quantiles) == 2
        assert '0.5' in quantiles
        assert '0.9' in quantiles
