"""
Tests for experiment mode functionality.
"""

import pytest
from src.utils.experiment import (
    is_experiment_mode,
    get_exp_runtime,
    generate_expect_error_prediction,
    generate_quantile_prediction,
    generate_experiment_prediction,
    generate_multimodal_samples
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
        assert result['error_margin_ms'] == 30.0  # 30% of 100 (new default CV)

    def test_error_margin_scales_with_runtime(self):
        """Error margin should be 30% of runtime (default CV)."""
        result = generate_expect_error_prediction(200.0)
        assert result['error_margin_ms'] == 60.0  # 30% of 200

    def test_error_margin_custom_cv(self):
        """Error margin should respect custom CV parameter."""
        result = generate_expect_error_prediction(100.0, cv=0.10)
        assert result['error_margin_ms'] == 10.0  # 10% of 100


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

        # Check values are based on normal distribution with mu=100, sigma=30 (cv=0.30)
        # These values come from sampling a normal distribution
        # 0.5 quantile (median) should be close to mu
        assert quantiles['0.5'] == pytest.approx(100.0, abs=5.0)
        # Higher quantiles should be greater than median
        assert quantiles['0.9'] > quantiles['0.5']
        assert quantiles['0.95'] > quantiles['0.9']
        assert quantiles['0.99'] > quantiles['0.95']
        # All values should be reasonable for N(100, 30)
        # 0.99 quantile should be around mu + 2.33*sigma ≈ 169.9
        assert quantiles['0.99'] == pytest.approx(100.0 + 2.33 * 30.0, abs=5.0)

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


class TestMultimodalDistribution:
    """Test multimodal (Gaussian Mixture Model) distribution support."""

    def test_bimodal_distribution(self):
        """Test bimodal distribution with two peaks."""
        features = {
            'exp_runtime': 275,  # Weighted average for reference
            'exp_modes': [
                {'mean': 50, 'weight': 0.8, 'cv': 0.10},   # Fast mode (80%)
                {'mean': 500, 'weight': 0.2, 'cv': 0.30}   # Slow mode (20%)
            ]
        }
        result = generate_experiment_prediction('quantile', features)

        quantiles = result['quantiles']
        q50 = float(quantiles['0.5'])
        q99 = float(quantiles['0.99'])

        # Bimodal distribution: q50 should be near the dominant mode (50)
        # q99 should be near the secondary mode (500)
        assert q50 < 200, f"Median {q50:.1f} should be closer to dominant mode (50)"
        assert q99 > 400, f"q99 {q99:.1f} should reach secondary mode region (~500)"

    def test_trimodal_distribution(self):
        """Test trimodal distribution with three peaks."""
        features = {
            'exp_runtime': 5000,
            'exp_modes': [
                {'mean': 1000, 'weight': 0.5, 'cv': 0.15},   # Small batch (50%)
                {'mean': 5000, 'weight': 0.35, 'cv': 0.20},  # Medium batch (35%)
                {'mean': 15000, 'weight': 0.15, 'cv': 0.10}  # Large batch (15%)
            ]
        }
        result = generate_experiment_prediction('quantile', features)

        quantiles = result['quantiles']
        q50 = float(quantiles['0.5'])
        q99 = float(quantiles['0.99'])

        # q50 should be in or near the dominant mode (1000)
        assert 500 < q50 < 3000, f"Median {q50:.1f} should be near dominant mode (1000)"
        # q99 should reach the large batch mode
        assert q99 > 10000, f"q99 {q99:.1f} should reach large batch mode (~15000)"

    def test_multimodal_expect_error(self):
        """Test expect_error calculation for multimodal distribution."""
        features = {
            'exp_runtime': 275,
            'exp_modes': [
                {'mean': 50, 'weight': 0.8, 'cv': 0.10},
                {'mean': 500, 'weight': 0.2, 'cv': 0.30}
            ]
        }
        result = generate_experiment_prediction('expect_error', features)

        # Error margin should account for both within-mode and between-mode variance
        assert 'expected_runtime_ms' in result
        assert 'error_margin_ms' in result
        assert result['expected_runtime_ms'] == 275

        # For bimodal with large separation, error margin should be significant
        error_margin = result['error_margin_ms']
        assert error_margin > 50, f"Error margin {error_margin:.1f} should be significant"

    def test_mode_with_skewness(self):
        """Test multimodal distribution where individual modes have skewness."""
        features = {
            'exp_runtime': 5000,
            'exp_modes': [
                {'mean': 2000, 'weight': 0.7, 'cv': 0.20, 'skewness': 0.0},  # Normal
                {'mean': 10000, 'weight': 0.3, 'cv': 0.50, 'skewness': 2.0}  # Skewed
            ]
        }
        result = generate_experiment_prediction('quantile', features)

        quantiles = result['quantiles']
        q50 = float(quantiles['0.5'])
        q99 = float(quantiles['0.99'])

        # The skewed mode should extend the tail further
        assert q99 > 15000, f"q99 {q99:.1f} should have extended tail due to skewed mode"

    def test_backward_compatible_unimodal(self):
        """Test that omitting exp_modes uses unimodal distribution (backward compatible)."""
        # Without exp_modes - should use unimodal normal
        features_no_modes = {'exp_runtime': 1000}
        result_no_modes = generate_experiment_prediction('quantile', features_no_modes)

        # With empty exp_modes - should also use unimodal
        features_empty = {'exp_runtime': 1000, 'exp_modes': []}
        result_empty = generate_experiment_prediction('quantile', features_empty)

        # Both should produce similar results (using normal distribution)
        q50_no_modes = float(result_no_modes['quantiles']['0.5'])
        q50_empty = float(result_empty['quantiles']['0.5'])

        # Both medians should be close to exp_runtime for normal distribution
        assert abs(q50_no_modes - 1000) < 100
        assert abs(q50_empty - 1000) < 100


class TestDistributionParameters:
    """Test configurable distribution parameters (CV and skewness)."""

    def test_custom_cv_expect_error(self):
        """Should respect custom CV for expect_error."""
        features = {'exp_runtime': 100.0, 'exp_cv': 0.10}
        result = generate_experiment_prediction('expect_error', features)

        # 10% CV means error = 10
        assert result['error_margin_ms'] == 10.0

    def test_custom_cv_quantile(self):
        """Should respect custom CV for quantile predictions."""
        # Low CV: spread should be smaller
        features_low_cv = {'exp_runtime': 1000.0, 'exp_cv': 0.10}
        result_low = generate_experiment_prediction('quantile', features_low_cv)

        # High CV: spread should be larger
        features_high_cv = {'exp_runtime': 1000.0, 'exp_cv': 0.50}
        result_high = generate_experiment_prediction('quantile', features_high_cv)

        # Compare spreads
        spread_low = float(result_low['quantiles']['0.99']) - float(result_low['quantiles']['0.5'])
        spread_high = float(result_high['quantiles']['0.99']) - float(result_high['quantiles']['0.5'])

        assert spread_high > spread_low * 2  # High CV should have much wider spread

    def test_skewness_creates_long_tail(self):
        """Positive skewness should create right-tailed distribution."""
        # No skewness: symmetric distribution
        features_sym = {'exp_runtime': 1000.0, 'exp_cv': 0.50, 'exp_skewness': 0.0}
        result_sym = generate_experiment_prediction('quantile', features_sym)

        # With skewness: right-tailed distribution
        features_skew = {'exp_runtime': 1000.0, 'exp_cv': 0.50, 'exp_skewness': 2.0}
        result_skew = generate_experiment_prediction('quantile', features_skew)

        # Calculate q99/q50 ratio
        ratio_sym = float(result_sym['quantiles']['0.99']) / float(result_sym['quantiles']['0.5'])
        ratio_skew = float(result_skew['quantiles']['0.99']) / float(result_skew['quantiles']['0.5'])

        # Skewed distribution should have higher ratio (heavier right tail)
        assert ratio_skew > ratio_sym * 1.5

    def test_type1_b_task_distribution(self):
        """Test distribution settings matching Type1 B task characteristics."""
        # B task settings: CV=1.0, skewness=2.5 (matching real T2VID data)
        features = {'exp_runtime': 10000.0, 'exp_cv': 1.0, 'exp_skewness': 2.5}
        result = generate_experiment_prediction('quantile', features)

        quantiles = result['quantiles']

        # q99 should be significantly higher than q50 for long-tail distribution
        q50 = float(quantiles['0.5'])
        q99 = float(quantiles['0.99'])

        # For heavy right-tail, q99/q50 should be > 5x
        ratio = q99 / q50
        assert ratio > 5.0, f"Expected ratio > 5.0 for long-tail, got {ratio:.2f}"

    def test_type1_a_task_distribution(self):
        """Test distribution settings matching Type1 A task characteristics."""
        # A task settings: CV=0.40, skewness=0.0 (matching real LLM data)
        features = {'exp_runtime': 3000.0, 'exp_cv': 0.40, 'exp_skewness': 0.0}
        result = generate_experiment_prediction('quantile', features)

        quantiles = result['quantiles']

        # q50 should be close to exp_runtime for symmetric distribution
        q50 = float(quantiles['0.5'])
        assert abs(q50 - 3000.0) < 500.0  # Within 500ms of expected

        # q99 should be reasonable for normal distribution (< 2.5x median)
        q99 = float(quantiles['0.99'])
        ratio = q99 / q50
        assert ratio < 2.5, f"Expected ratio < 2.5 for symmetric, got {ratio:.2f}"


class TestMultimodalEdgeCases:
    """Test multimodal edge cases and error handling."""

    def test_multimodal_without_rng(self):
        """Should use default random state when rng is None."""
        modes = [
            {'mean': 100, 'cv': 0.2, 'weight': 1.0}
        ]
        # No rng provided - should use default
        samples = generate_multimodal_samples(modes, num_samples=10)
        assert len(samples) == 10

    def test_multimodal_empty_modes_raises_error(self):
        """Should raise error when modes list is empty."""
        with pytest.raises(ValueError, match="At least one mode"):
            generate_multimodal_samples([], num_samples=10)

    def test_multimodal_with_zero_weight_mode(self):
        """Should handle mode with very low weight that produces 0 samples."""
        import numpy as np
        modes = [
            {'mean': 100, 'cv': 0.2, 'weight': 100.0},  # Dominant mode
            {'mean': 1000, 'cv': 0.2, 'weight': 0.001},  # Negligible weight
        ]
        # With small total samples, the tiny weight mode gets 0 samples
        samples = generate_multimodal_samples(modes, num_samples=10, rng=np.random.RandomState(42))
        assert len(samples) == 10

    def test_multimodal_sample_count_adjustment(self):
        """Should adjust sample counts to match total."""
        import numpy as np
        modes = [
            {'mean': 100, 'cv': 0.2, 'weight': 1.0},
            {'mean': 200, 'cv': 0.2, 'weight': 1.0},
            {'mean': 300, 'cv': 0.2, 'weight': 1.0},
        ]
        # 7 samples split 3 ways - needs adjustment
        samples = generate_multimodal_samples(modes, num_samples=7, rng=np.random.RandomState(42))
        assert len(samples) == 7
