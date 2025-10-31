"""
Experiment mode utilities for synthetic predictions.

Allows testing and integration without requiring trained models.
"""

from typing import Any, Dict


def is_experiment_mode(features: Dict[str, Any], platform_info: Dict[str, str]) -> bool:
    """
    Check if request is in experiment mode.

    Experiment mode is triggered if:
    1. features contains 'exp_runtime' field, OR
    2. All platform_info fields are set to "exp"

    Args:
        features: Feature dictionary from prediction request
        platform_info: Platform information dict

    Returns:
        True if experiment mode should be used
    """
    # Check for exp_runtime in features
    if 'exp_runtime' in features:
        return True

    # Check if all platform info fields are "exp"
    if (platform_info.get('software_name') == 'exp' and
        platform_info.get('software_version') == 'exp' and
        platform_info.get('hardware_name') == 'exp'):
        return True

    return False


def get_exp_runtime(features: Dict[str, Any]) -> float:
    """
    Extract experimental runtime value from features.

    Args:
        features: Feature dictionary

    Returns:
        Experimental runtime value

    Raises:
        ValueError: If exp_runtime not found or invalid
    """
    if 'exp_runtime' not in features:
        raise ValueError("Experiment mode requires 'exp_runtime' in features")

    exp_runtime = features['exp_runtime']

    if not isinstance(exp_runtime, (int, float)):
        raise ValueError(f"exp_runtime must be numeric, got {type(exp_runtime)}")

    if exp_runtime <= 0:
        raise ValueError(f"exp_runtime must be positive, got {exp_runtime}")

    return float(exp_runtime)


def generate_expect_error_prediction(exp_runtime: float) -> Dict[str, Any]:
    """
    Generate synthetic expect/error prediction.

    Args:
        exp_runtime: Expected runtime value

    Returns:
        Dict with expected_runtime_ms and error_margin_ms
    """
    return {
        'expected_runtime_ms': exp_runtime,
        'error_margin_ms': exp_runtime * 0.05  # 5% error margin
    }


def generate_quantile_prediction(exp_runtime: float, quantiles: list = None) -> Dict[str, Any]:
    """
    Generate synthetic quantile prediction.

    Uses fixed multipliers for common quantiles:
    - 0.5: exp_runtime (median)
    - 0.9: exp_runtime * 1.05
    - 0.95: exp_runtime * 1.075
    - 0.99: exp_runtime * 1.12

    For other quantiles, interpolates between these values.

    Args:
        exp_runtime: Expected runtime value
        quantiles: List of quantile levels (optional, for custom quantiles)

    Returns:
        Dict with 'quantiles' key containing quantile: value pairs
    """
    # Default quantiles and their multipliers
    quantile_multipliers = {
        0.5: 1.0,
        0.9: 1.05,
        0.95: 1.075,
        0.99: 1.12
    }

    if quantiles is None:
        # Use default quantiles
        quantiles = [0.5, 0.9, 0.95, 0.99]

    results = {}

    for q in quantiles:
        q_float = float(q)

        # Use exact multiplier if available
        if q_float in quantile_multipliers:
            multiplier = quantile_multipliers[q_float]
        else:
            # Interpolate for custom quantiles
            # Simple linear interpolation based on quantile value
            # Lower quantiles get lower multipliers
            multiplier = 1.0 + (q_float * 0.12)  # 0.12 is the max multiplier at q=1.0

        results[str(q)] = exp_runtime * multiplier

    return {
        'quantiles': results
    }


def generate_experiment_prediction(
    prediction_type: str,
    features: Dict[str, Any],
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate experiment mode prediction based on prediction type.

    Args:
        prediction_type: Either 'expect_error' or 'quantile'
        features: Feature dictionary with exp_runtime
        config: Optional config (used for quantiles list in quantile type)

    Returns:
        Prediction result dict

    Raises:
        ValueError: If prediction_type invalid or exp_runtime missing
    """
    exp_runtime = get_exp_runtime(features)

    if prediction_type == 'expect_error':
        return generate_expect_error_prediction(exp_runtime)

    elif prediction_type == 'quantile':
        quantiles = None
        if config and 'quantiles' in config:
            quantiles = config['quantiles']
        return generate_quantile_prediction(exp_runtime, quantiles)

    else:
        raise ValueError(f"Unknown prediction_type: {prediction_type}")
