"""Experiment mode utilities for synthetic predictions.

Allows testing and integration without requiring trained models.

Supported distribution types:
1. Unimodal distributions:
   - Normal (symmetric): skewness=0
   - Log-normal (right-skewed): skewness>0

2. Multimodal distributions (via exp_modes parameter):
   - Gaussian Mixture Model (GMM)
   - Each mode has: mean, weight, cv (optional)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from swarmpilot.predictor.utils.logging import get_logger

logger = get_logger()


def is_experiment_mode(
    features: dict[str, Any],
    platform_info: dict[str, str],
) -> bool:
    """Check if request is in experiment mode.

    Experiment mode is triggered if:
    1. features contains 'exp_runtime' field, OR
    2. All platform_info fields are set to "exp"

    Args:
        features: Feature dictionary from prediction request.
        platform_info: Platform information dict.

    Returns:
        True if experiment mode should be used.
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


def get_exp_runtime(features: dict[str, Any]) -> float:
    """Extract experimental runtime value from features.

    Args:
        features: Feature dictionary.

    Returns:
        Experimental runtime value.

    Raises:
        ValueError: If exp_runtime not found or invalid.
    """
    if 'exp_runtime' not in features:
        error_msg = "Experiment mode requires 'exp_runtime' in features"
        logger.error(
            f"Experiment mode validation failed\n"
            f"Error: {error_msg}\n"
            f"Features provided: {list(features.keys())}"
        )
        raise ValueError(error_msg)

    exp_runtime = features['exp_runtime']

    if not isinstance(exp_runtime, (int, float)):
        error_msg = f"exp_runtime must be numeric, got {type(exp_runtime)}"
        logger.error(
            f"Experiment mode validation failed\n"
            f"Error: {error_msg}\n"
            f"exp_runtime value: {exp_runtime}\n"
            f"exp_runtime type: {type(exp_runtime)}"
        )
        raise ValueError(error_msg)

    if exp_runtime <= 0:
        error_msg = f"exp_runtime must be positive, got {exp_runtime}"
        logger.error(
            f"Experiment mode validation failed\n"
            f"Error: {error_msg}\n"
            f"exp_runtime value: {exp_runtime}"
        )
        raise ValueError(error_msg)

    return float(exp_runtime)


def generate_expect_error_prediction(
    exp_runtime: float,
    cv: float = 0.30,
) -> dict[str, Any]:
    """Generate synthetic expect/error prediction.

    Args:
        exp_runtime: Expected runtime value.
        cv: Coefficient of variation (default: 0.30). Controls the error
            margin relative to expected runtime.

    Returns:
        Dict with expected_runtime_ms and error_margin_ms.
    """
    return {
        'expected_runtime_ms': exp_runtime,
        'error_margin_ms': exp_runtime * cv
    }


def generate_multimodal_samples(
    modes: list[dict[str, Any]],
    num_samples: int = 2000,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Generate samples from a Gaussian Mixture Model (multimodal).

    Each mode is specified as a dict with:
    - mean: Center of the mode (required)
    - weight: Probability weight (required, should sum to 1)
    - cv: Coefficient of variation (optional, default: 0.20)
    - skewness: Skewness for this mode (optional, default: 0)

    Args:
        modes: List of mode specifications.
        num_samples: Number of samples to generate.
        rng: Random state for reproducibility.

    Returns:
        Array of samples from the mixture distribution.

    Raises:
        ValueError: If modes are invalid.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    if not modes:
        error_msg = "At least one mode is required"
        logger.error(f"Multimodal sample generation failed: {error_msg}")
        raise ValueError(error_msg)

    # Normalize weights
    total_weight = sum(m.get('weight', 1.0) for m in modes)
    normalized_weights = [m.get('weight', 1.0) / total_weight for m in modes]

    # Calculate samples per mode based on weights
    samples_per_mode = [int(w * num_samples) for w in normalized_weights]

    # Adjust to ensure total samples equals num_samples
    diff = num_samples - sum(samples_per_mode)
    if diff > 0:
        samples_per_mode[0] += diff

    all_samples = []

    for mode, n_samples in zip(modes, samples_per_mode):
        if n_samples <= 0:
            continue

        mean = mode['mean']
        cv = mode.get('cv', 0.20)
        skewness = mode.get('skewness', 0.0)

        if skewness <= 0:
            # Normal distribution for this mode
            sigma = mean * cv
            mode_samples = rng.normal(loc=mean, scale=sigma, size=n_samples)
        else:
            # Log-normal distribution for this mode
            effective_cv = cv * (1 + 0.3 * skewness)
            sigma_ln = np.sqrt(np.log(1 + effective_cv ** 2))
            mu_ln = np.log(mean) - sigma_ln ** 2 / 2
            mode_samples = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=n_samples)

        all_samples.extend(mode_samples)

    # Shuffle to mix modes
    samples = np.array(all_samples)
    rng.shuffle(samples)

    return samples


def generate_quantile_prediction(
    exp_runtime: float,
    quantiles: list[float] | None = None,
    cv: float = 0.30,
    skewness: float = 0.0,
    modes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate synthetic quantile prediction using configurable distribution.

    Supports three distribution types:
    1. Unimodal Normal (default): skewness=0, modes=None
    2. Unimodal Log-normal: skewness>0, modes=None
    3. Multimodal (Gaussian Mixture): modes=[...]

    Args:
        exp_runtime: Expected runtime value.
        quantiles: List of quantile levels (defaults to [0.5, 0.9, 0.95, 0.99]).
        cv: Coefficient of variation (default: 0.30). Ignored if modes provided.
        skewness: Distribution skewness (default: 0.0). Ignored if modes provided.
        modes: List of mode specifications for multimodal distribution.

    Returns:
        Dict with quantiles key containing quantile: value pairs.
    """
    if quantiles is None:
        # Use default quantiles
        quantiles = [0.5, 0.9, 0.95, 0.99]

    # Use a fixed seed for reproducibility
    rng = np.random.RandomState(42)

    if modes is not None and len(modes) > 0:
        # Multimodal distribution (Gaussian Mixture Model)
        samples = generate_multimodal_samples(modes, num_samples=2000, rng=rng)
    elif skewness <= 0:
        # Normal distribution (symmetric)
        mu = exp_runtime
        sigma = exp_runtime * cv
        samples = rng.normal(loc=mu, scale=sigma, size=2000)
    else:
        # Log-normal distribution (right-skewed)
        # For log-normal: E[X] = exp(μ + σ²/2), CV = sqrt(exp(σ²) - 1)
        # Given target mean (exp_runtime) and CV, solve for log-normal params
        #
        # σ_ln = sqrt(ln(1 + CV²))
        # μ_ln = ln(mean) - σ_ln²/2
        #
        # Skewness of log-normal = (exp(σ²) + 2) * sqrt(exp(σ²) - 1)
        # We use skewness parameter to adjust the CV for more extreme tails

        # Adjust CV based on skewness for more realistic long-tail behavior
        effective_cv = cv * (1 + 0.3 * skewness)

        sigma_ln = np.sqrt(np.log(1 + effective_cv ** 2))
        mu_ln = np.log(exp_runtime) - sigma_ln ** 2 / 2

        samples = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=2000)

    # Ensure non-negative values
    samples = np.maximum(samples, 1.0)

    # Calculate quantile values
    quantile_value = np.quantile(samples, quantiles)
    results = {str(q): float(v) for q, v in zip(quantiles, quantile_value)}

    return {
        'quantiles': results
    }


def generate_experiment_prediction(
    prediction_type: str,
    features: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate experiment mode prediction based on prediction type.

    Distribution parameters can be specified in features:
    - exp_runtime (required): Expected runtime value in ms
    - exp_cv (optional): Coefficient of variation (default: 0.30)
    - exp_skewness (optional): Skewness parameter (default: 0.0)
    - exp_modes (optional): List of mode specifications for multimodal

    Args:
        prediction_type: Either 'expect_error' or 'quantile'.
        features: Feature dictionary with exp_runtime and optional params.
        config: Optional config (used for quantiles list in quantile type).

    Returns:
        Prediction result dict.

    Raises:
        ValueError: If prediction_type invalid or exp_runtime missing.
    """
    exp_runtime = get_exp_runtime(features)

    # Extract distribution parameters from features (with defaults for backward compatibility)
    cv = features.get('exp_cv', 0.30)
    skewness = features.get('exp_skewness', 0.0)
    modes = features.get('exp_modes', None)

    if prediction_type == 'expect_error':
        # For multimodal, calculate weighted average CV for error margin
        if modes is not None and len(modes) > 0:
            # Use the overall spread of the mixture for error margin
            # Approximate by using the weighted average of mode CVs plus inter-mode variance
            total_weight = sum(m.get('weight', 1.0) for m in modes)
            weighted_mean = sum(m['mean'] * m.get('weight', 1.0) for m in modes) / total_weight

            # Calculate variance: sum of (within-mode variance + between-mode variance)
            within_var = sum(
                m.get('weight', 1.0) * (m['mean'] * m.get('cv', 0.20)) ** 2
                for m in modes
            ) / total_weight
            between_var = sum(
                m.get('weight', 1.0) * (m['mean'] - weighted_mean) ** 2
                for m in modes
            ) / total_weight
            total_std = np.sqrt(within_var + between_var)

            effective_cv = total_std / weighted_mean if weighted_mean > 0 else cv
            return generate_expect_error_prediction(exp_runtime, cv=effective_cv)
        else:
            return generate_expect_error_prediction(exp_runtime, cv=cv)

    elif prediction_type == 'quantile':
        quantiles = None
        if config and 'quantiles' in config:
            quantiles = config['quantiles']
        return generate_quantile_prediction(
            exp_runtime,
            quantiles=quantiles,
            cv=cv,
            skewness=skewness,
            modes=modes
        )

    else:
        error_msg = f"Unknown prediction_type: {prediction_type}"
        logger.error(
            f"Experiment prediction generation failed\n"
            f"Error: {error_msg}\n"
            f"Valid types: 'expect_error', 'quantile'"
        )
        raise ValueError(error_msg)
