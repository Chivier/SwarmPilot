"""
Experiment mode utilities for synthetic predictions.

Allows testing and integration without requiring trained models.

Supported distribution types:
1. Unimodal distributions:
   - Normal (symmetric): skewness=0
   - Log-normal (right-skewed): skewness>0

2. Multimodal distributions (via exp_modes parameter):
   - Gaussian Mixture Model (GMM)
   - Each mode has: mean, weight, cv (optional)

Example multimodal configuration:
    features = {
        'exp_runtime': 5000,  # Overall expected runtime (used as reference)
        'exp_modes': [
            {'mean': 2000, 'weight': 0.6, 'cv': 0.15},  # Fast mode (60%)
            {'mean': 8000, 'weight': 0.3, 'cv': 0.20},  # Slow mode (30%)
            {'mean': 15000, 'weight': 0.1, 'cv': 0.10}  # Very slow mode (10%)
        ]
    }
"""

from typing import Any, Dict, List, Optional
import numpy as np


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


def generate_expect_error_prediction(
    exp_runtime: float,
    cv: float = 0.30
) -> Dict[str, Any]:
    """
    Generate synthetic expect/error prediction.

    Args:
        exp_runtime: Expected runtime value
        cv: Coefficient of variation (default: 0.30, i.e., 30% standard deviation)
            This parameter controls the error margin relative to expected runtime.
            Higher CV means more uncertainty in the prediction.

    Returns:
        Dict with expected_runtime_ms and error_margin_ms
    """
    return {
        'expected_runtime_ms': exp_runtime,
        'error_margin_ms': exp_runtime * cv
    }


def generate_multimodal_samples(
    modes: List[Dict[str, Any]],
    num_samples: int = 2000,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Generate samples from a Gaussian Mixture Model (multimodal distribution).

    Each mode is specified as a dict with:
    - mean: Center of the mode (required)
    - weight: Probability weight for this mode (required, should sum to 1)
    - cv: Coefficient of variation for this mode (optional, default: 0.20)
    - skewness: Skewness for this mode (optional, default: 0, uses normal if 0)

    Args:
        modes: List of mode specifications
        num_samples: Number of samples to generate
        rng: Random state for reproducibility

    Returns:
        Array of samples from the mixture distribution

    Raises:
        ValueError: If modes are invalid
    """
    if rng is None:
        rng = np.random.RandomState(42)

    if not modes:
        raise ValueError("At least one mode is required")

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
    quantiles: list = None,
    cv: float = 0.30,
    skewness: float = 0.0,
    modes: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate synthetic quantile prediction using configurable distribution.

    Supports three distribution types:

    1. Unimodal Normal (default): skewness=0, modes=None
       - Symmetric distribution around exp_runtime
       - Good for stable, predictable tasks

    2. Unimodal Log-normal: skewness>0, modes=None
       - Right-skewed (long-tail) distribution
       - Good for tasks with occasional long executions

    3. Multimodal (Gaussian Mixture): modes=[...] (overrides cv/skewness)
       - Multiple peaks in the distribution
       - Good for mixed workloads (e.g., cache hit vs miss, batch sizes)

    The coefficient of variation (CV) controls the spread of the distribution.
    For probabilistic scheduling to work well, the CV should match the actual
    runtime variability. Typical values:
    - Low variability tasks: CV = 0.10 - 0.20
    - Medium variability: CV = 0.30 - 0.50
    - High variability (long-tail): CV = 0.50 - 1.50

    Args:
        exp_runtime: Expected runtime value (used as distribution center for unimodal,
                     or as reference for error margin calculation in multimodal)
        quantiles: List of quantile levels (optional, defaults to [0.5, 0.9, 0.95, 0.99])
        cv: Coefficient of variation (default: 0.30, i.e., 30% standard deviation)
            Higher CV produces wider spread between quantiles, which is important
            for probabilistic scheduling to differentiate instances effectively.
            Ignored if modes is provided.
        skewness: Distribution skewness control (default: 0.0)
            - 0.0: Use normal distribution (symmetric)
            - > 0: Use log-normal distribution with right skew
            - Typical long-tail tasks have skewness 1.0 - 3.0
            Ignored if modes is provided.
        modes: List of mode specifications for multimodal distribution (optional)
            Each mode is a dict with:
            - mean: Center of the mode (required)
            - weight: Probability weight for this mode (required)
            - cv: Coefficient of variation for this mode (optional, default: 0.20)
            - skewness: Skewness for this mode (optional, default: 0)

            Example for bimodal (cache hit/miss):
            [
                {'mean': 50, 'weight': 0.8, 'cv': 0.10},   # Cache hit (80%)
                {'mean': 500, 'weight': 0.2, 'cv': 0.30}   # Cache miss (20%)
            ]

    Returns:
        Dict with 'quantiles' key containing quantile: value pairs
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
    features: Dict[str, Any],
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate experiment mode prediction based on prediction type.

    BACKWARD COMPATIBLE: All new parameters are optional with sensible defaults.
    Existing code using only 'exp_runtime' will continue to work unchanged.

    Distribution parameters can be specified in features:

    1. Unimodal distribution (default, backward compatible):
       - exp_runtime (required): Expected runtime value in ms
       - exp_cv (optional): Coefficient of variation (default: 0.30)
       - exp_skewness (optional): Skewness parameter (default: 0.0)

    2. Multimodal distribution (new feature):
       - exp_runtime (required): Used as reference for error margin calculation
       - exp_modes (optional): List of mode specifications, each with:
         - mean: Center of the mode (required)
         - weight: Probability weight (required)
         - cv: Coefficient of variation for this mode (optional, default: 0.20)
         - skewness: Skewness for this mode (optional, default: 0)

    Example configurations:

    # Backward compatible (unimodal normal)
    features = {'exp_runtime': 1000}

    # Unimodal with long tail
    features = {'exp_runtime': 1000, 'exp_cv': 1.0, 'exp_skewness': 2.5}

    # Bimodal (cache hit/miss scenario)
    features = {
        'exp_runtime': 275,  # Weighted average for error margin
        'exp_modes': [
            {'mean': 50, 'weight': 0.8, 'cv': 0.10},   # Cache hit (80%)
            {'mean': 500, 'weight': 0.2, 'cv': 0.30}   # Cache miss (20%)
        ]
    }

    # Trimodal (different batch sizes)
    features = {
        'exp_runtime': 5000,
        'exp_modes': [
            {'mean': 1000, 'weight': 0.5, 'cv': 0.15},   # Small batch (50%)
            {'mean': 5000, 'weight': 0.35, 'cv': 0.20},  # Medium batch (35%)
            {'mean': 15000, 'weight': 0.15, 'cv': 0.10}  # Large batch (15%)
        ]
    }

    Args:
        prediction_type: Either 'expect_error' or 'quantile'
        features: Feature dictionary with exp_runtime and optional distribution params
        config: Optional config (used for quantiles list in quantile type)

    Returns:
        Prediction result dict

    Raises:
        ValueError: If prediction_type invalid or exp_runtime missing
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
        raise ValueError(f"Unknown prediction_type: {prediction_type}")
