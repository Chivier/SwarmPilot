"""
OOD Task Data Structures and Sleep Time Loading.

This module provides:
- Sleep time distribution loading from training data
- OODTaskData dataclass for task tracking
- Pre-generation of task data
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# =============================================================================
# Training Data Configuration
# =============================================================================

# Path to training data (reuse from type2_deep_research)
DATA_DIR = Path(__file__).parent.parent / "type2_deep_research" / "data"
TRAINING_DATA_FILE = DATA_DIR / "training_data_llm_service_large_model.json"

# Simulation time scaling factor (consistent with type2_deep_research)
SIMULATION_TIME_SCALE = 0.5

# Cached sleep times
_BASE_SLEEP_TIMES_CACHE: Optional[List[float]] = None
_SMALL_PEAK_CACHE: Optional[List[float]] = None
_LARGE_PEAK_CACHE: Optional[List[float]] = None

# Cached medians for four-peak distribution
_SHORT_MEDIAN_CACHE: Optional[float] = None
_LONG_MEDIAN_CACHE: Optional[float] = None

# Default peak threshold (consistent with config.py default)
DEFAULT_PEAK_THRESHOLD = 20.0


def load_base_sleep_times() -> List[float]:
    """
    Load base sleep time distribution from training data.

    Loads ALL runtime_ms values from training data, ignoring task_type.
    Returns times in seconds, scaled by SIMULATION_TIME_SCALE.

    Returns:
        List of sleep times in seconds

    Raises:
        FileNotFoundError: If training data file doesn't exist
    """
    global _BASE_SLEEP_TIMES_CACHE

    if _BASE_SLEEP_TIMES_CACHE is not None:
        return _BASE_SLEEP_TIMES_CACHE

    if not TRAINING_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training data file not found: {TRAINING_DATA_FILE}\n"
            f"Please ensure the file exists."
        )

    with open(TRAINING_DATA_FILE, 'r') as f:
        data = json.load(f)

    all_times = []
    for sample in data.get('samples', []):
        runtime_ms = sample.get('runtime_ms', 0)
        if runtime_ms > 0:
            # Convert ms to seconds and apply simulation scale
            all_times.append(runtime_ms / 1000.0 * SIMULATION_TIME_SCALE)

    if not all_times:
        raise ValueError("No valid runtime samples found in training data")

    _BASE_SLEEP_TIMES_CACHE = all_times
    return all_times


def load_separated_peaks(
    threshold: float = DEFAULT_PEAK_THRESHOLD
) -> tuple[List[float], List[float]]:
    """
    Load base sleep times separated into small peak and large peak.

    The bimodal distribution is split at threshold:
    - Small peak: times < threshold (shorter tasks)
    - Large peak: times >= threshold (longer tasks)

    Args:
        threshold: Value to separate the two peaks (default: 20.0)

    Returns:
        Tuple of (small_peak_times, large_peak_times)
    """
    global _SMALL_PEAK_CACHE, _LARGE_PEAK_CACHE

    if _SMALL_PEAK_CACHE is not None and _LARGE_PEAK_CACHE is not None:
        return _SMALL_PEAK_CACHE, _LARGE_PEAK_CACHE

    all_times = load_base_sleep_times()

    small_peak = [t for t in all_times if t < threshold]
    large_peak = [t for t in all_times if t >= threshold]

    if not small_peak:
        raise ValueError(f"No samples found below threshold {threshold}")
    if not large_peak:
        raise ValueError(f"No samples found at or above threshold {threshold}")

    _SMALL_PEAK_CACHE = small_peak
    _LARGE_PEAK_CACHE = large_peak

    return small_peak, large_peak


def get_peak_medians(
    threshold: float = DEFAULT_PEAK_THRESHOLD
) -> tuple[float, float]:
    """
    Get median values for short and long peaks.

    Used by four-peak distribution to split each peak into two sub-peaks
    at the median, guaranteeing complete separation.

    Args:
        threshold: Value to separate the two peaks (default: 20.0)

    Returns:
        Tuple of (short_median, long_median)
    """
    global _SHORT_MEDIAN_CACHE, _LONG_MEDIAN_CACHE

    if _SHORT_MEDIAN_CACHE is not None and _LONG_MEDIAN_CACHE is not None:
        return _SHORT_MEDIAN_CACHE, _LONG_MEDIAN_CACHE

    small_peak, large_peak = load_separated_peaks(threshold)

    _SHORT_MEDIAN_CACHE = float(np.median(small_peak))
    _LONG_MEDIAN_CACHE = float(np.median(large_peak))

    return _SHORT_MEDIAN_CACHE, _LONG_MEDIAN_CACHE


def get_sleep_time_statistics() -> Dict[str, float]:
    """
    Get statistics about the sleep time distribution.

    Returns:
        Dict with min, max, mean, std, median values
    """
    times = load_base_sleep_times()
    return {
        "count": len(times),
        "min": min(times),
        "max": max(times),
        "mean": np.mean(times),
        "std": np.std(times),
        "median": np.median(times),
    }


def sample_base_sleep_time(
    rng: np.random.Generator,
    count: int = 1
) -> List[float]:
    """
    Sample base sleep times from the distribution.

    Args:
        rng: NumPy random generator for reproducibility
        count: Number of samples to draw

    Returns:
        List of sampled sleep times
    """
    times = load_base_sleep_times()
    indices = rng.integers(0, len(times), size=count)
    return [times[i] for i in indices]


def calculate_phase1_mean(config) -> float:
    """
    Calculate the mean runtime for Phase 1 tasks.

    Phase 1 uses weighted bimodal sampling (80% small / 20% large by default)
    with phase1_scale applied.

    Args:
        config: OODRecoveryConfig with phase1_scale and phase1_small_peak_ratio

    Returns:
        Mean runtime for Phase 1 tasks in seconds
    """
    small_peak, large_peak = load_separated_peaks(config.peak_threshold)
    small_peak_ratio = getattr(config, 'phase1_small_peak_ratio', 0.8)

    small_mean = np.mean(small_peak)
    large_mean = np.mean(large_peak)

    weighted_mean = small_peak_ratio * small_mean + (1 - small_peak_ratio) * large_mean
    return weighted_mean * config.phase1_scale


def calculate_phase23_mean(config) -> float:
    """
    Calculate the mean runtime for Phase 2/3 tasks based on distribution mode.

    Args:
        config: OODRecoveryConfig with distribution parameters

    Returns:
        Mean runtime for Phase 2/3 tasks in seconds
    """
    if config.phase23_distribution == "normal":
        # Normal distribution: mean = normal_mean
        return config.normal_mean
    elif config.phase23_distribution == "uniform":
        # Uniform distribution: mean = (min + max) / 2
        return (config.uniform_min + config.uniform_max) / 2.0
    elif config.phase23_distribution == "weighted_bimodal":
        # Weighted bimodal distribution:
        # Mean = (small_peak_ratio * mean(small_peak) + (1-small_peak_ratio) * mean(large_peak)) * scale
        small_peak, large_peak = load_separated_peaks(config.peak_threshold)
        small_peak_ratio = getattr(config, 'phase23_small_peak_ratio', 0.2)
        bimodal_scale = getattr(config, 'phase23_bimodal_scale', 2.0)

        small_mean = np.mean(small_peak)
        large_mean = np.mean(large_peak)

        weighted_mean = small_peak_ratio * small_mean + (1 - small_peak_ratio) * large_mean
        return weighted_mean * bimodal_scale
    elif config.phase23_distribution == "four_peak":
        # Four-peak distribution: weighted average based on peak membership and scale factors
        short_median, long_median = get_peak_medians(config.peak_threshold)
        small_peak, large_peak = load_separated_peaks(config.peak_threshold)

        # Split each peak at median
        short_below = [t for t in small_peak if t < short_median]
        short_above = [t for t in small_peak if t >= short_median]
        long_below = [t for t in large_peak if t < long_median]
        long_above = [t for t in large_peak if t >= long_median]

        # Calculate weighted mean
        total = 0.0
        count = 0
        if short_below:
            total += np.mean(short_below) * config.four_peak_scale_1a * len(short_below)
            count += len(short_below)
        if short_above:
            total += np.mean(short_above) * config.four_peak_scale_1b * len(short_above)
            count += len(short_above)
        if long_below:
            total += np.mean(long_below) * config.four_peak_scale_2a * len(long_below)
            count += len(long_below)
        if long_above:
            total += np.mean(long_above) * config.four_peak_scale_2b * len(long_above)
            count += len(long_above)

        return total / count if count > 0 else 0.0
    else:
        # Peak-dependent transformation
        times = load_base_sleep_times()

        # Apply peak-dependent transformation
        transformed_times = []
        for t in times:
            if t < config.peak_threshold:
                transformed_times.append(t * config.peak1_factor)
            else:
                transformed_times.append(t * config.peak2_factor)

        return np.mean(transformed_times)


def get_qps_scale_factor(config) -> float:
    """
    Get the QPS scale factor for Phase 1 → Phase 2 transition.

    QPS should scale inversely with the runtime ratio:
    - If Phase 2/3 runtime is longer than Phase 1, QPS should decrease
    - If Phase 2/3 runtime is shorter than Phase 1, QPS should increase

    Formula:
        scale_factor = Phase1_mean / Phase23_mean

    Example:
        Phase 1 mean = 3.43s, Phase 2/3 mean = 10.11s
        scale_factor = 3.43 / 10.11 = 0.34
        If original QPS = 10, new QPS = 10 * 0.34 = 3.4

    Args:
        config: OODRecoveryConfig with phase transformation parameters

    Returns:
        Factor to multiply original QPS by (< 1 means decrease QPS)
    """
    phase1_mean = calculate_phase1_mean(config)
    phase23_mean = calculate_phase23_mean(config)

    # Inverse ratio: if tasks are slower (longer), submit fewer per second
    return phase1_mean / phase23_mean if phase23_mean > 0 else 1.0


# =============================================================================
# Task Data Structure
# =============================================================================

@dataclass
class OODTaskData:
    """Data structure for a single OOD task."""

    task_id: str
    """Unique task identifier."""

    task_index: int
    """Task index in the sequence (0-based)."""

    phase: int = 1
    """Task phase: 1, 2, or 3."""

    base_sleep_time: float = 0.0
    """Base sleep time sampled from Phase 1 weighted bimodal distribution (before scaling)."""

    phase23_base_sleep_time: float = 0.0
    """Base sleep time sampled from Phase 2/3 weighted bimodal distribution (before scaling).
    Used when phase23_distribution='weighted_bimodal'."""

    exp_runtime_base: float = 0.0
    """Independent base for exp_runtime, sampled from Phase 1 distribution.
    Used in Phase 2 to simulate predictor trained on old (Phase 1) data."""

    exp_runtime_small_peak: float = 0.0
    """Pre-sampled exp_runtime base from small peak (Peak A, < threshold).
    Used for inverse correlation: large actual_sleep → small prediction."""

    exp_runtime_large_peak: float = 0.0
    """Pre-sampled exp_runtime base from large peak (Peak B, >= threshold).
    Used for inverse correlation: small actual_sleep → large prediction."""

    phase23_random_scale: float = 0.0
    """Random scaling factor for Phase 2/3 actual runtime, uniformly sampled from [1.0, 2.0].
    Used only when phase23_distribution='peak_dependent'."""

    uniform_ratio: float = 0.0
    """Random ratio in [0, 1] for uniform distribution sampling.
    Used to compute: actual = uniform_min + uniform_ratio * (uniform_max - uniform_min)
    This ensures reproducibility: same task_index + same seed → same actual_sleep_time."""

    normal_z: float = 0.0
    """Standard normal (z-score) for normal distribution sampling.
    Used to compute: actual = normal_mean + normal_z * normal_std
    This ensures reproducibility: same task_index + same seed → same actual_sleep_time."""

    actual_sleep_time: float = 0.0
    """Actual sleep time sent to model (after phase-based scaling)."""

    exp_runtime_ms: float = 0.0
    """Expected runtime sent to scheduler in milliseconds."""

    submit_time: Optional[float] = None
    """Timestamp when task was submitted."""

    complete_time: Optional[float] = None
    """Timestamp when task completed."""

    is_complete: bool = False
    """Whether the task has completed."""

    instance_id: Optional[str] = None
    """Instance ID that processed this task (populated from result callback)."""

    def _apply_peak_dependent_factor(self, config) -> float:
        """
        Apply peak-dependent scaling factor based on which peak the base_sleep_time belongs to.

        The original distribution is bimodal:
        - Peak 1 (short): ~5-17s (samples < peak_threshold)
        - Peak 2 (long): ~42-56s (samples >= peak_threshold)

        By applying different factors to each peak, we can transform the distribution:
        - peak1_factor=1.0, peak2_factor=0.2 → collapses Peak 2 into Peak 1 (unimodal)

        Args:
            config: OODRecoveryConfig instance

        Returns:
            Scaled sleep time after applying peak-dependent factor
        """
        if self.base_sleep_time < config.peak_threshold:
            # Peak 1 (short tasks): apply peak1_factor
            return self.base_sleep_time * config.peak1_factor
        else:
            # Peak 2 (long tasks): apply peak2_factor
            return self.base_sleep_time * config.peak2_factor

    def _compute_phase23_actual_sleep(self, config) -> float:
        """
        Compute actual_sleep_time for Phase 2/3 based on distribution mode.

        Args:
            config: OODRecoveryConfig instance

        Returns:
            Computed actual sleep time in seconds
        """
        if config.phase23_distribution == "normal":
            # Normal distribution: use pre-sampled normal_z
            # actual = mean + z * std, with minimum clamp at 0.1s
            value = config.normal_mean + self.normal_z * config.normal_std
            return max(0.1, value)  # Ensure positive sleep time
        elif config.phase23_distribution == "uniform":
            # Uniform distribution: use pre-sampled uniform_ratio
            return config.uniform_min + self.uniform_ratio * (config.uniform_max - config.uniform_min)
        elif config.phase23_distribution == "four_peak":
            # Four-peak distribution with guaranteed separation using median-based split:
            # - Peak 1 (short, < threshold): split at median into 1a (below) and 1b (above)
            # - Peak 2 (long, >= threshold): split at median into 2a (below) and 2b (above)
            # Each sub-peak has its own scale factor for maximum separation control
            short_median, long_median = get_peak_medians(config.peak_threshold)

            if self.base_sleep_time < config.peak_threshold:
                # Short task - use uniform_ratio to select below/above median
                if self.uniform_ratio < 0.5:
                    # Peak 1a: sample from below short median
                    small_peak, _ = load_separated_peaks(config.peak_threshold)
                    below_median = [t for t in small_peak if t < short_median]
                    if below_median:
                        idx = int(self.uniform_ratio * 2 * len(below_median)) % len(below_median)
                        base = sorted(below_median)[idx]
                    else:
                        base = self.base_sleep_time
                    scale = config.four_peak_scale_1a
                else:
                    # Peak 1b: sample from above short median
                    small_peak, _ = load_separated_peaks(config.peak_threshold)
                    above_median = [t for t in small_peak if t >= short_median]
                    if above_median:
                        idx = int((self.uniform_ratio - 0.5) * 2 * len(above_median)) % len(above_median)
                        base = sorted(above_median)[idx]
                    else:
                        base = self.base_sleep_time
                    scale = config.four_peak_scale_1b
            else:
                # Long task - use uniform_ratio to select below/above median
                if self.uniform_ratio < 0.5:
                    # Peak 2a: sample from below long median
                    _, large_peak = load_separated_peaks(config.peak_threshold)
                    below_median = [t for t in large_peak if t < long_median]
                    if below_median:
                        idx = int(self.uniform_ratio * 2 * len(below_median)) % len(below_median)
                        base = sorted(below_median)[idx]
                    else:
                        base = self.base_sleep_time
                    scale = config.four_peak_scale_2a
                else:
                    # Peak 2b: sample from above long median
                    _, large_peak = load_separated_peaks(config.peak_threshold)
                    above_median = [t for t in large_peak if t >= long_median]
                    if above_median:
                        idx = int((self.uniform_ratio - 0.5) * 2 * len(above_median)) % len(above_median)
                        base = sorted(above_median)[idx]
                    else:
                        base = self.base_sleep_time
                    scale = config.four_peak_scale_2b

            return base * scale
        elif config.phase23_distribution == "weighted_bimodal":
            # Weighted bimodal distribution:
            # - Use phase23_base_sleep_time (pre-sampled with inverted ratio: 20%/80%)
            # - Apply phase23_bimodal_scale factor (default 2x)
            return self.phase23_base_sleep_time * config.phase23_bimodal_scale
        else:
            # Peak-dependent transformation
            return self._apply_peak_dependent_factor(config)

    def calculate_times(self, config) -> None:
        """
        Calculate actual_sleep_time and exp_runtime based on phase and config.

        Phase 2/3 runtime distribution depends on config.phase23_distribution:
        - "normal": Normal distribution (concentrated, opposite of bimodal)
        - "uniform": Uniform[uniform_min, uniform_max]
        - "peak_dependent": Peak-dependent transformation of base distribution

        Phase 2 exp_runtime uses INVERSE CORRELATION sampling:
        - Large actual_sleep (normal_z > 0) → sample from small peak (underestimate)
        - Small actual_sleep (normal_z <= 0) → sample from large peak (overestimate)
        This creates maximally wrong predictions to stress-test the scheduler.

        IMPORTANT - Sleep Time Consistency:
        ====================================
        Phase 2 and Phase 3 use the SAME formula for actual_sleep_time:
            actual_sleep_time = _compute_phase23_actual_sleep(config)

        This ensures that for the same task index, the sleep_time is IDENTICAL
        whether it's assigned as Phase 2 (in both modes) or Phase 3 (in Recovery mode).
        This guarantees fair comparison between Recovery and Baseline experiments.

        The only difference between Phase 2 and Phase 3 is exp_runtime_ms:
        - Phase 2: exp_runtime uses inverse correlation (wrong prediction)
        - Phase 3: exp_runtime matches actual sleep (corrected prediction)

        Runtime Scale:
        ==============
        config.runtime_scale is applied as a global multiplier to both actual_sleep_time
        and exp_runtime_ms. This allows scaling all task durations uniformly for
        faster/slower experiments without changing the distribution characteristics.

        Args:
            config: OODRecoveryConfig instance
        """
        # Get runtime_scale (default to 1.0 if not set)
        runtime_scale = getattr(config, 'runtime_scale', 1.0)

        if self.phase == 1:
            # Phase 1: correct prediction
            # actual_sleep uses phase1_scale, exp_runtime matches actual
            self.actual_sleep_time = self.base_sleep_time * config.phase1_scale * runtime_scale
            self.exp_runtime_ms = self.actual_sleep_time * 1000.0
        elif self.phase == 2:
            # Phase 2: OOD - wrong prediction
            # actual_sleep uses phase23 distribution (uniform, normal, four_peak, or peak-dependent)
            self.actual_sleep_time = self._compute_phase23_actual_sleep(config) * runtime_scale

            # exp_runtime directly samples from Phase 1 distribution
            # This simulates a predictor trained on Phase 1 data that hasn't seen
            # the new Phase 2/3 distribution, creating prediction mismatch.
            # exp_runtime_base is independently sampled from the full bimodal distribution
            self.exp_runtime_ms = self.exp_runtime_base * config.phase1_scale * runtime_scale * 1000.0
        else:  # phase == 3
            # Phase 3: recovery - corrected prediction
            # actual_sleep uses phase23 distribution, exp_runtime matches actual
            self.actual_sleep_time = self._compute_phase23_actual_sleep(config) * runtime_scale
            self.exp_runtime_ms = self.actual_sleep_time * 1000.0

    @property
    def duration(self) -> Optional[float]:
        """Task duration in seconds (None if not completed)."""
        if self.submit_time is not None and self.complete_time is not None:
            return self.complete_time - self.submit_time
        return None


class TaskGenerator:
    """
    Dynamic task generator for OOD experiments.

    Generates tasks on-demand with reproducible random values based on seed.
    Ensures identical task data for the same task index across different runs
    with the same seed.
    """

    def __init__(self, seed: int = 42, config=None):
        """
        Initialize the task generator.

        Args:
            seed: Random seed for reproducibility
            config: Optional OODRecoveryConfig for weighted sampling ratios
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._generated_count = 0
        self.config = config

        # Pre-load base sleep times for sampling
        self._base_sleep_times = load_base_sleep_times()

        # Pre-load separated peaks for inverse correlation sampling
        self._small_peak, self._large_peak = load_separated_peaks()

    def _sample_weighted_bimodal(self, small_peak_ratio: float) -> float:
        """
        Sample from bimodal distribution with weighted peak selection.

        Args:
            small_peak_ratio: Probability of sampling from small peak (0.0-1.0)

        Returns:
            Sampled value from either small or large peak
        """
        if self.rng.random() < small_peak_ratio:
            # Sample from small peak
            idx = self.rng.integers(0, len(self._small_peak))
            return self._small_peak[idx]
        else:
            # Sample from large peak
            idx = self.rng.integers(0, len(self._large_peak))
            return self._large_peak[idx]

    def generate_task(self, task_index: int) -> OODTaskData:
        """
        Generate a single task with the given index.

        Note: Tasks should be generated in order (0, 1, 2, ...) for reproducibility.

        Args:
            task_index: The index of the task to generate

        Returns:
            OODTaskData instance with pre-sampled random values
        """
        # Get weighted sampling ratios from config
        phase1_small_peak_ratio = 0.8  # Default: 80% small peak for Phase 1
        phase23_small_peak_ratio = 0.2  # Default: 20% small peak for Phase 2/3
        if self.config is not None:
            phase1_small_peak_ratio = getattr(self.config, 'phase1_small_peak_ratio', 0.8)
            phase23_small_peak_ratio = getattr(self.config, 'phase23_small_peak_ratio', 0.2)

        # Sample base_sleep_time using weighted bimodal for Phase 1
        base_sleep_time = self._sample_weighted_bimodal(phase1_small_peak_ratio)

        # Sample phase23_base_sleep_time using weighted bimodal for Phase 2/3
        phase23_base_sleep_time = self._sample_weighted_bimodal(phase23_small_peak_ratio)

        # Sample exp_runtime_base independently using Phase 1 weighted bimodal
        # This is used for Phase 2 wrong predictions (predictor trained on Phase 1 data)
        exp_runtime_base = self._sample_weighted_bimodal(phase1_small_peak_ratio)

        # Sample phase23_random_scale uniformly from [1.0, 2.0]
        # Used only when phase23_distribution='peak_dependent'
        phase23_random_scale = self.rng.uniform(1.0, 2.0)

        # Sample uniform_ratio in [0, 1] for uniform distribution
        # Used only when phase23_distribution='uniform' or 'four_peak'
        uniform_ratio = self.rng.uniform(0.0, 1.0)

        # Sample normal_z from standard normal distribution
        # Used only when phase23_distribution='normal'
        normal_z = self.rng.standard_normal()

        # Sample from small peak (for inverse correlation: large actual → small prediction)
        idx = self.rng.integers(0, len(self._small_peak))
        exp_runtime_small_peak = self._small_peak[idx]

        # Sample from large peak (for inverse correlation: small actual → large prediction)
        idx = self.rng.integers(0, len(self._large_peak))
        exp_runtime_large_peak = self._large_peak[idx]

        self._generated_count += 1

        return OODTaskData(
            task_id=f"task-ood-{task_index:04d}",
            task_index=task_index,
            base_sleep_time=base_sleep_time,
            phase23_base_sleep_time=phase23_base_sleep_time,
            exp_runtime_base=exp_runtime_base,
            phase23_random_scale=phase23_random_scale,
            uniform_ratio=uniform_ratio,
            normal_z=normal_z,
            exp_runtime_small_peak=exp_runtime_small_peak,
            exp_runtime_large_peak=exp_runtime_large_peak,
        )

    def get_generated_count(self) -> int:
        """Get the number of tasks generated so far."""
        return self._generated_count


def pre_generate_tasks(
    config,
    seed: int = 42
) -> List[OODTaskData]:
    """
    Pre-generate all task data before experiment.

    Samples:
    - base_sleep_time: from Phase 1 weighted bimodal distribution (80% small / 20% large)
    - phase23_base_sleep_time: from Phase 2/3 weighted bimodal distribution (20% small / 80% large)
    - exp_runtime_base: independently from Phase 1 distribution (for Phase 2 wrong predictions)
    - phase23_random_scale: uniformly from [1.0, 2.0] (for Phase 2/3 actual runtime)

    Phase assignment happens at submission time based on current state.

    IMPORTANT - Data Consistency Guarantee:
    =========================================
    When running with the same seed, the following values are IDENTICAL for each task index:
    - base_sleep_time[i]
    - phase23_base_sleep_time[i]
    - exp_runtime_base[i]
    - phase23_random_scale[i]

    This guarantees that for the same task index i (where i >= phase1_count):
    - Recovery mode: actual_sleep_time uses phase23_base_sleep_time[i] * scale
    - Baseline mode: actual_sleep_time uses phase23_base_sleep_time[i] * scale

    Both Phase 2 and Phase 3 use the SAME formula for actual_sleep_time, so:
    - Task at index 30 in Recovery mode (whether Phase 2 or 3) has the same sleep_time
      as Task at index 30 in Baseline mode (Phase 2).

    This ensures fair comparison between Recovery and Baseline experiments.

    Args:
        config: OODRecoveryConfig instance
        seed: Random seed for reproducibility

    Returns:
        List of OODTaskData instances
    """
    generator = TaskGenerator(seed=seed, config=config)
    tasks = [generator.generate_task(i) for i in range(config.num_tasks)]
    return tasks
