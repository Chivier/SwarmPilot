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

    Phase 1 uses phase1_scale on the base sleep times.

    Args:
        config: OODRecoveryConfig with phase1_scale

    Returns:
        Mean runtime for Phase 1 tasks in seconds
    """
    times = load_base_sleep_times()
    return np.mean(times) * config.phase1_scale


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
    """Base sleep time sampled from distribution (before scaling)."""

    exp_runtime_base: float = 0.0
    """Independent base for exp_runtime, sampled from Phase 1 distribution.
    Used in Phase 2 to simulate predictor trained on old (Phase 1) data."""

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
        else:
            # Peak-dependent transformation
            return self._apply_peak_dependent_factor(config)

    def calculate_times(self, config) -> None:
        """
        Calculate actual_sleep_time and exp_runtime based on phase and config.

        Phase 2/3 runtime distribution depends on config.phase23_distribution:
        - "uniform": Uniform[uniform_min, uniform_max]
        - "peak_dependent": Peak-dependent transformation of base distribution

        Phase 2 exp_runtime uses independently sampled exp_runtime_base to simulate
        predictor trained on Phase 1 distribution making wrong predictions.

        IMPORTANT - Sleep Time Consistency:
        ====================================
        Phase 2 and Phase 3 use the SAME formula for actual_sleep_time:
            actual_sleep_time = _compute_phase23_actual_sleep(config)

        This ensures that for the same task index, the sleep_time is IDENTICAL
        whether it's assigned as Phase 2 (in both modes) or Phase 3 (in Recovery mode).
        This guarantees fair comparison between Recovery and Baseline experiments.

        The only difference between Phase 2 and Phase 3 is exp_runtime_ms:
        - Phase 2: exp_runtime uses wrong prediction (simulating outdated predictor)
        - Phase 3: exp_runtime matches actual sleep (simulating corrected predictor)

        Args:
            config: OODRecoveryConfig instance
        """
        if self.phase == 1:
            # Phase 1: correct prediction
            # actual_sleep uses phase1_scale, exp_runtime matches actual
            self.actual_sleep_time = self.base_sleep_time * config.phase1_scale
            self.exp_runtime_ms = self.actual_sleep_time * 1000.0
        elif self.phase == 2:
            # Phase 2: OOD - wrong prediction
            # actual_sleep uses phase23 distribution (uniform or peak-dependent)
            # exp_runtime uses independently sampled exp_runtime_base with phase1_scale
            # This simulates predictor trained on Phase 1 data predicting wrong values
            self.actual_sleep_time = self._compute_phase23_actual_sleep(config)
            self.exp_runtime_ms = self.exp_runtime_base * config.phase1_scale * 1000.0
        else:  # phase == 3
            # Phase 3: recovery - corrected prediction
            # actual_sleep uses phase23 distribution, exp_runtime matches actual
            self.actual_sleep_time = self._compute_phase23_actual_sleep(config)
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

    def __init__(self, seed: int = 42):
        """
        Initialize the task generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._generated_count = 0

        # Pre-load base sleep times for sampling
        self._base_sleep_times = load_base_sleep_times()

    def generate_task(self, task_index: int) -> OODTaskData:
        """
        Generate a single task with the given index.

        Note: Tasks should be generated in order (0, 1, 2, ...) for reproducibility.

        Args:
            task_index: The index of the task to generate

        Returns:
            OODTaskData instance with pre-sampled random values
        """
        # Sample base_sleep_time
        idx = self.rng.integers(0, len(self._base_sleep_times))
        base_sleep_time = self._base_sleep_times[idx]

        # Sample exp_runtime_base independently
        idx = self.rng.integers(0, len(self._base_sleep_times))
        exp_runtime_base = self._base_sleep_times[idx]

        # Sample phase23_random_scale uniformly from [1.0, 2.0]
        # Used only when phase23_distribution='peak_dependent'
        phase23_random_scale = self.rng.uniform(1.0, 2.0)

        # Sample uniform_ratio in [0, 1] for uniform distribution
        # Used only when phase23_distribution='uniform'
        uniform_ratio = self.rng.uniform(0.0, 1.0)

        # Sample normal_z from standard normal distribution
        # Used only when phase23_distribution='normal'
        normal_z = self.rng.standard_normal()

        self._generated_count += 1

        return OODTaskData(
            task_id=f"task-ood-{task_index:04d}",
            task_index=task_index,
            base_sleep_time=base_sleep_time,
            exp_runtime_base=exp_runtime_base,
            phase23_random_scale=phase23_random_scale,
            uniform_ratio=uniform_ratio,
            normal_z=normal_z,
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
    - base_sleep_time: from training data distribution
    - exp_runtime_base: independently from same distribution (for Phase 2 wrong predictions)
    - phase23_random_scale: uniformly from [1.0, 2.0] (for Phase 2/3 actual runtime)

    Phase assignment happens at submission time based on current state.

    IMPORTANT - Data Consistency Guarantee:
    =========================================
    When running with the same seed, the following values are IDENTICAL for each task index:
    - base_sleep_time[i]
    - exp_runtime_base[i]
    - phase23_random_scale[i]

    This guarantees that for the same task index i (where i >= phase1_count):
    - Recovery mode: actual_sleep_time = base_sleep_time[i] * phase23_random_scale[i]
    - Baseline mode: actual_sleep_time = base_sleep_time[i] * phase23_random_scale[i]

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
    generator = TaskGenerator(seed=seed)
    tasks = [generator.generate_task(i) for i in range(config.num_tasks)]
    return tasks
