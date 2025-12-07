"""
OOD Recovery Experiment Configuration.

This module provides configuration for the OOD (Out of Distribution) recovery
experiment, which tests the system's ability to recover when task runtime
distribution changes and the predictor re-trains.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class OODRecoveryConfig:
    """Configuration for OOD Recovery experiment."""

    # ==========================================================================
    # Task Parameters
    # ==========================================================================
    num_tasks: int = 100
    """Total number of tasks to submit."""

    qps: float = 1.0
    """Default queries per second for task submission."""

    phase1_qps: Optional[float] = None
    """Phase 1 QPS. If None, uses qps."""

    phase23_qps: Optional[float] = None
    """Phase 2/3 QPS. If None, uses qps."""

    runtime_scale: float = 1.0
    """Global scaling factor for task runtime."""

    duration: int = 600
    """Maximum experiment duration in seconds."""

    # ==========================================================================
    # Phase Configuration
    # ==========================================================================
    phase1_count: int = 300
    """Fixed number of tasks in Phase 1 (warmup with correct predictions)."""

    phase1_scale: float = 0.1
    """Scaling factor for Phase 1 sleep time and exp_runtime (0.1x base)."""

    phase1_small_peak_ratio: float = 0.8
    """Ratio of small peak samples in Phase 1 bimodal distribution.
    Default 0.8 means 80% from small peak, 20% from large peak."""

    # ==========================================================================
    # Phase 2/3 Runtime Distribution Mode
    # ==========================================================================
    phase23_distribution: str = "weighted_bimodal"
    """Distribution type for Phase 2/3 actual runtime.
    Options:
    - "normal": Concentrated normal distribution (opposite of Phase 1's bimodal)
    - "uniform": Uniform distribution between uniform_min and uniform_max
    - "peak_dependent": Peak-dependent transformation of base distribution
    - "four_peak": Four-peak distribution derived from bimodal with random factor selection
    - "weighted_bimodal": Weighted bimodal sampling with inverse ratio (20%/80%) and 2x scale
    """

    # ==========================================================================
    # Normal Distribution for Phase 2/3 (when phase23_distribution="normal")
    # ==========================================================================
    # Phase 1 has a bimodal distribution (two peaks at ~1s and ~5s after 0.1x scaling).
    # To create a clear OOD scenario, Phase 2/3 uses a concentrated normal distribution
    # with a single peak, which is the opposite characteristic of Phase 1's bimodal.
    #
    # The predictor trained on bimodal data will be unable to predict the
    # concentrated unimodal distribution accurately.

    normal_mean: float = 10.0
    """Mean of the normal distribution in seconds."""

    normal_std: float = 1.0
    """Standard deviation of the normal distribution in seconds.
    Small std creates a concentrated distribution (opposite of bimodal's spread)."""

    # ==========================================================================
    # Uniform Distribution for Phase 2/3 (when phase23_distribution="uniform")
    # ==========================================================================
    uniform_min: float = 5.0
    """Minimum value for uniform distribution in seconds."""

    uniform_max: float = 15.0
    """Maximum value for uniform distribution in seconds."""

    # ==========================================================================
    # Peak-Dependent Transformation for Phase 2/3 (when phase23_distribution="peak_dependent")
    # ==========================================================================
    # The original distribution is bimodal:
    #   - Peak 1 (short): ~5-17s (mean ~10s)
    #   - Peak 2 (long): ~42-56s (mean ~50s)
    #
    # To create an OOD scenario where the original distribution cannot describe
    # the new distribution, we apply different scaling factors to each peak:
    #   - Peak 1: factor = peak1_factor (e.g., 1.0 = keep as is)
    #   - Peak 2: factor = peak2_factor (e.g., 0.2 = collapse to ~10s)
    #
    # This merges the two peaks into a single peak, making the original
    # bimodal predictor completely unable to predict the new unimodal distribution.

    peak_threshold: float = 20.0
    """Threshold to separate Peak 1 (< threshold) from Peak 2 (>= threshold).
    Based on the valley in the bimodal distribution around 17-40s."""

    peak1_factor: float = 1.0
    """Scaling factor for Peak 1 samples in Phase 2/3 (short tasks).
    Factor 1.0 keeps Peak 1 values unchanged."""

    peak2_factor: float = 0.2
    """Scaling factor for Peak 2 samples in Phase 2/3 (long tasks).
    Factor 0.2 collapses Peak 2 (~50s) down to ~10s, merging with Peak 1."""

    # ==========================================================================
    # Four-Peak Distribution for Phase 2/3 (when phase23_distribution="four_peak")
    # ==========================================================================
    # Creates a well-separated four-peak distribution using median-based splitting:
    # - Peak 1 (short, < threshold) is split at median into 1a (below) and 1b (above)
    # - Peak 2 (long, >= threshold) is split at median into 2a (below) and 2b (above)
    # - Each sub-peak has its own scale factor for maximum separation control
    #
    # With default values:
    #   Peak 1a: short below median × 0.15 → ~0.03-1.6s
    #   Peak 1b: short above median × 0.35 → ~3.8-6.8s  (gap ~2.2s from 1a)
    #   Peak 2a: long below median × 0.25  → ~7.1-12.8s (gap ~0.3s from 1b)
    #   Peak 2b: long above median × 0.25  → ~12.8-14.1s

    four_peak_scale_1a: float = 0.15
    """Scale factor for Peak 1a (short tasks below median).
    Smaller value creates lower range, increasing gap to 1b."""

    four_peak_scale_1b: float = 0.35
    """Scale factor for Peak 1b (short tasks above median).
    Larger value creates higher range, increasing gap from 1a."""

    four_peak_scale_2a: float = 0.45
    """Scale factor for Peak 2a (long tasks below median).
    Increased to create >5s gap from Peak 1b."""

    four_peak_scale_2b: float = 0.65
    """Scale factor for Peak 2b (long tasks above median).
    Set to create ~10s gap from Peak 2a."""

    # ==========================================================================
    # Weighted Bimodal Distribution for Phase 2/3 (when phase23_distribution="weighted_bimodal")
    # ==========================================================================
    # Creates an OOD scenario by changing the bimodal sampling ratio and applying a scale factor:
    # - Phase 1: 80% small peak + 20% large peak (controlled by phase1_small_peak_ratio)
    # - Phase 2/3: 20% small peak + 80% large peak (inverse ratio) with 2x scale
    #
    # This creates a clear distribution shift:
    # - Phase 1 is dominated by short tasks
    # - Phase 2/3 is dominated by long tasks (scaled 2x)
    # The predictor trained on Phase 1 will be unable to predict Phase 2/3 accurately.

    phase23_small_peak_ratio: float = 0.2
    """Ratio of small peak samples in Phase 2/3 weighted bimodal distribution.
    Default 0.2 means 20% from small peak, 80% from large peak (inverse of Phase 1)."""

    phase23_bimodal_scale: float = 2.0
    """Scale factor applied to all Phase 2/3 weighted bimodal samples.
    Default 2.0 means all values are doubled."""

    # ==========================================================================
    # Phase Transition
    # ==========================================================================
    phase2_transition_count: int = 10
    """Number of Phase 2 tasks before triggering Phase 3 transition.
    Interpretation depends on transition_on_submit:
    - If transition_on_submit=True: count of Phase 2 tasks SUBMITTED
    - If transition_on_submit=False: count of Phase 2 tasks COMPLETED
    """

    transition_on_submit: bool = False
    """If True, trigger Phase 2→3 transition based on submission count.
    If False (default), trigger based on completion count (realistic behavior).

    Completion-based triggering (default) simulates real-world OOD detection:
    - System detects OOD after tasks complete and show misprediction
    - Predictor is retrained/updated based on observed errors
    - Future submissions use corrected predictions

    Note: With completion-based triggering, there will be queue backlog of
    Phase 2 tasks submitted before transition. This is expected behavior
    as it reflects the real latency between OOD detection and correction.
    """

    # ==========================================================================
    # Baseline Mode
    # ==========================================================================
    no_recovery: bool = False
    """If True, no Phase 2->3 transition (baseline mode)."""

    # ==========================================================================
    # Scheduler Configuration
    # ==========================================================================
    scheduler_url: str = "http://127.0.0.1:8100"
    """Scheduler endpoint URL."""

    model_id: str = "sleep_model_a"
    """Model ID for task submission."""

    # ==========================================================================
    # Strategy Configuration
    # ==========================================================================
    strategy: str = "probabilistic"
    """Scheduling strategy (fixed to probabilistic)."""

    target_quantile: float = 0.9
    """Target quantile for probabilistic strategy."""

    quantiles: list = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    """Quantiles for probabilistic strategy."""

    # ==========================================================================
    # Output Configuration
    # ==========================================================================
    output_dir: str = "output_ood"
    """Output directory for metrics and results."""

    metrics_file: str = "metrics.json"
    """Metrics output filename."""

    # ==========================================================================
    # Statistics Configuration
    # ==========================================================================
    num_warmup: int = 0
    """Number of warmup tasks (Phase 1 serves as warmup)."""

    portion_stats: float = 1.0
    """Portion of tasks to include in statistics."""

    @property
    def phase23_count(self) -> int:
        """Number of tasks in Phase 2 + Phase 3 combined (num_tasks - phase1_count)."""
        return max(0, self.num_tasks - self.phase1_count)

    def get_output_path(self) -> Path:
        """Get the output directory path."""
        return Path(self.output_dir)

    def get_metrics_path(self) -> Path:
        """Get the full path to the metrics file."""
        return self.get_output_path() / self.metrics_file

    def __str__(self) -> str:
        """Return a human-readable configuration summary."""
        mode = "Baseline (no recovery)" if self.no_recovery else "Recovery"
        trigger_type = "submitted" if self.transition_on_submit else "completed"

        # Phase 2/3 distribution description
        if self.phase23_distribution == "normal":
            phase23_dist_str = (
                f"  Phase 2/3 distribution: Normal(μ={self.normal_mean}s, σ={self.normal_std}s)\n"
                f"    - Concentrated unimodal (opposite of Phase 1's bimodal)\n"
            )
        elif self.phase23_distribution == "uniform":
            phase23_dist_str = (
                f"  Phase 2/3 distribution: Uniform[{self.uniform_min}s, {self.uniform_max}s]\n"
                f"    - Mean: {(self.uniform_min + self.uniform_max) / 2:.2f}s\n"
            )
        elif self.phase23_distribution == "peak_dependent":
            phase23_dist_str = (
                f"  Phase 2/3 distribution: Peak-dependent transformation\n"
                f"    - Peak 1 (< {self.peak_threshold}s): factor = {self.peak1_factor}x\n"
                f"    - Peak 2 (>= {self.peak_threshold}s): factor = {self.peak2_factor}x\n"
            )
        elif self.phase23_distribution == "four_peak":
            phase23_dist_str = (
                f"  Phase 2/3 distribution: Four-peak (median-split)\n"
                f"    - Peak 1a (short, below median): {self.four_peak_scale_1a}x\n"
                f"    - Peak 1b (short, above median): {self.four_peak_scale_1b}x\n"
                f"    - Peak 2a (long, below median):  {self.four_peak_scale_2a}x\n"
                f"    - Peak 2b (long, above median):  {self.four_peak_scale_2b}x\n"
            )
        else:  # weighted_bimodal
            phase23_dist_str = (
                f"  Phase 2/3 distribution: Weighted bimodal\n"
                f"    - Small peak ratio: {self.phase23_small_peak_ratio:.0%}\n"
                f"    - Large peak ratio: {1 - self.phase23_small_peak_ratio:.0%}\n"
                f"    - Scale factor: {self.phase23_bimodal_scale}x\n"
            )

        return (
            f"OODRecoveryConfig:\n"
            f"  Mode: {mode}\n"
            f"  Tasks: {self.num_tasks} (Phase1: {self.phase1_count}, Phase2+3: {self.phase23_count})\n"
            f"  QPS: {self.qps}\n"
            f"  Duration: {self.duration}s\n"
            f"  Phase 1: scale={self.phase1_scale}x, bimodal ratio={self.phase1_small_peak_ratio:.0%}/{1-self.phase1_small_peak_ratio:.0%} (small/large)\n"
            f"{phase23_dist_str}"
            f"  Phase 2 exp_runtime: Inverse correlation (large actual→small peak, small actual→large peak)\n"
            f"  Phase 3 exp_runtime: matches actual sleep (corrected)\n"
            f"  Phase 2→3 transition: after {self.phase2_transition_count} Phase 2 tasks {trigger_type}\n"
            f"  Scheduler: {self.scheduler_url}\n"
            f"  Model: {self.model_id}"
        )
