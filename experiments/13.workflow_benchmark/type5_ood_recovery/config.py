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
    """Target queries per second for task submission."""

    duration: int = 600
    """Maximum experiment duration in seconds."""

    # ==========================================================================
    # Phase Configuration
    # ==========================================================================
    phase1_count: int = 100
    """Fixed number of tasks in Phase 1 (warmup with correct predictions)."""

    phase1_scale: float = 0.1
    """Scaling factor for Phase 1 sleep time and exp_runtime (0.1x base)."""

    # ==========================================================================
    # Phase 2/3 Runtime Distribution Mode
    # ==========================================================================
    phase23_distribution: str = "normal"
    """Distribution type for Phase 2/3 actual runtime.
    Options:
    - "normal": Concentrated normal distribution (opposite of Phase 1's bimodal)
    - "uniform": Uniform distribution between uniform_min and uniform_max
    - "peak_dependent": Peak-dependent transformation of base distribution
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

    quantiles: list = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.99])
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
        else:  # peak_dependent
            phase23_dist_str = (
                f"  Phase 2/3 distribution: Peak-dependent transformation\n"
                f"    - Peak 1 (< {self.peak_threshold}s): factor = {self.peak1_factor}x\n"
                f"    - Peak 2 (>= {self.peak_threshold}s): factor = {self.peak2_factor}x\n"
            )

        return (
            f"OODRecoveryConfig:\n"
            f"  Mode: {mode}\n"
            f"  Tasks: {self.num_tasks} (Phase1: {self.phase1_count}, Phase2+3: {self.phase23_count})\n"
            f"  QPS: {self.qps}\n"
            f"  Duration: {self.duration}s\n"
            f"  Phase 1 scale: {self.phase1_scale}x (actual & exp_runtime)\n"
            f"{phase23_dist_str}"
            f"  Phase 2 exp_runtime: sampled from Phase 1 distribution × {self.phase1_scale}x (wrong)\n"
            f"  Phase 3 exp_runtime: matches actual sleep (corrected)\n"
            f"  Phase 2→3 transition: after {self.phase2_transition_count} Phase 2 tasks {trigger_type}\n"
            f"  Scheduler: {self.scheduler_url}\n"
            f"  Model: {self.model_id}"
        )
