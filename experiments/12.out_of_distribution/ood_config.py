#!/usr/bin/env python3
"""
Out-of-Distribution Experiment Configuration Module.

This module provides configuration classes for the two main experiments:
1. Experiment 1: B1 samples sleep_time from A1 distribution
2. Experiment 2: B1 scales sleep_time and exp_runtime by factors [0.2, 0.5, 0.8]

Both experiments have baseline and comparison configurations.
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class ExperimentType(Enum):
    """Type of out-of-distribution experiment."""
    EXP1_A1_SAMPLING = "exp1_a1_sampling"
    EXP2_SCALING = "exp2_scaling"


class B1DataSource(Enum):
    """Source of B1 sleep_time data."""
    STANDARD = "standard"  # Use dr_query (standard B1 distribution)
    A1_SAMPLED = "a1_sampled"  # Use dr_boot (A1 distribution)


@dataclass
class OODExperimentConfig:
    """Configuration for out-of-distribution experiments."""

    # Experiment identification
    experiment_type: ExperimentType
    experiment_name: str
    is_baseline: bool  # True for baseline, False for comparison

    # Workload configuration
    num_workflows: int = 100
    seed: int = 42

    # B1 configuration
    b1_data_source: B1DataSource = B1DataSource.STANDARD
    b1_sleep_time_scale: float = 1.0  # Scaling factor for sleep_time
    b1_exp_runtime_scale: float = 1.0  # Scaling factor for exp_runtime
    sync_exp_runtime_with_sleep_time: bool = False  # If True, exp_runtime = sleep_time

    # Scheduling configuration
    strategy: str = "probabilistic"  # Must be probabilistic for OOD experiments

    # Logging and metrics
    enable_detailed_logging: bool = True
    log_parameter_sources: bool = True  # Log where each parameter comes from

    def __post_init__(self):
        """Validate configuration."""
        if self.strategy != "probabilistic":
            raise ValueError(
                f"OOD experiments must use probabilistic strategy, got: {self.strategy}"
            )

        if self.b1_sleep_time_scale <= 0:
            raise ValueError(f"sleep_time scale must be positive, got: {self.b1_sleep_time_scale}")

        if self.b1_exp_runtime_scale <= 0:
            raise ValueError(f"exp_runtime scale must be positive, got: {self.b1_exp_runtime_scale}")

    def get_description(self) -> str:
        """Get human-readable description of this configuration."""
        config_type = "Baseline" if self.is_baseline else "Comparison"

        if self.experiment_type == ExperimentType.EXP1_A1_SAMPLING:
            if self.is_baseline:
                return (
                    f"Exp1 Baseline: B1 sleep_time from A1 (dr_boot), "
                    f"exp_runtime from standard B1 (dr_query)"
                )
            else:
                return (
                    f"Exp1 Comparison: B1 sleep_time from A1 (dr_boot), "
                    f"exp_runtime = sleep_time"
                )
        else:  # EXP2_SCALING
            if self.is_baseline:
                return (
                    f"Exp2 Baseline: B1 sleep_time scaled by {self.b1_sleep_time_scale}x, "
                    f"exp_runtime unchanged"
                )
            else:
                return (
                    f"Exp2 Comparison: B1 sleep_time and exp_runtime both scaled by "
                    f"{self.b1_sleep_time_scale}x"
                )


# Predefined configurations for Experiment 1

def get_exp1_baseline_config(num_workflows: int = 100, seed: int = 42) -> OODExperimentConfig:
    """
    Get configuration for Experiment 1 Baseline.

    B1 samples sleep_time from A1 data, but keeps exp_runtime from standard B1 distribution.
    """
    return OODExperimentConfig(
        experiment_type=ExperimentType.EXP1_A1_SAMPLING,
        experiment_name="exp1_baseline",
        is_baseline=True,
        num_workflows=num_workflows,
        seed=seed,
        b1_data_source=B1DataSource.A1_SAMPLED,
        b1_sleep_time_scale=1.0,
        b1_exp_runtime_scale=1.0,
        sync_exp_runtime_with_sleep_time=False,
        strategy="probabilistic"
    )


def get_exp1_comparison_config(num_workflows: int = 100, seed: int = 42) -> OODExperimentConfig:
    """
    Get configuration for Experiment 1 Comparison.

    B1 samples sleep_time from A1 data, and sets exp_runtime = sleep_time.
    """
    return OODExperimentConfig(
        experiment_type=ExperimentType.EXP1_A1_SAMPLING,
        experiment_name="exp1_comparison",
        is_baseline=False,
        num_workflows=num_workflows,
        seed=seed,
        b1_data_source=B1DataSource.A1_SAMPLED,
        b1_sleep_time_scale=1.0,
        b1_exp_runtime_scale=1.0,
        sync_exp_runtime_with_sleep_time=True,
        strategy="probabilistic"
    )


# Predefined configurations for Experiment 2

def get_exp2_baseline_configs(
    num_workflows: int = 100,
    seed: int = 42
) -> List[OODExperimentConfig]:
    """
    Get configurations for Experiment 2 Baseline with all scaling factors.

    B1 sleep_time is scaled by [0.2, 0.5, 0.8], exp_runtime remains unchanged.
    """
    scale_factors = [0.2, 0.5, 0.8]
    configs = []

    for scale in scale_factors:
        config = OODExperimentConfig(
            experiment_type=ExperimentType.EXP2_SCALING,
            experiment_name=f"exp2_baseline_scale_{scale}",
            is_baseline=True,
            num_workflows=num_workflows,
            seed=seed,
            b1_data_source=B1DataSource.STANDARD,
            b1_sleep_time_scale=scale,
            b1_exp_runtime_scale=1.0,  # Unchanged
            sync_exp_runtime_with_sleep_time=False,
            strategy="probabilistic"
        )
        configs.append(config)

    return configs


def get_exp2_comparison_configs(
    num_workflows: int = 100,
    seed: int = 42
) -> List[OODExperimentConfig]:
    """
    Get configurations for Experiment 2 Comparison with all scaling factors.

    Both B1 sleep_time and exp_runtime are scaled by [0.2, 0.5, 0.8].
    """
    scale_factors = [0.2, 0.5, 0.8]
    configs = []

    for scale in scale_factors:
        config = OODExperimentConfig(
            experiment_type=ExperimentType.EXP2_SCALING,
            experiment_name=f"exp2_comparison_scale_{scale}",
            is_baseline=False,
            num_workflows=num_workflows,
            seed=seed,
            b1_data_source=B1DataSource.STANDARD,
            b1_sleep_time_scale=scale,
            b1_exp_runtime_scale=scale,  # Same as sleep_time
            sync_exp_runtime_with_sleep_time=False,
            strategy="probabilistic"
        )
        configs.append(config)

    return configs


def get_all_experiment_configs(num_workflows: int = 100, seed: int = 42) -> List[OODExperimentConfig]:
    """
    Get all experiment configurations for automated testing.

    Returns:
        List of all 8 experiment configurations:
        - Exp1 Baseline
        - Exp1 Comparison
        - Exp2 Baseline (3 scaling factors)
        - Exp2 Comparison (3 scaling factors)
    """
    configs = []

    # Experiment 1 configurations
    configs.append(get_exp1_baseline_config(num_workflows, seed))
    configs.append(get_exp1_comparison_config(num_workflows, seed))

    # Experiment 2 configurations
    configs.extend(get_exp2_baseline_configs(num_workflows, seed))
    configs.extend(get_exp2_comparison_configs(num_workflows, seed))

    return configs


if __name__ == "__main__":
    print("=" * 60)
    print("Out-of-Distribution Experiment Configurations")
    print("=" * 60)

    all_configs = get_all_experiment_configs()

    print(f"\nTotal configurations: {len(all_configs)}\n")

    for i, config in enumerate(all_configs, 1):
        print(f"{i}. {config.experiment_name}")
        print(f"   {config.get_description()}")
        print(f"   B1 Data Source: {config.b1_data_source.value}")
        print(f"   Sleep Time Scale: {config.b1_sleep_time_scale}x")
        print(f"   Exp Runtime Scale: {config.b1_exp_runtime_scale}x")
        print(f"   Sync exp_runtime: {config.sync_exp_runtime_with_sleep_time}")
        print()
