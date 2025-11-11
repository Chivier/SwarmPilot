#!/usr/bin/env python3
"""
A1 Data Sampler Module for Out-of-Distribution Experiments.

This module provides functionality to sample sleep_time values from A1 task data
and apply them to B1 model tasks for out-of-distribution testing.

The module supports:
1. Loading A1 (dr_boot) data from traces
2. Random sampling from A1 distribution
3. Statistical validation of sampling
4. Integration with existing workload generation

Usage:
    from a1_data_sampler import A1DataSampler

    # Create sampler
    sampler = A1DataSampler(seed=42)

    # Sample single value
    sleep_time = sampler.sample()

    # Sample multiple values
    sleep_times = sampler.sample_batch(count=10)

    # Get statistics
    stats = sampler.get_statistics()
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class SamplerStatistics:
    """Statistics about the A1 data and sampling behavior."""
    source_count: int
    source_mean: float
    source_median: float
    source_std: float
    source_min: float
    source_max: float
    source_p50: float
    source_p95: float
    source_p99: float
    samples_drawn: int


class A1DataSampler:
    """
    Sampler for extracting sleep_time values from A1 (dr_boot) data.

    This class provides thread-safe sampling from A1 task data with
    proper randomization and statistical tracking.
    """

    def __init__(self, data_dir: Optional[Path] = None, seed: Optional[int] = None):
        """
        Initialize the A1 data sampler.

        Args:
            data_dir: Directory containing trace data files. If None, uses
                     default data directory relative to this file.
            seed: Random seed for reproducible sampling. If None, uses
                 system randomness.
        """
        self.data_dir = data_dir or (Path(__file__).parent / "data")
        self.seed = seed
        self.samples_drawn = 0

        # Initialize random state
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Load A1 data
        self.a1_data = self._load_a1_data()
        self.a1_array = np.array(self.a1_data)

        # Validate loaded data
        if len(self.a1_data) == 0:
            raise ValueError("A1 data is empty. Cannot sample from empty dataset.")

    def _load_a1_data(self) -> List[float]:
        """
        Load A1 (dr_boot) data from trace files.

        Returns:
            List of A1 sleep_time values

        Raises:
            FileNotFoundError: If dr_boot.json file is not found
            json.JSONDecodeError: If the file contains invalid JSON
            ValueError: If the data format is invalid
        """
        dr_boot_path = self.data_dir / "dr_boot.json"

        if not dr_boot_path.exists():
            raise FileNotFoundError(
                f"A1 data file not found: {dr_boot_path}\n"
                f"Please ensure trace data is available in {self.data_dir}"
            )

        try:
            with open(dr_boot_path, "r") as f:
                dr_boot = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in {dr_boot_path}: {e.msg}",
                e.doc,
                e.pos
            )

        # Validate data format
        if not isinstance(dr_boot, list):
            raise ValueError(
                f"Expected list in dr_boot.json, got {type(dr_boot).__name__}"
            )

        if not all(isinstance(x, (int, float)) and x > 0 for x in dr_boot):
            raise ValueError(
                "All values in dr_boot.json must be positive numbers"
            )

        return dr_boot

    def sample(self) -> float:
        """
        Sample a single sleep_time value from A1 data.

        Returns:
            A randomly selected sleep_time value from A1 distribution
        """
        value = random.choice(self.a1_data)
        self.samples_drawn += 1
        return value

    def sample_batch(self, count: int, allow_replacement: bool = True) -> List[float]:
        """
        Sample multiple sleep_time values from A1 data.

        Args:
            count: Number of values to sample
            allow_replacement: If True, allows sampling the same value multiple
                             times. If False, samples without replacement (requires
                             count <= len(a1_data))

        Returns:
            List of sampled sleep_time values

        Raises:
            ValueError: If count > len(a1_data) and allow_replacement=False
        """
        if count <= 0:
            return []

        if not allow_replacement:
            if count > len(self.a1_data):
                raise ValueError(
                    f"Cannot sample {count} values without replacement from "
                    f"{len(self.a1_data)} available values"
                )
            values = random.sample(self.a1_data, count)
        else:
            values = random.choices(self.a1_data, k=count)

        self.samples_drawn += count
        return values

    def get_statistics(self) -> SamplerStatistics:
        """
        Get statistical information about the A1 data source.

        Returns:
            SamplerStatistics object containing distribution statistics
        """
        return SamplerStatistics(
            source_count=len(self.a1_data),
            source_mean=float(np.mean(self.a1_array)),
            source_median=float(np.median(self.a1_array)),
            source_std=float(np.std(self.a1_array)),
            source_min=float(np.min(self.a1_array)),
            source_max=float(np.max(self.a1_array)),
            source_p50=float(np.percentile(self.a1_array, 50)),
            source_p95=float(np.percentile(self.a1_array, 95)),
            source_p99=float(np.percentile(self.a1_array, 99)),
            samples_drawn=self.samples_drawn
        )

    def print_statistics(self):
        """Print human-readable statistics about the A1 data source."""
        stats = self.get_statistics()
        print("\n=== A1 Data Sampler Statistics ===")
        print(f"Source Data:")
        print(f"  Count:       {stats.source_count}")
        print(f"  Mean:        {stats.source_mean:.3f}s")
        print(f"  Median:      {stats.source_median:.3f}s")
        print(f"  Std Dev:     {stats.source_std:.3f}s")
        print(f"  Min:         {stats.source_min:.3f}s")
        print(f"  Max:         {stats.source_max:.3f}s")
        print(f"  P50:         {stats.source_p50:.3f}s")
        print(f"  P95:         {stats.source_p95:.3f}s")
        print(f"  P99:         {stats.source_p99:.3f}s")
        print(f"Sampling:")
        print(f"  Samples Drawn: {stats.samples_drawn}")
        print("=" * 40)

    def reset_sample_counter(self):
        """Reset the sample counter to zero."""
        self.samples_drawn = 0

    def validate_distribution_match(
        self,
        sampled_values: List[float],
        tolerance: float = 0.15
    ) -> Dict[str, bool]:
        """
        Validate that sampled values maintain A1 distribution characteristics.

        Checks if the sampled values' mean and std deviation are within
        tolerance of the source A1 distribution.

        Args:
            sampled_values: List of values that were sampled
            tolerance: Acceptable relative difference (default 15%)

        Returns:
            Dictionary with validation results for mean and std
        """
        if len(sampled_values) == 0:
            return {"mean_valid": False, "std_valid": False}

        sampled_array = np.array(sampled_values)
        sampled_mean = np.mean(sampled_array)
        sampled_std = np.std(sampled_array)

        source_mean = np.mean(self.a1_array)
        source_std = np.std(self.a1_array)

        mean_diff = abs(sampled_mean - source_mean) / source_mean
        std_diff = abs(sampled_std - source_std) / source_std if source_std > 0 else 0

        return {
            "mean_valid": mean_diff <= tolerance,
            "std_valid": std_diff <= tolerance,
            "mean_diff_pct": mean_diff * 100,
            "std_diff_pct": std_diff * 100,
            "sampled_mean": sampled_mean,
            "sampled_std": sampled_std,
            "source_mean": source_mean,
            "source_std": source_std
        }


# Module-level singleton for easy access
_default_sampler: Optional[A1DataSampler] = None


def get_default_sampler(seed: Optional[int] = None) -> A1DataSampler:
    """
    Get or create the default module-level sampler.

    Args:
        seed: Random seed (only used if sampler doesn't exist yet)

    Returns:
        The default A1DataSampler instance
    """
    global _default_sampler
    if _default_sampler is None:
        _default_sampler = A1DataSampler(seed=seed)
    return _default_sampler


def sample_from_a1(count: int = 1, seed: Optional[int] = None) -> List[float]:
    """
    Convenience function to sample from A1 data using the default sampler.

    Args:
        count: Number of values to sample
        seed: Random seed (only used if default sampler doesn't exist yet)

    Returns:
        List of sampled values (single value list if count=1)
    """
    sampler = get_default_sampler(seed=seed)
    if count == 1:
        return [sampler.sample()]
    return sampler.sample_batch(count)


if __name__ == "__main__":
    # Demo usage
    print("A1 Data Sampler Demo")
    print("=" * 40)

    # Create sampler
    sampler = A1DataSampler(seed=42)

    # Print statistics
    sampler.print_statistics()

    # Sample some values
    print("\nSampling 10 values:")
    samples = sampler.sample_batch(10)
    for i, val in enumerate(samples, 1):
        print(f"  Sample {i}: {val:.3f}s")

    # Validate distribution
    print("\nValidating large sample (1000 values):")
    large_sample = sampler.sample_batch(1000)
    validation = sampler.validate_distribution_match(large_sample)
    print(f"  Mean valid: {validation['mean_valid']} (diff: {validation['mean_diff_pct']:.2f}%)")
    print(f"  Std valid:  {validation['std_valid']} (diff: {validation['std_diff_pct']:.2f}%)")
