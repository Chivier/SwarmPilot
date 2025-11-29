"""Generic distribution configurations for workflow parameters.

Supports four distribution types:
- static: Fixed value for all workflows
- uniform: Uniform distribution between min and max values
- two_peak: Two-peak (bimodal) distribution using two Gaussian peaks
- four_peak: Four-peak distribution using four Gaussian peaks

This module provides a reusable distribution framework that can be used for:
- Deep Research fanout count
- Text2Video frame count
- Text2Video max B loops
- Any other integer workflow parameter
"""

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class Distribution(ABC):
    """Abstract base class for integer distributions."""

    @abstractmethod
    def sample(self) -> int:
        """Sample a value from the distribution.

        Returns:
            Integer value (always >= min_value)
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert distribution to dictionary for serialization."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Distribution":
        """Create distribution from dictionary."""
        pass


@dataclass
class StaticDistribution(Distribution):
    """Fixed value for all samples.

    Example config:
        {
            "type": "static",
            "value": 4
        }

    Note: Returns float. Callers should convert to int if needed.
    """

    value: float = 3.0

    def __post_init__(self):
        if self.value < 0:
            raise ValueError(f"Static value must be >= 0, got {self.value}")

    def sample(self) -> float:
        return float(self.value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "static",
            "value": self.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StaticDistribution":
        return cls(value=float(data.get("value", 3)))


@dataclass
class UniformDistribution(Distribution):
    """Uniform distribution between min and max values.

    Example config:
        {
            "type": "uniform",
            "min": 2,
            "max": 8
        }

    Note: Returns float. Callers should convert to int if needed.
    """

    min_value: float = 1.0
    max_value: float = 5.0

    def __post_init__(self):
        if self.min_value < 0:
            raise ValueError(f"Min value must be >= 0, got {self.min_value}")
        if self.max_value < self.min_value:
            raise ValueError(f"Max value ({self.max_value}) must be >= min value ({self.min_value})")

    def sample(self) -> float:
        return random.uniform(self.min_value, self.max_value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "uniform",
            "min": self.min_value,
            "max": self.max_value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniformDistribution":
        return cls(
            min_value=float(data.get("min", 1)),
            max_value=float(data.get("max", 5))
        )


@dataclass
class NormalDistribution(Distribution):
    """Normal (Gaussian) distribution for continuous values like sleep times.

    Example config:
        {
            "type": "normal",
            "mean": 0.8,
            "std": 0.2,
            "min": 0.1,   # optional, clips negative values
            "max": 10.0   # optional, clips extreme values
        }
    """

    mean: float = 1.0
    std: float = 0.5
    min_value: float = 0.0
    max_value: float = float('inf')

    def __post_init__(self):
        if self.std < 0:
            raise ValueError(f"Standard deviation must be >= 0, got {self.std}")
        if self.min_value > self.max_value:
            raise ValueError(f"Min value ({self.min_value}) must be <= max value ({self.max_value})")

    def sample(self) -> float:
        """Sample from the normal distribution, clamped to [min, max]."""
        value = random.gauss(self.mean, self.std)
        return max(self.min_value, min(self.max_value, value))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "normal",
            "mean": self.mean,
            "std": self.std,
            "min": self.min_value,
            "max": self.max_value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalDistribution":
        return cls(
            mean=data.get("mean", 1.0),
            std=data.get("std", 0.5),
            min_value=data.get("min", 0.0),
            max_value=data.get("max", float('inf'))
        )


@dataclass
class GaussianPeak:
    """A single Gaussian peak in a multi-peak distribution.

    Attributes:
        mean: Center position of the peak
        std: Standard deviation (spread) of the peak
        weight: Relative weight of this peak (for sampling probability)
    """

    mean: float
    std: float = 1.0
    weight: float = 1.0

    def sample(self) -> float:
        """Sample from this Gaussian peak."""
        return random.gauss(self.mean, self.std)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "weight": self.weight
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GaussianPeak":
        return cls(
            mean=data["mean"],
            std=data.get("std", 1.0),
            weight=data.get("weight", 1.0)
        )


@dataclass
class TwoPeakDistribution(Distribution):
    """Two-peak (bimodal) distribution using two Gaussian peaks.

    Example config:
        {
            "type": "two_peak",
            "peaks": [
                {"mean": 3, "std": 0.5, "weight": 1.0},
                {"mean": 8, "std": 1.0, "weight": 1.0}
            ],
            "min": 1,
            "max": 12
        }

    Note: Returns float. Callers should convert to int if needed.
    """

    peaks: List[GaussianPeak] = field(default_factory=lambda: [
        GaussianPeak(mean=3, std=0.5),
        GaussianPeak(mean=8, std=1.0)
    ])
    min_value: float = 1.0
    max_value: float = 20.0

    def __post_init__(self):
        if len(self.peaks) != 2:
            raise ValueError(f"TwoPeakDistribution requires exactly 2 peaks, got {len(self.peaks)}")
        if self.min_value < 0:
            raise ValueError(f"Min value must be >= 0, got {self.min_value}")

    def sample(self) -> float:
        """Sample from the two-peak distribution."""
        # Select a peak based on weights
        total_weight = sum(p.weight for p in self.peaks)
        r = random.uniform(0, total_weight)

        cumulative = 0
        selected_peak = self.peaks[0]
        for peak in self.peaks:
            cumulative += peak.weight
            if r <= cumulative:
                selected_peak = peak
                break

        # Sample from the selected Gaussian and clamp to valid range
        value = selected_peak.sample()
        return max(self.min_value, min(self.max_value, value))

    def sample_with_peak(self) -> Tuple[float, int]:
        """Sample from the two-peak distribution and return peak index.

        Returns:
            Tuple of (value, peak_index) where peak_index is 0-based (0 or 1)
        """
        # Select a peak based on weights
        total_weight = sum(p.weight for p in self.peaks)
        r = random.uniform(0, total_weight)

        cumulative = 0
        selected_peak_idx = 0
        for idx, peak in enumerate(self.peaks):
            cumulative += peak.weight
            if r <= cumulative:
                selected_peak_idx = idx
                break

        # Sample from the selected Gaussian and clamp to valid range
        value = self.peaks[selected_peak_idx].sample()
        value = max(self.min_value, min(self.max_value, value))
        return value, selected_peak_idx

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "two_peak",
            "peaks": [p.to_dict() for p in self.peaks],
            "min": self.min_value,
            "max": self.max_value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TwoPeakDistribution":
        peaks_data = data.get("peaks", [])
        if len(peaks_data) != 2:
            raise ValueError(f"TwoPeakDistribution requires exactly 2 peaks in config")

        peaks = [GaussianPeak.from_dict(p) for p in peaks_data]
        return cls(
            peaks=peaks,
            min_value=float(data.get("min", 1)),
            max_value=float(data.get("max", 20))
        )


@dataclass
class FourPeakDistribution(Distribution):
    """Four-peak distribution using four Gaussian peaks.

    Example config:
        {
            "type": "four_peak",
            "peaks": [
                {"mean": 2, "std": 0.3, "weight": 1.0},
                {"mean": 5, "std": 0.5, "weight": 1.0},
                {"mean": 8, "std": 0.5, "weight": 1.0},
                {"mean": 12, "std": 1.0, "weight": 1.0}
            ],
            "min": 1,
            "max": 16
        }

    Note: Returns float. Callers should convert to int if needed.
    """

    peaks: List[GaussianPeak] = field(default_factory=lambda: [
        GaussianPeak(mean=2, std=0.3),
        GaussianPeak(mean=5, std=0.5),
        GaussianPeak(mean=8, std=0.5),
        GaussianPeak(mean=12, std=1.0)
    ])
    min_value: float = 1.0
    max_value: float = 20.0

    def __post_init__(self):
        if len(self.peaks) != 4:
            raise ValueError(f"FourPeakDistribution requires exactly 4 peaks, got {len(self.peaks)}")
        if self.min_value < 0:
            raise ValueError(f"Min value must be >= 0, got {self.min_value}")

    def sample(self) -> float:
        """Sample from the four-peak distribution."""
        # Select a peak based on weights
        total_weight = sum(p.weight for p in self.peaks)
        r = random.uniform(0, total_weight)

        cumulative = 0
        selected_peak = self.peaks[0]
        for peak in self.peaks:
            cumulative += peak.weight
            if r <= cumulative:
                selected_peak = peak
                break

        # Sample from the selected Gaussian and clamp to valid range
        value = selected_peak.sample()
        return max(self.min_value, min(self.max_value, value))

    def sample_with_peak(self) -> Tuple[float, int]:
        """Sample from the four-peak distribution and return peak index.

        Returns:
            Tuple of (value, peak_index) where peak_index is 0-based (0, 1, 2, or 3)
        """
        # Select a peak based on weights
        total_weight = sum(p.weight for p in self.peaks)
        r = random.uniform(0, total_weight)

        cumulative = 0
        selected_peak_idx = 0
        for idx, peak in enumerate(self.peaks):
            cumulative += peak.weight
            if r <= cumulative:
                selected_peak_idx = idx
                break

        # Sample from the selected Gaussian and clamp to valid range
        value = self.peaks[selected_peak_idx].sample()
        value = max(self.min_value, min(self.max_value, value))
        return value, selected_peak_idx

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "four_peak",
            "peaks": [p.to_dict() for p in self.peaks],
            "min": self.min_value,
            "max": self.max_value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FourPeakDistribution":
        peaks_data = data.get("peaks", [])
        if len(peaks_data) != 4:
            raise ValueError(f"FourPeakDistribution requires exactly 4 peaks in config")

        peaks = [GaussianPeak.from_dict(p) for p in peaks_data]
        return cls(
            peaks=peaks,
            min_value=float(data.get("min", 1)),
            max_value=float(data.get("max", 20))
        )


@dataclass
class WeightedChoiceDistribution(Distribution):
    """Weighted random choice between discrete values.

    Unlike other distributions that return integers, this returns
    the "value" field from the selected choice (can be any type).

    Example config:
        {
            "type": "weighted_choice",
            "choices": [
                {"value": "512x512", "weight": 0.7},
                {"value": "1024x1024", "weight": 0.3}
            ]
        }

    Note: The sample() method returns the value directly, which may not be an int.
    Use with DistributionSampler for consistent interface.
    """

    choices: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"value": "512x512", "weight": 0.7},
        {"value": "1024x1024", "weight": 0.3}
    ])

    def __post_init__(self):
        if not self.choices:
            raise ValueError("WeightedChoiceDistribution requires at least one choice")
        for choice in self.choices:
            if "value" not in choice:
                raise ValueError("Each choice must have a 'value' field")
            if "weight" not in choice:
                choice["weight"] = 1.0  # Default weight

    def sample(self) -> Any:
        """Sample a value based on weights.

        Returns:
            The selected choice's value (can be any type)
        """
        total_weight = sum(c["weight"] for c in self.choices)
        r = random.uniform(0, total_weight)

        cumulative = 0
        for choice in self.choices:
            cumulative += choice["weight"]
            if r <= cumulative:
                return choice["value"]

        return self.choices[-1]["value"]  # Fallback

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "weighted_choice",
            "choices": self.choices
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeightedChoiceDistribution":
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("WeightedChoiceDistribution requires 'choices' in config")
        return cls(choices=choices)


# Registry of distribution types
DISTRIBUTION_TYPES = {
    "static": StaticDistribution,
    "uniform": UniformDistribution,
    "normal": NormalDistribution,
    "two_peak": TwoPeakDistribution,
    "four_peak": FourPeakDistribution,
    "weighted_choice": WeightedChoiceDistribution,
}


def create_distribution(config: Dict[str, Any]) -> Distribution:
    """Create a distribution from a configuration dictionary.

    Args:
        config: Dictionary with "type" key and type-specific parameters

    Returns:
        Distribution instance

    Raises:
        ValueError: If distribution type is unknown
    """
    dist_type = config.get("type", "static")
    if dist_type not in DISTRIBUTION_TYPES:
        raise ValueError(
            f"Unknown distribution type: {dist_type}. "
            f"Supported types: {list(DISTRIBUTION_TYPES.keys())}"
        )

    return DISTRIBUTION_TYPES[dist_type].from_dict(config)


def load_distribution_config(config_path: Union[str, Path]) -> Distribution:
    """Load distribution configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        Distribution instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is not valid JSON
        ValueError: If config is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Distribution config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return create_distribution(config)


class DistributionSampler:
    """Generic sampler that generates values from a distribution.

    This class manages the distribution and provides methods
    to sample values for individual workflows.
    """

    def __init__(
        self,
        distribution: Optional[Distribution] = None,
        config_path: Optional[Union[str, Path]] = None,
        default_value: int = 3,
        seed: Optional[int] = None
    ):
        """Initialize the sampler.

        Args:
            distribution: Pre-configured distribution (takes precedence)
            config_path: Path to JSON config file
            default_value: Default static value if no config provided
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        if distribution is not None:
            self.distribution = distribution
        elif config_path is not None:
            self.distribution = load_distribution_config(config_path)
        else:
            self.distribution = StaticDistribution(value=default_value)

    def sample(self) -> float:
        """Sample a value from the distribution.

        Returns:
            Float value. Callers should convert to int if needed.
        """
        return self.distribution.sample()

    def sample_batch(self, count: int) -> List[float]:
        """Sample multiple values.

        Args:
            count: Number of values to sample

        Returns:
            List of float values. Callers should convert to int if needed.
        """
        return [self.sample() for _ in range(count)]

    def get_config(self) -> Dict[str, Any]:
        """Get the current distribution configuration."""
        return self.distribution.to_dict()


# Backward compatibility aliases
FanoutDistribution = Distribution
FanoutSampler = DistributionSampler
load_fanout_config = load_distribution_config
