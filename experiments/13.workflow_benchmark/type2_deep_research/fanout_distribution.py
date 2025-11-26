"""Fanout distribution configurations for Deep Research workflow.

Supports four distribution types:
- static: Fixed fanout value for all workflows
- uniform: Uniform distribution between min and max values
- two_peak: Two-peak (bimodal) distribution using two Gaussian peaks
- four_peak: Four-peak distribution using four Gaussian peaks
"""

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class FanoutDistribution(ABC):
    """Abstract base class for fanout distributions."""

    @abstractmethod
    def sample(self) -> int:
        """Sample a fanout value from the distribution.

        Returns:
            Integer fanout count (always >= 1)
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert distribution to dictionary for serialization."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FanoutDistribution":
        """Create distribution from dictionary."""
        pass


@dataclass
class StaticDistribution(FanoutDistribution):
    """Fixed fanout value for all workflows.

    Example config:
        {
            "type": "static",
            "value": 4
        }
    """

    value: int = 3

    def __post_init__(self):
        if self.value < 1:
            raise ValueError(f"Static value must be >= 1, got {self.value}")

    def sample(self) -> int:
        return self.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "static",
            "value": self.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StaticDistribution":
        return cls(value=data.get("value", 3))


@dataclass
class UniformDistribution(FanoutDistribution):
    """Uniform distribution between min and max values (inclusive).

    Example config:
        {
            "type": "uniform",
            "min": 2,
            "max": 8
        }
    """

    min_value: int = 1
    max_value: int = 5

    def __post_init__(self):
        if self.min_value < 1:
            raise ValueError(f"Min value must be >= 1, got {self.min_value}")
        if self.max_value < self.min_value:
            raise ValueError(f"Max value ({self.max_value}) must be >= min value ({self.min_value})")

    def sample(self) -> int:
        return random.randint(self.min_value, self.max_value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "uniform",
            "min": self.min_value,
            "max": self.max_value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniformDistribution":
        return cls(
            min_value=data.get("min", 1),
            max_value=data.get("max", 5)
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
class TwoPeakDistribution(FanoutDistribution):
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
    """

    peaks: List[GaussianPeak] = field(default_factory=lambda: [
        GaussianPeak(mean=3, std=0.5),
        GaussianPeak(mean=8, std=1.0)
    ])
    min_value: int = 1
    max_value: int = 20

    def __post_init__(self):
        if len(self.peaks) != 2:
            raise ValueError(f"TwoPeakDistribution requires exactly 2 peaks, got {len(self.peaks)}")
        if self.min_value < 1:
            raise ValueError(f"Min value must be >= 1, got {self.min_value}")

    def sample(self) -> int:
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
        return max(self.min_value, min(self.max_value, round(value)))

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
            min_value=data.get("min", 1),
            max_value=data.get("max", 20)
        )


@dataclass
class FourPeakDistribution(FanoutDistribution):
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
    """

    peaks: List[GaussianPeak] = field(default_factory=lambda: [
        GaussianPeak(mean=2, std=0.3),
        GaussianPeak(mean=5, std=0.5),
        GaussianPeak(mean=8, std=0.5),
        GaussianPeak(mean=12, std=1.0)
    ])
    min_value: int = 1
    max_value: int = 20

    def __post_init__(self):
        if len(self.peaks) != 4:
            raise ValueError(f"FourPeakDistribution requires exactly 4 peaks, got {len(self.peaks)}")
        if self.min_value < 1:
            raise ValueError(f"Min value must be >= 1, got {self.min_value}")

    def sample(self) -> int:
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
        return max(self.min_value, min(self.max_value, round(value)))

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
            min_value=data.get("min", 1),
            max_value=data.get("max", 20)
        )


# Registry of distribution types
DISTRIBUTION_TYPES = {
    "static": StaticDistribution,
    "uniform": UniformDistribution,
    "two_peak": TwoPeakDistribution,
    "four_peak": FourPeakDistribution,
}


def create_distribution(config: Dict[str, Any]) -> FanoutDistribution:
    """Create a distribution from a configuration dictionary.

    Args:
        config: Dictionary with "type" key and type-specific parameters

    Returns:
        FanoutDistribution instance

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


def load_fanout_config(config_path: Union[str, Path]) -> FanoutDistribution:
    """Load fanout distribution configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        FanoutDistribution instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is not valid JSON
        ValueError: If config is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Fanout config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return create_distribution(config)


def create_default_config(dist_type: str = "static") -> Dict[str, Any]:
    """Create a default configuration for a distribution type.

    Args:
        dist_type: Distribution type ("static", "uniform", "two_peak", "four_peak")

    Returns:
        Default configuration dictionary
    """
    defaults = {
        "static": {
            "type": "static",
            "value": 3
        },
        "uniform": {
            "type": "uniform",
            "min": 2,
            "max": 8
        },
        "two_peak": {
            "type": "two_peak",
            "peaks": [
                {"mean": 3, "std": 0.5, "weight": 1.0},
                {"mean": 8, "std": 1.0, "weight": 1.0}
            ],
            "min": 1,
            "max": 12
        },
        "four_peak": {
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
    }

    if dist_type not in defaults:
        raise ValueError(
            f"Unknown distribution type: {dist_type}. "
            f"Supported types: {list(defaults.keys())}"
        )

    return defaults[dist_type]


class FanoutSampler:
    """Fanout sampler that generates fanout values for workflows.

    This class manages the fanout distribution and provides methods
    to sample fanout values for individual workflows.
    """

    def __init__(
        self,
        distribution: Optional[FanoutDistribution] = None,
        config_path: Optional[Union[str, Path]] = None,
        default_value: int = 3,
        seed: Optional[int] = None
    ):
        """Initialize the fanout sampler.

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
            self.distribution = load_fanout_config(config_path)
        else:
            self.distribution = StaticDistribution(value=default_value)

    def sample(self) -> int:
        """Sample a fanout value."""
        return self.distribution.sample()

    def sample_batch(self, count: int) -> List[int]:
        """Sample multiple fanout values.

        Args:
            count: Number of values to sample

        Returns:
            List of fanout values
        """
        return [self.sample() for _ in range(count)]

    def get_config(self) -> Dict[str, Any]:
        """Get the current distribution configuration."""
        return self.distribution.to_dict()
