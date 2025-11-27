"""Data loader for Text2Image+Video simulation benchmark.

Extends the type1 RealDataLoader with FLUX timing model based on resolution.
"""

import json
import random
from pathlib import Path
from typing import List, Optional


# FLUX timing estimates (milliseconds) based on typical FLUX inference on H20-class GPUs
# These are estimates with ~10% variance applied during sampling
FLUX_TIMING_MS = {
    "512x512": 7000.0,    # ~7 seconds for 512x512
    "1024x1024": 19000.0,  # ~19 seconds for 1024x1024
}


class Type3DataLoader:
    """
    Data loader for Type3 (Text2Image+Video) simulation.

    Provides:
    - LLM timing (sample_llm_runtime_ms) for A task
    - FLUX timing (get_flux_runtime_ms) for C task based on resolution
    - T2VID timing (get_t2vid_runtime_ms) for B tasks

    Data sources:
    - training_config.json: Contains llm_samples (200) and t2vid_samples (80)
    - captions_10k.jsonl: Contains 10,000 entries with frame values
    """

    # Maximum sleep time in milliseconds (sleep_model constraint: 0-600 seconds)
    MAX_SLEEP_TIME_MS = 600 * 1000.0  # 600 seconds = 600,000 ms

    def __init__(
        self,
        training_config_path: str,
        captions_path: str,
        seed: int = 42,
        base_path: Optional[str] = None,
        max_sleep_time_ms: Optional[float] = None,
    ):
        """
        Initialize the data loader.

        Args:
            training_config_path: Path to training_config.json
            captions_path: Path to captions_10k.jsonl
            seed: Random seed for reproducibility
            base_path: Base path for resolving relative paths
            max_sleep_time_ms: Maximum sleep time in milliseconds (default: 600,000ms = 600s)
        """
        self._rng = random.Random(seed)
        self._seed = seed
        self._max_sleep_time_ms = max_sleep_time_ms or self.MAX_SLEEP_TIME_MS

        # Resolve paths
        if base_path:
            base = Path(base_path)
        else:
            base = Path(__file__).parent.parent  # experiments/13.workflow_benchmark

        training_config_path = base / training_config_path
        captions_path = base / captions_path

        # Load data
        self._llm_runtimes: List[float] = []
        self._t2vid_samples: List[dict] = []
        self._frame_values: List[int] = []

        self._load_training_config(training_config_path)
        self._load_captions(captions_path)

        # Build linear regression model for t2vid
        self._slope: float = 0.0
        self._intercept: float = 0.0
        self._build_regression_model()

        # Calculate scaling factor to fit all runtimes within max_sleep_time
        self._t2vid_scale_factor: float = self._compute_scale_factor()

        # Pre-compute averages for min_time strategy
        self._llm_avg: float = sum(self._llm_runtimes) / len(self._llm_runtimes) if self._llm_runtimes else 5000.0
        self._t2vid_avg: float = self._compute_t2vid_average()
        self._flux_avg: float = self._compute_flux_average()

    def _load_training_config(self, path: Path) -> None:
        """Load training_config.json and extract runtime data."""
        with open(path, "r") as f:
            data = json.load(f)

        # Extract LLM runtimes
        llm_samples = data.get("llm_samples", [])
        self._llm_runtimes = [s["runtime_ms"] for s in llm_samples if "runtime_ms" in s]

        # Extract T2VID samples (frames, runtime_ms pairs)
        self._t2vid_samples = data.get("t2vid_samples", [])

        if not self._llm_runtimes:
            raise ValueError(f"No llm_samples found in {path}")
        if not self._t2vid_samples:
            raise ValueError(f"No t2vid_samples found in {path}")

    def _load_captions(self, path: Path) -> None:
        """Load captions_10k.jsonl and extract frame values."""
        self._frame_values = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if "frame" in entry:
                        self._frame_values.append(int(entry["frame"]))

        if not self._frame_values:
            raise ValueError(f"No frame values found in {path}")

    def _build_regression_model(self) -> None:
        """
        Build linear regression model: runtime_ms = slope * frames + intercept
        """
        if not self._t2vid_samples:
            return

        x_values = [s["frames"] for s in self._t2vid_samples]
        y_values = [s["runtime_ms"] for s in self._t2vid_samples]

        n = len(x_values)
        if n == 0:
            return

        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            self._slope = 0.0
            self._intercept = y_mean
        else:
            self._slope = numerator / denominator
            self._intercept = y_mean - self._slope * x_mean

    def _compute_scale_factor(self) -> float:
        """Compute scale factor to proportionally scale all T2VID runtimes within max_sleep_time."""
        if not self._frame_values:
            return 1.0

        max_runtime = max(
            self._slope * frames + self._intercept for frames in self._frame_values
        )

        if max_runtime <= self._max_sleep_time_ms:
            return 1.0

        return self._max_sleep_time_ms / max_runtime

    def _compute_t2vid_average(self) -> float:
        """Compute average T2VID runtime across all frame values from captions."""
        if not self._frame_values:
            return sum(s["runtime_ms"] for s in self._t2vid_samples) / len(
                self._t2vid_samples
            )

        total = sum(self.get_t2vid_runtime_ms(f, scale=True) for f in self._frame_values)
        return total / len(self._frame_values)

    def _compute_flux_average(self) -> float:
        """Compute weighted average FLUX runtime (assuming 70/30 split for 512/1024)."""
        # Default assumption: 70% 512x512, 30% 1024x1024
        return 0.7 * FLUX_TIMING_MS["512x512"] + 0.3 * FLUX_TIMING_MS["1024x1024"]

    def sample_llm_runtime_ms(self) -> float:
        """Random sample from llm_samples runtime_ms values."""
        return self._rng.choice(self._llm_runtimes)

    def get_t2vid_runtime_ms(self, frames: int, scale: bool = True) -> float:
        """Calculate runtime_ms using linear regression: frames → runtime_ms."""
        runtime = self._slope * frames + self._intercept
        if scale:
            runtime = runtime * self._t2vid_scale_factor
        return runtime

    def get_flux_runtime_ms(self, resolution: str, add_variance: bool = True) -> float:
        """
        Get FLUX runtime for given resolution.

        Args:
            resolution: Resolution string ("512x512" or "1024x1024")
            add_variance: Whether to add 10% Gaussian variance

        Returns:
            Runtime in milliseconds
        """
        base_runtime = FLUX_TIMING_MS.get(resolution, FLUX_TIMING_MS["512x512"])

        if add_variance:
            # Add 10% Gaussian variance
            variance = self._rng.gauss(0, base_runtime * 0.1)
            runtime = base_runtime + variance
            # Ensure minimum of 1 second
            return max(1000.0, runtime)

        return base_runtime

    def sample_frame_count(self) -> int:
        """Random sample frame value from captions_10k.jsonl."""
        return self._rng.choice(self._frame_values)

    @property
    def llm_avg_runtime_ms(self) -> float:
        """Average LLM runtime for min_time strategy."""
        return self._llm_avg

    @property
    def t2vid_avg_runtime_ms(self) -> float:
        """Average T2VID runtime for min_time strategy."""
        return self._t2vid_avg

    @property
    def flux_avg_runtime_ms(self) -> float:
        """Average FLUX runtime for min_time strategy."""
        return self._flux_avg

    @property
    def regression_params(self) -> dict:
        """Return regression model parameters for debugging."""
        return {
            "slope": self._slope,
            "intercept": self._intercept,
            "scale_factor": self._t2vid_scale_factor,
        }

    def reset_rng(self, seed: Optional[int] = None) -> None:
        """Reset the random number generator with optional new seed."""
        self._rng = random.Random(seed if seed is not None else self._seed)
