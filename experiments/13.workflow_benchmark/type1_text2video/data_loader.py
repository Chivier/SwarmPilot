"""Real data loader for Text2Video simulation benchmark.

Loads benchmark data from training_config.json and captions_10k.jsonl
to provide realistic runtime values for simulation mode.

Data files are expected at fixed paths relative to the experiment root:
- type1_text2video/data/training_config.json
- type1_text2video/data/captions_10k.jsonl
"""

import json
import random
from pathlib import Path
from typing import List, Optional


class RealDataLoader:
    """
    Loads and provides access to training benchmark data for realistic simulation.

    Data sources:
    - training_config.json: Contains llm_samples (200) and t2vid_samples (80)
    - captions_10k.jsonl: Contains 10,000 entries with frame values

    For sleep_model_a (LLM): Random sample from llm_samples runtime_ms
    For sleep_model_b (T2VID): Linear regression on t2vid_samples, applied to frames from captions
    """

    # Default maximum sleep time in milliseconds (sleep_model constraint: 0-600 seconds)
    DEFAULT_MAX_SLEEP_TIME_MS = 600 * 1000.0  # 600 seconds = 600,000 ms

    def __init__(
        self,
        training_config_path: str,
        captions_path: str,
        seed: int = 42,
        max_sleep_time_ms: Optional[float] = None,
    ):
        """
        Initialize the data loader.

        Args:
            training_config_path: Path to training_config.json (relative to experiment root)
            captions_path: Path to captions_10k.jsonl (relative to experiment root)
            seed: Random seed for reproducibility
            max_sleep_time_ms: Maximum sleep time in milliseconds (default: 600,000ms = 600s)

        Raises:
            FileNotFoundError: If required data files are not found
            ValueError: If data files are invalid or empty
        """
        self._rng = random.Random(seed)
        self._seed = seed
        self._max_sleep_time_ms = max_sleep_time_ms or self.DEFAULT_MAX_SLEEP_TIME_MS

        # Resolve paths relative to experiment root (experiments/13.workflow_benchmark)
        base = Path(__file__).parent.parent
        training_config_full_path = base / training_config_path
        captions_full_path = base / captions_path

        # Verify data files exist before loading
        if not training_config_full_path.exists():
            raise FileNotFoundError(
                f"Required data file not found: {training_config_full_path}\n"
                f"Please ensure training_config.json exists at: {training_config_path}"
            )
        if not captions_full_path.exists():
            raise FileNotFoundError(
                f"Required data file not found: {captions_full_path}\n"
                f"Please ensure captions_10k.jsonl exists at: {captions_path}"
            )

        # Load data
        self._llm_runtimes: List[float] = []
        self._t2vid_samples: List[dict] = []
        self._frame_values: List[int] = []

        self._load_training_config(training_config_full_path)
        self._load_captions(captions_full_path)

        # Build linear regression model for t2vid
        self._slope: float = 0.0
        self._intercept: float = 0.0
        self._build_regression_model()

        # Calculate scaling factor to fit all runtimes within max_sleep_time
        self._t2vid_scale_factor: float = self._compute_scale_factor()

        # Pre-compute averages for min_time strategy
        self._llm_avg: float = sum(self._llm_runtimes) / len(self._llm_runtimes)
        self._t2vid_avg: float = self._compute_t2vid_average()

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

        Uses least squares method on t2vid_samples data.
        """
        if not self._t2vid_samples:
            return

        # Extract (frames, runtime_ms) pairs
        x_values = [s["frames"] for s in self._t2vid_samples]
        y_values = [s["runtime_ms"] for s in self._t2vid_samples]

        n = len(x_values)
        if n == 0:
            return

        # Calculate means
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        # Calculate slope and intercept using least squares
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            # All x values are the same, use mean y
            self._slope = 0.0
            self._intercept = y_mean
        else:
            self._slope = numerator / denominator
            self._intercept = y_mean - self._slope * x_mean

    # Minimum sleep time in milliseconds (1 second)
    MIN_SLEEP_TIME_MS = 1000.0

    def _compute_scale_factor(self) -> float:
        """
        Compute scale factor to proportionally scale all T2VID runtimes.

        The scale factor ensures:
        1. Maximum runtime does not exceed max_sleep_time_ms (upper bound) - PRIMARY
        2. Minimum runtime is at least MIN_SLEEP_TIME_MS (lower bound, 1 second) - SECONDARY

        Priority: Upper bound is primary (user-specified max_sleep_time).
        Lower bound only applies when unscaled min_runtime is already < MIN_SLEEP_TIME_MS.

        This preserves the original distribution shape while fitting values
        within the sleep_model constraints.

        Returns:
            Scale factor (factor > 0)
        """
        if not self._frame_values:
            return 1.0

        # Find min and max runtime from all frame values (unscaled)
        runtimes = [self._slope * frames + self._intercept for frames in self._frame_values]
        min_runtime = min(runtimes)
        max_runtime = max(runtimes)

        if max_runtime <= 0:
            return 1.0

        # Primary constraint: upper bound (user-specified max_sleep_time)
        # Scale down if max runtime exceeds the limit
        scale_for_upper = 1.0
        if max_runtime > self._max_sleep_time_ms:
            scale_for_upper = self._max_sleep_time_ms / max_runtime

        # Secondary constraint: lower bound (only if unscaled min < MIN_SLEEP_TIME)
        # This handles cases where the original data has very small minimum values
        # Scale up to ensure min >= MIN_SLEEP_TIME_MS
        scale_for_lower = 1.0
        if min_runtime > 0 and min_runtime < self.MIN_SLEEP_TIME_MS:
            scale_for_lower = self.MIN_SLEEP_TIME_MS / min_runtime

        # Determine final scale factor
        # - Upper bound takes priority (user specified)
        # - Lower bound only applies if original data has small minimums
        import logging
        if scale_for_lower > 1.0 and scale_for_lower > scale_for_upper:
            # Lower bound requires scaling up, but upper bound requires scaling down
            # This is a conflict - prioritize upper bound (user's explicit setting)
            scale_factor = scale_for_upper
            scaled_min = min_runtime * scale_factor
            if scaled_min < self.MIN_SLEEP_TIME_MS:
                logging.warning(
                    f"Scale factor conflict: unscaled min={min_runtime:.1f}ms requires "
                    f"scale_up={scale_for_lower:.4f}, but max_sleep_time requires "
                    f"scale_down={scale_for_upper:.4f}. Using {scale_factor:.4f} "
                    f"(honoring user's max_sleep_time). Scaled min={scaled_min:.1f}ms "
                    f"is below {self.MIN_SLEEP_TIME_MS:.1f}ms limit."
                )
        elif scale_for_lower > 1.0:
            # Need to scale up to meet minimum, and it doesn't conflict with upper
            scale_factor = scale_for_lower
            logging.info(
                f"Scaling up by {scale_factor:.4f} to ensure min runtime >= {self.MIN_SLEEP_TIME_MS:.1f}ms"
            )
        else:
            # Normal case: just apply upper bound scaling (or 1.0 if not needed)
            scale_factor = scale_for_upper

        return scale_factor

    def _compute_t2vid_average(self) -> float:
        """
        Compute average T2VID runtime across all frame values from captions.

        This applies the regression model to all frame values and returns the mean.
        The average uses scaled values to be consistent with actual simulation behavior.
        """
        if not self._frame_values:
            # Fallback to average of t2vid_samples
            return sum(s["runtime_ms"] for s in self._t2vid_samples) / len(
                self._t2vid_samples
            )

        # Use scaled values for the average to be consistent with actual simulation
        total = sum(self.get_t2vid_runtime_ms(f, scale=True) for f in self._frame_values)
        return total / len(self._frame_values)

    def sample_llm_runtime_ms(self) -> float:
        """
        Random sample from llm_samples runtime_ms values.

        Returns:
            Runtime in milliseconds for LLM task
        """
        return self._rng.choice(self._llm_runtimes)

    def get_t2vid_runtime_ms(self, frames: int, scale: bool = True) -> float:
        """
        Calculate runtime_ms using linear regression: frames → runtime_ms.

        Args:
            frames: Number of video frames
            scale: Whether to apply proportional scaling to fit within max_sleep_time (default: True)

        Returns:
            Predicted runtime in milliseconds (scaled proportionally if scale=True)
        """
        runtime = self._slope * frames + self._intercept
        if scale:
            runtime = runtime * self._t2vid_scale_factor
        return runtime

    def sample_frame_count(self) -> int:
        """
        Random sample frame value from captions_10k.jsonl.

        Returns:
            Frame count value
        """
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
    def regression_params(self) -> dict:
        """Return regression model parameters for debugging."""
        return {
            "slope": self._slope,
            "intercept": self._intercept,
            "scale_factor": self._t2vid_scale_factor,
        }

    @property
    def llm_sample_count(self) -> int:
        """Number of LLM runtime samples."""
        return len(self._llm_runtimes)

    @property
    def t2vid_sample_count(self) -> int:
        """Number of T2VID training samples."""
        return len(self._t2vid_samples)

    @property
    def frame_value_count(self) -> int:
        """Number of frame values from captions."""
        return len(self._frame_values)

    @property
    def llm_runtime_range_ms(self) -> tuple:
        """Return (min, max) LLM runtime in milliseconds."""
        return (min(self._llm_runtimes), max(self._llm_runtimes))

    @property
    def frame_value_range(self) -> tuple:
        """Return (min, max) frame values."""
        return (min(self._frame_values), max(self._frame_values))

    @property
    def max_sleep_time_ms(self) -> float:
        """Return the configured maximum sleep time in milliseconds."""
        return self._max_sleep_time_ms

    def get_statistics_summary(self) -> dict:
        """Return comprehensive statistics for logging.

        Returns:
            Dict containing all relevant statistics for display.
        """
        llm_min, llm_max = self.llm_runtime_range_ms
        frame_min, frame_max = self.frame_value_range

        # Calculate T2VID runtime range based on frame range
        t2vid_min = self.get_t2vid_runtime_ms(frame_min, scale=True)
        t2vid_max = self.get_t2vid_runtime_ms(frame_max, scale=True)

        return {
            "llm": {
                "sample_count": self.llm_sample_count,
                "avg_runtime_ms": self._llm_avg,
                "min_runtime_ms": llm_min,
                "max_runtime_ms": llm_max,
            },
            "t2vid": {
                "training_sample_count": self.t2vid_sample_count,
                "avg_runtime_ms": self._t2vid_avg,
                "min_runtime_ms": t2vid_min,
                "max_runtime_ms": t2vid_max,
                "regression_slope": self._slope,
                "regression_intercept": self._intercept,
                "scale_factor": self._t2vid_scale_factor,
            },
            "frames": {
                "caption_count": self.frame_value_count,
                "min_frames": frame_min,
                "max_frames": frame_max,
            },
            "config": {
                "max_sleep_time_ms": self._max_sleep_time_ms,
                "min_sleep_time_ms": self.MIN_SLEEP_TIME_MS,
                "seed": self._seed,
            },
        }

    def reset_rng(self, seed: Optional[int] = None) -> None:
        """Reset the random number generator with optional new seed."""
        self._rng = random.Random(seed if seed is not None else self._seed)
