"""Configuration for Text2Video workflow experiments."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from common.distribution import (
    Distribution,
    DistributionSampler,
    StaticDistribution,
    create_distribution,
    load_distribution_config,
)


@dataclass
class Text2VideoConfig:
    """Configuration for Text2Video workflow experiments."""

    # Experiment mode
    mode: str = "simulation"  # "simulation" or "real"

    # Experiment parameters
    qps: float = 1.0
    duration: int = 600  # seconds
    num_workflows: int = 600
    max_b_loops: int = 4  # B task iterations per workflow

    # Model IDs
    model_a_id: str = "llm_service_small_model"
    model_b_id: str = "t2vid"

    # Scheduler URLs
    scheduler_a_url: str = "http://127.0.0.1:8100"
    scheduler_b_url: str = "http://127.0.0.1:8200"  # Consistent with type2 and from_env()

    # Predictor and planner URLs
    predictor_url: str = "http://127.0.0.1:8102"
    planner_url: str = "http://127.0.0.1:8103"

    # Task parameters (simulation mode)
    sleep_time_min: float = 5.0
    sleep_time_max: float = 15.0

    # Task parameters (real mode)
    max_tokens: int = 512  # For A1/A2 LLM tasks

    # Caption file
    caption_file: str = "captions_10k.json"

    # Output paths
    output_dir: str = "output"
    metrics_file: str = "metrics.json"

    # Scheduling strategy testing
    strategies: Optional[list] = None  # List of strategy names to test
    target_quantile: Optional[float] = None  # Target quantile for probabilistic strategy
    quantiles: Optional[list] = None  # Custom quantiles for probabilistic strategy

    # Additional fields
    strategy: str = "probabilistic"  # Single strategy name for workflow generation
    num_warmup: int = 0  # Number of warmup workflows
    frame_count: int = 16  # Default frame count for video generation
    portion_stats: float = 1.0  # Portion of non-warmup workflows to include in statistics (0.0-1.0)

    # Distribution configurations for frame_count and max_b_loops
    # These can be:
    #   - None: Use the static values from frame_count/max_b_loops fields
    #   - str/Path: Path to JSON config file
    #   - Dict: Inline distribution configuration
    frame_count_config: Optional[Union[str, Path, Dict[str, Any]]] = None
    max_b_loops_config: Optional[Union[str, Path, Dict[str, Any]]] = None
    frame_count_seed: Optional[int] = None  # Random seed for frame_count distribution
    max_b_loops_seed: Optional[int] = None  # Random seed for max_b_loops distribution

    # Internal samplers (initialized in __post_init__)
    _frame_count_sampler: Optional[DistributionSampler] = field(default=None, repr=False)
    _max_b_loops_sampler: Optional[DistributionSampler] = field(default=None, repr=False)

    def __post_init__(self):
        """Post-initialization to set mode-specific model IDs and scheduler URLs."""
        if self.mode == "simulation":
            # Override model IDs for simulation mode
            self.model_a_id = "sleep_model_a"  # For A1/A2 tasks
            self.model_b_id = "sleep_model_b"  # For B tasks
            # Use local schedulers for simulation
            self.scheduler_a_url = "http://127.0.0.1:8100"
            self.scheduler_b_url = "http://127.0.0.1:8200"
        else:  # real mode
            # Keep or set real mode model IDs
            if self.model_a_id == "sleep_model_a":  # Fix if incorrectly set
                self.model_a_id = "llm_service_small_model"
            if self.model_b_id == "sleep_model_b":  # Fix if incorrectly set
                self.model_b_id = "t2vid"
            # Use remote schedulers for real mode
            self.scheduler_a_url = "http://29.209.114.51:8100"
            self.scheduler_b_url = "http://29.209.113.228:8100"

        # Initialize distribution samplers
        self._init_samplers()

    def _init_samplers(self):
        """Initialize distribution samplers for frame_count and max_b_loops."""
        # Frame count sampler
        if self.frame_count_config is not None:
            if isinstance(self.frame_count_config, (str, Path)):
                dist = load_distribution_config(self.frame_count_config)
            else:
                dist = create_distribution(self.frame_count_config)
            self._frame_count_sampler = DistributionSampler(
                distribution=dist,
                seed=self.frame_count_seed
            )
        else:
            # Use static value from frame_count field
            self._frame_count_sampler = DistributionSampler(
                distribution=StaticDistribution(value=self.frame_count),
                seed=self.frame_count_seed
            )

        # Max B loops sampler
        if self.max_b_loops_config is not None:
            if isinstance(self.max_b_loops_config, (str, Path)):
                dist = load_distribution_config(self.max_b_loops_config)
            else:
                dist = create_distribution(self.max_b_loops_config)
            self._max_b_loops_sampler = DistributionSampler(
                distribution=dist,
                seed=self.max_b_loops_seed
            )
        elif self.max_b_loops == 0:
            # Special case: max_b_loops=0 means uniform [1, 4]
            from common.distribution import UniformDistribution
            self._max_b_loops_sampler = DistributionSampler(
                distribution=UniformDistribution(min_value=1, max_value=4),
                seed=self.max_b_loops_seed
            )
        else:
            # Use static value from max_b_loops field
            self._max_b_loops_sampler = DistributionSampler(
                distribution=StaticDistribution(value=self.max_b_loops),
                seed=self.max_b_loops_seed
            )

    def sample_frame_count(self) -> int:
        """Sample a frame count value from the configured distribution."""
        if self._frame_count_sampler is None:
            self._init_samplers()
        return self._frame_count_sampler.sample()

    def sample_max_b_loops(self) -> int:
        """Sample a max_b_loops value from the configured distribution."""
        if self._max_b_loops_sampler is None:
            self._init_samplers()
        return self._max_b_loops_sampler.sample()

    def get_frame_count_config(self) -> Dict[str, Any]:
        """Get the frame count distribution configuration."""
        if self._frame_count_sampler is None:
            self._init_samplers()
        return self._frame_count_sampler.get_config()

    def get_max_b_loops_config(self) -> Dict[str, Any]:
        """Get the max_b_loops distribution configuration."""
        if self._max_b_loops_sampler is None:
            self._init_samplers()
        return self._max_b_loops_sampler.get_config()

    @classmethod
    def from_env(cls) -> "Text2VideoConfig":
        """Create configuration from environment variables."""
        # Parse strategies from comma-separated string
        strategies_str = os.getenv("STRATEGIES")
        strategies = None
        if strategies_str:
            strategies = [s.strip() for s in strategies_str.split(",")]

        # Parse quantiles from comma-separated string
        quantiles_str = os.getenv("QUANTILES")
        quantiles = None
        if quantiles_str:
            quantiles = [float(q.strip()) for q in quantiles_str.split(",")]

        # Parse target quantile
        target_quantile = None
        target_quantile_str = os.getenv("TARGET_QUANTILE")
        if target_quantile_str:
            target_quantile = float(target_quantile_str)

        # Parse distribution config paths
        frame_count_config = os.getenv("FRAME_COUNT_CONFIG")
        max_b_loops_config = os.getenv("MAX_B_LOOPS_CONFIG")

        # Parse distribution seeds
        frame_count_seed = None
        frame_count_seed_str = os.getenv("FRAME_COUNT_SEED")
        if frame_count_seed_str:
            frame_count_seed = int(frame_count_seed_str)

        max_b_loops_seed = None
        max_b_loops_seed_str = os.getenv("MAX_B_LOOPS_SEED")
        if max_b_loops_seed_str:
            max_b_loops_seed = int(max_b_loops_seed_str)

        return cls(
            mode=os.getenv("MODE", "simulation"),
            qps=float(os.getenv("QPS", "1.0")),
            duration=int(os.getenv("DURATION", "600")),
            num_workflows=int(os.getenv("NUM_WORKFLOWS", "600")),
            max_b_loops=int(os.getenv("MAX_B_LOOPS", "4")),
            model_a_id=os.getenv("MODEL_A_ID", "llm_service_small_model"),
            model_b_id=os.getenv("MODEL_B_ID", "t2vid"),
            scheduler_a_url=os.getenv("SCHEDULER_A_URL", "http://127.0.0.1:8100"),
            scheduler_b_url=os.getenv("SCHEDULER_B_URL", "http://127.0.0.1:8200"),
            predictor_url=os.getenv("PREDICTOR_URL", "http://127.0.0.1:8102"),
            planner_url=os.getenv("PLANNER_URL", "http://127.0.0.1:8103"),
            sleep_time_min=float(os.getenv("SLEEP_TIME_MIN", "5.0")),
            sleep_time_max=float(os.getenv("SLEEP_TIME_MAX", "15.0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "512")),
            caption_file=os.getenv("CAPTION_FILE", "captions_10k.json"),
            output_dir=os.getenv("OUTPUT_DIR", "output"),
            metrics_file=os.getenv("METRICS_FILE", "metrics.json"),
            strategies=strategies,
            target_quantile=target_quantile,
            quantiles=quantiles,
            strategy=os.getenv("STRATEGY", "probabilistic"),
            num_warmup=int(os.getenv("NUM_WARMUP", "0")),
            frame_count=int(os.getenv("FRAME_COUNT", "16")),
            portion_stats=float(os.getenv("PORTION_STATS", "1.0")),
            frame_count_config=frame_count_config,
            max_b_loops_config=max_b_loops_config,
            frame_count_seed=frame_count_seed,
            max_b_loops_seed=max_b_loops_seed,
        )
