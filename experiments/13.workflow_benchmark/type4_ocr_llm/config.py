"""Configuration for OCR+LLM workflow experiments."""

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
class OCRLLMConfig:
    """Configuration for OCR+LLM workflow experiments.

    This workflow has two stages:
    - Stage A: OCR (EasyOCR) - CPU-only, extract text from images
    - Stage B: LLM (llm_service_small_model) - GPU, process extracted text

    Workflow pattern: A → B (simple sequential, no loops)
    """

    # Experiment mode
    mode: str = "simulation"  # "simulation" or "real"

    # Experiment parameters
    qps: float = 1.0
    duration: int = 600  # seconds
    num_workflows: int = 600

    # Model IDs (will be auto-set in __post_init__ based on mode)
    model_a_id: str = "ocr_model"  # OCR model (CPU)
    model_b_id: str = "llm_service_small_model"  # LLM model (GPU)

    # Scheduler URLs
    scheduler_a_url: str = "http://127.0.0.1:8100"  # OCR scheduler
    scheduler_b_url: str = "http://127.0.0.1:8200"  # LLM scheduler

    # Predictor and planner URLs
    predictor_url: str = "http://127.0.0.1:8102"
    planner_url: str = "http://127.0.0.1:8103"

    # Task parameters (simulation mode)
    sleep_time_min: float = 1.0  # OCR is typically faster
    sleep_time_max: float = 5.0
    sleep_time_b_min: float = 5.0  # LLM is typically slower
    sleep_time_b_max: float = 15.0

    # Task parameters (real mode)
    max_tokens: int = 512  # For LLM tasks
    ocr_languages: str = "en"  # OCR languages (comma-separated)
    ocr_detail_level: str = "standard"  # OCR detail level

    # Strategy for task scheduling
    strategy: str = "probabilistic"

    # Warmup workflows
    num_warmup: int = 10

    # Output paths
    output_dir: str = "output"
    metrics_file: str = "metrics.json"

    # Scheduling strategy testing
    strategies: Optional[list] = None  # List of strategy names to test
    target_quantile: Optional[float] = None  # Target quantile for probabilistic strategy
    quantiles: Optional[list] = None  # Custom quantiles for probabilistic strategy

    # Statistics filtering
    portion_stats: float = 1.0  # Portion of non-warmup workflows to include in statistics

    # Distribution configurations for sleep times
    sleep_time_a_config: Optional[Union[str, Path, Dict[str, Any]]] = None
    sleep_time_b_config: Optional[Union[str, Path, Dict[str, Any]]] = None
    sleep_time_seed: Optional[int] = 42  # Random seed for reproducibility

    # Internal samplers (initialized in __post_init__)
    _sleep_time_a_sampler: Optional[DistributionSampler] = field(default=None, repr=False)
    _sleep_time_b_sampler: Optional[DistributionSampler] = field(default=None, repr=False)

    def __post_init__(self):
        """Post-initialization to set model IDs and scheduler URLs based on mode."""
        if self.mode == "simulation":
            # Use sleep models for simulation
            # NOTE: Using sleep_model_a/sleep_model_b to be compatible with
            # the unified startup script (scripts/start_all_services.sh)
            self.model_a_id = "sleep_model_a"  # Simulates OCR
            self.model_b_id = "sleep_model_b"  # Simulates LLM
            # Use local schedulers for simulation
            self.scheduler_a_url = "http://127.0.0.1:8100"
            self.scheduler_b_url = "http://127.0.0.1:8200"
        else:  # real mode
            # Use real models
            self.model_a_id = "ocr_model"  # EasyOCR
            self.model_b_id = "llm_service_small_model"  # LLM
            # Use remote schedulers for real mode (configured via env or params)
            # Default to local if not specified
            if self.scheduler_a_url == "http://127.0.0.1:8100":
                self.scheduler_a_url = os.getenv("SCHEDULER_A_URL", "http://127.0.0.1:8100")
            if self.scheduler_b_url == "http://127.0.0.1:8200":
                self.scheduler_b_url = os.getenv("SCHEDULER_B_URL", "http://127.0.0.1:8200")

        # Initialize distribution samplers
        self._init_samplers()

    def _parse_config(self, config: Union[str, Path, Dict[str, Any]]) -> Distribution:
        """Parse a distribution config from string, path, or dict.

        Args:
            config: Can be:
                - A dict with distribution parameters
                - A JSON string (starts with '{')
                - A file path to a JSON config file

        Returns:
            Distribution object
        """
        import json

        if isinstance(config, dict):
            return create_distribution(config)
        elif isinstance(config, str):
            # Check if it's a JSON string (starts with '{')
            if config.strip().startswith('{'):
                try:
                    config_dict = json.loads(config)
                    return create_distribution(config_dict)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON config string: {e}")
            else:
                # Treat as file path
                return load_distribution_config(config)
        elif isinstance(config, Path):
            return load_distribution_config(config)
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

    def _init_samplers(self):
        """Initialize distribution samplers for sleep times."""
        # Sleep time A sampler (OCR)
        if self.sleep_time_a_config is not None:
            dist = self._parse_config(self.sleep_time_a_config)
            self._sleep_time_a_sampler = DistributionSampler(
                distribution=dist,
                seed=self.sleep_time_seed
            )
        else:
            # Use uniform distribution between min and max
            from common.distribution import UniformDistribution
            self._sleep_time_a_sampler = DistributionSampler(
                distribution=UniformDistribution(
                    min_value=self.sleep_time_min,
                    max_value=self.sleep_time_max
                ),
                seed=self.sleep_time_seed
            )

        # Sleep time B sampler (LLM)
        if self.sleep_time_b_config is not None:
            dist = self._parse_config(self.sleep_time_b_config)
            self._sleep_time_b_sampler = DistributionSampler(
                distribution=dist,
                seed=self.sleep_time_seed + 1 if self.sleep_time_seed else None
            )
        else:
            # Use uniform distribution between min and max
            from common.distribution import UniformDistribution
            self._sleep_time_b_sampler = DistributionSampler(
                distribution=UniformDistribution(
                    min_value=self.sleep_time_b_min,
                    max_value=self.sleep_time_b_max
                ),
                seed=self.sleep_time_seed + 1 if self.sleep_time_seed else None
            )

    def sample_sleep_time_a(self) -> float:
        """Sample a sleep time for task A (OCR) from the configured distribution."""
        if self._sleep_time_a_sampler is None:
            self._init_samplers()
        return self._sleep_time_a_sampler.sample()

    def sample_sleep_time_b(self) -> float:
        """Sample a sleep time for task B (LLM) from the configured distribution."""
        if self._sleep_time_b_sampler is None:
            self._init_samplers()
        return self._sleep_time_b_sampler.sample()

    @classmethod
    def from_env(cls) -> "OCRLLMConfig":
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
        sleep_time_a_config = os.getenv("SLEEP_TIME_A_CONFIG")
        sleep_time_b_config = os.getenv("SLEEP_TIME_B_CONFIG")

        # Parse sleep time seed (default to 42 for reproducibility)
        sleep_time_seed = int(os.getenv("SLEEP_TIME_SEED", "42"))

        return cls(
            mode=os.getenv("MODE", "simulation"),
            qps=float(os.getenv("QPS", "1.0")),
            duration=int(os.getenv("DURATION", "600")),
            num_workflows=int(os.getenv("NUM_WORKFLOWS", "600")),
            model_a_id=os.getenv("MODEL_A_ID", "ocr_model"),
            model_b_id=os.getenv("MODEL_B_ID", "llm_service_small_model"),
            scheduler_a_url=os.getenv("SCHEDULER_A_URL", "http://127.0.0.1:8100"),
            scheduler_b_url=os.getenv("SCHEDULER_B_URL", "http://127.0.0.1:8200"),
            predictor_url=os.getenv("PREDICTOR_URL", "http://127.0.0.1:8102"),
            planner_url=os.getenv("PLANNER_URL", "http://127.0.0.1:8103"),
            sleep_time_min=float(os.getenv("SLEEP_TIME_MIN", "1.0")),
            sleep_time_max=float(os.getenv("SLEEP_TIME_MAX", "5.0")),
            sleep_time_b_min=float(os.getenv("SLEEP_TIME_B_MIN", "5.0")),
            sleep_time_b_max=float(os.getenv("SLEEP_TIME_B_MAX", "15.0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "512")),
            ocr_languages=os.getenv("OCR_LANGUAGES", "en"),
            ocr_detail_level=os.getenv("OCR_DETAIL_LEVEL", "standard"),
            strategy=os.getenv("STRATEGY", "probabilistic"),
            num_warmup=int(os.getenv("NUM_WARMUP", "10")),
            output_dir=os.getenv("OUTPUT_DIR", "output"),
            metrics_file=os.getenv("METRICS_FILE", "metrics.json"),
            strategies=strategies,
            target_quantile=target_quantile,
            quantiles=quantiles,
            portion_stats=float(os.getenv("PORTION_STATS", "1.0")),
            sleep_time_a_config=sleep_time_a_config,
            sleep_time_b_config=sleep_time_b_config,
            sleep_time_seed=sleep_time_seed,
        )
