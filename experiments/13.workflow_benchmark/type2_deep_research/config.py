"""Configuration for Deep Research workflow experiments."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DeepResearchConfig:
    """Configuration for Deep Research workflow experiments."""

    # Experiment mode
    mode: str = "simulation"  # "simulation" or "real"

    # Experiment parameters
    qps: float = 1.0
    duration: int = 600  # seconds
    num_workflows: int = 600
    fanout_count: int = 3  # Number of B1/B2 tasks per workflow

    # Strategy for task scheduling (used in task IDs)
    strategy: str = "probabilistic"

    # Warmup workflows
    num_warmup: int = 10

    # Model IDs (will be auto-set in __post_init__ based on mode)
    model_a_id: str = "llm_service_small_model"
    model_b_id: str = "llm_service_small_model"
    model_merge_id: str = "llm_service_small_model"

    # Scheduler URLs
    scheduler_a_url: str = "http://127.0.0.1:8100"
    scheduler_b_url: str = "http://127.0.0.1:8200"

    # Predictor and planner URLs
    predictor_url: str = "http://127.0.0.1:8102"
    planner_url: str = "http://127.0.0.1:8103"

    # Task parameters (simulation mode)
    sleep_time_min: float = 5.0
    sleep_time_max: float = 15.0

    # Task parameters (real mode)
    max_tokens: int = 512  # For LLM tasks

    # Output paths
    output_dir: str = "output"
    metrics_file: str = "metrics.json"

    # Scheduling strategy testing
    strategies: Optional[list] = None  # List of strategy names to test
    target_quantile: Optional[float] = None  # Target quantile for probabilistic strategy
    quantiles: Optional[list] = None  # Custom quantiles for probabilistic strategy

    def __post_init__(self):
        """Post-initialization to set model IDs and scheduler URLs based on mode."""
        if self.mode == "simulation":
            # Use sleep models for simulation
            self.model_a_id = "sleep_model_a"
            self.model_b_id = "sleep_model_b"
            self.model_merge_id = "sleep_model_a"
            # Use local schedulers for simulation
            self.scheduler_a_url = "http://127.0.0.1:8100"
            self.scheduler_b_url = "http://127.0.0.1:8200"
        else:  # real mode
            # Use LLM models for real workload
            self.model_a_id = "llm_service_small_model"
            self.model_b_id = "llm_service_large_model"  # B tasks use large model
            self.model_merge_id = "llm_service_small_model"  # Merge uses small model
            # Use remote schedulers for real mode
            self.scheduler_a_url = "http://29.209.114.51:8100"
            self.scheduler_b_url = "http://29.209.113.228:8100"

    @classmethod
    def from_env(cls) -> "DeepResearchConfig":
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

        return cls(
            mode=os.getenv("MODE", "simulation"),
            qps=float(os.getenv("QPS", "1.0")),
            duration=int(os.getenv("DURATION", "600")),
            num_workflows=int(os.getenv("NUM_WORKFLOWS", "600")),
            fanout_count=int(os.getenv("FANOUT_COUNT", "3")),
            strategy=os.getenv("STRATEGY", "probabilistic"),
            num_warmup=int(os.getenv("NUM_WARMUP", "10")),
            model_a_id=os.getenv("MODEL_A_ID", "llm_service_small_model"),
            model_b_id=os.getenv("MODEL_B_ID", "llm_service_small_model"),
            model_merge_id=os.getenv("MODEL_MERGE_ID", "llm_service_small_model"),
            scheduler_a_url=os.getenv("SCHEDULER_A_URL", "http://127.0.0.1:8100"),
            scheduler_b_url=os.getenv("SCHEDULER_B_URL", "http://127.0.0.1:8200"),
            predictor_url=os.getenv("PREDICTOR_URL", "http://127.0.0.1:8102"),
            planner_url=os.getenv("PLANNER_URL", "http://127.0.0.1:8103"),
            sleep_time_min=float(os.getenv("SLEEP_TIME_MIN", "5.0")),
            sleep_time_max=float(os.getenv("SLEEP_TIME_MAX", "15.0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "512")),
            output_dir=os.getenv("OUTPUT_DIR", "output"),
            metrics_file=os.getenv("METRICS_FILE", "metrics.json"),
            strategies=strategies,
            target_quantile=target_quantile,
            quantiles=quantiles,
        )
