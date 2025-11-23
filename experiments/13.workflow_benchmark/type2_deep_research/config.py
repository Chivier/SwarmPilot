"""Configuration for Deep Research workflow experiments."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DeepResearchConfig:
    """Configuration for Deep Research workflow experiments."""

    # Experiment parameters
    qps: float = 1.0
    duration: int = 600  # seconds
    num_workflows: int = 600
    fanout_count: int = 3  # Number of B1/B2 tasks per workflow

    # Model IDs
    model_a_id: str = "llm_service_small_model"
    model_b_id: str = "llm_service_small_model"
    model_merge_id: str = "llm_service_small_model"

    # Scheduler URLs
    scheduler_a_url: str = "http://127.0.0.1:8100"
    scheduler_b_url: str = "http://127.0.0.1:8101"

    # Predictor and planner URLs
    predictor_url: str = "http://127.0.0.1:8102"
    planner_url: str = "http://127.0.0.1:8103"

    # Task parameters (simulation mode)
    sleep_time_min: float = 5.0
    sleep_time_max: float = 15.0

    # Output paths
    output_dir: str = "output"
    metrics_file: str = "metrics.json"

    @classmethod
    def from_env(cls) -> "DeepResearchConfig":
        """Create configuration from environment variables."""
        return cls(
            qps=float(os.getenv("QPS", "1.0")),
            duration=int(os.getenv("DURATION", "600")),
            num_workflows=int(os.getenv("NUM_WORKFLOWS", "600")),
            fanout_count=int(os.getenv("FANOUT_COUNT", "3")),
            model_a_id=os.getenv("MODEL_A_ID", "llm_service_small_model"),
            model_b_id=os.getenv("MODEL_B_ID", "llm_service_small_model"),
            model_merge_id=os.getenv("MODEL_MERGE_ID", "llm_service_small_model"),
            scheduler_a_url=os.getenv("SCHEDULER_A_URL", "http://127.0.0.1:8100"),
            scheduler_b_url=os.getenv("SCHEDULER_B_URL", "http://127.0.0.1:8101"),
            predictor_url=os.getenv("PREDICTOR_URL", "http://127.0.0.1:8102"),
            planner_url=os.getenv("PLANNER_URL", "http://127.0.0.1:8103"),
            sleep_time_min=float(os.getenv("SLEEP_TIME_MIN", "5.0")),
            sleep_time_max=float(os.getenv("SLEEP_TIME_MAX", "15.0")),
            output_dir=os.getenv("OUTPUT_DIR", "output"),
            metrics_file=os.getenv("METRICS_FILE", "metrics.json"),
        )
