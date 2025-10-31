"""
Configuration management for the scheduler service.

This module centralizes all configuration settings with support for
environment variables and sensible defaults.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class PredictorConfig:
    """Configuration for predictor service integration."""

    # Predictor service URL
    url: str = os.getenv("PREDICTOR_URL", "http://localhost:8001")

    # Request timeout in seconds
    timeout: float = float(os.getenv("PREDICTOR_TIMEOUT", "5.0"))

    # Maximum retry attempts for transient failures
    max_retries: int = int(os.getenv("PREDICTOR_MAX_RETRIES", "3"))

    # Initial retry delay in seconds (exponential backoff)
    retry_delay: float = float(os.getenv("PREDICTOR_RETRY_DELAY", "1.0"))

    # Prediction cache TTL in seconds
    cache_ttl: int = int(os.getenv("PREDICTOR_CACHE_TTL", "300"))  # 5 minutes

    # Enable prediction caching
    enable_cache: bool = os.getenv("PREDICTOR_ENABLE_CACHE", "true").lower() == "true"


@dataclass
class SchedulingConfig:
    """Configuration for scheduling behavior."""

    # Default scheduling strategy: "min_time", "probabilistic", "round_robin"
    default_strategy: str = os.getenv("SCHEDULING_STRATEGY", "probabilistic")

    # Target quantile for probabilistic scheduling (0.0 - 1.0)
    probabilistic_quantile: float = float(
        os.getenv("SCHEDULING_PROBABILISTIC_QUANTILE", "0.9")
    )


@dataclass
class TrainingConfig:
    """Configuration for model training pipeline."""

    # Enable automatic training
    enable_auto_training: bool = (
        os.getenv("TRAINING_ENABLE_AUTO", "false").lower() == "true"
    )

    # Batch size for training data collection
    batch_size: int = int(os.getenv("TRAINING_BATCH_SIZE", "100"))

    # Training frequency in seconds
    frequency_seconds: int = int(os.getenv("TRAINING_FREQUENCY", "3600"))  # 1 hour

    # Minimum samples required before training
    min_samples: int = int(os.getenv("TRAINING_MIN_SAMPLES", "10"))


@dataclass
class ServerConfig:
    """Configuration for the scheduler server."""

    # Server host
    host: str = os.getenv("SCHEDULER_HOST", "0.0.0.0")

    # Server port
    port: int = int(os.getenv("SCHEDULER_PORT", "8000"))

    # Enable CORS
    enable_cors: bool = os.getenv("SCHEDULER_ENABLE_CORS", "true").lower() == "true"

    # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_level: str = os.getenv("SCHEDULER_LOG_LEVEL", "INFO")

    # API version
    version: str = "1.0.0"


@dataclass
class Config:
    """Main configuration object combining all settings."""

    predictor: PredictorConfig
    scheduling: SchedulingConfig
    training: TrainingConfig
    server: ServerConfig

    @classmethod
    def load(cls) -> "Config":
        """
        Load configuration from environment variables.

        Returns:
            Config object with all settings
        """
        return cls(
            predictor=PredictorConfig(),
            scheduling=SchedulingConfig(),
            training=TrainingConfig(),
            server=ServerConfig(),
        )

    def __repr__(self) -> str:
        """Return string representation hiding sensitive info."""
        return (
            f"Config(\n"
            f"  predictor=PredictorConfig(url='{self.predictor.url}', ...),\n"
            f"  scheduling=SchedulingConfig(strategy='{self.scheduling.default_strategy}'),\n"
            f"  training=TrainingConfig(auto={self.training.enable_auto_training}),\n"
            f"  server=ServerConfig(host='{self.server.host}', port={self.server.port})\n"
            f")"
        )


# Global configuration instance
config = Config.load()
