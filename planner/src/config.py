"""Configuration management for Planner service."""

import os

from loguru import logger


class PlannerConfig:
    """Configuration settings for the Planner service."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        # Scheduler URL configuration
        self.scheduler_url: str | None = os.getenv("SCHEDULER_URL")

        # Instance deployment configuration
        self.instance_timeout: int = int(os.getenv("INSTANCE_TIMEOUT", "30"))
        self.instance_max_retries: int = int(
            os.getenv("INSTANCE_MAX_RETRIES", "3")
        )
        self.instance_retry_delay: float = float(
            os.getenv("INSTANCE_RETRY_DELAY", "1.0")
        )

        # Planner service configuration
        self.planner_port: int = int(os.getenv("PLANNER_PORT", "8000"))
        self.planner_host: str = os.getenv("PLANNER_HOST", "0.0.0.0")

        # Auto-optimization configuration
        self.auto_optimize_enabled: bool = os.getenv(
            "AUTO_OPTIMIZE_ENABLED", "false"
        ).lower() in ("true", "1", "yes")
        self.auto_optimize_interval: float = float(
            os.getenv("AUTO_OPTIMIZE_INTERVAL", "60.0")
        )

    def get_scheduler_url(self, override: str | None = None) -> str | None:
        """Get scheduler URL with optional override.

        Args:
            override: Optional override URL (takes precedence)

        Returns:
            Scheduler URL (override > env var > None)
        """
        return override if override is not None else self.scheduler_url

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.instance_timeout <= 0:
            error_msg = f"INSTANCE_TIMEOUT must be positive, got {self.instance_timeout}"
            logger.error(f"Configuration validation failed: {error_msg}")
            raise ValueError(error_msg)

        if self.instance_max_retries < 0:
            error_msg = f"INSTANCE_MAX_RETRIES must be non-negative, got {self.instance_max_retries}"
            logger.error(f"Configuration validation failed: {error_msg}")
            raise ValueError(error_msg)

        if self.instance_retry_delay < 0:
            error_msg = f"INSTANCE_RETRY_DELAY must be non-negative, got {self.instance_retry_delay}"
            logger.error(f"Configuration validation failed: {error_msg}")
            raise ValueError(error_msg)

        if not (0 < self.planner_port < 65536):
            error_msg = f"PLANNER_PORT must be 1-65535, got {self.planner_port}"
            logger.error(f"Configuration validation failed: {error_msg}")
            raise ValueError(error_msg)

        if self.auto_optimize_interval <= 0:
            error_msg = f"AUTO_OPTIMIZE_INTERVAL must be positive, got {self.auto_optimize_interval}"
            logger.error(f"Configuration validation failed: {error_msg}")
            raise ValueError(error_msg)


# Global configuration instance
config = PlannerConfig()
