"""Configuration management for Planner service."""

import os
from typing import Optional


class PlannerConfig:
    """Configuration settings for the Planner service."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        # Scheduler URL configuration
        self.scheduler_url: Optional[str] = os.getenv("SCHEDULER_URL")

        # Instance deployment configuration
        self.instance_timeout: int = int(os.getenv("INSTANCE_TIMEOUT", "30"))
        self.instance_max_retries: int = int(os.getenv("INSTANCE_MAX_RETRIES", "3"))
        self.instance_retry_delay: float = float(os.getenv("INSTANCE_RETRY_DELAY", "1.0"))

        # Planner service configuration
        self.planner_port: int = int(os.getenv("PLANNER_PORT", "8000"))
        self.planner_host: str = os.getenv("PLANNER_HOST", "0.0.0.0")

        # Auto-optimization configuration
        self.auto_optimize_enabled: bool = os.getenv("AUTO_OPTIMIZE_ENABLED", "false").lower() in ("true", "1", "yes")
        self.auto_optimize_interval: float = float(os.getenv("AUTO_OPTIMIZE_INTERVAL", "60.0"))
        self.auto_optimize_cooldown: float = float(os.getenv("AUTO_OPTIMIZE_COOLDOWN", "5.0"))

    def get_scheduler_url(self, override: Optional[str] = None) -> Optional[str]:
        """
        Get scheduler URL with optional override.

        Args:
            override: Optional override URL (takes precedence)

        Returns:
            Scheduler URL (override > env var > None)
        """
        return override if override is not None else self.scheduler_url

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.instance_timeout <= 0:
            raise ValueError(f"INSTANCE_TIMEOUT must be positive, got {self.instance_timeout}")

        if self.instance_max_retries < 0:
            raise ValueError(f"INSTANCE_MAX_RETRIES must be non-negative, got {self.instance_max_retries}")

        if self.instance_retry_delay < 0:
            raise ValueError(f"INSTANCE_RETRY_DELAY must be non-negative, got {self.instance_retry_delay}")

        if not (0 < self.planner_port < 65536):
            raise ValueError(f"PLANNER_PORT must be 1-65535, got {self.planner_port}")

        if self.auto_optimize_interval <= 0:
            raise ValueError(f"AUTO_OPTIMIZE_INTERVAL must be positive, got {self.auto_optimize_interval}")

        if self.auto_optimize_cooldown < 0:
            raise ValueError(f"AUTO_OPTIMIZE_COOLDOWN must be non-negative, got {self.auto_optimize_cooldown}")


# Global configuration instance
config = PlannerConfig()
