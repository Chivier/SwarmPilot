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

        # PyLet configuration
        self.pylet_enabled: bool = os.getenv(
            "PYLET_ENABLED", "false"
        ).lower() in ("true", "1", "yes")
        self.pylet_head_url: str | None = os.getenv("PYLET_HEAD_URL")
        self.pylet_backend: str = os.getenv("PYLET_BACKEND", "vllm")
        self.pylet_gpu_count: int = int(os.getenv("PYLET_GPU_COUNT", "1"))
        self.pylet_deploy_timeout: float = float(
            os.getenv("PYLET_DEPLOY_TIMEOUT", "300.0")
        )
        self.pylet_drain_timeout: float = float(
            os.getenv("PYLET_DRAIN_TIMEOUT", "30.0")
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

        # PyLet validation
        if self.pylet_enabled:
            if not self.pylet_head_url:
                error_msg = "PYLET_HEAD_URL is required when PYLET_ENABLED=true"
                logger.error(f"Configuration validation failed: {error_msg}")
                raise ValueError(error_msg)

            if self.pylet_backend not in ("vllm", "sglang"):
                error_msg = f"PYLET_BACKEND must be 'vllm' or 'sglang', got {self.pylet_backend}"
                logger.error(f"Configuration validation failed: {error_msg}")
                raise ValueError(error_msg)

            if self.pylet_gpu_count <= 0:
                error_msg = f"PYLET_GPU_COUNT must be positive, got {self.pylet_gpu_count}"
                logger.error(f"Configuration validation failed: {error_msg}")
                raise ValueError(error_msg)

            if self.pylet_deploy_timeout <= 0:
                error_msg = f"PYLET_DEPLOY_TIMEOUT must be positive, got {self.pylet_deploy_timeout}"
                logger.error(f"Configuration validation failed: {error_msg}")
                raise ValueError(error_msg)


# Global configuration instance
config = PlannerConfig()
