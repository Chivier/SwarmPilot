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
        # Act only when the objective improves by at least this much.
        self.auto_optimize_change_threshold: float = float(
            os.getenv("AUTO_OPTIMIZE_CHANGE_THRESHOLD", "0.05")
        )
        # Share of machines that may change model in one round.
        self.auto_optimize_change_factor: float = float(
            os.getenv("AUTO_OPTIMIZE_CHANGE_FACTOR", "0.3")
        )
        self.auto_optimize_min_replicas: int = int(
            os.getenv("AUTO_OPTIMIZE_MIN_REPLICAS", "1")
        )
        self.auto_optimize_algorithm: str = os.getenv(
            "AUTO_OPTIMIZE_ALGORITHM", "integer_programming"
        )
        # "tail" sizes models by arrival rate, q0.9 backlog and calls
        # between models; "count" by outstanding tasks only.
        self.auto_optimize_scaler: str = os.getenv(
            "AUTO_OPTIMIZE_SCALER", "tail"
        ).lower()

        # Elastic pool (tail scaler): grow into free GPUs when the models
        # need more than TARGET_CONCURRENCY tasks in flight per instance,
        # shrink after SCALE_DOWN_ROUNDS rounds that want fewer.
        self.auto_optimize_elastic: bool = os.getenv(
            "AUTO_OPTIMIZE_ELASTIC", "true"
        ).lower() in ("true", "1", "yes")
        self.auto_optimize_target_concurrency: float = float(
            os.getenv("AUTO_OPTIMIZE_TARGET_CONCURRENCY", "64")
        )
        self.auto_optimize_min_instances: int = int(
            os.getenv("AUTO_OPTIMIZE_MIN_INSTANCES", "1")
        )
        # 0: no limit but the free GPUs.
        self.auto_optimize_max_instances: int = int(
            os.getenv("AUTO_OPTIMIZE_MAX_INSTANCES", "0")
        )
        self.auto_optimize_scale_down_rounds: int = int(
            os.getenv("AUTO_OPTIMIZE_SCALE_DOWN_ROUNDS", "3")
        )
        # Weigh machines by hardware speed learned from execution times
        # (tail scaler): name which instances move, size the pool in
        # capacity, and serve arrivals before backlog when short.
        self.auto_optimize_hardware_aware: bool = os.getenv(
            "AUTO_OPTIMIZE_HARDWARE_AWARE", "true"
        ).lower() in ("true", "1", "yes")

        # PyLet configuration
        self.pylet_enabled: bool = os.getenv(
            "PYLET_ENABLED", "false"
        ).lower() in (
            "true",
            "1",
            "yes",
        )
        self.pylet_head_url: str | None = os.getenv("PYLET_HEAD_URL")
        self.pylet_backend: str = os.getenv("PYLET_BACKEND", "vllm")
        self.pylet_gpu_count: int = int(os.getenv("PYLET_GPU_COUNT", "1"))
        self.pylet_cpu_count: int = int(os.getenv("PYLET_CPU_COUNT", "1"))
        self.pylet_deploy_timeout: float = float(
            os.getenv("PYLET_DEPLOY_TIMEOUT", "300.0")
        )
        self.pylet_drain_timeout: float = float(
            os.getenv("PYLET_DRAIN_TIMEOUT", "30.0")
        )
        # Custom command template for testing (overrides backend command)
        # Use $PORT placeholder for auto-allocated port
        # Example: "python dummy_model_server.py"
        self.pylet_custom_command: str | None = os.getenv(
            "PYLET_CUSTOM_COMMAND"
        )
        # Whether to reuse an existing PyLet cluster (skip pylet.init if already initialized)
        self.pylet_reuse_cluster: bool = os.getenv(
            "PYLET_REUSE_CLUSTER", "false"
        ).lower() in ("true", "1", "yes")
        # Per-model deployment overrides as JSON.
        # Example: '{"Qwen/Qwen3-Next-80B": {"gpu_count": 2, "extra_args": "--max-model-len 8192"}}'
        self.pylet_model_configs: dict[str, dict] = {}
        _raw_model_configs = os.getenv("PYLET_MODEL_CONFIGS")
        if _raw_model_configs:
            import json

            self.pylet_model_configs = json.loads(_raw_model_configs)

        # Local PyLet mode — Planner starts its own PyLet cluster
        # as subprocesses (head + workers) and auto-stops on shutdown.
        self.pylet_local_mode: bool = os.getenv(
            "PYLET_LOCAL_MODE", "false"
        ).lower() in ("true", "1", "yes")
        self.pylet_local_port: int = int(os.getenv("PYLET_LOCAL_PORT", "5100"))
        self.pylet_local_num_workers: int = int(
            os.getenv("PYLET_LOCAL_NUM_WORKERS", "1")
        )
        self.pylet_local_cpu_per_worker: int = int(
            os.getenv("PYLET_LOCAL_CPU_PER_WORKER", "8")
        )
        self.pylet_local_gpu_per_worker: int = int(
            os.getenv("PYLET_LOCAL_GPU_PER_WORKER", "4")
        )
        self.pylet_local_worker_port_start: int = int(
            os.getenv("PYLET_LOCAL_WORKER_PORT_START", "5300")
        )
        self.pylet_local_worker_port_gap: int = int(
            os.getenv("PYLET_LOCAL_WORKER_PORT_GAP", "200")
        )
        self.pylet_local_memory_per_worker: int = int(
            os.getenv("PYLET_LOCAL_MEMORY_PER_WORKER", "65536")
        )

        # Simulated PyLet mode — like local mode, but every node of a
        # cluster spec is a simulated worker (swarmpilot-sim) running
        # virtual engines, so no GPUs are needed. Implies local mode.
        self.pylet_sim_mode: bool = os.getenv(
            "PYLET_SIM_MODE", "false"
        ).lower() in ("true", "1", "yes")
        self.pylet_sim_spec: str | None = os.getenv("PYLET_SIM_SPEC")
        self.pylet_sim_time_scale: float = float(
            os.getenv("PYLET_SIM_TIME_SCALE", "1.0")
        )
        self.pylet_sim_port_start: int = int(
            os.getenv("PYLET_SIM_PORT_START", "21000")
        )
        self.pylet_sim_port_gap: int = int(
            os.getenv("PYLET_SIM_PORT_GAP", "100")
        )
        self.pylet_sim_control_port: int = int(
            os.getenv("PYLET_SIM_CONTROL_PORT", "20999")
        )
        if self.pylet_sim_mode:
            self.pylet_local_mode = True

        # Seconds between checks for instances whose engine died (PyLet
        # FAILED); they are removed from their Scheduler. 0 disables.
        self.instance_watch_interval: float = float(
            os.getenv("PLANNER_INSTANCE_WATCH_INTERVAL", "2.0")
        )
        # Deploy a replacement for each instance found dead.
        self.replace_failed_instances: bool = os.getenv(
            "PLANNER_REPLACE_FAILED", "true"
        ).lower() in ("true", "1", "yes")

        # Auto-activate local mode when PyLet is enabled without
        # an explicit head URL, so a local cluster is started.
        if self.pylet_enabled and not self.pylet_head_url:
            self.pylet_local_mode = True
            logger.info(
                "PYLET_ENABLED is set without PYLET_HEAD_URL; "
                "auto-activating PYLET_LOCAL_MODE"
            )

        # Auto-derive PyLet settings when local mode is enabled
        if self.pylet_local_mode:
            self.pylet_enabled = True
            if not self.pylet_head_url:
                self.pylet_head_url = (
                    f"http://localhost:{self.pylet_local_port}"
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

        if not 0 < self.auto_optimize_change_factor <= 1:
            error_msg = (
                "AUTO_OPTIMIZE_CHANGE_FACTOR must be in (0, 1], got "
                f"{self.auto_optimize_change_factor}"
            )
            logger.error(f"Configuration validation failed: {error_msg}")
            raise ValueError(error_msg)

        if self.auto_optimize_scaler not in ("tail", "count"):
            error_msg = (
                "AUTO_OPTIMIZE_SCALER must be 'tail' or 'count', got "
                f"{self.auto_optimize_scaler!r}"
            )
            logger.error(f"Configuration validation failed: {error_msg}")
            raise ValueError(error_msg)

        if self.auto_optimize_target_concurrency <= 0:
            error_msg = (
                "AUTO_OPTIMIZE_TARGET_CONCURRENCY must be positive, got "
                f"{self.auto_optimize_target_concurrency}"
            )
            logger.error(f"Configuration validation failed: {error_msg}")
            raise ValueError(error_msg)

        # PyLet validation
        if self.pylet_enabled:
            if not self.pylet_head_url:
                error_msg = "PYLET_HEAD_URL is required when PYLET_ENABLED=true"
                logger.error(f"Configuration validation failed: {error_msg}")
                raise ValueError(error_msg)

            # Only validate backend if no custom command is provided
            from swarmpilot.engines import adapter_names

            if (
                not self.pylet_custom_command
                and self.pylet_backend not in adapter_names()
            ):
                error_msg = f"PYLET_BACKEND must be one of {adapter_names()}, got {self.pylet_backend}"
                logger.error(f"Configuration validation failed: {error_msg}")
                raise ValueError(error_msg)

            if self.pylet_gpu_count < 0:
                error_msg = f"PYLET_GPU_COUNT must be non-negative, got {self.pylet_gpu_count}"
                logger.error(f"Configuration validation failed: {error_msg}")
                raise ValueError(error_msg)

            if self.pylet_cpu_count <= 0:
                error_msg = f"PYLET_CPU_COUNT must be positive, got {self.pylet_cpu_count}"
                logger.error(f"Configuration validation failed: {error_msg}")
                raise ValueError(error_msg)

            if self.pylet_deploy_timeout <= 0:
                error_msg = f"PYLET_DEPLOY_TIMEOUT must be positive, got {self.pylet_deploy_timeout}"
                logger.error(f"Configuration validation failed: {error_msg}")
                raise ValueError(error_msg)


# Global configuration instance
config = PlannerConfig()
