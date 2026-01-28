"""Configuration dataclasses for multi-scheduler E2E experiments.

This module defines the configuration structure for running experiments
with multiple schedulers, each handling a specific model type.

Example:
    config = ExperimentConfig(
        name="sleep_model_test",
        models=[
            ModelConfig("sleep_a", instance_count=4, scheduler_port=8010),
            ModelConfig("sleep_b", instance_count=3, scheduler_port=8011),
        ],
        planner=PlannerConfig(port=8003),
        cluster=PyLetClusterConfig(num_workers=10),
        total_qps=5.0,
        duration_seconds=60.0,
    )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PyLetClusterConfig:
    """Configuration for PyLet cluster (head + workers).

    Attributes:
        num_workers: Number of PyLet worker nodes.
        pylet_head_port: Port for PyLet head node.
        worker_port_gap: Gap between worker ports (for instance endpoints).
        gpu_per_worker: GPUs per worker (0 for CPU-only testing).
        cpu_per_worker: CPUs per worker.
        reuse_cluster: Whether to reuse an existing cluster.
    """

    num_workers: int = 10
    pylet_head_port: int = 6380
    worker_port_gap: int = 1
    gpu_per_worker: int = 0
    cpu_per_worker: int = 1
    reuse_cluster: bool = False


@dataclass
class ModelConfig:
    """Configuration for a single model type and its scheduler.

    Each model gets its own scheduler process. The scheduler registers
    with the planner using SCHEDULER_MODEL_ID and receives tasks
    routed by model_id.

    Attributes:
        model_id: Unique identifier for this model type.
        instance_count: Number of instances to deploy for this model.
        scheduler_port: Port for this model's scheduler.
        qps_ratio: Relative QPS share for this model (1.0 = equal share).
        metadata: Additional metadata for scheduler registration.
    """

    model_id: str
    instance_count: int = 1
    scheduler_port: int = 8010
    qps_ratio: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def scheduler_url(self) -> str:
        """Get scheduler URL for this model."""
        return f"http://localhost:{self.scheduler_port}"


@dataclass
class PlannerConfig:
    """Configuration for the planner service.

    The planner manages the scheduler registry and deploys instances
    via PyLet. It looks up scheduler URLs from the registry when
    deploying instances.

    Attributes:
        port: Port for the planner service.
        pylet_custom_command: Custom command for PyLet instance deployment.
        pylet_backend: PyLet backend type (e.g., "local", "kubernetes").
        pylet_deploy_timeout: Timeout for PyLet deployment operations.
        pylet_drain_timeout: Timeout for draining instances.
    """

    port: int = 8003
    pylet_custom_command: str = ""
    pylet_backend: str = "local"
    pylet_deploy_timeout: float = 60.0
    pylet_drain_timeout: float = 30.0

    @property
    def planner_url(self) -> str:
        """Get planner URL."""
        return f"http://localhost:{self.port}"


@dataclass
class ExperimentConfig:
    """Complete configuration for a multi-scheduler experiment.

    An experiment specifies:
    - PyLet cluster configuration
    - Multiple model configurations (each with its own scheduler)
    - Planner configuration
    - Workload parameters (QPS, duration)

    Attributes:
        name: Experiment name (used for logs and reports).
        cluster: PyLet cluster configuration.
        models: List of model configurations (one scheduler per model).
        planner: Planner service configuration.
        total_qps: Total queries per second across all models.
        duration_seconds: Test duration in seconds.
        sleep_time_range: Range for sleep time generation (min, max).
        task_completion_timeout: Timeout for waiting on task completion.
        output_dir: Directory for test outputs (logs, reports).
        log_dir: Directory for log files.
    """

    name: str
    cluster: PyLetClusterConfig
    models: list[ModelConfig]
    planner: PlannerConfig
    total_qps: float = 5.0
    duration_seconds: float = 60.0
    sleep_time_range: tuple[float, float] = (0.1, 1.0)
    task_completion_timeout: float = 300.0
    output_dir: Path = field(
        default_factory=lambda: Path("./e2e_multi_scheduler_results")
    )
    log_dir: Path = field(default_factory=lambda: Path("/tmp/e2e_multi_scheduler_logs"))

    @property
    def total_instances(self) -> int:
        """Total number of instances across all models."""
        return sum(m.instance_count for m in self.models)

    @property
    def model_ids(self) -> list[str]:
        """List of all model IDs."""
        return [m.model_id for m in self.models]

    @property
    def scheduler_urls(self) -> dict[str, str]:
        """Mapping of model_id to scheduler URL."""
        return {m.model_id: m.scheduler_url for m in self.models}

    @property
    def model_distribution(self) -> dict[str, int]:
        """Mapping of model_id to instance count."""
        return {m.model_id: m.instance_count for m in self.models}

    def get_model_config(self, model_id: str) -> ModelConfig | None:
        """Get configuration for a specific model.

        Args:
            model_id: Model identifier.

        Returns:
            ModelConfig or None if not found.
        """
        for m in self.models:
            if m.model_id == model_id:
                return m
        return None

    def get_qps_distribution(self) -> dict[str, float]:
        """Calculate QPS per model based on ratios.

        Returns:
            Dict mapping model_id to QPS.
        """
        total_ratio = sum(m.qps_ratio for m in self.models)
        return {
            m.model_id: self.total_qps * (m.qps_ratio / total_ratio)
            for m in self.models
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Configuration as dictionary.
        """
        return {
            "name": self.name,
            "cluster": {
                "num_workers": self.cluster.num_workers,
                "pylet_head_port": self.cluster.pylet_head_port,
                "gpu_per_worker": self.cluster.gpu_per_worker,
                "cpu_per_worker": self.cluster.cpu_per_worker,
            },
            "models": [
                {
                    "model_id": m.model_id,
                    "instance_count": m.instance_count,
                    "scheduler_port": m.scheduler_port,
                    "qps_ratio": m.qps_ratio,
                }
                for m in self.models
            ],
            "planner": {
                "port": self.planner.port,
                "pylet_custom_command": self.planner.pylet_custom_command,
            },
            "total_qps": self.total_qps,
            "duration_seconds": self.duration_seconds,
            "sleep_time_range": self.sleep_time_range,
            "total_instances": self.total_instances,
        }
