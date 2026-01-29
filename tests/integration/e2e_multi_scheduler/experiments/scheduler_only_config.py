"""Configuration for single-scheduler isolation experiments.

This module provides a simplified configuration for testing scheduler behavior
in isolation without planner or PyLet complexity.

Key differences from multi-scheduler experiments:
- No planner (scheduler runs standalone)
- No PyLet (instances launched as direct subprocesses)
- Single scheduler handling a single model type
- Uses llm_slow latency characteristics (~2000ms gamma distribution)

Example:
    from tests.integration.e2e_multi_scheduler.experiments import (
        create_scheduler_only_config,
    )

    config = create_scheduler_only_config(
        num_instances=4,
        total_qps=2.0,
        duration_seconds=60.0,
    )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SchedulerOnlyConfig:
    """Configuration for single-scheduler isolation experiment.

    This configuration defines a minimal setup with:
    - One scheduler (no planner registration)
    - Multiple mock LLM instances (launched as subprocesses)
    - Single model type (llm_slow with ~2000ms gamma distribution)

    Attributes:
        name: Experiment name (used for logs and reports).
        scheduler_port: Port for the scheduler service.
        model_id: Model identifier (determines latency characteristics).
        num_instances: Number of mock LLM instances to launch.
        instance_port_start: Starting port for instances.
        total_qps: Queries per second to generate.
        duration_seconds: Test duration in seconds.
        task_completion_timeout: Timeout for waiting on task completion.
        output_dir: Directory for test outputs (logs, reports).
        log_dir: Directory for log files.
    """

    name: str = "scheduler_only_test"
    scheduler_port: int = 8001
    model_id: str = "llm_slow"
    num_instances: int = 4
    instance_port_start: int = 8101
    total_qps: float = 2.0
    duration_seconds: float = 60.0
    task_completion_timeout: float = 600.0
    output_dir: Path = field(
        default_factory=lambda: Path("./e2e_scheduler_only_results")
    )
    log_dir: Path = field(default_factory=lambda: Path("/tmp/e2e_scheduler_only_logs"))

    @property
    def scheduler_url(self) -> str:
        """Get scheduler URL."""
        return f"http://localhost:{self.scheduler_port}"

    @property
    def instance_ports(self) -> list[int]:
        """Get list of instance ports."""
        return [self.instance_port_start + i for i in range(self.num_instances)]

    def get_instance_endpoint(self, index: int) -> str:
        """Get endpoint URL for an instance.

        Args:
            index: Instance index (0-based).

        Returns:
            Instance endpoint URL.
        """
        port = self.instance_port_start + index
        return f"http://127.0.0.1:{port}"

    def get_instance_id(self, index: int) -> str:
        """Get instance ID for an instance.

        Args:
            index: Instance index (0-based).

        Returns:
            Instance ID string.
        """
        return f"{self.model_id}-instance-{index:02d}"

    @property
    def expected_tasks(self) -> int:
        """Expected number of tasks based on QPS and duration."""
        return int(self.total_qps * self.duration_seconds)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Configuration as dictionary.
        """
        return {
            "name": self.name,
            "scheduler_port": self.scheduler_port,
            "model_id": self.model_id,
            "num_instances": self.num_instances,
            "instance_port_start": self.instance_port_start,
            "total_qps": self.total_qps,
            "duration_seconds": self.duration_seconds,
            "task_completion_timeout": self.task_completion_timeout,
            "expected_tasks": self.expected_tasks,
        }


def create_scheduler_only_config(
    num_instances: int = 4,
    total_qps: float = 2.0,
    duration_seconds: float = 60.0,
    model_id: str = "llm_slow",
    scheduler_port: int = 8001,
    instance_port_start: int = 8101,
    output_dir: Path | None = None,
    log_dir: Path | None = None,
) -> SchedulerOnlyConfig:
    """Create a scheduler-only experiment configuration.

    This factory function creates a configuration for testing scheduler
    behavior in isolation. The default model_id "llm_slow" uses a gamma
    distribution with ~2000ms mean latency.

    Latency characteristics for llm_slow:
    - Distribution: Gamma
    - Mean: 2000ms
    - Shape: 3.0 (bell-like with right tail)
    - Cap: 6000ms (3x mean)

    Args:
        num_instances: Number of mock LLM instances (default: 4).
        total_qps: Queries per second (default: 2.0, suitable for slow model).
        duration_seconds: Test duration (default: 60 seconds).
        model_id: Model identifier (default: "llm_slow").
        scheduler_port: Scheduler port (default: 8001).
        instance_port_start: Starting port for instances (default: 8101).
        output_dir: Output directory (default: ./e2e_scheduler_only_results).
        log_dir: Log directory (default: /tmp/e2e_scheduler_only_logs).

    Returns:
        SchedulerOnlyConfig instance.

    Example:
        # Default configuration: 4 instances, 2 QPS, 60 seconds
        config = create_scheduler_only_config()

        # Custom configuration: 8 instances, higher load
        config = create_scheduler_only_config(
            num_instances=8,
            total_qps=4.0,
            duration_seconds=120.0,
        )
    """
    return SchedulerOnlyConfig(
        name=f"scheduler_only_{model_id}_{num_instances}inst_{total_qps}qps",
        scheduler_port=scheduler_port,
        model_id=model_id,
        num_instances=num_instances,
        instance_port_start=instance_port_start,
        total_qps=total_qps,
        duration_seconds=duration_seconds,
        task_completion_timeout=max(600.0, duration_seconds * 5),
        output_dir=output_dir or Path(f"./e2e_scheduler_only_results/{model_id}"),
        log_dir=log_dir or Path(f"/tmp/e2e_scheduler_only_logs/{model_id}"),
    )
