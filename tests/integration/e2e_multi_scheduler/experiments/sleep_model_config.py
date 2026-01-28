"""Sleep model experiment configuration.

This module provides a pre-configured experiment for testing the
multi-scheduler architecture with sleep model instances.

The sleep model is a simple model that sleeps for a configurable
duration, useful for testing scheduling behavior without actual
compute workload.

Usage:
    from experiments.sleep_model_config import create_sleep_model_config

    config = create_sleep_model_config(
        num_workers=10,
        total_qps=5.0,
        duration_seconds=60.0,
    )
"""

from pathlib import Path

from ..config import (
    ExperimentConfig,
    ModelConfig,
    PlannerConfig,
    PyLetClusterConfig,
)

# Project root for constructing paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent


def create_sleep_model_config(
    num_workers: int = 10,
    total_qps: float = 5.0,
    duration_seconds: float = 60.0,
    output_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    planner_port: int = 8003,
    scheduler_port_start: int = 8010,
    reuse_cluster: bool = False,
) -> ExperimentConfig:
    """Create sleep model experiment configuration.

    This configuration creates 3 model types with different instance counts:
    - sleep_model_a: 4 instances (40% capacity)
    - sleep_model_b: 3 instances (30% capacity)
    - sleep_model_c: 3 instances (30% capacity)

    Each model gets its own scheduler, and QPS is distributed equally.

    Args:
        num_workers: Number of PyLet workers (should be >= total instances).
        total_qps: Total queries per second across all models.
        duration_seconds: Test duration in seconds.
        output_dir: Directory for test outputs (default: ./e2e_multi_scheduler_results).
        log_dir: Directory for log files (default: /tmp/e2e_multi_scheduler_logs).
        planner_port: Port for planner service.
        scheduler_port_start: Starting port for schedulers.
        reuse_cluster: Whether to reuse existing PyLet cluster.

    Returns:
        ExperimentConfig for sleep model test.
    """
    # Default directories
    if output_dir is None:
        output_dir = Path("./e2e_multi_scheduler_results/sleep_model")
    if log_dir is None:
        log_dir = Path("/tmp/e2e_multi_scheduler_logs/sleep_model")

    # Build path to sleep model script
    sleep_model_script = (
        PROJECT_ROOT
        / "tests"
        / "integration"
        / "e2e_pylet_benchmark"
        / "pylet_sleep_model.py"
    )

    # Model configurations with dedicated scheduler ports
    models = [
        ModelConfig(
            model_id="sleep_model_a",
            instance_count=4,
            scheduler_port=scheduler_port_start,
            qps_ratio=1.0,  # Equal QPS share
        ),
        ModelConfig(
            model_id="sleep_model_b",
            instance_count=3,
            scheduler_port=scheduler_port_start + 1,
            qps_ratio=1.0,
        ),
        ModelConfig(
            model_id="sleep_model_c",
            instance_count=3,
            scheduler_port=scheduler_port_start + 2,
            qps_ratio=1.0,
        ),
    ]

    return ExperimentConfig(
        name="sleep_model_multi_scheduler",
        cluster=PyLetClusterConfig(
            num_workers=num_workers,
            gpu_per_worker=0,  # CPU only for sleep model
            cpu_per_worker=1,
            reuse_cluster=reuse_cluster,
        ),
        models=models,
        planner=PlannerConfig(
            port=planner_port,
            pylet_custom_command=f"python {sleep_model_script}",
        ),
        total_qps=total_qps,
        duration_seconds=duration_seconds,
        sleep_time_range=(0.1, 1.0),
        task_completion_timeout=300.0,
        output_dir=Path(output_dir),
        log_dir=Path(log_dir),
    )


def create_high_load_sleep_config(
    num_workers: int = 20,
    total_qps: float = 20.0,
    duration_seconds: float = 120.0,
) -> ExperimentConfig:
    """Create high-load sleep model experiment.

    This configuration tests the system under higher load:
    - 5 model types
    - 20 total instances
    - 20 QPS total

    Args:
        num_workers: Number of PyLet workers.
        total_qps: Total queries per second.
        duration_seconds: Test duration.

    Returns:
        ExperimentConfig for high-load test.
    """
    sleep_model_script = (
        PROJECT_ROOT
        / "tests"
        / "integration"
        / "e2e_pylet_benchmark"
        / "pylet_sleep_model.py"
    )

    models = [
        ModelConfig(
            model_id="sleep_fast",
            instance_count=6,
            scheduler_port=8010,
            qps_ratio=2.0,  # Higher QPS share
        ),
        ModelConfig(
            model_id="sleep_medium_a",
            instance_count=4,
            scheduler_port=8011,
            qps_ratio=1.0,
        ),
        ModelConfig(
            model_id="sleep_medium_b",
            instance_count=4,
            scheduler_port=8012,
            qps_ratio=1.0,
        ),
        ModelConfig(
            model_id="sleep_slow_a",
            instance_count=3,
            scheduler_port=8013,
            qps_ratio=0.5,  # Lower QPS share
        ),
        ModelConfig(
            model_id="sleep_slow_b",
            instance_count=3,
            scheduler_port=8014,
            qps_ratio=0.5,
        ),
    ]

    return ExperimentConfig(
        name="sleep_model_high_load",
        cluster=PyLetClusterConfig(
            num_workers=num_workers,
            gpu_per_worker=0,
            cpu_per_worker=1,
            reuse_cluster=False,
        ),
        models=models,
        planner=PlannerConfig(
            port=8003,
            pylet_custom_command=f"python {sleep_model_script}",
        ),
        total_qps=total_qps,
        duration_seconds=duration_seconds,
        sleep_time_range=(0.05, 0.5),  # Shorter sleep times for higher load
        task_completion_timeout=600.0,
        output_dir=Path("./e2e_multi_scheduler_results/sleep_high_load"),
        log_dir=Path("/tmp/e2e_multi_scheduler_logs/sleep_high_load"),
    )
