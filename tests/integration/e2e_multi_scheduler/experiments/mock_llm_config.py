"""Mock LLM experiment configuration.

This module provides a pre-configured experiment for testing the
multi-scheduler architecture with mock LLM (vLLM-like) instances.

The mock LLM simulates a language model inference server with
configurable response times, useful for testing realistic scheduling
behavior.

Usage:
    from experiments.mock_llm_config import create_mock_llm_config

    config = create_mock_llm_config(
        num_workers=20,
        total_qps=10.0,
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


def create_mock_llm_config(
    num_workers: int = 20,
    total_qps: float = 10.0,
    duration_seconds: float = 60.0,
    output_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    planner_port: int = 8003,
    scheduler_port_start: int = 8010,
    reuse_cluster: bool = False,
) -> ExperimentConfig:
    """Create mock LLM experiment configuration.

    This configuration creates 3 model types simulating different LLM sizes:
    - llm_fast: 10 instances (small model, fast inference)
    - llm_medium: 6 instances (medium model)
    - llm_slow: 4 instances (large model, slow inference)

    QPS is distributed based on expected throughput (fast models get more).

    Args:
        num_workers: Number of PyLet workers (should be >= total instances).
        total_qps: Total queries per second across all models.
        duration_seconds: Test duration in seconds.
        output_dir: Directory for test outputs.
        log_dir: Directory for log files.
        planner_port: Port for planner service.
        scheduler_port_start: Starting port for schedulers.
        reuse_cluster: Whether to reuse existing PyLet cluster.

    Returns:
        ExperimentConfig for mock LLM test.
    """
    # Default directories
    if output_dir is None:
        output_dir = Path("./e2e_multi_scheduler_results/mock_llm")
    if log_dir is None:
        log_dir = Path("/tmp/e2e_multi_scheduler_logs/mock_llm")

    # Build path to mock vLLM server script
    mock_llm_script = (
        PROJECT_ROOT
        / "tests"
        / "integration"
        / "e2e_llm_cluster"
        / "mock_vllm_server.py"
    )

    # Model configurations with dedicated scheduler ports
    # Higher QPS ratio for faster models (they can handle more throughput)
    models = [
        ModelConfig(
            model_id="llm_fast",
            instance_count=10,
            scheduler_port=scheduler_port_start,
            qps_ratio=2.0,  # Fast model gets 2x QPS share
        ),
        ModelConfig(
            model_id="llm_medium",
            instance_count=6,
            scheduler_port=scheduler_port_start + 1,
            qps_ratio=1.0,
        ),
        ModelConfig(
            model_id="llm_slow",
            instance_count=4,
            scheduler_port=scheduler_port_start + 2,
            qps_ratio=0.5,  # Slow model gets 0.5x QPS share
        ),
    ]

    return ExperimentConfig(
        name="mock_llm_multi_scheduler",
        cluster=PyLetClusterConfig(
            num_workers=num_workers,
            gpu_per_worker=0,  # Mock LLM runs on CPU for testing
            cpu_per_worker=1,
            reuse_cluster=reuse_cluster,
        ),
        models=models,
        planner=PlannerConfig(
            port=planner_port,
            pylet_custom_command=f"python {mock_llm_script}",
        ),
        total_qps=total_qps,
        duration_seconds=duration_seconds,
        sleep_time_range=(0.5, 3.0),  # Longer times for LLM simulation
        task_completion_timeout=600.0,
        output_dir=Path(output_dir),
        log_dir=Path(log_dir),
    )


def create_llm_scaling_config(
    total_instances: int = 30,
    total_qps: float = 15.0,
    duration_seconds: float = 180.0,
) -> ExperimentConfig:
    """Create LLM scaling experiment.

    This configuration tests scaling behavior with more model variants:
    - 4 model sizes (tiny, small, medium, large)
    - Variable instance counts based on model size
    - QPS distributed by throughput capacity

    Args:
        total_instances: Total instances across all models.
        total_qps: Total queries per second.
        duration_seconds: Test duration.

    Returns:
        ExperimentConfig for LLM scaling test.
    """
    mock_llm_script = (
        PROJECT_ROOT
        / "tests"
        / "integration"
        / "e2e_llm_cluster"
        / "mock_vllm_server.py"
    )

    # Distribute instances: more for smaller/faster models
    # tiny: 40%, small: 30%, medium: 20%, large: 10%
    tiny_count = int(total_instances * 0.40)
    small_count = int(total_instances * 0.30)
    medium_count = int(total_instances * 0.20)
    large_count = total_instances - tiny_count - small_count - medium_count

    models = [
        ModelConfig(
            model_id="llm_tiny",
            instance_count=tiny_count,
            scheduler_port=8010,
            qps_ratio=3.0,  # Highest throughput
        ),
        ModelConfig(
            model_id="llm_small",
            instance_count=small_count,
            scheduler_port=8011,
            qps_ratio=2.0,
        ),
        ModelConfig(
            model_id="llm_medium",
            instance_count=medium_count,
            scheduler_port=8012,
            qps_ratio=1.0,
        ),
        ModelConfig(
            model_id="llm_large",
            instance_count=large_count,
            scheduler_port=8013,
            qps_ratio=0.5,  # Lowest throughput
        ),
    ]

    return ExperimentConfig(
        name="llm_scaling_test",
        cluster=PyLetClusterConfig(
            num_workers=total_instances + 5,  # Some headroom
            gpu_per_worker=0,
            cpu_per_worker=1,
            reuse_cluster=False,
        ),
        models=models,
        planner=PlannerConfig(
            port=8003,
            pylet_custom_command=f"python {mock_llm_script}",
        ),
        total_qps=total_qps,
        duration_seconds=duration_seconds,
        sleep_time_range=(0.3, 5.0),  # Wide range for variety
        task_completion_timeout=900.0,
        output_dir=Path("./e2e_multi_scheduler_results/llm_scaling"),
        log_dir=Path("/tmp/e2e_multi_scheduler_logs/llm_scaling"),
    )
