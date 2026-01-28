#!/usr/bin/env python3
"""Generic CLI runner for multi-scheduler E2E experiments.

This script provides a command-line interface for running multi-scheduler
experiments with configurable parameters.

Usage:
    # Run sleep model experiment with defaults
    python -m tests.integration.e2e_multi_scheduler.run_experiment \
        --experiment sleep_model

    # Run with custom parameters
    python -m tests.integration.e2e_multi_scheduler.run_experiment \
        --experiment sleep_model \
        --workers 10 \
        --qps 5 \
        --duration 60 \
        --output-dir ./results

    # Run mock LLM experiment
    python -m tests.integration.e2e_multi_scheduler.run_experiment \
        --experiment mock_llm \
        --workers 20 \
        --qps 10 \
        --duration 120

Available experiments:
    - sleep_model: Basic sleep model test (10 instances, 3 schedulers)
    - sleep_high_load: High load sleep test (20 instances, 5 schedulers)
    - mock_llm: Mock LLM test (20 instances, 3 schedulers)
    - llm_scaling: LLM scaling test (30 instances, 4 schedulers)
"""

import argparse
import asyncio
import signal
import sys
from pathlib import Path

from loguru import logger

from .config import ExperimentConfig
from .orchestrator import MultiSchedulerOrchestrator
from .experiments.sleep_model_config import (
    create_sleep_model_config,
    create_high_load_sleep_config,
)
from .experiments.mock_llm_config import (
    create_mock_llm_config,
    create_llm_scaling_config,
)


# Registry of available experiments
EXPERIMENT_REGISTRY = {
    "sleep_model": create_sleep_model_config,
    "sleep_high_load": create_high_load_sleep_config,
    "mock_llm": create_mock_llm_config,
    "llm_scaling": create_llm_scaling_config,
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Scheduler E2E Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        required=True,
        choices=list(EXPERIMENT_REGISTRY.keys()),
        help="Experiment type to run",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of PyLet workers (default: experiment-specific)",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=None,
        help="Total queries per second (default: experiment-specific)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="Test duration in seconds (default: experiment-specific)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for reports",
    )
    parser.add_argument(
        "--log-dir",
        "-l",
        type=str,
        default=None,
        help="Directory for log files",
    )
    parser.add_argument(
        "--planner-port",
        type=int,
        default=8003,
        help="Port for planner service (default: 8003)",
    )
    parser.add_argument(
        "--scheduler-port-start",
        type=int,
        default=8010,
        help="Starting port for schedulers (default: 8010)",
    )
    parser.add_argument(
        "--reuse-cluster",
        action="store_true",
        help="Reuse existing PyLet cluster",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Create experiment configuration from CLI arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        ExperimentConfig with overrides applied.
    """
    # Get base config from experiment type
    experiment_fn = EXPERIMENT_REGISTRY[args.experiment]

    # Build kwargs for config function
    kwargs = {}

    if args.workers is not None:
        kwargs["num_workers"] = args.workers
    if args.qps is not None:
        kwargs["total_qps"] = args.qps
    if args.duration is not None:
        kwargs["duration_seconds"] = args.duration
    if args.output_dir is not None:
        kwargs["output_dir"] = Path(args.output_dir)
    if args.log_dir is not None:
        kwargs["log_dir"] = Path(args.log_dir)

    # These are only supported by base configs
    if experiment_fn in (create_sleep_model_config, create_mock_llm_config):
        kwargs["planner_port"] = args.planner_port
        kwargs["scheduler_port_start"] = args.scheduler_port_start
        kwargs["reuse_cluster"] = args.reuse_cluster

    return experiment_fn(**kwargs)


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        )

    # Create configuration
    config = create_config_from_args(args)

    logger.info(f"Starting experiment: {config.name}")
    logger.info(f"  Models: {config.model_ids}")
    logger.info(f"  Total instances: {config.total_instances}")
    logger.info(f"  Total QPS: {config.total_qps}")
    logger.info(f"  Duration: {config.duration_seconds}s")

    # Create orchestrator
    orchestrator = MultiSchedulerOrchestrator(config)

    # Handle signals
    def signal_handler(signum, frame):
        logger.warning(f"Received signal {signum}, shutting down...")
        asyncio.create_task(orchestrator._cleanup())
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run experiment
    result = await orchestrator.run()

    if result["success"]:
        logger.success("Experiment completed successfully")
        logger.info(f"Report: {result['report_md']}")
        return 0
    else:
        logger.error(f"Experiment failed: {result.get('error')}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
