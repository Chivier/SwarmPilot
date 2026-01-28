#!/usr/bin/env python3
"""Convenience script for running mock LLM multi-scheduler experiment.

This is a simplified entry point for the mock LLM experiment with
sensible defaults for testing LLM-like workloads.

Usage:
    # Run with defaults (20 workers, 10 QPS, 60s)
    python -m tests.integration.e2e_multi_scheduler.run_llm_experiment

    # Run with custom settings
    python -m tests.integration.e2e_multi_scheduler.run_llm_experiment \
        --qps 15 --duration 180

    # Run scaling test variant
    python -m tests.integration.e2e_multi_scheduler.run_llm_experiment \
        --scaling-test
"""

import argparse
import asyncio
import signal
import sys

from loguru import logger

from .orchestrator import MultiSchedulerOrchestrator
from .experiments.mock_llm_config import (
    create_mock_llm_config,
    create_llm_scaling_config,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Mock LLM Multi-Scheduler Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--qps",
        type=float,
        default=10.0,
        help="Target queries per second",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Test duration in seconds",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of PyLet workers",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./e2e_multi_scheduler_results/mock_llm",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--scaling-test",
        action="store_true",
        help="Use scaling test configuration (4 schedulers, 30 instances)",
    )
    parser.add_argument(
        "--reuse-cluster",
        action="store_true",
        help="Reuse existing PyLet cluster",
    )

    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Create configuration
    if args.scaling_test:
        config = create_llm_scaling_config(
            total_instances=30,
            total_qps=max(args.qps, 15.0),
            duration_seconds=args.duration,
        )
    else:
        config = create_mock_llm_config(
            num_workers=args.workers,
            total_qps=args.qps,
            duration_seconds=args.duration,
            output_dir=args.output_dir,
            reuse_cluster=args.reuse_cluster,
        )

    logger.info(f"Mock LLM Experiment: {config.name}")
    logger.info(f"  Schedulers: {len(config.models)}")
    logger.info(f"  Total instances: {config.total_instances}")
    logger.info(f"  QPS: {config.total_qps}")
    logger.info(f"  Duration: {config.duration_seconds}s")
    logger.info("  Model distribution:")
    for model in config.models:
        qps = config.get_qps_distribution()[model.model_id]
        logger.info(
            f"    {model.model_id}: {model.instance_count} instances, {qps:.2f} QPS"
        )

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
        logger.success("Mock LLM experiment completed successfully")
        return 0
    else:
        logger.error(f"Experiment failed: {result.get('error')}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
