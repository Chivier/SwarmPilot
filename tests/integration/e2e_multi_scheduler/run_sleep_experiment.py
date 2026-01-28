#!/usr/bin/env python3
"""Convenience script for running sleep model multi-scheduler experiment.

This is a simplified entry point for the sleep model experiment with
sensible defaults for quick testing.

Usage:
    # Run with defaults (10 workers, 5 QPS, 60s)
    python -m tests.integration.e2e_multi_scheduler.run_sleep_experiment

    # Run with custom settings
    python -m tests.integration.e2e_multi_scheduler.run_sleep_experiment \
        --qps 10 --duration 120

    # Run high load variant
    python -m tests.integration.e2e_multi_scheduler.run_sleep_experiment \
        --high-load
"""

import argparse
import asyncio
import signal
import sys

from loguru import logger

from .orchestrator import MultiSchedulerOrchestrator
from .experiments.sleep_model_config import (
    create_sleep_model_config,
    create_high_load_sleep_config,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sleep Model Multi-Scheduler Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--qps",
        type=float,
        default=5.0,
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
        default=10,
        help="Number of PyLet workers",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./e2e_multi_scheduler_results/sleep_model",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--high-load",
        action="store_true",
        help="Use high-load configuration (5 schedulers, 20 instances)",
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
    if args.high_load:
        config = create_high_load_sleep_config(
            num_workers=max(args.workers, 20),
            total_qps=max(args.qps, 20.0),
            duration_seconds=args.duration,
        )
    else:
        config = create_sleep_model_config(
            num_workers=args.workers,
            total_qps=args.qps,
            duration_seconds=args.duration,
            output_dir=args.output_dir,
            reuse_cluster=args.reuse_cluster,
        )

    logger.info(f"Sleep Model Experiment: {config.name}")
    logger.info(f"  Schedulers: {len(config.models)}")
    logger.info(f"  Total instances: {config.total_instances}")
    logger.info(f"  QPS: {config.total_qps}")
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
        logger.success("Sleep model experiment completed successfully")
        return 0
    else:
        logger.error(f"Experiment failed: {result.get('error')}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
