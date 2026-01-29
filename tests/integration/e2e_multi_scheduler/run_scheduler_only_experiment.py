#!/usr/bin/env python3
"""Run Single-Scheduler Isolation Experiment.

This script runs an experiment that tests scheduler behavior in isolation:
- Single scheduler (no planner registration)
- Mock LLM instances launched as direct subprocesses
- All instances register with the single scheduler
- Uses llm_slow latency characteristics (~2000ms gamma distribution)

This is useful for testing:
- Instance registration flow
- Task dispatch to workers
- Work stealing between instances
- Queue management under load
- Scheduler performance without planner overhead

Usage:
    # Default: 4 instances, 2 QPS, 60 seconds
    python -m tests.integration.e2e_multi_scheduler.run_scheduler_only_experiment

    # Custom configuration
    python -m tests.integration.e2e_multi_scheduler.run_scheduler_only_experiment \\
        --instances 8 --qps 4 --duration 120

    # Different model (llm_fast for faster tests)
    python -m tests.integration.e2e_multi_scheduler.run_scheduler_only_experiment \\
        --instances 4 --qps 10 --duration 30 --model llm_fast
"""

import argparse
import asyncio
import sys
from pathlib import Path

from .experiments.scheduler_only_config import create_scheduler_only_config
from .scheduler_only_orchestrator import SchedulerOnlyOrchestrator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run single-scheduler isolation experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default configuration (4 instances, 2 QPS, 60s)
    python -m tests.integration.e2e_multi_scheduler.run_scheduler_only_experiment

    # High load test
    python -m tests.integration.e2e_multi_scheduler.run_scheduler_only_experiment \\
        --instances 8 --qps 4 --duration 120

    # Fast model for quick testing
    python -m tests.integration.e2e_multi_scheduler.run_scheduler_only_experiment \\
        --instances 4 --qps 10 --duration 30 --model llm_fast

Model latency characteristics:
    - llm_fast:   Exponential dist, mean ~100ms  (quick tests)
    - llm_medium: Log-normal dist, mean ~500ms   (moderate load)
    - llm_slow:   Gamma dist, mean ~2000ms       (realistic LLM simulation)
        """,
    )

    parser.add_argument(
        "--instances",
        type=int,
        default=4,
        help="Number of mock LLM instances (default: 4)",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=2.0,
        help="Queries per second (default: 2.0, suitable for llm_slow)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Test duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llm_slow",
        choices=["llm_fast", "llm_medium", "llm_slow"],
        help="Model type (default: llm_slow)",
    )
    parser.add_argument(
        "--scheduler-port",
        type=int,
        default=8001,
        help="Scheduler port (default: 8001)",
    )
    parser.add_argument(
        "--instance-port-start",
        type=int,
        default=8101,
        help="Starting port for instances (default: 8101)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reports (default: ./e2e_scheduler_only_results/<model>)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Log directory (default: /tmp/e2e_scheduler_only_logs/<model>)",
    )

    return parser.parse_args()


async def main() -> int:
    """Run the experiment.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()

    # Create configuration
    config = create_scheduler_only_config(
        num_instances=args.instances,
        total_qps=args.qps,
        duration_seconds=args.duration,
        model_id=args.model,
        scheduler_port=args.scheduler_port,
        instance_port_start=args.instance_port_start,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
    )

    print(f"\n{'='*60}")
    print("SCHEDULER ISOLATION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Model:     {config.model_id}")
    print(f"Instances: {config.num_instances}")
    print(f"QPS:       {config.total_qps}")
    print(f"Duration:  {config.duration_seconds}s")
    print(f"Expected:  ~{config.expected_tasks} tasks")
    print(f"Output:    {config.output_dir}")
    print(f"{'='*60}\n")

    # Run experiment
    orchestrator = SchedulerOnlyOrchestrator(config)
    result = await orchestrator.run()

    if result["success"]:
        print(f"\n{'='*60}")
        print("EXPERIMENT SUCCEEDED")
        print(f"{'='*60}")
        summary = result["summary"]
        print(f"Tasks submitted: {summary['tasks_submitted']}")
        print(f"Tasks completed: {summary['tasks_completed']}")
        print(f"Tasks failed:    {summary['tasks_failed']}")
        print(f"Actual QPS:      {summary['actual_qps']:.2f}")
        print(f"\nReport: {result['report_md']}")

        if result["problems"]:
            print(f"\nProblems found: {len(result['problems'])}")
            for problem in result["problems"]:
                print(f"  - {problem}")

        print(f"{'='*60}\n")
        return 0
    else:
        print(f"\n{'='*60}")
        print("EXPERIMENT FAILED")
        print(f"{'='*60}")
        print(f"Error: {result.get('error', 'Unknown error')}")
        print(f"{'='*60}\n")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
