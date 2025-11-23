#!/usr/bin/env python3
"""
Command-line interface for workflow benchmarks.

Provides simple commands for running experiments:
  - run-text2video-sim: Run Text2Video simulation
  - run-text2video-real: Run Text2Video real cluster mode
  - run-deep-research-sim: Run Deep Research simulation
  - run-deep-research-real: Run Deep Research real cluster mode
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.experiment_runner import ExperimentRunner
from common import configure_logging


def main():
    parser = argparse.ArgumentParser(
        description="Unified Workflow Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Text2Video simulation for 1 minute
  %(prog)s run-text2video-sim --duration 60 --num-workflows 120

  # Run Deep Research simulation with custom QPS
  %(prog)s run-deep-research-sim --qps 1.5 --duration 300

  # Run Text2Video in real cluster mode
  %(prog)s run-text2video-real --qps 2.0 --duration 600
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ========================================================================
    # Text2Video Simulation
    # ========================================================================
    t2v_sim = subparsers.add_parser(
        "run-text2video-sim",
        help="Run Text2Video workflow in simulation mode"
    )
    t2v_sim.add_argument("--qps", type=float, default=2.0, help="Query per second (default: 2.0)")
    t2v_sim.add_argument("--duration", type=int, default=300, help="Duration in seconds (default: 300)")
    t2v_sim.add_argument("--num-workflows", type=int, default=600, help="Number of workflows (default: 600)")
    t2v_sim.add_argument("--max-b-loops", type=int, default=4, help="Max B iterations (default: 4)")
    t2v_sim.add_argument("--output-dir", type=str, default="output", help="Output directory (default: output)")
    t2v_sim.add_argument("--strategies", type=str, help="Comma-separated list of scheduling strategies to test (e.g., min_time,probabilistic,round_robin)")
    t2v_sim.add_argument("--target-quantile", type=float, help="Target quantile for probabilistic strategy (default: 0.9)")
    t2v_sim.add_argument("--quantiles", type=str, help="Comma-separated quantiles for probabilistic (default: 0.1,0.25,0.5,0.75,0.99)")

    # ========================================================================
    # Text2Video Real
    # ========================================================================
    t2v_real = subparsers.add_parser(
        "run-text2video-real",
        help="Run Text2Video workflow in real cluster mode"
    )
    t2v_real.add_argument("--qps", type=float, default=2.0, help="Query per second (default: 2.0)")
    t2v_real.add_argument("--duration", type=int, default=300, help="Duration in seconds (default: 300)")
    t2v_real.add_argument("--num-workflows", type=int, default=100, help="Number of workflows (default: 100)")
    t2v_real.add_argument("--max-b-loops", type=int, default=4, help="Max B iterations (default: 4)")
    t2v_real.add_argument("--output-dir", type=str, default="output", help="Output directory (default: output)")
    t2v_real.add_argument("--scheduler-a-url", type=str, help="Scheduler A URL")
    t2v_real.add_argument("--scheduler-b-url", type=str, help="Scheduler B URL")
    t2v_real.add_argument("--strategies", type=str, help="Comma-separated list of scheduling strategies to test (e.g., min_time,probabilistic,round_robin)")
    t2v_real.add_argument("--target-quantile", type=float, help="Target quantile for probabilistic strategy (default: 0.9)")
    t2v_real.add_argument("--quantiles", type=str, help="Comma-separated quantiles for probabilistic (default: 0.1,0.25,0.5,0.75,0.99)")

    # ========================================================================
    # Deep Research Simulation
    # ========================================================================
    dr_sim = subparsers.add_parser(
        "run-deep-research-sim",
        help="Run Deep Research workflow in simulation mode"
    )
    dr_sim.add_argument("--qps", type=float, default=1.0, help="Query per second (default: 1.0)")
    dr_sim.add_argument("--duration", type=int, default=600, help="Duration in seconds (default: 600)")
    dr_sim.add_argument("--num-workflows", type=int, default=600, help="Number of workflows (default: 600)")
    dr_sim.add_argument("--fanout-count", type=int, default=3, help="Fanout count (default: 3)")
    dr_sim.add_argument("--output-dir", type=str, default="output", help="Output directory (default: output)")
    dr_sim.add_argument("--strategies", type=str, help="Comma-separated list of scheduling strategies to test (e.g., min_time,probabilistic,round_robin)")
    dr_sim.add_argument("--target-quantile", type=float, help="Target quantile for probabilistic strategy (default: 0.9)")
    dr_sim.add_argument("--quantiles", type=str, help="Comma-separated quantiles for probabilistic (default: 0.1,0.25,0.5,0.75,0.99)")

    # ========================================================================
    # Deep Research Real
    # ========================================================================
    dr_real = subparsers.add_parser(
        "run-deep-research-real",
        help="Run Deep Research workflow in real cluster mode"
    )
    dr_real.add_argument("--qps", type=float, default=1.0, help="Query per second (default: 1.0)")
    dr_real.add_argument("--duration", type=int, default=600, help="Duration in seconds (default: 600)")
    dr_real.add_argument("--num-workflows", type=int, default=100, help="Number of workflows (default: 100)")
    dr_real.add_argument("--fanout-count", type=int, default=3, help="Fanout count (default: 3)")
    dr_real.add_argument("--output-dir", type=str, default="output", help="Output directory (default: output)")
    dr_real.add_argument("--scheduler-a-url", type=str, help="Scheduler A URL")
    dr_real.add_argument("--scheduler-b-url", type=str, help="Scheduler B URL")
    dr_real.add_argument("--strategies", type=str, help="Comma-separated list of scheduling strategies to test (e.g., min_time,probabilistic,round_robin)")
    dr_real.add_argument("--target-quantile", type=float, help="Target quantile for probabilistic strategy (default: 0.9)")
    dr_real.add_argument("--quantiles", type=str, help="Comma-separated quantiles for probabilistic (default: 0.1,0.25,0.5,0.75,0.99)")

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    logger = configure_logging(level=logging.INFO)

    # Create runner
    runner = ExperimentRunner(logger=logger)

    # Execute command
    try:
        if args.command == "run-text2video-sim":
            kwargs = {}
            if args.strategies:
                kwargs["STRATEGIES"] = args.strategies
            if args.target_quantile:
                kwargs["TARGET_QUANTILE"] = str(args.target_quantile)
            if args.quantiles:
                kwargs["QUANTILES"] = args.quantiles

            result = runner.run_text2video_simulation(
                qps=args.qps,
                duration=args.duration,
                num_workflows=args.num_workflows,
                max_b_loops=args.max_b_loops,
                output_dir=args.output_dir,
                **kwargs
            )

        elif args.command == "run-text2video-real":
            kwargs = {}
            if args.scheduler_a_url:
                kwargs["SCHEDULER_A_URL"] = args.scheduler_a_url
            if args.scheduler_b_url:
                kwargs["SCHEDULER_B_URL"] = args.scheduler_b_url
            if args.strategies:
                kwargs["STRATEGIES"] = args.strategies
            if args.target_quantile:
                kwargs["TARGET_QUANTILE"] = str(args.target_quantile)
            if args.quantiles:
                kwargs["QUANTILES"] = args.quantiles

            result = runner.run_text2video_real(
                qps=args.qps,
                duration=args.duration,
                num_workflows=args.num_workflows,
                max_b_loops=args.max_b_loops,
                output_dir=args.output_dir,
                **kwargs
            )

        elif args.command == "run-deep-research-sim":
            kwargs = {}
            if args.strategies:
                kwargs["STRATEGIES"] = args.strategies
            if args.target_quantile:
                kwargs["TARGET_QUANTILE"] = str(args.target_quantile)
            if args.quantiles:
                kwargs["QUANTILES"] = args.quantiles

            result = runner.run_deep_research_simulation(
                qps=args.qps,
                duration=args.duration,
                num_workflows=args.num_workflows,
                fanout_count=args.fanout_count,
                output_dir=args.output_dir,
                **kwargs
            )

        elif args.command == "run-deep-research-real":
            kwargs = {}
            if args.scheduler_a_url:
                kwargs["SCHEDULER_A_URL"] = args.scheduler_a_url
            if args.scheduler_b_url:
                kwargs["SCHEDULER_B_URL"] = args.scheduler_b_url
            if args.strategies:
                kwargs["STRATEGIES"] = args.strategies
            if args.target_quantile:
                kwargs["TARGET_QUANTILE"] = str(args.target_quantile)
            if args.quantiles:
                kwargs["QUANTILES"] = args.quantiles

            result = runner.run_deep_research_real(
                qps=args.qps,
                duration=args.duration,
                num_workflows=args.num_workflows,
                fanout_count=args.fanout_count,
                output_dir=args.output_dir,
                **kwargs
            )

        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)

        # Print results
        print("\n" + "=" * 80)
        print("Experiment Complete!")
        print("=" * 80)
        print(f"Success: {result['success']}")
        print(f"Elapsed Time: {result['elapsed_time']:.2f}s")
        print(f"Metrics File: {result['metrics_path']}")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
