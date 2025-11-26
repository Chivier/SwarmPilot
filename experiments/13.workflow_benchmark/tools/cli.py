#!/usr/bin/env python3
"""
Command-line interface for workflow benchmarks.

Provides simple commands for running experiments:
  - run-text2video-sim: Run Text2Video simulation
  - run-text2video-real: Run Text2Video real cluster mode
  - run-deep-research-sim: Run Deep Research simulation
  - run-deep-research-real: Run Deep Research real cluster mode

CLI parameters are unified with the individual test scripts.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.experiment_runner import ExperimentRunner
from common.utils import configure_logging
from common.cli_utils import VALID_STRATEGIES, DEFAULT_STRATEGIES


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common arguments shared by all workflow commands."""
    parser.add_argument(
        "--num-workflows",
        type=int,
        default=10,
        help="Number of workflows to run (default: 10)"
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=2.0,
        help="Queries per second rate (default: 2.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="default",
        help=f"Comma-separated strategies, 'default', or 'all'. "
             f"Default: {DEFAULT_STRATEGIES}. All: {VALID_STRATEGIES}"
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=0.2,
        help="Warmup ratio (0.0-1.0). Warmup workflows = num_workflows * warmup. "
             "E.g., num_workflows=100, warmup=0.2 submits 120 total, first 20 are warmup (default: 0.2)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Maximum experiment duration in seconds (default: 120)"
    )
    parser.add_argument(
        "--portion-stats",
        type=float,
        default=1.0,
        help="Portion of non-warmup workflows to include in statistics (0.0-1.0). "
             "E.g., num_workflows=100, warmup=0.2, portion_stats=0.5 submits 120 total, "
             "statistics include only the first 50 of the 100 non-warmup workflows (default: 1.0)"
    )
    return parser


def main():
    parser = argparse.ArgumentParser(
        description="Unified Workflow Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Default strategies: {', '.join(DEFAULT_STRATEGIES)}
All valid strategies: {', '.join(VALID_STRATEGIES)}

Examples:
  # Run Text2Video simulation with default strategies
  %(prog)s run-text2video-sim --num-workflows 50

  # Run Deep Research simulation with specific strategies
  %(prog)s run-deep-research-sim --qps 1.5 --strategies probabilistic,min_time

  # Run Text2Video in real cluster mode with all strategies
  %(prog)s run-text2video-real --num-workflows 20 --strategies all
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
    add_common_args(t2v_sim)
    t2v_sim.add_argument(
        "--max-b-loops",
        type=int,
        default=3,
        help="Maximum B task iterations (default: 3)"
    )

    # ========================================================================
    # Text2Video Real
    # ========================================================================
    t2v_real = subparsers.add_parser(
        "run-text2video-real",
        help="Run Text2Video workflow in real cluster mode"
    )
    add_common_args(t2v_real)
    t2v_real.add_argument(
        "--max-b-loops",
        type=int,
        default=3,
        help="Maximum B task iterations (default: 3)"
    )

    # ========================================================================
    # Deep Research Simulation
    # ========================================================================
    dr_sim = subparsers.add_parser(
        "run-deep-research-sim",
        help="Run Deep Research workflow in simulation mode"
    )
    add_common_args(dr_sim)
    dr_sim.add_argument(
        "--fanout",
        type=int,
        default=4,
        help="Default fanout count for parallel B tasks (default: 4). "
             "Ignored if --fanout-config is specified."
    )
    dr_sim.add_argument(
        "--fanout-config",
        type=str,
        default=None,
        help="Path to JSON config file for fanout distribution. "
             "Supports: static, uniform, two_peak, four_peak distributions."
    )
    dr_sim.add_argument(
        "--fanout-seed",
        type=int,
        default=None,
        help="Random seed for fanout distribution sampling."
    )

    # ========================================================================
    # Deep Research Real
    # ========================================================================
    dr_real = subparsers.add_parser(
        "run-deep-research-real",
        help="Run Deep Research workflow in real cluster mode"
    )
    add_common_args(dr_real)
    dr_real.add_argument(
        "--fanout",
        type=int,
        default=4,
        help="Default fanout count for parallel B tasks (default: 4). "
             "Ignored if --fanout-config is specified."
    )
    dr_real.add_argument(
        "--fanout-config",
        type=str,
        default=None,
        help="Path to JSON config file for fanout distribution. "
             "Supports: static, uniform, two_peak, four_peak distributions."
    )
    dr_real.add_argument(
        "--fanout-seed",
        type=int,
        default=None,
        help="Random seed for fanout distribution sampling."
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    logger = configure_logging(level="INFO")

    # Create runner
    runner = ExperimentRunner(custom_logger=logger)

    # Execute command
    try:
        if args.command == "run-text2video-sim":
            result = runner.run_text2video_simulation(
                num_workflows=args.num_workflows,
                qps=args.qps,
                seed=args.seed,
                strategies=args.strategies,
                warmup=args.warmup,
                duration=args.duration,
                max_b_loops=args.max_b_loops,
                portion_stats=args.portion_stats,
            )

        elif args.command == "run-text2video-real":
            result = runner.run_text2video_real(
                num_workflows=args.num_workflows,
                qps=args.qps,
                seed=args.seed,
                strategies=args.strategies,
                warmup=args.warmup,
                duration=args.duration,
                max_b_loops=args.max_b_loops,
                portion_stats=args.portion_stats,
            )

        elif args.command == "run-deep-research-sim":
            result = runner.run_deep_research_simulation(
                num_workflows=args.num_workflows,
                qps=args.qps,
                seed=args.seed,
                strategies=args.strategies,
                warmup=args.warmup,
                duration=args.duration,
                fanout=args.fanout,
                fanout_config=args.fanout_config,
                fanout_seed=args.fanout_seed,
                portion_stats=args.portion_stats,
            )

        elif args.command == "run-deep-research-real":
            result = runner.run_deep_research_real(
                num_workflows=args.num_workflows,
                qps=args.qps,
                seed=args.seed,
                strategies=args.strategies,
                warmup=args.warmup,
                duration=args.duration,
                fanout=args.fanout,
                fanout_config=args.fanout_config,
                fanout_seed=args.fanout_seed,
                portion_stats=args.portion_stats,
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
