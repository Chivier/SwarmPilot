#!/usr/bin/env python3
"""
Command-line interface for workflow benchmarks.

Provides simple commands for running experiments:
  - run-text2video-sim: Run Text2Video simulation
  - run-text2video-real: Run Text2Video real cluster mode
  - run-deep-research-sim: Run Deep Research simulation
  - run-deep-research-real: Run Deep Research real cluster mode
  - run-ocr-llm-sim: Run OCR+LLM simulation
  - run-ocr-llm-real: Run OCR+LLM real cluster mode

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

  # Run OCR+LLM simulation
  %(prog)s run-ocr-llm-sim --num-workflows 50 --qps 2.0

  # Run OCR+LLM in real cluster mode with images
  %(prog)s run-ocr-llm-real --num-workflows 20 --image-dir ./images

  # Run Text2Image+Video simulation (A→C→B workflow)
  %(prog)s run-text2image-video-sim --num-workflows 50 --qps 2.0

  # Run Text2Image+Video in real cluster mode
  %(prog)s run-text2image-video-real --num-workflows 20 --resolution 512
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
        help="Maximum B task iterations (default: 3). "
             "Ignored if --max-b-loops-config is specified."
    )
    t2v_sim.add_argument(
        "--max-b-loops-config",
        type=str,
        default=None,
        help="Path to JSON config file for max_b_loops distribution. "
             "Supports: static, uniform, two_peak, four_peak distributions."
    )
    t2v_sim.add_argument(
        "--max-b-loops-seed",
        type=int,
        default=None,
        help="Random seed for max_b_loops distribution sampling."
    )
    t2v_sim.add_argument(
        "--frame-count",
        type=int,
        default=16,
        help="Frame count for video generation (default: 16). "
             "Ignored if --frame-count-config is specified."
    )
    t2v_sim.add_argument(
        "--frame-count-config",
        type=str,
        default=None,
        help="Path to JSON config file for frame_count distribution. "
             "Supports: static, uniform, two_peak, four_peak distributions."
    )
    t2v_sim.add_argument(
        "--frame-count-seed",
        type=int,
        default=None,
        help="Random seed for frame_count distribution sampling."
    )
    t2v_sim.add_argument(
        "--max-sleep-time",
        type=float,
        default=600.0,
        help="Maximum sleep time in seconds for simulation mode (default: 600.0). "
             "Sleep times from benchmark data will be scaled proportionally to fit within this limit."
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
        help="Maximum B task iterations (default: 3). "
             "Ignored if --max-b-loops-config is specified."
    )
    t2v_real.add_argument(
        "--max-b-loops-config",
        type=str,
        default=None,
        help="Path to JSON config file for max_b_loops distribution. "
             "Supports: static, uniform, two_peak, four_peak distributions."
    )
    t2v_real.add_argument(
        "--max-b-loops-seed",
        type=int,
        default=None,
        help="Random seed for max_b_loops distribution sampling."
    )
    t2v_real.add_argument(
        "--frame-count",
        type=int,
        default=16,
        help="Frame count for video generation (default: 16). "
             "Ignored if --frame-count-config is specified."
    )
    t2v_real.add_argument(
        "--frame-count-config",
        type=str,
        default=None,
        help="Path to JSON config file for frame_count distribution. "
             "Supports: static, uniform, two_peak, four_peak distributions."
    )
    t2v_real.add_argument(
        "--frame-count-seed",
        type=int,
        default=None,
        help="Random seed for frame_count distribution sampling."
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

    # ========================================================================
    # OCR+LLM Simulation
    # ========================================================================
    ocr_sim = subparsers.add_parser(
        "run-ocr-llm-sim",
        help="Run OCR+LLM workflow in simulation mode"
    )
    add_common_args(ocr_sim)
    ocr_sim.add_argument(
        "--sleep-time-a-config",
        type=str,
        default=None,
        help="Path to JSON config file for A (OCR) sleep time distribution."
    )
    ocr_sim.add_argument(
        "--sleep-time-b-config",
        type=str,
        default=None,
        help="Path to JSON config file for B (LLM) sleep time distribution."
    )
    ocr_sim.add_argument(
        "--sleep-time-seed",
        type=int,
        default=42,
        help="Random seed for sleep time distribution sampling (default: 42)."
    )

    # ========================================================================
    # OCR+LLM Real
    # ========================================================================
    ocr_real = subparsers.add_parser(
        "run-ocr-llm-real",
        help="Run OCR+LLM workflow in real cluster mode"
    )
    add_common_args(ocr_real)
    ocr_real.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing images for OCR (supports jpg, png, etc.)"
    )
    ocr_real.add_argument(
        "--image-json",
        type=str,
        default=None,
        help="JSON file containing base64-encoded images"
    )
    ocr_real.add_argument(
        "--ocr-languages",
        type=str,
        default="en",
        help="OCR languages (comma-separated, e.g., 'en,ch_sim') (default: en)"
    )
    ocr_real.add_argument(
        "--ocr-detail-level",
        type=str,
        default="standard",
        choices=["minimal", "standard", "detailed"],
        help="OCR detail level (default: standard)"
    )
    ocr_real.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens for LLM generation (default: 512)"
    )
    ocr_real.add_argument(
        "--scheduler-a-url",
        type=str,
        default=None,
        help="URL for Scheduler A (OCR). If not specified, uses env or default."
    )
    ocr_real.add_argument(
        "--scheduler-b-url",
        type=str,
        default=None,
        help="URL for Scheduler B (LLM). If not specified, uses env or default."
    )

    # ========================================================================
    # Text2Image+Video Simulation (Type 3)
    # ========================================================================
    t2iv_sim = subparsers.add_parser(
        "run-text2image-video-sim",
        help="Run Text2Image+Video workflow in simulation mode (A→C→B)"
    )
    add_common_args(t2iv_sim)
    t2iv_sim.add_argument(
        "--max-b-loops",
        type=int,
        default=3,
        help="Maximum B task iterations (default: 3). "
             "Ignored if --max-b-loops-config is specified."
    )
    t2iv_sim.add_argument(
        "--max-b-loops-config",
        type=str,
        default=None,
        help="Path to JSON config file for max_b_loops distribution. "
             "Supports: static, uniform, two_peak distributions."
    )
    t2iv_sim.add_argument(
        "--max-b-loops-seed",
        type=int,
        default=None,
        help="Random seed for max_b_loops distribution sampling."
    )
    t2iv_sim.add_argument(
        "--frame-count",
        type=int,
        default=16,
        help="Frame count for video generation (default: 16). "
             "Ignored if --frame-count-config is specified."
    )
    t2iv_sim.add_argument(
        "--frame-count-config",
        type=str,
        default=None,
        help="Path to JSON config file for frame_count distribution. "
             "Supports: static, uniform, two_peak, four_peak distributions."
    )
    t2iv_sim.add_argument(
        "--frame-count-seed",
        type=int,
        default=None,
        help="Random seed for frame_count distribution sampling."
    )
    t2iv_sim.add_argument(
        "--resolution",
        type=int,
        default=512,
        choices=[512, 1024],
        help="Image resolution for FLUX (512 or 1024, default: 512). "
             "Ignored if --resolution-config is specified."
    )
    t2iv_sim.add_argument(
        "--resolution-config",
        type=str,
        default=None,
        help="Path to JSON config file for resolution distribution. "
             "Supports: static_512, static_1024, 50_50, 70_30 distributions."
    )
    t2iv_sim.add_argument(
        "--resolution-seed",
        type=int,
        default=None,
        help="Random seed for resolution distribution sampling."
    )

    # ========================================================================
    # Text2Image+Video Real (Type 3)
    # ========================================================================
    t2iv_real = subparsers.add_parser(
        "run-text2image-video-real",
        help="Run Text2Image+Video workflow in real cluster mode (A→C→B)"
    )
    add_common_args(t2iv_real)
    t2iv_real.add_argument(
        "--max-b-loops",
        type=int,
        default=3,
        help="Maximum B task iterations (default: 3). "
             "Ignored if --max-b-loops-config is specified."
    )
    t2iv_real.add_argument(
        "--max-b-loops-config",
        type=str,
        default=None,
        help="Path to JSON config file for max_b_loops distribution. "
             "Supports: static, uniform, two_peak distributions."
    )
    t2iv_real.add_argument(
        "--max-b-loops-seed",
        type=int,
        default=None,
        help="Random seed for max_b_loops distribution sampling."
    )
    t2iv_real.add_argument(
        "--frame-count",
        type=int,
        default=16,
        help="Frame count for video generation (default: 16). "
             "Ignored if --frame-count-config is specified."
    )
    t2iv_real.add_argument(
        "--frame-count-config",
        type=str,
        default=None,
        help="Path to JSON config file for frame_count distribution. "
             "Supports: static, uniform, two_peak, four_peak distributions."
    )
    t2iv_real.add_argument(
        "--frame-count-seed",
        type=int,
        default=None,
        help="Random seed for frame_count distribution sampling."
    )
    t2iv_real.add_argument(
        "--resolution",
        type=int,
        default=512,
        choices=[512, 1024],
        help="Image resolution for FLUX (512 or 1024, default: 512). "
             "Ignored if --resolution-config is specified."
    )
    t2iv_real.add_argument(
        "--resolution-config",
        type=str,
        default=None,
        help="Path to JSON config file for resolution distribution. "
             "Supports: static_512, static_1024, 50_50, 70_30 distributions."
    )
    t2iv_real.add_argument(
        "--resolution-seed",
        type=int,
        default=None,
        help="Random seed for resolution distribution sampling."
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
                frame_count=args.frame_count,
                frame_count_config=args.frame_count_config,
                frame_count_seed=args.frame_count_seed,
                max_b_loops_config=args.max_b_loops_config,
                max_b_loops_seed=args.max_b_loops_seed,
                portion_stats=args.portion_stats,
                max_sleep_time=args.max_sleep_time,
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
                frame_count=args.frame_count,
                frame_count_config=args.frame_count_config,
                frame_count_seed=args.frame_count_seed,
                max_b_loops_config=args.max_b_loops_config,
                max_b_loops_seed=args.max_b_loops_seed,
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

        elif args.command == "run-ocr-llm-sim":
            result = runner.run_ocr_llm_simulation(
                num_workflows=args.num_workflows,
                qps=args.qps,
                seed=args.seed,
                strategies=args.strategies,
                warmup=args.warmup,
                duration=args.duration,
                sleep_time_a_config=args.sleep_time_a_config,
                sleep_time_b_config=args.sleep_time_b_config,
                sleep_time_seed=args.sleep_time_seed,
                portion_stats=args.portion_stats,
            )

        elif args.command == "run-ocr-llm-real":
            result = runner.run_ocr_llm_real(
                num_workflows=args.num_workflows,
                qps=args.qps,
                seed=args.seed,
                strategies=args.strategies,
                warmup=args.warmup,
                duration=args.duration,
                image_dir=args.image_dir,
                image_json=args.image_json,
                ocr_languages=args.ocr_languages,
                ocr_detail_level=args.ocr_detail_level,
                max_tokens=args.max_tokens,
                scheduler_a_url=args.scheduler_a_url,
                scheduler_b_url=args.scheduler_b_url,
                portion_stats=args.portion_stats,
            )

        elif args.command == "run-text2image-video-sim":
            result = runner.run_text2image_video_simulation(
                num_workflows=args.num_workflows,
                qps=args.qps,
                seed=args.seed,
                strategies=args.strategies,
                warmup=args.warmup,
                duration=args.duration,
                max_b_loops=args.max_b_loops,
                frame_count=args.frame_count,
                frame_count_config=args.frame_count_config,
                frame_count_seed=args.frame_count_seed,
                max_b_loops_config=args.max_b_loops_config,
                max_b_loops_seed=args.max_b_loops_seed,
                resolution=args.resolution,
                resolution_config=args.resolution_config,
                resolution_seed=args.resolution_seed,
                portion_stats=args.portion_stats,
            )

        elif args.command == "run-text2image-video-real":
            result = runner.run_text2image_video_real(
                num_workflows=args.num_workflows,
                qps=args.qps,
                seed=args.seed,
                strategies=args.strategies,
                warmup=args.warmup,
                duration=args.duration,
                max_b_loops=args.max_b_loops,
                frame_count=args.frame_count,
                frame_count_config=args.frame_count_config,
                frame_count_seed=args.frame_count_seed,
                max_b_loops_config=args.max_b_loops_config,
                max_b_loops_seed=args.max_b_loops_seed,
                resolution=args.resolution,
                resolution_config=args.resolution_config,
                resolution_seed=args.resolution_seed,
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
