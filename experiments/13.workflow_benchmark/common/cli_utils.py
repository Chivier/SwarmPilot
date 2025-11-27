"""
Command-line interface utilities for workflow experiments.

Provides unified argument parsing for all workflow test scripts,
ensuring consistent CLI across simulation and real modes.
"""

import argparse
from typing import List

# Valid strategies from scheduler/src/model.py StrategyType
VALID_STRATEGIES = [
    "min_time",
    "probabilistic",
    "round_robin",
    "random",
    "po2",
    "serverless",
]

# Default strategies for experiments (probabilistic first for baseline)
DEFAULT_STRATEGIES = [
    "probabilistic",
    "min_time",
    "round_robin",
    "random",
    "po2",
]


def parse_strategies(strategies_str: str) -> List[str]:
    """
    Parse and validate strategy string.

    Args:
        strategies_str: Comma-separated strategy names, "all", or "default"

    Returns:
        List of valid strategy names

    Raises:
        ValueError: If any strategy is invalid
    """
    if strategies_str.lower() == "all":
        return VALID_STRATEGIES.copy()

    if strategies_str.lower() == "default":
        return DEFAULT_STRATEGIES.copy()

    strategies = [s.strip() for s in strategies_str.split(",")]
    invalid = [s for s in strategies if s not in VALID_STRATEGIES]

    if invalid:
        raise ValueError(
            f"Invalid strategies: {invalid}. "
            f"Valid options: {VALID_STRATEGIES}"
        )

    return strategies


def create_base_parser(description: str = "Workflow benchmark test") -> argparse.ArgumentParser:
    """
    Create argument parser with common parameters.

    Args:
        description: Parser description

    Returns:
        Configured ArgumentParser with common arguments
    """
    parser = argparse.ArgumentParser(description=description)

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


def add_type1_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add Type1 (Text2Video) specific arguments.

    Args:
        parser: ArgumentParser to extend

    Returns:
        Parser with Type1 arguments added
    """
    parser.add_argument(
        "--max-b-loops",
        type=int,
        default=3,
        help="Maximum B task iterations (default: 3). "
             "Ignored if --max-b-loops-config is specified."
    )

    parser.add_argument(
        "--max-b-loops-config",
        type=str,
        default=None,
        help="Path to JSON config file for max_b_loops distribution. "
             "Supports: static, uniform, two_peak, four_peak distributions. "
             "If not specified, uses --max-b-loops as static value."
    )

    parser.add_argument(
        "--max-b-loops-seed",
        type=int,
        default=None,
        help="Random seed for max_b_loops distribution sampling (default: None). "
             "Use for reproducible max_b_loops values across runs."
    )

    parser.add_argument(
        "--frame-count",
        type=int,
        default=16,
        help="Frame count for video generation (default: 16). "
             "Ignored if --frame-count-config is specified."
    )

    parser.add_argument(
        "--frame-count-config",
        type=str,
        default=None,
        help="Path to JSON config file for frame_count distribution. "
             "Supports: static, uniform, two_peak, four_peak distributions. "
             "If not specified, uses --frame-count as static value."
    )

    parser.add_argument(
        "--frame-count-seed",
        type=int,
        default=None,
        help="Random seed for frame_count distribution sampling (default: None). "
             "Use for reproducible frame_count values across runs."
    )

    return parser


def add_type2_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add Type2 (Deep Research) specific arguments.

    Args:
        parser: ArgumentParser to extend

    Returns:
        Parser with Type2 arguments added
    """
    parser.add_argument(
        "--fanout",
        type=int,
        default=4,
        help="Default fanout count for parallel B tasks (default: 4). "
             "Ignored if --fanout-config is specified."
    )

    parser.add_argument(
        "--fanout-config",
        type=str,
        default=None,
        help="Path to JSON config file for fanout distribution. "
             "Supports: static, uniform, two_peak, four_peak distributions. "
             "If not specified, uses --fanout as static value."
    )

    parser.add_argument(
        "--fanout-seed",
        type=int,
        default=None,
        help="Random seed for fanout distribution sampling (default: None). "
             "Use for reproducible fanout values across runs."
    )

    return parser


def add_type3_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add Type3 (Text2Image+Video) specific arguments.

    Type3 extends Type1 with a FLUX text-to-image step (C) between LLM (A) and T2VID (B).
    Workflow: LLM (A) -> FLUX (C) -> T2VID (B loops)

    Args:
        parser: ArgumentParser to extend

    Returns:
        Parser with Type3 arguments added
    """
    parser.add_argument(
        "--max-b-loops",
        type=int,
        default=3,
        help="Maximum B task iterations (default: 3). "
             "Ignored if --max-b-loops-config is specified."
    )

    parser.add_argument(
        "--max-b-loops-config",
        type=str,
        default=None,
        help="Path to JSON config file for max_b_loops distribution. "
             "Supports: static, uniform, two_peak, four_peak distributions. "
             "If not specified, uses --max-b-loops as static value."
    )

    parser.add_argument(
        "--max-b-loops-seed",
        type=int,
        default=None,
        help="Random seed for max_b_loops distribution sampling (default: None). "
             "Use for reproducible max_b_loops values across runs."
    )

    parser.add_argument(
        "--frame-count",
        type=int,
        default=16,
        help="Frame count for video generation (default: 16). "
             "Ignored if --frame-count-config is specified."
    )

    parser.add_argument(
        "--frame-count-config",
        type=str,
        default=None,
        help="Path to JSON config file for frame_count distribution. "
             "Supports: static, uniform, two_peak, four_peak distributions. "
             "If not specified, uses --frame-count as static value."
    )

    parser.add_argument(
        "--frame-count-seed",
        type=int,
        default=None,
        help="Random seed for frame_count distribution sampling (default: None). "
             "Use for reproducible frame_count values across runs."
    )

    parser.add_argument(
        "--resolution",
        type=str,
        default="512x512",
        help="Default resolution for FLUX image generation (default: 512x512). "
             "Supported: 512x512, 1024x1024. Ignored if --resolution-config is specified."
    )

    parser.add_argument(
        "--resolution-config",
        type=str,
        default=None,
        help="Path to JSON config file for resolution distribution. "
             "Supports weighted_choice distribution for selecting between resolutions. "
             "If not specified, uses --resolution as static value."
    )

    parser.add_argument(
        "--resolution-seed",
        type=int,
        default=None,
        help="Random seed for resolution distribution sampling (default: None). "
             "Use for reproducible resolution values across runs."
    )

    return parser
