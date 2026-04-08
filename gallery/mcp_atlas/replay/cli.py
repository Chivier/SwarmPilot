"""CLI entry point with two subcommands: prepare and run."""

from __future__ import annotations

import argparse
import asyncio

import yaml

from replay.main import prepare, run
from replay.models import ExperimentConfig


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with prepare/run subcommands.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="replay",
        description="Replay MCP-Atlas agent interactions for latency benchmarking",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- prepare subcommand ---
    prep = subparsers.add_parser(
        "prepare",
        help="Reshape MCP-Atlas dataset into prepared replay data",
    )
    prep.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of conversations to include",
    )
    prep.add_argument(
        "--output",
        required=True,
        help="Output JSONL file path for prepared data",
    )

    # --- run subcommand ---
    run_parser = subparsers.add_parser(
        "run",
        help="Execute a replay experiment using prepared data and YAML config",
    )
    run_parser.add_argument(
        "--data",
        required=True,
        help="Path to prepared data JSONL (output of 'prepare')",
    )
    run_parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML experiment config file",
    )
    run_parser.add_argument(
        "--output",
        default="./results.json",
        help="Output JSON file path for results (default: ./results.json)",
    )
    run_parser.add_argument(
        "--no-deploy",
        action="store_true",
        help="Skip automatic model deployment via Planner",
    )
    run_parser.add_argument(
        "--no-teardown",
        action="store_true",
        help="Skip automatic model teardown after experiment",
    )

    return parser


def _load_experiment_config(config_path: str) -> ExperimentConfig:
    """Load and validate an ExperimentConfig from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated ExperimentConfig.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        pydantic.ValidationError: If the YAML content doesn't match the schema.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw)


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate command."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        prepare(
            output_path=args.output,
            limit=args.limit,
        )
    elif args.command == "run":
        config = _load_experiment_config(args.config)
        asyncio.run(run(
            data_path=args.data,
            config=config,
            output_path=args.output,
            no_deploy=args.no_deploy,
            no_teardown=args.no_teardown,
        ))


if __name__ == "__main__":
    main()
