from __future__ import annotations

import argparse
import os

from swarmbench.models import RunConfig
from swarmbench.runner import report_results, run_dataset, evaluate_saved_logs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swarmbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--dataset", required=True, choices=["mcp-atlas", "dataclaw", "swe-bench-pro"]
    )
    run_parser.add_argument("--mode", required=True)
    run_parser.add_argument("--model", default="gpt-4o")
    run_parser.add_argument(
        "--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    run_parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--max-turns", type=int, default=30)
    run_parser.add_argument("--output", default="./output")
    run_parser.add_argument("--workspace", default=None)
    run_parser.add_argument("--report", default=None)
    run_parser.add_argument("--docker-timeout", type=int, default=1800)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument(
        "--dataset", required=True, choices=["mcp-atlas", "dataclaw", "swe-bench-pro"]
    )
    eval_parser.add_argument("--mode", default="mock")
    eval_parser.add_argument("--model", default="gpt-4o")
    eval_parser.add_argument(
        "--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    eval_parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    eval_parser.add_argument("--limit", type=int, default=None)
    eval_parser.add_argument("--output", default="./output")
    eval_parser.add_argument("--report", default=None)
    eval_parser.add_argument("--workspace", default=None)

    subparsers.add_parser("list-datasets")
    return parser


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    payload = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "max_turns": getattr(args, "max_turns", 30),
        "dataset_name": args.dataset,
        "mode": args.mode,
        "limit": args.limit,
        "output_dir": args.output,
        "workspace": getattr(args, "workspace", None),
        "docker_timeout": getattr(args, "docker_timeout", 1800),
    }
    return RunConfig.model_validate(payload)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list-datasets":
        print("mcp-atlas: mock, real")
        print("dataclaw: trajectory, live")
        print("swe-bench-pro: dry-run, live")
        return

    config = _config_from_args(args)
    if args.command == "run":
        results = run_dataset(config)
        report_results(results, args.report)
        return
    if args.command == "evaluate":
        results = evaluate_saved_logs(config)
        report_results(results, args.report)
        return

    raise ValueError(f"Unknown command: {args.command}")
