#!/usr/bin/env python3
"""Run comparison between Probabilistic and Round-Robin scheduling.

This script runs the E2E LLM cluster test twice:
1. With probabilistic scheduling
2. With round-robin scheduling

Then compares the results.

Usage:
    python -m tests.integration.e2e_llm_cluster.run_comparison
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class ComparisonConfig:
    """Configuration for comparison experiment."""

    total_requests: int = 1000
    total_qps: float = 20.0  # 1000 requests / 50 seconds
    duration_seconds: float = 50.0
    num_workers: int = 8
    output_dir: Path = Path("./e2e_comparison_results")


def setup_logging():
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )


def cleanup_processes():
    """Kill any lingering processes."""
    for pattern in ["pylet", "uvicorn.*scheduler", "uvicorn.*planner"]:
        subprocess.run(
            f"pkill -9 -f '{pattern}' 2>/dev/null || true",
            shell=True,
            capture_output=True,
        )
    time.sleep(2)

    # Clean PyLet state
    pylet_state = Path.home() / ".pylet"
    for subdir in ["state", "workers"]:
        subdir_path = pylet_state / subdir
        if subdir_path.exists():
            subprocess.run(f"rm -rf {subdir_path}", shell=True, capture_output=True)
    for db_file in pylet_state.glob("pylet.db*"):
        db_file.unlink(missing_ok=True)


def run_experiment(
    strategy: str,
    config: ComparisonConfig,
    output_file: Path,
) -> dict:
    """Run a single E2E experiment with the specified scheduling strategy.

    Args:
        strategy: Scheduling strategy ("probabilistic" or "round_robin")
        config: Experiment configuration
        output_file: Path to save the report

    Returns:
        Report dict
    """
    logger.info(f"=" * 60)
    logger.info(f"Running experiment with {strategy.upper()} scheduling")
    logger.info(f"=" * 60)

    cleanup_processes()

    # Set environment for scheduler
    env = os.environ.copy()
    env["SCHEDULING_STRATEGY"] = strategy

    # Run the E2E test
    cmd = [
        sys.executable,
        "-m",
        "tests.integration.e2e_llm_cluster.run_e2e_llm_cluster",
        "--workers", str(config.num_workers),
        "--total-qps", str(config.total_qps),
        "--duration", str(config.duration_seconds),
    ]

    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Strategy: {strategy}")

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,  # 10 minute timeout
    )

    # Print output
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-20:]:
            logger.info(f"  {line}")

    if result.returncode != 0:
        logger.error(f"Experiment failed with return code {result.returncode}")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-10:]:
                logger.error(f"  {line}")
        return {"error": "Experiment failed", "strategy": strategy}

    # Read and copy report
    report_path = Path("e2e_llm_cluster_results/report.json")
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)

        # Add strategy info
        report["scheduling_strategy"] = strategy

        # Save to output file
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.success(f"Report saved to {output_file}")
        return report
    else:
        logger.error("Report file not found")
        return {"error": "Report not found", "strategy": strategy}


def compare_results(
    probabilistic_report: dict,
    round_robin_report: dict,
) -> dict:
    """Compare results from both experiments.

    Args:
        probabilistic_report: Report from probabilistic scheduling
        round_robin_report: Report from round-robin scheduling

    Returns:
        Comparison summary dict
    """
    comparison = {
        "probabilistic": {},
        "round_robin": {},
        "improvement": {},
    }

    for strategy, report in [
        ("probabilistic", probabilistic_report),
        ("round_robin", round_robin_report),
    ]:
        if "error" in report:
            comparison[strategy] = {"error": report["error"]}
            continue

        exec_stats = report.get("execution", {}).get("execution_stats", {})

        # Calculate overall metrics
        total_latency = 0
        total_count = 0
        model_metrics = {}

        for model_id, stats in exec_stats.items():
            count = stats.get("count", 0)
            avg_ms = stats.get("avg_ms", 0)
            p50_ms = stats.get("p50_ms", 0)
            p90_ms = stats.get("p90_ms", 0)
            p99_ms = stats.get("p99_ms", 0)

            total_latency += avg_ms * count
            total_count += count

            model_metrics[model_id] = {
                "count": count,
                "avg_ms": round(avg_ms, 2),
                "p50_ms": round(p50_ms, 2),
                "p90_ms": round(p90_ms, 2),
                "p99_ms": round(p99_ms, 2),
            }

        overall_avg = total_latency / total_count if total_count > 0 else 0

        comparison[strategy] = {
            "total_tasks": report.get("workload", {}).get("tasks_submitted", 0),
            "tasks_completed": report.get("execution", {}).get("tasks_completed", 0),
            "success_rate": report.get("execution", {}).get("success_rate", 0),
            "overall_avg_ms": round(overall_avg, 2),
            "models": model_metrics,
        }

    # Calculate improvement (probabilistic vs round_robin)
    if "error" not in comparison["probabilistic"] and "error" not in comparison["round_robin"]:
        prob_avg = comparison["probabilistic"]["overall_avg_ms"]
        rr_avg = comparison["round_robin"]["overall_avg_ms"]

        if rr_avg > 0:
            improvement_pct = ((rr_avg - prob_avg) / rr_avg) * 100
            comparison["improvement"] = {
                "latency_reduction_pct": round(improvement_pct, 2),
                "probabilistic_avg_ms": prob_avg,
                "round_robin_avg_ms": rr_avg,
            }

    return comparison


def print_comparison(comparison: dict):
    """Print comparison results in a nice format."""
    print("\n" + "=" * 70)
    print("SCHEDULING ALGORITHM COMPARISON")
    print("=" * 70)

    for strategy in ["probabilistic", "round_robin"]:
        data = comparison.get(strategy, {})
        if "error" in data:
            print(f"\n{strategy.upper()}: ERROR - {data['error']}")
            continue

        print(f"\n{strategy.upper()} SCHEDULING:")
        print(f"  Total Tasks: {data.get('total_tasks', 'N/A')}")
        print(f"  Completed: {data.get('tasks_completed', 'N/A')}")
        print(f"  Success Rate: {data.get('success_rate', 0) * 100:.1f}%")
        print(f"  Overall Avg Latency: {data.get('overall_avg_ms', 'N/A'):.2f}ms")

        print("\n  Per-Model Metrics:")
        print(f"  {'Model':<12} {'Count':>6} {'Avg':>10} {'P50':>10} {'P90':>10} {'P99':>10}")
        print(f"  {'-' * 60}")

        for model_id, metrics in data.get("models", {}).items():
            print(
                f"  {model_id:<12} {metrics['count']:>6} "
                f"{metrics['avg_ms']:>10.2f} {metrics['p50_ms']:>10.2f} "
                f"{metrics['p90_ms']:>10.2f} {metrics['p99_ms']:>10.2f}"
            )

    # Print improvement summary
    improvement = comparison.get("improvement", {})
    if improvement:
        print("\n" + "-" * 70)
        print("IMPROVEMENT SUMMARY (Probabilistic vs Round-Robin)")
        print("-" * 70)
        print(f"  Probabilistic Avg: {improvement.get('probabilistic_avg_ms', 'N/A'):.2f}ms")
        print(f"  Round-Robin Avg:   {improvement.get('round_robin_avg_ms', 'N/A'):.2f}ms")

        reduction = improvement.get("latency_reduction_pct", 0)
        if reduction > 0:
            print(f"  Latency Reduction: {reduction:.2f}% (Probabilistic is FASTER)")
        elif reduction < 0:
            print(f"  Latency Increase:  {abs(reduction):.2f}% (Round-Robin is FASTER)")
        else:
            print(f"  No significant difference")

    print("\n" + "=" * 70)


async def main():
    """Run comparison experiment."""
    setup_logging()

    config = ComparisonConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting comparison experiment")
    logger.info(f"  Total requests: {config.total_requests}")
    logger.info(f"  QPS: {config.total_qps}")
    logger.info(f"  Duration: {config.duration_seconds}s")
    logger.info(f"  Workers: {config.num_workers}")

    # Run probabilistic experiment
    probabilistic_report = run_experiment(
        strategy="probabilistic",
        config=config,
        output_file=config.output_dir / "probabilistic_report.json",
    )

    # Run round-robin experiment
    round_robin_report = run_experiment(
        strategy="round_robin",
        config=config,
        output_file=config.output_dir / "round_robin_report.json",
    )

    # Compare results
    comparison = compare_results(probabilistic_report, round_robin_report)

    # Save comparison
    comparison_file = config.output_dir / "comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Comparison saved to {comparison_file}")

    # Print results
    print_comparison(comparison)

    # Cleanup
    cleanup_processes()


if __name__ == "__main__":
    asyncio.run(main())
