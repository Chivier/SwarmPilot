#!/usr/bin/env python3
"""
SLO Violation Rate Analysis - Sweep through ratio thresholds

This script analyzes SLO violations by comparing latency to execution time.
SLO is defined as: latency / execution_time <= threshold

For each threshold value, it generates a comparison plot showing violation
rates per phase for Baseline vs Recovery scenarios.

Usage:
    # Run with default paths (from experiments/13.workflow_benchmark)
    python -m type5_ood_recovery.plot_slo_violation

    # Specify custom paths
    python -m type5_ood_recovery.plot_slo_violation \
        --baseline output_realtime_diff/baseline/metrics.json \
        --recovery output_realtime_diff/recovery/metrics.json \
        --output-dir type5_ood_recovery/results/slo_analysis
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(filepath: str) -> Dict:
    """Load metrics from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def extract_task_slo_data(metrics: Dict) -> List[Dict]:
    """
    Extract SLO-relevant data from task executions.

    Returns list of dicts with:
        - phase: int (1, 2, or 3)
        - execution_time: float (complete_time - exec_start_time)
        - latency: float (complete_time - submit_time)
        - ratio: float (latency / execution_time)
    """
    tasks = metrics.get("task_executions", [])
    slo_data = []

    for task in tasks:
        exec_start = task.get("exec_start_time")
        complete = task.get("complete_time")
        submit = task.get("submit_time")
        phase = task.get("phase")

        if exec_start is None or complete is None or submit is None:
            continue

        execution_time = complete - exec_start
        latency = complete - submit

        # Avoid division by zero
        if execution_time <= 0:
            continue

        ratio = latency / execution_time

        slo_data.append(
            {
                "phase": phase,
                "execution_time": execution_time,
                "latency": latency,
                "ratio": ratio,
            }
        )

    return slo_data


def calculate_violation_rate(
    tasks: List[Dict], threshold: float, phase: Optional[int] = None
) -> float:
    """
    Calculate SLO violation rate for given threshold.

    Args:
        tasks: List of task SLO data
        threshold: SLO ratio threshold (violation if ratio > threshold)
        phase: Optional phase filter (None = all phases)

    Returns:
        Violation rate as percentage (0-100)
    """
    if phase is not None:
        filtered = [t for t in tasks if t["phase"] == phase]
    else:
        filtered = tasks

    if not filtered:
        return 0.0

    violations = sum(1 for t in filtered if t["ratio"] > threshold)
    return (violations / len(filtered)) * 100


def plot_slo_comparison(
    baseline_data: List[Dict],
    recovery_data: List[Dict],
    threshold: float,
    output_path: str,
    baseline_config: Dict,
    recovery_config: Dict,
) -> None:
    """
    Create bar chart comparing SLO violation rates by phase.

    Excludes Phase 1 (warmup) to focus on OOD and Recovery phases only.

    Args:
        baseline_data: List of task SLO data for baseline
        recovery_data: List of task SLO data for recovery
        threshold: SLO ratio threshold
        output_path: Path to save the plot
        baseline_config: Config dict from baseline metrics
        recovery_config: Config dict from recovery metrics
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter out Phase 1 (warmup/transition) data
    baseline_filtered = [t for t in baseline_data if t["phase"] != 1]
    recovery_filtered = [t for t in recovery_data if t["phase"] != 1]

    # Categories: Phase 2 (OOD), Phase 3 (Recovery), All (P2+P3)
    categories = ["Phase 2\n(OOD)", "Phase 3\n(Post-Recovery)", "All Tasks\n(P2+P3)"]
    phases = [2, 3, None]

    # Calculate violation rates (using filtered data for "All")
    baseline_rates = [
        calculate_violation_rate(baseline_filtered, threshold, p) for p in phases
    ]
    recovery_rates = [
        calculate_violation_rate(recovery_filtered, threshold, p) for p in phases
    ]

    # Bar positions
    x = np.arange(len(categories))
    width = 0.35

    # Colors
    baseline_color = "#E94F37"  # Red
    recovery_color = "#2E86AB"  # Blue

    # Create bars
    bars1 = ax.bar(
        x - width / 2,
        baseline_rates,
        width,
        label="Baseline",
        color=baseline_color,
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        recovery_rates,
        width,
        label="Recovery",
        color=recovery_color,
        alpha=0.8,
    )

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    add_labels(bars1)
    add_labels(bars2)

    # Labels and title
    ax.set_xlabel("Phase", fontsize=12)
    ax.set_ylabel("SLO Violation Rate (%)", fontsize=12)
    ax.set_title(
        f"SLO Violation Rate Comparison (Excluding Warmup Phase)\n"
        f"(Ratio Threshold = {threshold:.1f}, SLO: latency/exec_time <= {threshold:.1f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # Set y-axis limit to at least 0-100 for percentage
    max_rate = max(baseline_rates + recovery_rates) if baseline_rates + recovery_rates else 0
    ax.set_ylim(0, max(100, max_rate * 1.1))

    # Add task count info (excluding Phase 1)
    baseline_counts = {p: sum(1 for t in baseline_data if t["phase"] == p) for p in [2, 3]}
    recovery_counts = {p: sum(1 for t in recovery_data if t["phase"] == p) for p in [2, 3]}

    info_text = (
        f"Baseline: P2={baseline_counts[2]}, P3={baseline_counts[3]}, Total={len(baseline_filtered)}\n"
        f"Recovery: P2={recovery_counts[2]}, P3={recovery_counts[3]}, Total={len(recovery_filtered)}"
    )
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def sweep_slo_thresholds(
    baseline_path: str,
    recovery_path: str,
    output_dir: str,
    thresholds: List[float],
) -> None:
    """
    Sweep through SLO thresholds and generate comparison plots.

    Args:
        baseline_path: Path to baseline metrics.json
        recovery_path: Path to recovery metrics.json
        output_dir: Directory to save output plots
        thresholds: List of threshold values to test
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading baseline metrics from: {baseline_path}")
    baseline_metrics = load_metrics(baseline_path)
    baseline_data = extract_task_slo_data(baseline_metrics)
    baseline_config = baseline_metrics.get("config", {})
    print(f"  Loaded {len(baseline_data)} tasks")

    print(f"Loading recovery metrics from: {recovery_path}")
    recovery_metrics = load_metrics(recovery_path)
    recovery_data = extract_task_slo_data(recovery_metrics)
    recovery_config = recovery_metrics.get("config", {})
    print(f"  Loaded {len(recovery_data)} tasks")

    print(f"\nGenerating {len(thresholds)} plots for thresholds: {thresholds}")

    for threshold in thresholds:
        plot_path = output_path / f"slo_ratio_{threshold:.1f}.png"
        plot_slo_comparison(
            baseline_data,
            recovery_data,
            threshold,
            str(plot_path),
            baseline_config,
            recovery_config,
        )

    print(f"\nDone! {len(thresholds)} plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SLO violations across different ratio thresholds"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="output_realtime_diff/baseline/metrics.json",
        help="Path to baseline metrics.json",
    )
    parser.add_argument(
        "--recovery",
        type=str,
        default="output_realtime_diff/recovery/metrics.json",
        help="Path to recovery metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="type5_ood_recovery/results/slo_analysis",
        help="Directory to save output plots",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=1.0,
        help="Minimum SLO ratio threshold",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=10.0,
        help="Maximum SLO ratio threshold",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.5,
        help="Step size for threshold sweep",
    )

    args = parser.parse_args()

    # Generate threshold list
    thresholds = list(
        np.arange(args.min_threshold, args.max_threshold + args.step / 2, args.step)
    )

    sweep_slo_thresholds(
        args.baseline,
        args.recovery,
        args.output_dir,
        thresholds,
    )


if __name__ == "__main__":
    main()
