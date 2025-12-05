#!/usr/bin/env python3
"""
OOD Recovery Experiment - Throughput Visualization

This script plots throughput trends from experiment results.

Usage:
    # Plot single experiment
    python -m type5_ood_recovery.plot_throughput output_ood/metrics_recovery.json

    # Compare recovery vs baseline
    python -m type5_ood_recovery.plot_throughput \
        output_ood/metrics_recovery.json \
        output_ood/metrics_baseline.json \
        --compare

    # Specify output file
    python -m type5_ood_recovery.plot_throughput \
        output_ood/metrics_recovery.json \
        --output throughput_plot.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_metrics(filepath: str) -> Dict:
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_single_experiment(
    metrics: Dict,
    ax: plt.Axes,
    label: str = "",
    color: str = "blue",
    show_phases: bool = True,
) -> None:
    """
    Plot throughput trend for a single experiment.

    Args:
        metrics: Metrics dictionary from experiment
        ax: Matplotlib axes to plot on
        label: Label for the plot
        color: Line color
        show_phases: Whether to show phase completion markers
    """
    trend = metrics.get("throughput_trend", [])
    if not trend:
        print(f"Warning: No throughput_trend data in metrics")
        return

    # Extract data
    elapsed = [p["elapsed_s"] for p in trend]
    cumulative = [p["cumulative_throughput"] for p in trend]
    instantaneous = [p["instantaneous_throughput"] for p in trend]

    # Plot cumulative throughput (main line)
    line_label = f"{label} (cumulative)" if label else "Cumulative Throughput"
    ax.plot(elapsed, cumulative, color=color, linewidth=2, label=line_label)

    # Plot instantaneous throughput (lighter, dashed)
    inst_label = f"{label} (instant)" if label else "Instantaneous Throughput"
    ax.plot(elapsed, instantaneous, color=color, linewidth=1, linestyle='--',
            alpha=0.5, label=inst_label)

    # Mark phase transition point
    transition_time = metrics.get("summary", {}).get("phase_transition_delay_s")
    if transition_time is not None:
        ax.axvline(x=transition_time, color=color, linestyle=':', alpha=0.7)
        # Find throughput at transition
        for p in trend:
            if p["elapsed_s"] >= transition_time:
                ax.scatter([transition_time], [p["cumulative_throughput"]],
                          color=color, marker='o', s=100, zorder=5,
                          label=f"{label} Phase Transition" if label else "Phase Transition")
                break

    # Show phase completion areas if requested
    if show_phases and not label:  # Only for single plot
        phase_colors = {1: 'green', 2: 'orange', 3: 'blue'}
        for p in trend:
            phase_completed = p.get("phase_completed", {})
            # This is more complex - skip for now


def plot_comparison(
    metrics_list: List[Dict],
    labels: List[str],
    output_path: Optional[str] = None,
    title: str = "OOD Recovery Experiment - Throughput Comparison"
) -> None:
    """
    Plot comparison of multiple experiments.

    Args:
        metrics_list: List of metrics dictionaries
        labels: Labels for each experiment
        output_path: Path to save the plot (None for display)
        title: Plot title
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    colors = ['#2E86AB', '#E94F37', '#A23B72', '#F18F01']

    # Top plot: Cumulative throughput comparison
    ax1 = axes[0]
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        trend = metrics.get("throughput_trend", [])
        if not trend:
            continue

        elapsed = [p["elapsed_s"] for p in trend]
        cumulative = [p["cumulative_throughput"] for p in trend]
        color = colors[i % len(colors)]

        ax1.plot(elapsed, cumulative, color=color, linewidth=2, label=label)

        # Mark phase transition
        transition_time = metrics.get("summary", {}).get("phase_transition_delay_s")
        if transition_time is not None:
            ax1.axvline(x=transition_time, color=color, linestyle=':', alpha=0.7)
            for p in trend:
                if p["elapsed_s"] >= transition_time:
                    ax1.scatter([transition_time], [p["cumulative_throughput"]],
                               color=color, marker='o', s=80, zorder=5)
                    break

    ax1.set_ylabel("Cumulative Throughput (tasks/s)", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Phase completion over time
    ax2 = axes[1]
    phase_colors = {'P1': '#4CAF50', 'P2': '#FF9800', 'P3': '#2196F3'}

    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        trend = metrics.get("throughput_trend", [])
        if not trend:
            continue

        elapsed = [p["elapsed_s"] for p in trend]
        color = colors[i % len(colors)]

        # Extract phase completions
        p1_completed = [p.get("phase_completed", {}).get("1", 0) for p in trend]
        p2_completed = [p.get("phase_completed", {}).get("2", 0) for p in trend]
        p3_completed = [p.get("phase_completed", {}).get("3", 0) for p in trend]

        linestyle = '-' if i == 0 else '--'
        ax2.plot(elapsed, p1_completed, color=phase_colors['P1'],
                linewidth=1.5, linestyle=linestyle, alpha=0.8,
                label=f"{label} P1" if i == 0 else None)
        ax2.plot(elapsed, p2_completed, color=phase_colors['P2'],
                linewidth=1.5, linestyle=linestyle, alpha=0.8,
                label=f"{label} P2" if i == 0 else None)
        ax2.plot(elapsed, p3_completed, color=phase_colors['P3'],
                linewidth=1.5, linestyle=linestyle, alpha=0.8,
                label=f"{label} P3" if i == 0 else None)

    ax2.set_xlabel("Elapsed Time (s)", fontsize=12)
    ax2.set_ylabel("Tasks Completed", fontsize=12)
    ax2.set_title("Phase Completion Over Time", fontsize=12)
    ax2.legend(loc='upper left', ncol=3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_single(
    metrics: Dict,
    output_path: Optional[str] = None,
    title: str = "OOD Recovery Experiment - Throughput"
) -> None:
    """
    Plot a single experiment's throughput.

    Args:
        metrics: Metrics dictionary
        output_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    trend = metrics.get("throughput_trend", [])
    if not trend:
        print("Warning: No throughput_trend data")
        return

    mode = metrics.get("config", {}).get("mode", "Unknown")
    elapsed = [p["elapsed_s"] for p in trend]
    cumulative = [p["cumulative_throughput"] for p in trend]
    instantaneous = [p["instantaneous_throughput"] for p in trend]

    # Plot 1: Throughput over time
    ax1 = axes[0]
    ax1.plot(elapsed, cumulative, 'b-', linewidth=2, label='Cumulative')
    ax1.plot(elapsed, instantaneous, 'b--', linewidth=1, alpha=0.5, label='Instantaneous')

    # Mark phase transition
    transition_time = metrics.get("summary", {}).get("phase_transition_delay_s")
    if transition_time is not None:
        ax1.axvline(x=transition_time, color='red', linestyle=':', linewidth=2,
                   label=f'Phase Transition ({transition_time:.1f}s)')

    ax1.set_ylabel("Throughput (tasks/s)", fontsize=12)
    ax1.set_title(f"{title} - {mode}", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Task completion over time
    ax2 = axes[1]
    total_completed = [p["total_completed"] for p in trend]
    ax2.plot(elapsed, total_completed, 'g-', linewidth=2, label='Total Completed')

    # Phase breakdown
    p1 = [p.get("phase_completed", {}).get("1", 0) for p in trend]
    p2 = [p.get("phase_completed", {}).get("2", 0) for p in trend]
    p3 = [p.get("phase_completed", {}).get("3", 0) for p in trend]

    ax2.fill_between(elapsed, 0, p1, alpha=0.3, color='green', label='Phase 1')
    ax2.fill_between(elapsed, p1, [a+b for a,b in zip(p1, p2)],
                    alpha=0.3, color='orange', label='Phase 2')
    ax2.fill_between(elapsed, [a+b for a,b in zip(p1, p2)],
                    [a+b+c for a,b,c in zip(p1, p2, p3)],
                    alpha=0.3, color='blue', label='Phase 3')

    if transition_time is not None:
        ax2.axvline(x=transition_time, color='red', linestyle=':', linewidth=2)

    ax2.set_ylabel("Tasks Completed", fontsize=12)
    ax2.set_title("Task Completion by Phase", fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Instantaneous throughput histogram/distribution
    ax3 = axes[2]
    # Rolling average of instantaneous throughput
    window = 5
    if len(instantaneous) >= window:
        rolling_avg = np.convolve(instantaneous, np.ones(window)/window, mode='valid')
        rolling_elapsed = elapsed[window-1:]
        ax3.plot(rolling_elapsed, rolling_avg, 'purple', linewidth=2,
                label=f'Rolling Avg (window={window}s)')
    ax3.plot(elapsed, instantaneous, 'gray', linewidth=0.5, alpha=0.5, label='Raw')

    if transition_time is not None:
        ax3.axvline(x=transition_time, color='red', linestyle=':', linewidth=2)

    ax3.set_xlabel("Elapsed Time (s)", fontsize=12)
    ax3.set_ylabel("Instantaneous Throughput (tasks/s)", fontsize=12)
    ax3.set_title("Instantaneous Throughput (Smoothed)", fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot throughput trends from OOD experiment results"
    )
    parser.add_argument(
        "metrics_files",
        nargs='+',
        help="Path to metrics JSON file(s)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path for the plot (default: display)"
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare multiple experiments side by side"
    )
    parser.add_argument(
        "--title", "-t",
        type=str,
        default="OOD Recovery Experiment - Throughput",
        help="Plot title"
    )

    args = parser.parse_args()

    # Load all metrics files
    metrics_list = []
    labels = []
    for filepath in args.metrics_files:
        try:
            metrics = load_metrics(filepath)
            metrics_list.append(metrics)
            # Generate label from filename or mode
            mode = metrics.get("config", {}).get("mode", Path(filepath).stem)
            labels.append(mode)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

    if not metrics_list:
        print("No valid metrics files found")
        sys.exit(1)

    # Plot
    if len(metrics_list) == 1 and not args.compare:
        plot_single(metrics_list[0], args.output, args.title)
    else:
        plot_comparison(metrics_list, labels, args.output, args.title)


if __name__ == "__main__":
    main()
