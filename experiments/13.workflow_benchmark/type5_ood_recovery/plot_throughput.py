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


def plot_recovery_vs_baseline(
    recovery_metrics: Dict,
    baseline_metrics: Dict,
    output_path: Optional[str] = None,
    title: str = "Recovery vs Baseline - Real-time Throughput Comparison",
    smoothing_window: int = 5,
) -> None:
    """
    Plot Recovery and Baseline real-time throughput (QPS) overlaid on the same chart.

    Creates a multi-panel figure showing:
    1. Real-time throughput comparison with smoothing
    2. Cumulative throughput comparison
    3. Task completion over time

    Args:
        recovery_metrics: Metrics dictionary from Recovery experiment
        baseline_metrics: Metrics dictionary from Baseline experiment
        output_path: Path to save the plot (None for display)
        title: Plot title
        smoothing_window: Window size for rolling average smoothing
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Colors
    recovery_color = '#2E86AB'  # Blue for Recovery
    baseline_color = '#E94F37'  # Red for Baseline

    # Extract data
    rec_trend = recovery_metrics.get("throughput_trend", [])
    base_trend = baseline_metrics.get("throughput_trend", [])

    if not rec_trend or not base_trend:
        print("Warning: Missing throughput_trend data")
        return

    rec_elapsed = [p["elapsed_s"] for p in rec_trend]
    base_elapsed = [p["elapsed_s"] for p in base_trend]

    # Support both field names (realtime_throughput and instantaneous_throughput)
    rec_realtime = [p.get("realtime_throughput", p.get("instantaneous_throughput", 0)) for p in rec_trend]
    base_realtime = [p.get("realtime_throughput", p.get("instantaneous_throughput", 0)) for p in base_trend]

    rec_cumulative = [p["cumulative_throughput"] for p in rec_trend]
    base_cumulative = [p["cumulative_throughput"] for p in base_trend]

    rec_completed = [p["total_completed"] for p in rec_trend]
    base_completed = [p["total_completed"] for p in base_trend]

    # Phase transition time (only Recovery has it)
    rec_transition = recovery_metrics.get("summary", {}).get("phase_transition_delay_s")

    # =========================================================================
    # Plot 1: Real-time Throughput (QPS) Comparison
    # =========================================================================
    ax1 = axes[0]

    # Raw data (light, thin lines)
    ax1.plot(rec_elapsed, rec_realtime, color=recovery_color, linewidth=0.5, alpha=0.3)
    ax1.plot(base_elapsed, base_realtime, color=baseline_color, linewidth=0.5, alpha=0.3)

    # Smoothed data (bold lines)
    if len(rec_realtime) >= smoothing_window:
        rec_smooth = np.convolve(rec_realtime, np.ones(smoothing_window)/smoothing_window, mode='valid')
        rec_smooth_elapsed = rec_elapsed[smoothing_window-1:]
        ax1.plot(rec_smooth_elapsed, rec_smooth, color=recovery_color, linewidth=2.5,
                label=f'Recovery (smoothed, window={smoothing_window}s)')
    else:
        ax1.plot(rec_elapsed, rec_realtime, color=recovery_color, linewidth=2, label='Recovery')

    if len(base_realtime) >= smoothing_window:
        base_smooth = np.convolve(base_realtime, np.ones(smoothing_window)/smoothing_window, mode='valid')
        base_smooth_elapsed = base_elapsed[smoothing_window-1:]
        ax1.plot(base_smooth_elapsed, base_smooth, color=baseline_color, linewidth=2.5,
                label=f'Baseline (smoothed, window={smoothing_window}s)')
    else:
        ax1.plot(base_elapsed, base_realtime, color=baseline_color, linewidth=2, label='Baseline')

    # Mark phase transition
    if rec_transition is not None:
        ax1.axvline(x=rec_transition, color='green', linestyle='--', linewidth=2, alpha=0.8,
                   label=f'Phase Transition (t={rec_transition:.1f}s)')

    ax1.set_ylabel("Real-time Throughput (tasks/s)", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add summary stats as text
    rec_summary = recovery_metrics.get("summary", {})
    base_summary = baseline_metrics.get("summary", {})
    rec_avg = rec_summary.get("average_throughput", 0)
    base_avg = base_summary.get("average_throughput", 0)
    improvement = ((rec_avg - base_avg) / base_avg * 100) if base_avg > 0 else 0

    stats_text = (
        f"Recovery Avg: {rec_avg:.2f} tasks/s\n"
        f"Baseline Avg: {base_avg:.2f} tasks/s\n"
        f"Improvement: {improvement:+.2f}%"
    )
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # =========================================================================
    # Plot 2: Cumulative Throughput Comparison
    # =========================================================================
    ax2 = axes[1]

    ax2.plot(rec_elapsed, rec_cumulative, color=recovery_color, linewidth=2, label='Recovery')
    ax2.plot(base_elapsed, base_cumulative, color=baseline_color, linewidth=2, label='Baseline')

    if rec_transition is not None:
        ax2.axvline(x=rec_transition, color='green', linestyle='--', linewidth=2, alpha=0.8)

    ax2.set_ylabel("Cumulative Throughput (tasks/s)", fontsize=12)
    ax2.set_title("Cumulative Throughput Over Time", fontsize=12)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 3: Task Completion Comparison
    # =========================================================================
    ax3 = axes[2]

    ax3.plot(rec_elapsed, rec_completed, color=recovery_color, linewidth=2, label='Recovery')
    ax3.plot(base_elapsed, base_completed, color=baseline_color, linewidth=2, label='Baseline')

    if rec_transition is not None:
        ax3.axvline(x=rec_transition, color='green', linestyle='--', linewidth=2, alpha=0.8)

    # Shade the difference (Recovery ahead)
    # Interpolate to common time points
    common_time = np.linspace(0, min(max(rec_elapsed), max(base_elapsed)), 200)
    rec_interp = np.interp(common_time, rec_elapsed, rec_completed)
    base_interp = np.interp(common_time, base_elapsed, base_completed)

    ax3.fill_between(common_time, rec_interp, base_interp,
                    where=(rec_interp > base_interp),
                    color=recovery_color, alpha=0.2, label='Recovery ahead')
    ax3.fill_between(common_time, rec_interp, base_interp,
                    where=(rec_interp < base_interp),
                    color=baseline_color, alpha=0.2, label='Baseline ahead')

    ax3.set_xlabel("Elapsed Time (s)", fontsize=12)
    ax3.set_ylabel("Tasks Completed", fontsize=12)
    ax3.set_title("Task Completion Progress", fontsize=12)
    ax3.legend(loc='lower right', fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
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
    # Support both old field name (instantaneous_throughput) and new name (realtime_throughput)
    instantaneous = [p.get("realtime_throughput", p.get("instantaneous_throughput", 0)) for p in trend]

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
