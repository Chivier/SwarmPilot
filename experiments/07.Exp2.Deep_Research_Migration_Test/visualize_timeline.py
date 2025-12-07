#!/usr/bin/env python3
"""
Visualize instance deployment timeline from experiment results.

This script reads the timeline data from experiment results and creates
visualizations showing how instance allocations change over time.

Usage:
    python visualize_timeline.py results/results_workflow_b1b2_20231219_120000.json
    python visualize_timeline.py results/results_workflow_b1b2_20231219_120000.json --strategy min_time
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


def load_timeline_from_results(results_file: str, strategy: Optional[str] = None) -> Dict:
    """
    Load timeline data from experiment results file.

    Args:
        results_file: Path to results JSON file
        strategy: Strategy name to extract (if None, use first strategy)

    Returns:
        Dictionary with timeline data
    """
    with open(results_file, 'r') as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("Error: No results found in file")
        return None

    # Find matching strategy or use first
    target_result = None
    if strategy:
        for result in results:
            if result.get("strategy") == strategy:
                target_result = result
                break
        if not target_result:
            print(f"Error: Strategy '{strategy}' not found in results")
            print(f"Available strategies: {[r.get('strategy') for r in results]}")
            return None
    else:
        target_result = results[0]
        strategy = target_result.get("strategy")

    timeline = target_result.get("planner_timeline")
    if not timeline:
        print(f"Error: No timeline data found for strategy '{strategy}'")
        return None

    if not timeline.get("success"):
        print(f"Error: Timeline retrieval failed for strategy '{strategy}'")
        return None

    print(f"Loaded timeline for strategy '{strategy}'")
    print(f"  Entry count: {timeline.get('entry_count', 0)}")

    return {
        "strategy": strategy,
        "timeline": timeline,
        "config": data.get("config", {})
    }


def print_timeline_summary(timeline_data: Dict):
    """
    Print a text summary of the timeline.

    Args:
        timeline_data: Timeline data dictionary
    """
    entries = timeline_data["timeline"]["entries"]
    strategy = timeline_data["strategy"]

    print("\n" + "=" * 80)
    print(f"Timeline Summary - Strategy: {strategy}")
    print("=" * 80)

    if not entries:
        print("No timeline entries found")
        return

    print(f"\nTotal events: {len(entries)}")

    # Count event types
    deploy_count = sum(1 for e in entries if e["event_type"] == "deploy_migration")
    auto_opt_count = sum(1 for e in entries if e["event_type"] == "auto_optimize")
    print(f"  - Initial deployments: {deploy_count}")
    print(f"  - Auto-optimizations: {auto_opt_count}")

    # Success rate
    successful = sum(1 for e in entries if e["success"])
    print(f"  - Successful: {successful}/{len(entries)} ({successful/len(entries)*100:.1f}%)")

    # Time range
    first_time = datetime.fromisoformat(entries[0]["timestamp_iso"])
    last_time = datetime.fromisoformat(entries[-1]["timestamp_iso"])
    duration = (last_time - first_time).total_seconds()
    print(f"\nTime range:")
    print(f"  - First event: {first_time}")
    print(f"  - Last event:  {last_time}")
    print(f"  - Duration:    {duration:.1f} seconds ({duration/60:.1f} minutes)")

    # Instance allocation changes
    print(f"\nInstance allocation timeline:")
    for i, entry in enumerate(entries):
        timestamp = datetime.fromisoformat(entry["timestamp_iso"])
        event_type = entry["event_type"]
        counts = entry["instance_counts"]
        changes = entry["changes_count"]
        success_icon = "✓" if entry["success"] else "✗"

        print(f"\n  {i+1}. [{timestamp.strftime('%H:%M:%S')}] {event_type} {success_icon}")
        for model_id, count in sorted(counts.items()):
            print(f"     - {model_id}: {count} instances")
        if changes > 0:
            print(f"     Changes: {changes}")
        if entry.get("score") is not None:
            print(f"     Score: {entry['score']:.4f}")


def plot_timeline(timeline_data: Dict, output_file: Optional[str] = None):
    """
    Create visualization of instance allocation over time.

    Args:
        timeline_data: Timeline data dictionary
        output_file: Output file path (if None, display plot)
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        return

    entries = timeline_data["timeline"]["entries"]
    strategy = timeline_data["strategy"]

    if not entries:
        print("No timeline entries to plot")
        return

    # Extract data
    timestamps = [datetime.fromisoformat(e["timestamp_iso"]) for e in entries]

    # Get all model IDs
    all_model_ids = set()
    for entry in entries:
        all_model_ids.update(entry["instance_counts"].keys())
    all_model_ids = sorted(all_model_ids)

    # Extract counts for each model
    model_counts = {model_id: [] for model_id in all_model_ids}
    for entry in entries:
        counts = entry["instance_counts"]
        for model_id in all_model_ids:
            model_counts[model_id].append(counts.get(model_id, 0))

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Instance counts over time (stacked area)
    colors = plt.cm.Set3(range(len(all_model_ids)))
    ax1.stackplot(timestamps, *[model_counts[mid] for mid in all_model_ids],
                  labels=all_model_ids, colors=colors, alpha=0.8)

    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Instance Count', fontsize=12)
    ax1.set_title(f'Instance Allocation Over Time - Strategy: {strategy}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Mark event types with vertical lines
    for i, entry in enumerate(entries):
        timestamp = timestamps[i]
        if entry["event_type"] == "deploy_migration":
            ax1.axvline(timestamp, color='green', linestyle='--', alpha=0.5, linewidth=1)
        elif entry["event_type"] == "auto_optimize":
            ax1.axvline(timestamp, color='orange', linestyle=':', alpha=0.5, linewidth=1)

    # Plot 2: Instance counts per model (line plot)
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    for i, model_id in enumerate(all_model_ids):
        marker = markers[i % len(markers)]
        ax2.plot(timestamps, model_counts[model_id], marker=marker,
                label=model_id, linewidth=2, markersize=8, alpha=0.8)

    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Instance Count', fontsize=12)
    ax2.set_title('Instance Count by Model', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add event markers
    for i, entry in enumerate(entries):
        timestamp = timestamps[i]
        if entry["event_type"] == "deploy_migration":
            ax2.axvline(timestamp, color='green', linestyle='--', alpha=0.5, linewidth=1,
                       label='Deploy' if i == 0 else '')
        elif entry["event_type"] == "auto_optimize":
            ax2.axvline(timestamp, color='orange', linestyle=':', alpha=0.5, linewidth=1,
                       label='Auto-opt' if i == 0 else '')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    else:
        print("\nDisplaying plot...")
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize instance deployment timeline from experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "results_file",
        type=str,
        help="Path to results JSON file"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Strategy name to visualize (default: first strategy in file)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for plot (default: display plot)"
    )

    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print summary only, no visualization"
    )

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.results_file).exists():
        print(f"Error: File not found: {args.results_file}")
        sys.exit(1)

    # Load timeline data
    timeline_data = load_timeline_from_results(args.results_file, args.strategy)
    if not timeline_data:
        sys.exit(1)

    # Print summary
    print_timeline_summary(timeline_data)

    # Create visualization if requested
    if not args.summary_only:
        if args.output:
            output_file = args.output
        else:
            # Auto-generate output filename
            strategy = timeline_data["strategy"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"timeline_{strategy}_{timestamp}.png"

        plot_timeline(timeline_data, output_file if args.output else None)


if __name__ == "__main__":
    main()
