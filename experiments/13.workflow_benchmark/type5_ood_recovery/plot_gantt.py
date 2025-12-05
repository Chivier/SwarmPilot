#!/usr/bin/env python3
"""
Gantt Chart Visualization for OOD Recovery Experiment.

This script visualizes task distribution across instances as a Gantt chart.
It shows:
- Task execution timeline on each instance
- Phase-based coloring (Phase 1/2/3)
- Task start time = completion_time - sleep_time (calculated)

Data sources:
- Task submit/complete times: From experiment metrics (completed_tasks)
- Instance assignment: From scheduler /task/schedule_info endpoint
- Sleep time: From task_data
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import httpx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


@dataclass
class TaskExecution:
    """Task execution data for Gantt chart."""
    task_id: str
    task_index: int
    phase: int
    instance_id: str
    submit_time: float
    complete_time: float
    sleep_time_s: float  # Actual sleep time in seconds
    exec_start_time: float  # Calculated: complete_time - sleep_time

    @property
    def exec_duration(self) -> float:
        """Execution duration (same as sleep_time for simulation)."""
        return self.sleep_time_s


def fetch_schedule_info(
    scheduler_url: str,
    model_id: str = "sleep_model_a",
    limit: int = 10000,
) -> Dict[str, str]:
    """
    Fetch task-to-instance mapping from scheduler.

    Args:
        scheduler_url: Scheduler endpoint URL
        model_id: Model ID to filter by
        limit: Maximum tasks to fetch

    Returns:
        Dict mapping task_id to assigned_instance
    """
    url = f"{scheduler_url}/task/schedule_info"
    params = {"model_id": model_id, "limit": limit}

    response = httpx.get(url, params=params, timeout=30.0)
    response.raise_for_status()

    data = response.json()

    # Build task_id -> instance_id mapping
    task_to_instance = {}
    for task in data.get("tasks", []):
        task_id = task.get("task_id", "")
        instance_id = task.get("assigned_instance", "")
        if task_id and instance_id:
            task_to_instance[task_id] = instance_id

    return task_to_instance


def load_experiment_data(
    metrics_file: Path,
    scheduler_url: str,
) -> Tuple[List[TaskExecution], float]:
    """
    Load experiment data and combine with schedule info.

    Args:
        metrics_file: Path to metrics JSON file
        scheduler_url: Scheduler URL for fetching schedule_info

    Returns:
        Tuple of (list of TaskExecution, experiment_start_time)
    """
    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Get model_id from config
    model_id = metrics.get("config", {}).get("model_id", "sleep_model_a")

    # Fetch schedule info from scheduler
    print(f"Fetching schedule info from {scheduler_url}...")
    task_to_instance = fetch_schedule_info(scheduler_url, model_id)
    print(f"  Found {len(task_to_instance)} task-instance mappings")

    # Check if we have task-level data in metrics
    # The experiment stores task data in receiver.get_completed_tasks()
    # which gets saved as part of the metrics

    # For now, we need to reconstruct from the throughput_trend
    # which doesn't have per-task data.
    #
    # SOLUTION: Modify the experiment to also export per-task data
    # For this script, we'll need that data or run a new experiment

    print("Note: Per-task execution data needs to be exported from experiment.")
    print("This script expects task data in metrics['task_executions']")

    task_executions = []
    experiment_start = 0.0

    if "task_executions" in metrics:
        experiment_start = metrics.get("experiment_start_time", 0.0)
        for task_data in metrics["task_executions"]:
            task_id = task_data["task_id"]
            instance_id = task_to_instance.get(task_id, "unknown")

            task_executions.append(TaskExecution(
                task_id=task_id,
                task_index=task_data["task_index"],
                phase=task_data["phase"],
                instance_id=instance_id,
                submit_time=task_data["submit_time"] - experiment_start,
                complete_time=task_data["complete_time"] - experiment_start,
                sleep_time_s=task_data["sleep_time_s"],
                exec_start_time=task_data["complete_time"] - experiment_start - task_data["sleep_time_s"],
            ))

    return task_executions, experiment_start


def collect_task_data_live(
    config,
    tasks: List,
    task_lookup: Dict,
    scheduler_url: str,
    experiment_start: float,
) -> List[TaskExecution]:
    """
    Collect task execution data after experiment completes.

    Args:
        config: OODRecoveryConfig
        tasks: List of OODTaskData from experiment
        task_lookup: Dict mapping task_id to OODTaskData
        scheduler_url: Scheduler URL
        experiment_start: Experiment start timestamp

    Returns:
        List of TaskExecution for Gantt chart
    """
    # Fetch schedule info
    task_to_instance = fetch_schedule_info(scheduler_url, config.model_id)

    task_executions = []
    for task in task_lookup.values():
        if task.is_complete and task.submit_time and task.complete_time:
            instance_id = task_to_instance.get(task.task_id, "unknown")

            task_executions.append(TaskExecution(
                task_id=task.task_id,
                task_index=task.task_index,
                phase=task.phase,
                instance_id=instance_id,
                submit_time=task.submit_time - experiment_start,
                complete_time=task.complete_time - experiment_start,
                sleep_time_s=task.actual_sleep_time,
                exec_start_time=task.complete_time - experiment_start - task.actual_sleep_time,
            ))

    # Sort by execution start time
    task_executions.sort(key=lambda t: t.exec_start_time)

    return task_executions


def plot_gantt_chart(
    task_executions: List[TaskExecution],
    output_file: Path,
    title: str = "Task Execution Gantt Chart",
) -> None:
    """
    Plot Gantt chart showing task distribution across instances.

    Args:
        task_executions: List of TaskExecution data
        output_file: Path to save the plot
        title: Plot title
    """
    if not task_executions:
        print("No task execution data to plot!")
        return

    # Group tasks by instance
    instances = sorted(set(t.instance_id for t in task_executions))
    instance_to_idx = {inst: i for i, inst in enumerate(instances)}

    # Phase colors
    phase_colors = {
        1: '#4CAF50',  # Green - Warmup (correct prediction)
        2: '#FF9800',  # Orange - OOD (wrong prediction)
        3: '#2196F3',  # Blue - Recovery (corrected prediction)
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(6, len(instances) * 0.8)))

    # Plot each task as a horizontal bar
    bar_height = 0.6
    for task in task_executions:
        y = instance_to_idx[task.instance_id]
        color = phase_colors.get(task.phase, '#9E9E9E')

        # Draw execution bar (from exec_start to complete)
        ax.barh(
            y=y,
            width=task.exec_duration,
            left=task.exec_start_time,
            height=bar_height,
            color=color,
            edgecolor='white',
            linewidth=0.5,
            alpha=0.8,
        )

    # Configure axes
    ax.set_yticks(range(len(instances)))
    ax.set_yticklabels(instances)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Instance')
    ax.set_title(title)

    # Add legend
    legend_patches = [
        mpatches.Patch(color=phase_colors[1], label='Phase 1 (Warmup)', alpha=0.8),
        mpatches.Patch(color=phase_colors[2], label='Phase 2 (OOD)', alpha=0.8),
        mpatches.Patch(color=phase_colors[3], label='Phase 3 (Recovery)', alpha=0.8),
    ]
    ax.legend(handles=legend_patches, loc='upper right')

    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Save
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Gantt chart saved to: {output_file}")
    plt.close()


def plot_gantt_with_queuing(
    task_executions: List[TaskExecution],
    output_file: Path,
    title: str = "Task Execution Gantt Chart (with Queueing)",
) -> None:
    """
    Plot Gantt chart showing both queueing time and execution time.

    Args:
        task_executions: List of TaskExecution data
        output_file: Path to save the plot
        title: Plot title
    """
    if not task_executions:
        print("No task execution data to plot!")
        return

    # Group tasks by instance
    instances = sorted(set(t.instance_id for t in task_executions))
    instance_to_idx = {inst: i for i, inst in enumerate(instances)}

    # Phase colors
    phase_colors = {
        1: '#4CAF50',  # Green
        2: '#FF9800',  # Orange
        3: '#2196F3',  # Blue
    }

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, max(10, len(instances) * 1.5)),
                              gridspec_kw={'height_ratios': [3, 1]})

    ax_gantt = axes[0]
    ax_queue = axes[1]

    # Plot Gantt chart
    bar_height = 0.6
    for task in task_executions:
        y = instance_to_idx[task.instance_id]
        color = phase_colors.get(task.phase, '#9E9E9E')

        # Queue time (from submit to exec_start) - lighter color
        queue_time = task.exec_start_time - task.submit_time
        if queue_time > 0:
            ax_gantt.barh(
                y=y,
                width=queue_time,
                left=task.submit_time,
                height=bar_height,
                color=color,
                edgecolor='white',
                linewidth=0.3,
                alpha=0.3,
                hatch='//',
            )

        # Execution time (from exec_start to complete)
        ax_gantt.barh(
            y=y,
            width=task.exec_duration,
            left=task.exec_start_time,
            height=bar_height,
            color=color,
            edgecolor='white',
            linewidth=0.5,
            alpha=0.9,
        )

    # Configure Gantt axes
    ax_gantt.set_yticks(range(len(instances)))
    ax_gantt.set_yticklabels(instances)
    ax_gantt.set_xlabel('Time (seconds)')
    ax_gantt.set_ylabel('Instance')
    ax_gantt.set_title(title)
    ax_gantt.grid(True, axis='x', alpha=0.3)
    ax_gantt.set_axisbelow(True)

    # Add legend
    legend_patches = [
        mpatches.Patch(color=phase_colors[1], label='Phase 1 (Warmup)', alpha=0.9),
        mpatches.Patch(color=phase_colors[2], label='Phase 2 (OOD)', alpha=0.9),
        mpatches.Patch(color=phase_colors[3], label='Phase 3 (Recovery)', alpha=0.9),
        mpatches.Patch(facecolor='gray', alpha=0.3, hatch='//', label='Queue Time'),
    ]
    ax_gantt.legend(handles=legend_patches, loc='upper right')

    # Plot queue length over time
    times = []
    queue_lengths = {inst: [] for inst in instances}

    # Calculate queue length at each event
    events = []
    for task in task_executions:
        events.append((task.submit_time, 'submit', task.instance_id))
        events.append((task.complete_time, 'complete', task.instance_id))
    events.sort(key=lambda x: x[0])

    current_queue = {inst: 0 for inst in instances}
    for event_time, event_type, instance_id in events:
        if event_type == 'submit':
            current_queue[instance_id] += 1
        else:
            current_queue[instance_id] = max(0, current_queue[instance_id] - 1)

        times.append(event_time)
        for inst in instances:
            queue_lengths[inst].append(current_queue[inst])

    # Plot queue lengths
    for inst, lengths in queue_lengths.items():
        ax_queue.step(times, lengths, where='post', label=inst, alpha=0.8)

    ax_queue.set_xlabel('Time (seconds)')
    ax_queue.set_ylabel('Queue Length')
    ax_queue.set_title('Queue Length per Instance Over Time')
    ax_queue.legend(loc='upper right')
    ax_queue.grid(True, alpha=0.3)
    ax_queue.set_xlim(ax_gantt.get_xlim())

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Gantt chart with queueing saved to: {output_file}")
    plt.close()


def main():
    """Main entry point for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Generate Gantt chart from OOD experiment data"
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        required=True,
        help="Path to metrics JSON file"
    )
    parser.add_argument(
        "--scheduler-url",
        type=str,
        default="http://127.0.0.1:8100",
        help="Scheduler URL for fetching schedule_info"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gantt_chart.png",
        help="Output file path"
    )
    parser.add_argument(
        "--with-queueing",
        action="store_true",
        help="Include queueing time visualization"
    )

    args = parser.parse_args()

    metrics_file = Path(args.metrics_file)
    if not metrics_file.exists():
        print(f"Error: Metrics file not found: {metrics_file}")
        sys.exit(1)

    task_executions, _ = load_experiment_data(metrics_file, args.scheduler_url)

    if not task_executions:
        print("No task execution data found. Make sure the experiment exports task_executions.")
        sys.exit(1)

    output_file = Path(args.output)

    if args.with_queueing:
        plot_gantt_with_queuing(task_executions, output_file)
    else:
        plot_gantt_chart(task_executions, output_file)


if __name__ == "__main__":
    main()
