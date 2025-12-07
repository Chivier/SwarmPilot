#!/usr/bin/env python3
"""
OOD Recovery Experiment - Simulation Mode

This script tests the system's ability to recover when task runtime distribution
changes and the predictor re-trains.

Three-Phase Pattern:
- Phase 1 (30%): Correct prediction - sleep and exp_runtime both use 0.2x scale
- Phase 2 (OOD): Wrong prediction - sleep uses 0.5x, exp_runtime uses 0.2x (mismatch)
- Phase 3 (Recovery): Corrected prediction - sleep and exp_runtime both use 0.5x

Phase Transition:
- Phase 1 → Phase 2: After phase1_count tasks submitted
- Phase 2 → Phase 3: When first Phase 2 task completes (triggered by receiver)

Baseline Mode (--no-recovery):
- No Phase 2 → Phase 3 transition
- exp_runtime stays at wrong 0.2x scale throughout

Usage:
    python -m type5_ood_recovery.simulation.test_ood_sim \\
        --num-tasks 100 --qps 2.0 --duration 300

    # Baseline mode (no recovery)
    python -m type5_ood_recovery.simulation.test_ood_sim \\
        --num-tasks 100 --qps 2.0 --no-recovery
"""

import argparse
import random
import sys
import threading
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common import (
    configure_logging,
    MetricsCollector,
    RateLimiter,
    ensure_directory,
    setup_scheduler_strategies,
    clear_scheduler_tasks,
)
from type5_ood_recovery.config import OODRecoveryConfig
from type5_ood_recovery.task_data import (
    OODTaskData,
    TaskGenerator,
    pre_generate_tasks,
    get_sleep_time_statistics,
)
from type5_ood_recovery.submitter import OODTaskSubmitter
from type5_ood_recovery.receiver import OODTaskReceiver


def run_experiment(config: OODRecoveryConfig, logger, seed: int = 42) -> Dict:
    """
    Run a single OOD recovery experiment.

    Args:
        config: OODRecoveryConfig instance
        logger: Logger instance
        seed: Random seed for task generation (default: 42)

    Returns:
        Dictionary of results including throughput and per-phase metrics
    """
    mode_str = "Baseline (no recovery)" if config.no_recovery else "Recovery"

    # Clear all tasks from scheduler before starting
    logger.info("Clearing all tasks from scheduler before experiment...")
    clear_scheduler_tasks(config.scheduler_url, logger)
    logger.info("Task clearing complete")

    logger.info("=" * 70)
    logger.info(f"OOD Recovery Experiment - {mode_str}")
    logger.info("=" * 70)
    logger.info(f"Target completions: {config.num_tasks}")
    logger.info(f"  Phase 1: {config.phase1_count} initial tasks (correct prediction)")
    logger.info(f"  Phase 2+3: {config.phase23_count} initial tasks (OOD + recovery)")
    logger.info(f"  Continuous mode: Submit until {config.num_tasks} completions received")
    logger.info(f"QPS: {config.qps}")
    logger.info(f"Duration: {config.duration}s")
    logger.info(f"Phase2 transition count: {config.phase2_transition_count}")
    logger.info(f"Scheduler: {config.scheduler_url}")
    logger.info(f"Model: {config.model_id}")
    logger.info(f"Strategy: {config.strategy}")
    logger.info("=" * 70)

    # ========================================================================
    # Pre-generate Tasks and Create Task Generator
    # ========================================================================

    logger.info(f"Pre-generating initial task data (seed={seed})...")
    tasks = pre_generate_tasks(config, seed=seed)
    logger.info(f"Pre-generated {len(tasks)} initial tasks")

    # Create task generator for continuous submission mode
    # Uses a different seed offset to ensure unique tasks after initial batch
    continuous_seed = seed + 10000  # Offset to avoid overlap with pre-generated tasks
    task_generator = TaskGenerator(seed=continuous_seed)
    # Skip indices already used by pre-generated tasks
    for i in range(len(tasks)):
        task_generator.generate_task(i)  # Consume RNG state
    logger.info(f"Task generator initialized for continuous mode (offset seed={continuous_seed})")

    # Print first few tasks for reproducibility verification
    logger.info("First 5 tasks (for reproducibility verification):")
    for t in tasks[:5]:
        logger.info(
            f"  {t.task_id}: base_sleep={t.base_sleep_time:.4f}s, "
            f"exp_runtime_base={t.exp_runtime_base:.4f}s, "
            f"phase23_scale={t.phase23_random_scale:.4f}"
        )

    # Verify Phase 2/3 sleep_time consistency
    # This shows the actual_sleep_time that Phase 2/3 tasks will have
    # (same formula for both phases: base_sleep_time * phase23_random_scale)
    logger.info("")
    logger.info("Phase 2/3 tasks sleep_time verification (first 5 after Phase 1):")
    logger.info("  NOTE: These values are IDENTICAL between Recovery and Baseline modes")
    logger.info("        because both Phase 2 and Phase 3 use the same formula:")
    logger.info("        actual_sleep_time = base_sleep_time * phase23_random_scale")
    phase23_start = config.phase1_count
    for i, t in enumerate(tasks[phase23_start:phase23_start + 5]):
        # Calculate what the actual_sleep_time will be (same for Phase 2 or 3)
        expected_sleep = t.base_sleep_time * t.phase23_random_scale
        logger.info(
            f"  {t.task_id} (idx={t.task_index}): "
            f"sleep_time={expected_sleep:.4f}s "
            f"(base={t.base_sleep_time:.4f} × scale={t.phase23_random_scale:.4f})"
        )

    # Print sleep time statistics
    stats = get_sleep_time_statistics()
    logger.info(f"Base sleep time distribution:")
    logger.info(f"  Count: {stats['count']}")
    logger.info(f"  Mean: {stats['mean']:.3f}s, Std: {stats['std']:.3f}s")
    logger.info(f"  Min: {stats['min']:.3f}s, Max: {stats['max']:.3f}s")

    # ========================================================================
    # Shared State
    # ========================================================================

    # Task lookup (shared between submitter and receiver)
    task_lookup: Dict[str, OODTaskData] = {}

    # Phase transition event (set by receiver when first Phase 2 completes)
    phase_transition_event = threading.Event()

    # Metrics collector
    metrics = MetricsCollector(custom_logger=logger)

    # Rate limiter - use phase1_qps for initial rate
    initial_qps = config.phase1_qps if config.phase1_qps is not None else config.qps
    rate_limiter = RateLimiter(rate=initial_qps)

    # ========================================================================
    # Create Components
    # ========================================================================

    logger.info("Initializing components...")

    # Task Receiver (created first so submitter can access its completion count)
    receiver = OODTaskReceiver(
        config=config,
        task_lookup=task_lookup,
        phase_transition_event=phase_transition_event,
        metrics=metrics,
    )

    # Task Submitter (with task generator and completion callback for continuous mode)
    submitter = OODTaskSubmitter(
        config=config,
        tasks=tasks,
        task_lookup=task_lookup,
        phase_transition_event=phase_transition_event,
        task_generator=task_generator,
        get_total_completed=receiver.get_total_completed,
        metrics=metrics,
        rate_limiter=rate_limiter,
    )

    # ========================================================================
    # Start Threads
    # ========================================================================

    logger.info("Starting threads...")
    start_time = time.time()

    # Start receiver first (so it's ready to receive)
    receiver.start()

    # Wait for receiver to connect
    time.sleep(0.2)  # Brief wait for receiver to connect

    # Start submitter
    submitter.start()

    logger.info("All threads started successfully")

    # ========================================================================
    # Wait for Completion or Duration
    # ========================================================================

    logger.info(f"Running experiment for up to {config.duration} seconds...")

    # Throughput trend tracking
    throughput_trend = []  # List of (timestamp, elapsed, completed, throughput, phase_completed)
    last_completed = 0
    trend_interval = 1.0  # Record throughput every 1 second
    last_trend_time = start_time

    # Print progress with real-time throughput
    elapsed = 0
    progress_interval = 5  # Print progress every 5 seconds
    last_progress_elapsed = 0  # Track last printed elapsed to avoid duplicates
    last_progress_completed = 0  # Track completed count at last progress output
    last_progress_time = start_time  # Track time at last progress output

    while elapsed < config.duration:
        time.sleep(0.1)  # Fast polling for responsive throughput tracking
        current_time = time.time()
        elapsed = current_time - start_time

        # Get current counts
        submitted = submitter.get_phase_submit_counts()
        completed = receiver.get_phase_complete_counts()
        total_submitted = sum(submitted.values())
        total_completed = sum(completed.values())

        # Record throughput trend (every trend_interval seconds)
        if current_time - last_trend_time >= trend_interval:
            # Calculate instantaneous throughput (tasks completed in last interval)
            tasks_in_interval = total_completed - last_completed
            interval_duration = current_time - last_trend_time
            instantaneous_throughput = tasks_in_interval / interval_duration if interval_duration > 0 else 0.0

            # Calculate cumulative throughput
            cumulative_throughput = total_completed / elapsed if elapsed > 0 else 0.0

            # Record trend point
            throughput_trend.append({
                "timestamp": current_time,
                "elapsed_s": round(elapsed, 2),
                "total_completed": total_completed,
                "phase_completed": dict(completed),
                "instantaneous_throughput": round(instantaneous_throughput, 4),
                "cumulative_throughput": round(cumulative_throughput, 4),
                "phase_transition": phase_transition_event.is_set(),
            })

            last_completed = total_completed
            last_trend_time = current_time

        # Check for phase transition
        transition_status = "Yes" if phase_transition_event.is_set() else "No"

        # Print real-time progress every progress_interval seconds
        current_progress_elapsed = int(elapsed // progress_interval) * progress_interval
        if current_progress_elapsed > last_progress_elapsed and current_progress_elapsed > 0:
            # Calculate throughput for this output interval (between two log outputs)
            interval_completed = total_completed - last_progress_completed
            interval_time = current_time - last_progress_time
            interval_throughput = interval_completed / interval_time if interval_time > 0 else 0.0

            # Calculate cumulative throughput
            cumul_tp = total_completed / elapsed if elapsed > 0 else 0.0

            # Calculate current phase
            if not phase_transition_event.is_set():
                if total_submitted <= config.phase1_count:
                    current_phase = "P1"
                else:
                    current_phase = "P2"
            else:
                current_phase = "P3"

            logger.info(
                f"[{elapsed:6.1f}s] "
                f"Phase={current_phase} | "
                f"Submitted: {total_submitted:3d} (P1:{submitted.get(1, 0):2d} P2:{submitted.get(2, 0):2d} P3:{submitted.get(3, 0):2d}) | "
                f"Completed: {total_completed:3d} (P1:{completed.get(1, 0):2d} P2:{completed.get(2, 0):2d} P3:{completed.get(3, 0):2d}) | "
                f"Throughput: {interval_throughput:5.2f} tasks/s (avg: {cumul_tp:5.2f}) | "
                f"Recovery: {transition_status}"
            )

            # Update tracking for next interval
            last_progress_elapsed = current_progress_elapsed
            last_progress_completed = total_completed
            last_progress_time = current_time

        # Early stopping: all tasks completed
        if total_completed >= config.num_tasks:
            logger.info("All tasks completed - stopping early")
            break

        # Also stop if submitter finished and receiver caught up
        if total_submitted >= config.num_tasks and total_completed >= total_submitted:
            logger.info("All submitted tasks completed - stopping early")
            break

    # ========================================================================
    # Stop Threads
    # ========================================================================

    logger.info("Stopping threads...")

    # Stop submitter first
    submitter.stop()
    submitter.join(timeout=10)

    # Stop receiver
    receiver.stop()
    receiver.join(timeout=10)

    logger.info("All threads stopped")

    # ========================================================================
    # Calculate Results
    # ========================================================================

    end_time = time.time()
    total_time = end_time - start_time

    # Get final counts
    submit_counts = submitter.get_phase_submit_counts()
    complete_counts = receiver.get_phase_complete_counts()
    total_submitted = sum(submit_counts.values())
    total_completed = sum(complete_counts.values())

    # Calculate throughput
    throughput = total_completed / total_time if total_time > 0 else 0.0

    # Calculate per-phase latencies from completed tasks
    completed_tasks = receiver.get_completed_tasks()
    phase_latencies = {1: [], 2: [], 3: []}
    for task in completed_tasks:
        if task.duration is not None:
            phase_latencies[task.phase].append(task.duration)

    # Calculate statistics per phase
    phase_stats = {}
    for phase, latencies in phase_latencies.items():
        if latencies:
            phase_stats[phase] = {
                "count": len(latencies),
                "avg": float(np.mean(latencies)),
                "p50": float(np.percentile(latencies, 50)),
                "p90": float(np.percentile(latencies, 90)),
                "p99": float(np.percentile(latencies, 99)),
                "min": float(np.min(latencies)),
                "max": float(np.max(latencies)),
            }
        else:
            phase_stats[phase] = {
                "count": 0,
                "avg": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p99": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

    # Phase transition time
    transition_time = receiver.get_phase_transition_time()
    transition_delay = None
    if transition_time and start_time:
        transition_delay = transition_time - start_time

    # ========================================================================
    # Print Results
    # ========================================================================

    logger.info("=" * 70)
    logger.info("Experiment Complete")
    logger.info("=" * 70)
    logger.info(f"Mode: {mode_str}")
    logger.info(f"Total runtime: {total_time:.2f}s")
    logger.info(f"Tasks submitted: {total_submitted}")
    logger.info(f"Tasks completed: {total_completed}")
    logger.info(f"Throughput: {throughput:.4f} tasks/second")

    if transition_delay is not None:
        logger.info(f"Phase 2→3 transition at: {transition_delay:.2f}s")

    logger.info("")
    logger.info("Per-Phase Results:")
    logger.info("-" * 50)
    for phase in [1, 2, 3]:
        stats = phase_stats[phase]
        if stats["count"] > 0:
            logger.info(
                f"Phase {phase}: "
                f"n={stats['count']}, "
                f"avg={stats['avg']:.3f}s, "
                f"p50={stats['p50']:.3f}s, "
                f"p90={stats['p90']:.3f}s, "
                f"p99={stats['p99']:.3f}s"
            )
        else:
            logger.info(f"Phase {phase}: No completed tasks")

    # Print throughput trend summary
    if throughput_trend:
        logger.info("")
        logger.info("Throughput Trend Summary:")
        logger.info("-" * 50)
        # Show key points: start, phase transition, end
        trend_len = len(throughput_trend)
        key_points = []

        # First point
        if trend_len > 0:
            key_points.append(("Start", throughput_trend[0]))

        # Phase transition point (if exists)
        for i, point in enumerate(throughput_trend):
            if point["phase_transition"] and (i == 0 or not throughput_trend[i-1]["phase_transition"]):
                key_points.append(("Transition", point))
                break

        # Middle points (sample every 20%)
        for pct in [25, 50, 75]:
            idx = int(trend_len * pct / 100)
            if 0 < idx < trend_len - 1:
                key_points.append((f"{pct}%", throughput_trend[idx]))

        # Last point
        if trend_len > 1:
            key_points.append(("End", throughput_trend[-1]))

        for label, point in key_points:
            logger.info(
                f"  {label:12s} [{point['elapsed_s']:6.1f}s]: "
                f"completed={point['total_completed']:3d}, "
                f"instant={point['instantaneous_throughput']:.2f}, "
                f"cumulative={point['cumulative_throughput']:.2f} tasks/s"
            )
        logger.info(f"  Total trend points recorded: {trend_len}")

    logger.info("=" * 70)

    # ========================================================================
    # Export Results
    # ========================================================================

    output_dir = ensure_directory(config.output_dir)

    # Build task_executions list for Gantt chart visualization
    # Instance ID is now captured directly from task result callback
    task_executions = []
    tasks_with_instance = 0
    for task in completed_tasks:
        if task.submit_time is not None and task.complete_time is not None:
            instance_id = task.instance_id if task.instance_id else "unknown"
            if task.instance_id:
                tasks_with_instance += 1
            task_executions.append({
                "task_id": task.task_id,
                "task_index": task.task_index,
                "phase": task.phase,
                "submit_time": task.submit_time,
                "complete_time": task.complete_time,
                "sleep_time_s": task.actual_sleep_time,
                "exp_runtime_ms": task.exp_runtime_ms,
                "instance_id": instance_id,
            })
    logger.info(f"Built {len(task_executions)} task executions ({tasks_with_instance} with instance_id)")

    # Build results dictionary
    results = {
        "config": {
            "mode": mode_str,
            "num_tasks": config.num_tasks,
            "phase1_count": config.phase1_count,
            "phase23_count": config.phase23_count,
            "qps": config.qps,
            "duration": config.duration,
            "no_recovery": config.no_recovery,
            "phase1_scale": config.phase1_scale,
            "peak_threshold": config.peak_threshold,
            "peak1_factor": config.peak1_factor,
            "peak2_factor": config.peak2_factor,
            "phase2_transition_count": config.phase2_transition_count,
            "strategy": config.strategy,
            "scheduler_url": config.scheduler_url,
            "model_id": config.model_id,
        },
        "summary": {
            "total_runtime_s": total_time,
            "tasks_submitted": total_submitted,
            "tasks_completed": total_completed,
            "throughput_tasks_per_second": throughput,
            "phase_transition_delay_s": transition_delay,
        },
        "submit_counts": submit_counts,
        "complete_counts": complete_counts,
        "phase_stats": phase_stats,
        "throughput_trend": throughput_trend,
        "task_executions": task_executions,
        "experiment_start_time": start_time,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results to JSON
    import json
    results_file = output_dir / config.metrics_file
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results exported to: {results_file}")

    # Store results_file path in results for plotting
    results["_results_file"] = str(results_file)

    return results


def generate_throughput_plot(
    results_file: Path,
    output_dir: Path,
    mode_suffix: str,
    logger,
) -> Optional[Path]:
    """
    Generate throughput plot from experiment results.

    Args:
        results_file: Path to the metrics JSON file
        output_dir: Directory to save the plot
        mode_suffix: Suffix for the plot filename (e.g., "_recovery" or "_baseline")
        logger: Logger instance

    Returns:
        Path to the generated plot, or None if failed
    """
    try:
        from type5_ood_recovery.plot_throughput import plot_single, load_metrics

        plot_file = output_dir / f"throughput_plot{mode_suffix}.png"
        metrics = load_metrics(str(results_file))
        plot_single(metrics, str(plot_file))
        logger.info(f"Throughput plot saved to: {plot_file}")
        return plot_file
    except ImportError as e:
        logger.warning(f"Could not import plotting module: {e}")
        logger.warning("Skipping plot generation. Run manually with:")
        logger.warning(f"  python -m type5_ood_recovery.plot_throughput {results_file}")
        return None
    except Exception as e:
        logger.warning(f"Failed to generate plot: {e}")
        return None


def generate_gantt_chart(
    results_file: Path,
    output_dir: Path,
    mode_suffix: str,
    scheduler_url: str,
    logger,
) -> Optional[Path]:
    """
    Generate Gantt chart from experiment results.

    Args:
        results_file: Path to the metrics JSON file
        output_dir: Directory to save the plot
        mode_suffix: Suffix for the plot filename (e.g., "_recovery" or "_baseline")
        scheduler_url: Scheduler URL for fetching instance assignments
        logger: Logger instance

    Returns:
        Path to the generated plot, or None if failed
    """
    try:
        from type5_ood_recovery.plot_gantt import (
            load_experiment_data,
            plot_gantt_chart,
            plot_gantt_with_queuing,
        )

        # Load experiment data
        task_executions, _ = load_experiment_data(results_file, scheduler_url)

        if not task_executions:
            logger.warning("No task execution data found for Gantt chart")
            return None

        # Determine title based on mode
        mode_name = "Recovery" if "recovery" in mode_suffix else "Baseline"
        title = f"Task Execution Gantt Chart ({mode_name} Mode)"

        # Generate simple Gantt chart
        gantt_file = output_dir / f"gantt_chart{mode_suffix}.png"
        plot_gantt_chart(task_executions, gantt_file, title=title)
        logger.info(f"Gantt chart saved to: {gantt_file}")

        # Generate Gantt chart with queueing visualization
        gantt_queue_file = output_dir / f"gantt_chart_with_queue{mode_suffix}.png"
        plot_gantt_with_queuing(
            task_executions,
            gantt_queue_file,
            title=f"Task Execution with Queueing ({mode_name} Mode)"
        )
        logger.info(f"Gantt chart with queueing saved to: {gantt_queue_file}")

        return gantt_file
    except ImportError as e:
        logger.warning(f"Could not import Gantt plotting module: {e}")
        logger.warning("Skipping Gantt chart generation.")
        return None
    except Exception as e:
        logger.warning(f"Failed to generate Gantt chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_all_plots(
    results_file: Path,
    output_dir: Path,
    mode_suffix: str,
    scheduler_url: str,
    logger,
) -> None:
    """
    Generate all plots (throughput and Gantt charts) from experiment results.

    Args:
        results_file: Path to the metrics JSON file
        output_dir: Directory to save plots
        mode_suffix: Suffix for plot filenames
        scheduler_url: Scheduler URL for Gantt chart
        logger: Logger instance
    """
    logger.info("")
    logger.info("=" * 50)
    logger.info("Generating all plots...")
    logger.info("=" * 50)

    # 1. Generate throughput plot
    logger.info("")
    logger.info("1. Generating throughput plot...")
    generate_throughput_plot(
        results_file=results_file,
        output_dir=output_dir,
        mode_suffix=mode_suffix,
        logger=logger,
    )

    # 2. Generate Gantt charts
    logger.info("")
    logger.info("2. Generating Gantt charts...")
    generate_gantt_chart(
        results_file=results_file,
        output_dir=output_dir,
        mode_suffix=mode_suffix,
        scheduler_url=scheduler_url,
        logger=logger,
    )

    logger.info("")
    logger.info("All plots generated successfully!")
    logger.info("=" * 50)


def main():
    """Main entry point."""

    # ========================================================================
    # Parse Command Line Arguments (early for seed)
    # ========================================================================

    parser = argparse.ArgumentParser(
        description="OOD Recovery Experiment - Simulation Mode"
    )

    # Task parameters
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=100,
        help="Total number of tasks to submit (default: 100)"
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=1.0,
        help="Target queries per second (default: 1.0)"
    )
    parser.add_argument(
        "--phase1-qps",
        type=float,
        default=None,
        help="Phase 1 QPS (default: same as --qps)"
    )
    parser.add_argument(
        "--phase23-qps",
        type=float,
        default=None,
        help="Phase 2/3 QPS (default: same as --qps)"
    )
    parser.add_argument(
        "--runtime-scale",
        type=float,
        default=1.0,
        help="Global scaling factor for task runtime (default: 1.0)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=600,
        help="Maximum experiment duration in seconds (default: 600)"
    )

    # Phase configuration
    parser.add_argument(
        "--phase1-count",
        type=int,
        default=100,
        help="Fixed number of tasks in Phase 1 (default: 100)"
    )
    parser.add_argument(
        "--phase2-transition-count",
        type=int,
        default=10,
        help="Number of Phase 2 tasks before transitioning to Phase 3 (default: 10). "
             "Interpreted as submission or completion count based on --transition-on-submit."
    )
    parser.add_argument(
        "--transition-on-submit",
        action="store_true",
        default=False,
        help="Trigger Phase 2→3 transition based on submission count"
    )
    parser.add_argument(
        "--transition-on-complete",
        action="store_true",
        default=True,
        help="Trigger Phase 2→3 transition based on completion count (default, realistic behavior)"
    )

    # Baseline mode
    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help="Baseline mode: no Phase 2→3 transition (exp_runtime stays wrong)"
    )

    # Scheduler configuration
    parser.add_argument(
        "--scheduler-url",
        type=str,
        default="http://127.0.0.1:8100",
        help="Scheduler endpoint URL (default: http://127.0.0.1:8100)"
    )
    parser.add_argument(
        "--predictor-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Predictor endpoint URL (default: http://127.0.0.1:8000)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="sleep_model_a",
        help="Model ID for task submission (default: sleep_model_a)"
    )
    parser.add_argument(
        "--skip-service-check",
        action="store_true",
        help="Skip automatic service health checks (use when services are user-started)"
    )

    # Phase 2/3 distribution configuration
    parser.add_argument(
        "--phase23-distribution",
        type=str,
        default="four_peak",
        choices=["normal", "uniform", "peak_dependent", "four_peak"],
        help="Distribution mode for Phase 2/3 sleep times (default: four_peak)"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_ood",
        help="Output directory for results (default: output_ood)"
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    # Plotting
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable automatic plot generation after experiment"
    )

    args = parser.parse_args()

    # ========================================================================
    # Set Random Seeds for Reproducibility
    # ========================================================================
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ========================================================================
    # Configuration
    # ========================================================================

    # Determine output suffix based on mode
    if args.no_recovery:
        output_suffix = "_baseline"
    else:
        output_suffix = "_recovery"

    # Determine transition mode: --transition-on-submit overrides default completion-based
    transition_on_submit = args.transition_on_submit

    config = OODRecoveryConfig(
        num_tasks=args.num_tasks,
        qps=args.qps,
        phase1_qps=args.phase1_qps,
        phase23_qps=args.phase23_qps,
        runtime_scale=args.runtime_scale,
        duration=args.duration,
        phase1_count=args.phase1_count,
        phase2_transition_count=args.phase2_transition_count,
        transition_on_submit=transition_on_submit,
        no_recovery=args.no_recovery,
        scheduler_url=args.scheduler_url,
        model_id=args.model_id,
        output_dir=args.output_dir,
        metrics_file=f"metrics{output_suffix}.json",
        phase23_distribution=args.phase23_distribution,
    )

    logger = configure_logging(level="INFO")

    logger.info("=" * 70)
    logger.info("OOD Recovery Experiment Configuration")
    logger.info("=" * 70)
    logger.info(str(config))
    logger.info("")
    logger.info("Phase-Specific QPS Parameters:")
    logger.info(f"  --qps (default): {config.qps}")
    logger.info(f"  --phase1-qps:    {config.phase1_qps} (effective: {config.phase1_qps if config.phase1_qps else config.qps})")
    logger.info(f"  --phase23-qps:   {config.phase23_qps} (effective: {config.phase23_qps if config.phase23_qps else config.qps})")
    logger.info(f"  --runtime-scale: {config.runtime_scale}")
    logger.info("=" * 70)

    # ========================================================================
    # Setup Scheduler Strategy
    # ========================================================================

    logger.info("")
    logger.info("Setting up scheduler strategy...")

    # Clear tasks first
    clear_scheduler_tasks(config.scheduler_url, logger)

    # Setup probabilistic strategy
    strategy_results = setup_scheduler_strategies(
        strategy_name=config.strategy,
        scheduler_a_url=config.scheduler_url,
        scheduler_b_url=None,  # Only one scheduler
        target_quantile=config.target_quantile,
        quantiles=config.quantiles,
        custom_logger=logger,
    )

    if not strategy_results.get(config.strategy, False):
        logger.error(f"Failed to setup strategy: {config.strategy}")
        sys.exit(1)

    logger.info(f"Strategy '{config.strategy}' set up successfully")

    # ========================================================================
    # Run Experiment
    # ========================================================================

    results = run_experiment(config, logger, seed=args.seed)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Experiment finished!")
    logger.info(f"Throughput: {results['summary']['throughput_tasks_per_second']:.4f} tasks/second")
    logger.info("=" * 70)

    # ========================================================================
    # Generate All Plots (Throughput + Gantt Charts)
    # ========================================================================

    if not args.no_plot:
        results_file = Path(results.get("_results_file", ""))
        if results_file.exists():
            generate_all_plots(
                results_file=results_file,
                output_dir=Path(args.output_dir),
                mode_suffix=output_suffix,
                scheduler_url=args.scheduler_url,
                logger=logger,
            )
        else:
            logger.warning(f"Results file not found: {results_file}")
    else:
        logger.info("Plot generation skipped (--no-plot specified)")


if __name__ == "__main__":
    main()
