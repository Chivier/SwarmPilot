#!/usr/bin/env python3
"""
Quick test script to run OOD experiment and generate Gantt chart.

This script:
1. Starts services (scheduler + sleep model instances)
2. Runs a short OOD experiment
3. Collects task execution data
4. Generates Gantt chart visualization
5. Stops services

Usage:
    python -m type5_ood_recovery.test_gantt_visualization
"""

import sys
import time
import json
import threading
import random
import numpy as np
from pathlib import Path
from typing import Dict, List

import httpx

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import (
    configure_logging,
    MetricsCollector,
    RateLimiter,
    setup_scheduler_strategies,
    clear_scheduler_tasks,
)
from type5_ood_recovery.config import OODRecoveryConfig
from type5_ood_recovery.task_data import TaskGenerator, pre_generate_tasks
from type5_ood_recovery.submitter import OODTaskSubmitter
from type5_ood_recovery.receiver import OODTaskReceiver
from type5_ood_recovery.plot_gantt import (
    TaskExecution,
    fetch_schedule_info,
    plot_gantt_chart,
    plot_gantt_with_queuing,
)


def run_experiment_and_plot_gantt(
    config: OODRecoveryConfig,
    logger,
    seed: int = 42,
    output_dir: Path = Path("output_gantt"),
) -> None:
    """
    Run experiment and generate Gantt chart.

    Args:
        config: OODRecoveryConfig
        logger: Logger instance
        seed: Random seed
        output_dir: Output directory for plots
    """
    # Clear tasks before starting
    logger.info("Clearing tasks from scheduler...")
    clear_scheduler_tasks(config.scheduler_url, logger)

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)

    # Pre-generate tasks
    logger.info(f"Pre-generating {config.num_tasks} tasks (seed={seed})...")
    tasks = pre_generate_tasks(config, seed=seed)

    # Create task generator for continuous mode
    task_generator = TaskGenerator(seed=seed + 10000)
    for i in range(len(tasks)):
        task_generator.generate_task(i)

    # Shared state
    task_lookup: Dict = {}
    phase_transition_event = threading.Event()
    metrics = MetricsCollector(custom_logger=logger)
    rate_limiter = RateLimiter(rate=config.qps)

    # Create components
    receiver = OODTaskReceiver(
        config=config,
        task_lookup=task_lookup,
        phase_transition_event=phase_transition_event,
        metrics=metrics,
    )

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

    # Start threads
    logger.info("Starting receiver and submitter threads...")
    start_time = time.time()

    receiver.start()
    time.sleep(1.0)  # Wait for receiver to connect
    submitter.start()

    # Wait for completion
    logger.info(f"Running experiment for up to {config.duration} seconds...")
    elapsed = 0
    while elapsed < config.duration:
        time.sleep(2)
        elapsed = time.time() - start_time

        submitted = submitter.get_phase_submit_counts()
        completed = receiver.get_phase_complete_counts()
        total_submitted = sum(submitted.values())
        total_completed = sum(completed.values())

        logger.info(
            f"[{elapsed:.0f}s] "
            f"Submitted: P1={submitted.get(1, 0)}, P2={submitted.get(2, 0)}, P3={submitted.get(3, 0)} | "
            f"Completed: P1={completed.get(1, 0)}, P2={completed.get(2, 0)}, P3={completed.get(3, 0)} | "
            f"Transition: {'Yes' if phase_transition_event.is_set() else 'No'}"
        )

        if total_completed >= config.num_tasks:
            logger.info("All tasks completed!")
            break

    # Stop threads
    logger.info("Stopping threads...")
    submitter.stop()
    submitter.join(timeout=10)
    receiver.stop()
    receiver.join(timeout=10)

    end_time = time.time()
    total_time = end_time - start_time

    # Summary
    logger.info("=" * 60)
    logger.info("Experiment Complete")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Tasks completed: {sum(receiver.get_phase_complete_counts().values())}")

    # Collect task execution data
    logger.info("")
    logger.info("Collecting task execution data for Gantt chart...")

    # Fetch schedule info from scheduler
    task_to_instance = fetch_schedule_info(config.scheduler_url, config.model_id)
    logger.info(f"  Found {len(task_to_instance)} task-instance mappings")

    # Build TaskExecution list
    task_executions = []
    for task_id, task in task_lookup.items():
        if task.is_complete and task.submit_time and task.complete_time:
            instance_id = task_to_instance.get(task_id, "unknown")
            exec_start = task.complete_time - task.actual_sleep_time

            task_executions.append(TaskExecution(
                task_id=task_id,
                task_index=task.task_index,
                phase=task.phase,
                instance_id=instance_id,
                submit_time=task.submit_time - start_time,
                complete_time=task.complete_time - start_time,
                sleep_time_s=task.actual_sleep_time,
                exec_start_time=exec_start - start_time,
            ))

    task_executions.sort(key=lambda t: t.exec_start_time)
    logger.info(f"  Collected {len(task_executions)} task executions")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate Gantt charts
    logger.info("")
    logger.info("Generating Gantt charts...")

    mode_suffix = "_baseline" if config.no_recovery else "_recovery"

    # Simple Gantt chart
    gantt_file = output_dir / f"gantt_chart{mode_suffix}.png"
    plot_gantt_chart(
        task_executions,
        gantt_file,
        title=f"Task Execution Gantt Chart ({'Baseline' if config.no_recovery else 'Recovery'} Mode)"
    )

    # Gantt chart with queueing
    gantt_queue_file = output_dir / f"gantt_chart_with_queue{mode_suffix}.png"
    plot_gantt_with_queuing(
        task_executions,
        gantt_queue_file,
        title=f"Task Execution with Queueing ({'Baseline' if config.no_recovery else 'Recovery'} Mode)"
    )

    # Export task execution data
    export_data = {
        "config": {
            "num_tasks": config.num_tasks,
            "phase1_count": config.phase1_count,
            "qps": config.qps,
            "no_recovery": config.no_recovery,
            "phase23_distribution": config.phase23_distribution,
            "normal_mean": config.normal_mean,
            "normal_std": config.normal_std,
        },
        "summary": {
            "total_time_s": total_time,
            "tasks_completed": len(task_executions),
            "instances": list(set(t.instance_id for t in task_executions)),
        },
        "task_executions": [
            {
                "task_id": t.task_id,
                "task_index": t.task_index,
                "phase": t.phase,
                "instance_id": t.instance_id,
                "submit_time": t.submit_time,
                "exec_start_time": t.exec_start_time,
                "complete_time": t.complete_time,
                "sleep_time_s": t.sleep_time_s,
            }
            for t in task_executions
        ],
    }

    data_file = output_dir / f"task_executions{mode_suffix}.json"
    with open(data_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    logger.info(f"Task execution data exported to: {data_file}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info(f"Gantt charts saved to: {output_dir}")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run OOD experiment and generate Gantt chart"
    )
    parser.add_argument("--num-tasks", type=int, default=50, help="Number of tasks")
    parser.add_argument("--qps", type=float, default=2.0, help="Queries per second")
    parser.add_argument("--duration", type=int, default=300, help="Max duration (s)")
    parser.add_argument("--phase1-count", type=int, default=20, help="Phase 1 tasks")
    parser.add_argument("--phase2-transition-count", type=int, default=5, help="Phase 2 count before transition")
    parser.add_argument("--no-recovery", action="store_true", help="Baseline mode")
    parser.add_argument("--scheduler-url", type=str, default="http://127.0.0.1:8100")
    parser.add_argument("--model-id", type=str, default="sleep_model_a")
    parser.add_argument("--output-dir", type=str, default="output_gantt")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    logger = configure_logging(level="INFO")

    config = OODRecoveryConfig(
        num_tasks=args.num_tasks,
        qps=args.qps,
        duration=args.duration,
        phase1_count=args.phase1_count,
        phase2_transition_count=args.phase2_transition_count,
        no_recovery=args.no_recovery,
        scheduler_url=args.scheduler_url,
        model_id=args.model_id,
        phase23_distribution="normal",
        normal_mean=10.0,
        normal_std=1.0,
    )

    logger.info("Configuration:")
    logger.info(str(config))

    # Setup scheduler strategy
    logger.info("")
    logger.info("Setting up scheduler strategy...")
    setup_scheduler_strategies(
        strategy_name="probabilistic",
        scheduler_a_url=args.scheduler_url,
        scheduler_b_url=None,
        target_quantile=0.9,
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.99],
        custom_logger=logger,
    )

    run_experiment_and_plot_gantt(
        config=config,
        logger=logger,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
