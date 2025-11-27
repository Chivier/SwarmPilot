#!/usr/bin/env python3
"""
OCR+LLM Workflow - Simulation Mode

This script runs the OCR+LLM workflow (A→B) using sleep models for simulation.

Architecture:
- Thread 1 (A Submitter): Submit A tasks (OCR) with Poisson arrivals
- Thread 2 (A Receiver): Receive A results → trigger B
- Thread 3 (B Submitter): Submit B tasks (LLM)
- Thread 4 (B Receiver): Receive B results → mark complete

Workflow Pattern:
- A: OCR - extract text from image (simulated)
- B: LLM - process extracted text (simulated)

Usage:
    python -m type4_ocr_llm.simulation.test_workflow_sim \\
        --num-workflows 50 --qps 2.0 --strategies min_time,probabilistic --duration 300
"""

import random
import sys
import threading
import time
from pathlib import Path
from queue import Queue

import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common import (
    configure_logging,
    MetricsCollector,
    RateLimiter,
    ensure_directory,
    setup_scheduler_strategies,
    clear_scheduler_tasks,
    create_base_parser,
    parse_strategies,
    generate_strategy_comparison_table,
)
from type4_ocr_llm.config import OCRLLMConfig
from type4_ocr_llm.workflow_data import generate_dummy_images
from type4_ocr_llm.submitters import ATaskSubmitter, BTaskSubmitter
from type4_ocr_llm.receivers import ATaskReceiver, BTaskReceiver


def add_type4_args(parser):
    """Add type4 OCR+LLM specific arguments."""
    parser.add_argument(
        "--sleep-time-a-config",
        type=str,
        default=None,
        help="JSON config for A (OCR) sleep time distribution. "
             "Can be a file path or inline JSON string. "
             "Example: '{\"type\": \"normal\", \"mean\": 0.8, \"std\": 0.2}'"
    )
    parser.add_argument(
        "--sleep-time-b-config",
        type=str,
        default=None,
        help="JSON config for B (LLM) sleep time distribution. "
             "Can be a file path or inline JSON string. "
             "Example: '{\"type\": \"normal\", \"mean\": 5.0, \"std\": 0.5}'"
    )
    parser.add_argument(
        "--sleep-time-seed",
        type=int,
        default=42,
        help="Random seed for sleep time distribution sampling (default: 42)."
    )
    return parser


def run_single_experiment(config, images, logger, strategy_name=None):
    """Run a single experiment with the given configuration.

    Args:
        config: OCRLLMConfig instance
        images: List of base64-encoded images to use
        logger: Logger instance
        strategy_name: Optional strategy name (for logging and task ID generation)

    Returns:
        Dict with strategy results containing task_metrics and workflow_metrics
    """

    # Update config.strategy if strategy_name is provided
    if strategy_name:
        config.strategy = strategy_name
        logger.info(f"Using strategy: {strategy_name}")

    logger.info("="*70)
    if strategy_name:
        logger.info(f"OCR+LLM Workflow Simulation - Strategy: {strategy_name}")
    else:
        logger.info("OCR+LLM Workflow Simulation")
    logger.info("="*70)
    logger.info(f"QPS: {config.qps}")
    logger.info(f"Duration: {config.duration}s")
    logger.info(f"Workflows: {config.num_workflows}")
    logger.info(f"Scheduler A (OCR): {config.scheduler_a_url}")
    logger.info(f"Scheduler B (LLM): {config.scheduler_b_url}")
    if strategy_name:
        logger.info(f"Strategy: {strategy_name}")
    logger.info("="*70)

    # ========================================================================
    # Shared State
    # ========================================================================

    # Workflow states (shared across all threads)
    workflow_states = {}
    state_lock = threading.Lock()

    # Queues for inter-thread communication
    a_result_queue = Queue()  # A receiver → B submitter

    # Metrics collector
    metrics = MetricsCollector(custom_logger=logger)

    # Rate limiter (shared across all submitters)
    rate_limiter = RateLimiter(rate=config.qps)

    # ========================================================================
    # Create Components
    # ========================================================================

    logger.info("Initializing components...")

    # A Submitter (Thread 1) - OCR
    a_submitter = ATaskSubmitter(
        name="ASubmitter",
        images=images,
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        scheduler_url=config.scheduler_a_url,
        qps=config.qps,
        duration=config.duration,
        rate_limiter=rate_limiter,
        metrics=metrics,
    )

    # Pre-generate task IDs for receivers to subscribe to
    strategy = getattr(config, 'strategy', 'probabilistic')
    a_task_ids = [f"task-A-{strategy}-workflow-{i:04d}" for i in range(config.num_workflows)]
    b_task_ids = [f"task-B-{strategy}-workflow-{i:04d}" for i in range(config.num_workflows)]

    # B Submitter (Thread 3) - LLM
    b_submitter = BTaskSubmitter(
        name="BSubmitter",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        a_result_queue=a_result_queue,
        scheduler_url=config.scheduler_b_url,
        rate_limiter=rate_limiter,
        metrics=metrics,
    )

    # A Receiver (Thread 2)
    a_receiver = ATaskReceiver(
        name="AReceiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        b_submitter=b_submitter,
        a_result_queue=a_result_queue,
        task_ids=a_task_ids,
        scheduler_url=config.scheduler_a_url,
        model_id=config.model_a_id,
        metrics=metrics,
    )

    # B Receiver (Thread 4)
    b_receiver = BTaskReceiver(
        name="BReceiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        task_ids=b_task_ids,
        scheduler_url=config.scheduler_b_url,
        model_id=config.model_b_id,
        metrics=metrics,
    )

    # ========================================================================
    # Start All Threads
    # ========================================================================

    logger.info("Starting all threads...")
    start_time = time.time()

    # Start receivers first (so they're ready to receive)
    a_receiver.start()
    b_receiver.start()

    # Wait a moment for receivers to connect
    time.sleep(1.0)

    # Start submitters
    b_submitter.start()
    a_submitter.start()  # Start A last (it drives the workflow)

    logger.info("All threads started successfully")

    # ========================================================================
    # Wait for Duration or Early Completion
    # ========================================================================

    logger.info(f"Running experiment for up to {config.duration} seconds...")

    # Calculate target workflows for early stopping
    import math
    num_warmup = config.num_warmup
    num_non_warmup = config.num_workflows - num_warmup
    target_non_warmup = math.ceil(num_non_warmup * config.portion_stats)

    # Generate the list of workflow IDs that must complete for metrics
    target_workflow_ids = set()
    for i in range(num_warmup):
        target_workflow_ids.add(f"workflow-{i:04d}")
    for i in range(num_warmup, num_warmup + target_non_warmup):
        target_workflow_ids.add(f"workflow-{i:04d}")

    target_completion = len(target_workflow_ids)
    logger.info(f"Early stopping target: {target_completion} specific workflows by submission order "
                f"(warmup={num_warmup}, non-warmup needed for stats={target_non_warmup})")

    # Print progress every 10 seconds
    elapsed = 0
    early_stop = False
    while elapsed < config.duration:
        time.sleep(10)
        elapsed = time.time() - start_time

        with state_lock:
            total_workflows = len(workflow_states)
            completed = sum(1 for w in workflow_states.values() if w.is_complete())
            # Count how many of the TARGET workflows have completed
            target_completed = sum(
                1 for wid, w in workflow_states.items()
                if wid in target_workflow_ids and w.is_complete()
            )

        target_pct = 100 * target_completed / target_completion if target_completion > 0 else 0
        total_pct = 100 * completed / total_workflows if total_workflows > 0 else 0
        logger.info(
            f"Progress [{elapsed:.1f}s]: "
            f"target={target_completed}/{target_completion} ({target_pct:.1f}%), "
            f"total={completed}/{total_workflows} ({total_pct:.1f}%)"
        )

        # Early stopping
        if target_completed >= target_completion:
            logger.info(f"Early stopping: {target_completed} target workflows completed")
            logger.info("All workflows needed for metrics calculation are complete!")
            early_stop = True
            break

    # ========================================================================
    # Stop All Threads
    # ========================================================================

    logger.info("Stopping all threads...")

    # Stop submitters first (no new tasks)
    a_submitter.stop()
    b_submitter.stop()

    # Wait for submitters to finish
    a_submitter.join(timeout=10)
    b_submitter.join(timeout=10)

    # Stop receivers
    a_receiver.stop()
    b_receiver.stop()

    # Wait for receivers to finish
    a_receiver.join(timeout=10)
    b_receiver.join(timeout=10)

    logger.info("All threads stopped")

    # ========================================================================
    # Final Statistics
    # ========================================================================

    end_time = time.time()
    total_time = end_time - start_time

    logger.info("="*70)
    logger.info("Experiment Complete")
    logger.info("="*70)
    logger.info(f"Total runtime: {total_time:.2f}s")
    if early_stop:
        logger.info(f"Stopped early: All {target_completion} workflows needed for metrics completed")

    with state_lock:
        total_workflows = len(workflow_states)
        completed_workflows = sum(1 for w in workflow_states.values() if w.is_complete())

    logger.info(f"Total workflows: {total_workflows}")
    logger.info(f"Completed workflows: {completed_workflows}")
    logger.info(f"Completion rate: {100*completed_workflows/total_workflows if total_workflows > 0 else 0:.1f}%")

    logger.info("\nSubmitter Statistics:")
    logger.info(f"  A (OCR): {a_submitter.submitted_count} submitted, {a_submitter.failed_count} failed")
    logger.info(f"  B (LLM): {b_submitter.submitted_count} submitted, {b_submitter.failed_count} failed")

    logger.info("\nReceiver Statistics:")
    logger.info(f"  A (OCR): {a_receiver.received_count} received")
    logger.info(f"  B (LLM): {b_receiver.received_count} received, {b_receiver.completed_workflows} workflows completed")

    # ========================================================================
    # Export Metrics
    # ========================================================================

    output_dir = Path(config.output_dir)
    ensure_directory(str(output_dir))

    # Generate strategy-specific metrics filename
    base_metrics_file = config.metrics_file
    if strategy_name:
        if '.' in base_metrics_file:
            name, ext = base_metrics_file.rsplit('.', 1)
            metrics_filename = f"{name}_{strategy_name}.{ext}"
        else:
            metrics_filename = f"{base_metrics_file}_{strategy_name}"
    else:
        metrics_filename = base_metrics_file

    metrics_path = output_dir / metrics_filename
    logger.info(f"\nExporting metrics to: {metrics_path}")

    # Export to JSON (with portion_stats filtering)
    metrics.export_to_json(str(metrics_path), portion_stats=config.portion_stats)

    # Generate detailed text report with per-task-type metrics (Avg, P90, P99)
    report = metrics.generate_detailed_text_report(
        task_types=["A", "B"],
        portion_stats=config.portion_stats
    )
    print("\n" + report)

    logger.info("="*70)
    if strategy_name:
        logger.info(f"Simulation complete for strategy: {strategy_name}!")
    else:
        logger.info("Simulation complete!")
    logger.info("="*70)

    # Collect and return strategy results for comparison table
    task_types = ["A", "B"]
    task_metrics_result = {}
    for task_type in task_types:
        task_metrics_result[task_type] = metrics.get_task_metrics_by_type(
            task_type, exclude_warmup=True, portion_stats=config.portion_stats
        )

    # Get workflow metrics
    workflow_durations = []
    with metrics._workflow_lock:
        total_non_warmup = len(metrics._non_warmup_workflow_order)
        num_to_include = int(total_non_warmup * config.portion_stats)
        included_workflow_ids = set(metrics._non_warmup_workflow_order[:num_to_include])

        completed_workflows = [
            w for w in metrics.workflow_metrics
            if w.workflow_id in included_workflow_ids and w.end_time
        ]
        workflow_durations = [w.total_duration for w in completed_workflows if w.total_duration]

    if workflow_durations:
        wf_avg = float(np.mean(workflow_durations))
        wf_p90 = float(np.percentile(workflow_durations, 90))
        wf_p99 = float(np.percentile(workflow_durations, 99))
    else:
        wf_avg = wf_p90 = wf_p99 = 0.0

    return {
        'task_metrics': task_metrics_result,
        'workflow_metrics': {
            'avg': wf_avg,
            'p90': wf_p90,
            'p99': wf_p99,
        }
    }


def main():
    """Main entry point - handles strategy management and experiment orchestration."""

    # ========================================================================
    # Set Random Seeds for Reproducibility
    # ========================================================================
    random.seed(42)
    np.random.seed(42)

    # ========================================================================
    # Parse Command Line Arguments
    # ========================================================================

    parser = create_base_parser(description="OCR+LLM Workflow - Simulation Mode")
    parser = add_type4_args(parser)
    args = parser.parse_args()

    # Parse and validate strategies
    try:
        strategies = parse_strategies(args.strategies)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # ========================================================================
    # Configuration and Setup
    # ========================================================================

    # Compute warmup count from ratio
    warmup_count = int(args.num_workflows * args.warmup)

    # Create config with hardcoded simulation mode
    config = OCRLLMConfig(
        mode="simulation",  # Hardcoded for simulation
        qps=args.qps,
        duration=args.duration,
        num_workflows=args.num_workflows,
        num_warmup=warmup_count,
        strategies=strategies,
        portion_stats=args.portion_stats,
        sleep_time_a_config=args.sleep_time_a_config,
        sleep_time_b_config=args.sleep_time_b_config,
        sleep_time_seed=args.sleep_time_seed,
    )

    logger = configure_logging(level="INFO")

    logger.info(f"Mode: {config.mode} (hardcoded for simulation)")
    logger.info(f"Model A (OCR): {config.model_a_id}")
    logger.info(f"Model B (LLM): {config.model_b_id}")
    logger.info(f"Scheduler A: {config.scheduler_a_url}")
    logger.info(f"Scheduler B: {config.scheduler_b_url}")

    # Generate dummy images for simulation
    logger.info(f"Generating {config.num_workflows} dummy images for simulation...")
    images = generate_dummy_images(config.num_workflows)
    logger.info(f"Generated {len(images)} dummy images")

    # ========================================================================
    # Strategy Management
    # ========================================================================

    logger.info("="*70)
    logger.info("Strategy-based Testing Mode")
    logger.info(f"Will test {len(strategies)} strategies: {', '.join(strategies)}")
    logger.info("="*70)

    # Set default quantiles if not specified
    quantiles = config.quantiles if config.quantiles else [0.1, 0.25, 0.5, 0.75, 0.99]

    # Collect results from all strategies for comparison
    all_strategy_results = {}

    # Run experiment for each strategy
    for strategy_name in strategies:
        logger.info("\n" + "="*70)
        logger.info(f"Setting up strategy: {strategy_name}")
        logger.info("="*70)

        # Clear all tasks from schedulers before each strategy test
        logger.info("Clearing scheduler task queues...")
        if not clear_scheduler_tasks(config.scheduler_a_url, logger):
            logger.error("Failed to clear Scheduler A tasks, skipping strategy")
            continue
        if config.scheduler_b_url and config.scheduler_b_url != config.scheduler_a_url:
            if not clear_scheduler_tasks(config.scheduler_b_url, logger):
                logger.error("Failed to clear Scheduler B tasks, skipping strategy")
                continue

        # Setup this strategy on schedulers
        strategy_results = setup_scheduler_strategies(
            strategy_name=strategy_name,
            scheduler_a_url=config.scheduler_a_url,
            scheduler_b_url=config.scheduler_b_url,
            target_quantile=config.target_quantile,
            quantiles=quantiles,
            custom_logger=logger
        )

        if not strategy_results.get(strategy_name, False):
            logger.error(f"Failed to setup strategy: {strategy_name}")
            logger.error("Skipping this strategy")
            continue

        logger.info(f"Strategy {strategy_name} set up successfully")

        # Run the experiment
        logger.info("\n" + "="*70)
        logger.info(f"Running experiment with strategy: {strategy_name}")
        logger.info("="*70)
        experiment_results = run_single_experiment(config, images, logger, strategy_name=strategy_name)

        # Store results for comparison
        all_strategy_results[strategy_name] = experiment_results

        logger.info(f"\nCompleted experiment for strategy: {strategy_name}")

    # Generate strategy comparison table
    if len(all_strategy_results) > 1:
        comparison_table = generate_strategy_comparison_table(
            all_strategy_results,
            task_types=["A", "B"]
        )
        print(comparison_table)
        logger.info("Strategy comparison table generated")

    logger.info("\n" + "="*70)
    logger.info("All strategy experiments completed!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
