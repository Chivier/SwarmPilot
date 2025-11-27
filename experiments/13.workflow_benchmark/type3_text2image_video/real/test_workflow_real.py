#!/usr/bin/env python3
"""
Text2Image+Video Workflow - Real Cluster Mode

This script runs the Text2Image+Video workflow (A→C→B) in real cluster mode
using actual model IDs and task inputs instead of sleep simulations.

Key differences from simulation mode:
- Uses real model inputs instead of sleep_time
- A: Uses sentence + max_tokens for LLM
- C: Uses sentence + width/height for FLUX image generation
- B: Uses prompt + negative_prompt + frames for T2VID

Architecture:
- Thread 1: A submitter (Poisson process, QPS-controlled)
- Thread 2: A receiver → triggers C
- Thread 3: C submitter
- Thread 4: C receiver → triggers B
- Thread 5: B submitter
- Thread 6: B receiver (with loop control)

Workflow Pattern:
- A: caption → positive prompt (LLM)
- C: positive prompt → image (FLUX, with resolution)
- B: video generation (1-4 iterations, negative_prompt="blur")

Usage:
    python -m type3_text2image_video.real.test_workflow_real \\
        --num-workflows 50 --qps 2.0 --strategies min_time,probabilistic --duration 300
"""

import random
import sys
import threading
import time
from pathlib import Path
from queue import Queue

import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common import (
    configure_logging,
    MetricsCollector,
    RateLimiter,
    ensure_directory,
    setup_scheduler_strategies,
    clear_scheduler_tasks,
    create_base_parser,
    add_type3_args,
    parse_strategies,
    generate_strategy_comparison_table,
)
from type3_text2image_video.config import Text2ImageVideoConfig
from type3_text2image_video.workflow_data import load_captions
from type3_text2image_video.submitters import ATaskSubmitter, CTaskSubmitter, BTaskSubmitter
from type3_text2image_video.receivers import ATaskReceiver, CTaskReceiver, BTaskReceiver


def run_single_experiment(config, captions, logger, strategy_name=None):
    """Run a single experiment with the given configuration.

    Returns:
        Dict with strategy results containing task_metrics and workflow_metrics
    """

    # Update config.strategy if strategy_name is provided
    if strategy_name:
        config.strategy = strategy_name

    logger.info("=" * 80)
    logger.info(f"Text2Image+Video Workflow - Real Cluster Mode - Strategy: {strategy_name or 'default'}")
    logger.info("=" * 80)
    logger.info(f"QPS: {config.qps}")
    logger.info(f"Duration: {config.duration}s")
    logger.info(f"Workflows: {config.num_workflows}")
    logger.info(f"Max B loops config: {config.get_max_b_loops_config()}")
    logger.info(f"Frame count config: {config.get_frame_count_config()}")
    logger.info(f"Resolution config: {config.get_resolution_config()}")
    logger.info(f"Scheduler A (LLM): {config.scheduler_a_url}")
    logger.info(f"Scheduler C (FLUX): {config.scheduler_c_url}")
    logger.info(f"Scheduler B (T2VID): {config.scheduler_b_url}")

    # Initialize components
    metrics = MetricsCollector(logger)
    rate_limiter = RateLimiter(rate=config.qps)
    workflow_states = {}
    state_lock = threading.Lock()

    # Create inter-thread queues
    a_result_queue = Queue()
    c_result_queue = Queue()

    # Pre-generate task IDs for receivers to subscribe to
    strategy = getattr(config, 'strategy', 'probabilistic')
    a_task_ids = [f"task-A-{strategy}-workflow-{i:04d}" for i in range(config.num_workflows)]
    c_task_ids = [f"task-C-{strategy}-workflow-{i:04d}" for i in range(config.num_workflows)]

    # Create submitters first (ATaskSubmitter populates workflow_states)
    a_submitter = ATaskSubmitter(
        name="ASubmitter",
        captions=captions,
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        scheduler_url=config.scheduler_a_url,
        qps=config.qps,
        duration=config.duration,
        rate_limiter=rate_limiter,
        metrics=metrics,
        custom_logger=logger
    )

    c_submitter = CTaskSubmitter(
        name="CSubmitter",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        a_result_queue=a_result_queue,
        scheduler_url=config.scheduler_c_url,
        rate_limiter=None,
        metrics=metrics,
        custom_logger=logger
    )

    b_submitter = BTaskSubmitter(
        name="BSubmitter",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        c_result_queue=c_result_queue,
        scheduler_url=config.scheduler_b_url,
        rate_limiter=None,
        metrics=metrics,
        custom_logger=logger
    )

    # Create receivers
    a_receiver = ATaskReceiver(
        name="AReceiver",
        config=config,
        model_id=config.model_a_id,
        c_submitter=c_submitter,
        a_result_queue=a_result_queue,
        task_ids=a_task_ids,
        workflow_states=workflow_states,
        state_lock=state_lock,
        scheduler_url=config.scheduler_a_url,
        metrics=metrics,
        custom_logger=logger
    )

    c_receiver = CTaskReceiver(
        name="CReceiver",
        config=config,
        model_id=config.model_c_id,
        b_submitter=b_submitter,
        c_result_queue=c_result_queue,
        task_ids=c_task_ids,
        workflow_states=workflow_states,
        state_lock=state_lock,
        scheduler_url=config.scheduler_c_url,
        metrics=metrics,
        custom_logger=logger
    )

    # Generate all B task IDs (including all loop iterations)
    # With distribution-based sampling, each workflow may have different max_b_loops,
    # so we use each workflow's actual max_b_loops from the pre-generated workflow data
    b_task_ids = []
    with state_lock:
        for workflow_id, workflow_data in workflow_states.items():
            workflow_num = workflow_id.split('-')[-1]
            for loop in range(1, workflow_data.max_b_loops + 1):
                b_task_ids.append(f"task-B{loop}-{strategy}-workflow-{workflow_num}")

    b_receiver = BTaskReceiver(
        name="BReceiver",
        config=config,
        model_id=config.model_b_id,
        b_submitter=b_submitter,
        task_ids=b_task_ids,
        workflow_states=workflow_states,
        state_lock=state_lock,
        scheduler_url=config.scheduler_b_url,
        metrics=metrics,
        custom_logger=logger
    )

    # Start all threads
    logger.info("Starting experiment...")
    start_time = time.time()

    a_submitter.start()
    a_receiver.start()
    c_submitter.start()
    c_receiver.start()
    b_submitter.start()
    b_receiver.start()

    # Calculate target workflows for early stopping based on SUBMISSION ORDER
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

    # Monitor progress
    early_stop = False
    try:
        while time.time() - start_time < config.duration:
            time.sleep(10)

            with state_lock:
                completed = b_receiver.completed_workflows
                total = config.num_workflows
                progress_pct = (completed / total * 100) if total > 0 else 0

                # Count how many of the TARGET workflows have completed
                target_completed = sum(
                    1 for wid, w in workflow_states.items()
                    if wid in target_workflow_ids and w.is_complete()
                )

                target_pct = 100 * target_completed / target_completion if target_completion > 0 else 0
                elapsed = time.time() - start_time
                logger.info(
                    f"Progress [{elapsed:.1f}s]: "
                    f"target={target_completed}/{target_completion} ({target_pct:.1f}%), "
                    f"total={completed}/{total} ({progress_pct:.1f}%)"
                )

                # Early stopping: check if all TARGET workflows (by submission order) have completed
                if target_completed >= target_completion:
                    logger.info(f"Early stopping: {target_completed} target workflows completed")
                    logger.info("All workflows needed for metrics calculation are complete!")
                    early_stop = True
                    break

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")

    # Stop all threads
    logger.info("Stopping all threads...")
    a_submitter.stop()
    a_receiver.stop()
    c_submitter.stop()
    c_receiver.stop()
    b_submitter.stop()
    b_receiver.stop()

    # Wait for threads to finish
    a_submitter.join()
    a_receiver.join()
    c_submitter.join()
    c_receiver.join()
    b_submitter.join()
    b_receiver.join()

    # Calculate final statistics
    elapsed_time = time.time() - start_time
    with state_lock:
        completed_workflows = b_receiver.completed_workflows

    logger.info("=" * 80)
    logger.info("Experiment Complete")
    logger.info("=" * 80)
    logger.info(f"Duration: {elapsed_time:.2f}s")
    if early_stop:
        logger.info(f"Stopped early: All {target_completion} workflows needed for metrics completed")
    logger.info(f"Completed workflows: {completed_workflows}/{config.num_workflows}")
    logger.info(f"Completion rate: {completed_workflows / config.num_workflows * 100:.1f}%")

    # Export metrics (with portion_stats filtering)
    output_dir = ensure_directory(config.output_dir)

    # Generate strategy-specific metrics filename to prevent overwriting
    base_metrics_file = config.metrics_file
    if strategy_name:
        # Insert strategy name before extension: metrics.json -> metrics_probabilistic.json
        if '.' in base_metrics_file:
            name, ext = base_metrics_file.rsplit('.', 1)
            metrics_filename = f"{name}_{strategy_name}.{ext}"
        else:
            metrics_filename = f"{base_metrics_file}_{strategy_name}"
    else:
        metrics_filename = base_metrics_file

    metrics_file = output_dir / metrics_filename
    metrics.export_to_json(str(metrics_file), portion_stats=config.portion_stats)
    logger.info(f"Metrics exported to: {metrics_file}")

    # Print detailed metrics report with per-task-type metrics (Avg, P90, P99)
    print("\n" + metrics.generate_detailed_text_report(
        task_types=["A", "C", "B"],
        portion_stats=config.portion_stats
    ))

    # Collect and return strategy results for comparison table
    task_types = ["A", "C", "B"]
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

    parser = create_base_parser(description="Text2Image+Video Workflow - Real Cluster Mode")
    parser = add_type3_args(parser)
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

    # Create config with hardcoded real mode
    config = Text2ImageVideoConfig(
        mode="real",  # Hardcoded for real cluster
        qps=args.qps,
        duration=args.duration,
        num_workflows=args.num_workflows,
        max_b_loops=args.max_b_loops,
        num_warmup=warmup_count,
        strategies=strategies,
        portion_stats=args.portion_stats,
        frame_count=args.frame_count,
        frame_count_config=args.frame_count_config,
        max_b_loops_config=args.max_b_loops_config,
        frame_count_seed=args.frame_count_seed,
        max_b_loops_seed=args.max_b_loops_seed,
        resolution=args.resolution,
        resolution_config=args.resolution_config,
        resolution_seed=args.resolution_seed,
    )

    logger = configure_logging(level="INFO")

    logger.info(f"Mode: {config.mode} (hardcoded for real cluster)")
    logger.info(f"Model A (LLM): {config.model_a_id}")
    logger.info(f"Model C (FLUX): {config.model_c_id}")
    logger.info(f"Model B (T2VID): {config.model_b_id}")
    logger.info(f"Scheduler A: {config.scheduler_a_url}")
    logger.info(f"Scheduler C: {config.scheduler_c_url}")
    logger.info(f"Scheduler B: {config.scheduler_b_url}")

    # Load captions
    caption_path = Path(__file__).parent.parent.parent.parent.parent / config.caption_file
    if not caption_path.exists():
        logger.warning(f"Caption file not found: {caption_path}")
        logger.warning("Using dummy captions")
        captions = [f"Sample caption {i}" for i in range(100)]
    else:
        captions = load_captions(str(caption_path))

    logger.info(f"Loaded {len(captions)} captions")

    # ========================================================================
    # Strategy Management
    # ========================================================================

    logger.info("=" * 70)
    logger.info("Strategy-based Testing Mode")
    logger.info(f"Will test {len(strategies)} strategies: {', '.join(strategies)}")
    logger.info("=" * 70)

    # Set default quantiles if not specified
    quantiles = config.quantiles if config.quantiles else [0.1, 0.25, 0.5, 0.75, 0.99]

    # Collect results from all strategies for comparison
    all_strategy_results = {}

    # Run experiment for each strategy
    for strategy_name in strategies:
        logger.info("\n" + "=" * 70)
        logger.info(f"Setting up strategy: {strategy_name}")
        logger.info("=" * 70)

        # Clear all tasks from schedulers before each strategy test
        logger.info("Clearing scheduler task queues...")
        if not clear_scheduler_tasks(config.scheduler_a_url, logger):
            logger.error("Failed to clear Scheduler A tasks, skipping strategy")
            continue
        if config.scheduler_c_url and config.scheduler_c_url != config.scheduler_a_url:
            if not clear_scheduler_tasks(config.scheduler_c_url, logger):
                logger.error("Failed to clear Scheduler C tasks, skipping strategy")
                continue
        if config.scheduler_b_url and config.scheduler_b_url != config.scheduler_a_url:
            if not clear_scheduler_tasks(config.scheduler_b_url, logger):
                logger.error("Failed to clear Scheduler B tasks, skipping strategy")
                continue

        # Setup this strategy on all three schedulers
        strategy_results = setup_scheduler_strategies(
            strategy_name=strategy_name,
            scheduler_a_url=config.scheduler_a_url,
            scheduler_b_url=config.scheduler_b_url,
            target_quantile=config.target_quantile,
            quantiles=quantiles,
            custom_logger=logger
        )

        # Also setup strategy on Scheduler C (FLUX)
        if config.scheduler_c_url and config.scheduler_c_url != config.scheduler_a_url:
            c_results = setup_scheduler_strategies(
                strategy_name=strategy_name,
                scheduler_a_url=config.scheduler_c_url,
                scheduler_b_url=None,
                target_quantile=config.target_quantile,
                quantiles=quantiles,
                custom_logger=logger
            )
            strategy_results.update(c_results)

        if not strategy_results.get(strategy_name, False):
            logger.error(f"Failed to setup strategy: {strategy_name}")
            logger.error("Skipping this strategy")
            continue

        logger.info(f"Strategy {strategy_name} set up successfully on all schedulers")

        # Run the experiment
        logger.info("\n" + "=" * 70)
        logger.info(f"Running experiment with strategy: {strategy_name}")
        logger.info("=" * 70)
        experiment_results = run_single_experiment(config, captions, logger, strategy_name=strategy_name)

        # Store results for comparison
        all_strategy_results[strategy_name] = experiment_results

        logger.info(f"\nCompleted experiment for strategy: {strategy_name}")

    # Generate strategy comparison table
    if len(all_strategy_results) > 1:
        comparison_table = generate_strategy_comparison_table(
            all_strategy_results,
            task_types=["A", "C", "B"]
        )
        print(comparison_table)
        logger.info("Strategy comparison table generated")

    logger.info("\n" + "=" * 70)
    logger.info("All strategy experiments completed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
