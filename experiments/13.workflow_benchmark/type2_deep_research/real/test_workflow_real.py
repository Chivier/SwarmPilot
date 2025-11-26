#!/usr/bin/env python3
"""
Deep Research Workflow - Real Cluster Mode

This script runs the Deep Research workflow (A→n×B1→n×B2→Merge) in real cluster mode
using actual model IDs and task inputs instead of sleep simulations.

Key differences from simulation mode:
- Uses real model inputs (sentence, max_tokens) instead of sleep_time
- Includes token estimation in metadata for performance prediction
- Supports predictor and planner integration
- Can be deployed across distributed schedulers

Architecture:
- Thread 1: A submitter (Poisson process, QPS-controlled)
- Thread 2: A receiver → fans out to n B1 tasks
- Thread 3: B1 submitter
- Thread 4: B1 receiver → triggers B2 tasks
- Thread 5: B2 submitter
- Thread 6: B2 receiver → synchronizes and triggers Merge
- Thread 7: Merge submitter
- Thread 8: Merge receiver

Usage:
    python -m type2_deep_research.real.test_workflow_real \\
        --num-workflows 50 --qps 2.0 --fanout 4 --strategies min_time,probabilistic --duration 300
"""

import sys
import threading
import time
from pathlib import Path
from queue import Queue

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
    add_type2_args,
    parse_strategies,
)
from type2_deep_research.config import DeepResearchConfig
from type2_deep_research.submitters import (
    ATaskSubmitter,
    B1TaskSubmitter,
    B2TaskSubmitter,
    MergeTaskSubmitter
)
from type2_deep_research.receivers import (
    ATaskReceiver,
    B1TaskReceiver,
    B2TaskReceiver,
    MergeTaskReceiver
)


def run_single_experiment(config, logger, strategy_name=None):
    """Run a single experiment with the given configuration."""

    # Update config.strategy if strategy_name is provided
    if strategy_name:
        config.strategy = strategy_name

    logger.info("=" * 80)
    logger.info(f"Deep Research Workflow - Real Cluster Mode - Strategy: {strategy_name or 'default'}")
    logger.info("=" * 80)
    logger.info(f"QPS: {config.qps}")
    logger.info(f"Duration: {config.duration}s")
    logger.info(f"Workflows: {config.num_workflows}")
    logger.info(f"Fanout count: {config.fanout_count}")
    logger.info(f"Scheduler A: {config.scheduler_a_url}")
    logger.info(f"Scheduler B: {config.scheduler_b_url}")

    # Initialize components
    metrics = MetricsCollector(logger)
    rate_limiter = RateLimiter(rate=config.qps)
    workflow_states = {}
    state_lock = threading.Lock()

    # Create inter-thread queues
    a_result_queue = Queue()
    b1_result_queue = Queue()
    merge_trigger_queue = Queue()

    # Pre-generate task IDs for receivers
    strategy = getattr(config, 'strategy', 'probabilistic')

    # A task IDs (one per workflow)
    a_task_ids = [
        f"task-A-{strategy}-workflow-{i:04d}"
        for i in range(config.num_workflows)
    ]

    # B1 task IDs (fanout_count per workflow)
    b1_task_ids = [
        f"task-B1-{strategy}-workflow-{i:04d}-{j}"
        for i in range(config.num_workflows)
        for j in range(config.fanout_count)
    ]

    # B2 task IDs (fanout_count per workflow)
    b2_task_ids = [
        f"task-B2-{strategy}-workflow-{i:04d}-{j}"
        for i in range(config.num_workflows)
        for j in range(config.fanout_count)
    ]

    # Merge task IDs (one per workflow)
    merge_task_ids = [
        f"task-merge-{strategy}-workflow-{i:04d}"
        for i in range(config.num_workflows)
    ]

    logger.info(f"Pre-generated {len(a_task_ids)} A, {len(b1_task_ids)} B1, "
                f"{len(b2_task_ids)} B2, {len(merge_task_ids)} Merge task IDs")

    # Create submitters
    a_submitter = ATaskSubmitter(
        name="ASubmitter",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        scheduler_url=config.scheduler_a_url,
        rate_limiter=rate_limiter,
        metrics=metrics,
        custom_logger=logger
    )

    b1_submitter = B1TaskSubmitter(
        name="B1Submitter",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        a_result_queue=a_result_queue,
        scheduler_url=config.scheduler_b_url,
        rate_limiter=None,
        metrics=metrics,
        custom_logger=logger
    )

    b2_submitter = B2TaskSubmitter(
        name="B2Submitter",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        b1_result_queue=b1_result_queue,
        scheduler_url=config.scheduler_b_url,
        rate_limiter=None,
        metrics=metrics,
        custom_logger=logger
    )

    merge_submitter = MergeTaskSubmitter(
        name="MergeSubmitter",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        merge_trigger_queue=merge_trigger_queue,
        scheduler_url=config.scheduler_a_url,
        rate_limiter=None,
        metrics=metrics,
        custom_logger=logger
    )

    # Create receivers
    a_receiver = ATaskReceiver(
        name="AReceiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        b1_submitter=b1_submitter,
        a_result_queue=a_result_queue,
        task_ids=a_task_ids,
        scheduler_url=config.scheduler_a_url,
        model_id=config.model_a_id,
        metrics=metrics,
        custom_logger=logger
    )

    b1_receiver = B1TaskReceiver(
        name="B1Receiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        b2_submitter=b2_submitter,
        b1_result_queue=b1_result_queue,
        task_ids=b1_task_ids,
        scheduler_url=config.scheduler_b_url,
        model_id=config.model_b_id,
        metrics=metrics,
        custom_logger=logger
    )

    b2_receiver = B2TaskReceiver(
        name="B2Receiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        merge_submitter=merge_submitter,
        merge_trigger_queue=merge_trigger_queue,
        task_ids=b2_task_ids,
        scheduler_url=config.scheduler_b_url,
        model_id=config.model_b_id,
        metrics=metrics,
        custom_logger=logger
    )

    merge_receiver = MergeTaskReceiver(
        name="MergeReceiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        task_ids=merge_task_ids,
        scheduler_url=config.scheduler_a_url,
        model_id=config.model_merge_id,
        metrics=metrics,
        custom_logger=logger
    )

    # Start all threads
    logger.info("Starting experiment...")
    start_time = time.time()

    a_submitter.start()
    a_receiver.start()
    b1_submitter.start()
    b1_receiver.start()
    b2_submitter.start()
    b2_receiver.start()
    merge_submitter.start()
    merge_receiver.start()

    # Monitor progress
    try:
        while time.time() - start_time < config.duration:
            time.sleep(10)

            with state_lock:
                completed = merge_receiver.completed_workflows
                total = config.num_workflows
                progress_pct = (completed / total * 100) if total > 0 else 0

                logger.info(
                    f"Progress: {completed}/{total} workflows "
                    f"({progress_pct:.1f}%) - "
                    f"Elapsed: {time.time() - start_time:.1f}s"
                )

                if completed >= total:
                    logger.info("All workflows completed! Stopping early.")
                    break

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")

    # Stop all threads
    logger.info("Stopping all threads...")
    a_submitter.stop()
    a_receiver.stop()
    b1_submitter.stop()
    b1_receiver.stop()
    b2_submitter.stop()
    b2_receiver.stop()
    merge_submitter.stop()
    merge_receiver.stop()

    # Wait for threads to finish
    a_submitter.join()
    a_receiver.join()
    b1_submitter.join()
    b1_receiver.join()
    b2_submitter.join()
    b2_receiver.join()
    merge_submitter.join()
    merge_receiver.join()

    # Calculate final statistics
    elapsed_time = time.time() - start_time
    with state_lock:
        completed_workflows = merge_receiver.completed_workflows

    logger.info("=" * 80)
    logger.info("Experiment Complete")
    logger.info("=" * 80)
    logger.info(f"Duration: {elapsed_time:.2f}s")
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

    # Print metrics report (with portion_stats filtering)
    print("\n" + metrics.generate_text_report(portion_stats=config.portion_stats))


def main():
    """Main entry point - handles strategy management and experiment orchestration."""

    # ========================================================================
    # Parse Command Line Arguments
    # ========================================================================

    parser = create_base_parser(description="Deep Research Workflow - Real Cluster Mode")
    parser = add_type2_args(parser)
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
    # warmup_count = num_workflows * warmup_ratio
    warmup_count = int(args.num_workflows * args.warmup)

    # Create config with hardcoded real mode
    config = DeepResearchConfig(
        mode="real",  # Hardcoded for real cluster
        qps=args.qps,
        duration=args.duration,
        num_workflows=args.num_workflows,
        fanout_count=args.fanout,
        fanout_config=args.fanout_config,
        fanout_seed=args.fanout_seed,
        num_warmup=warmup_count,
        strategies=strategies,
        portion_stats=args.portion_stats,
    )

    logger = configure_logging(level="INFO")

    logger.info(f"Mode: {config.mode} (hardcoded for real cluster)")
    logger.info(f"Model A: {config.model_a_id}")
    logger.info(f"Model B: {config.model_b_id}")
    logger.info(f"Model Merge: {config.model_merge_id}")
    logger.info(f"Scheduler A: {config.scheduler_a_url}")
    logger.info(f"Scheduler B: {config.scheduler_b_url}")

    # Log fanout distribution info
    fanout_info = config.get_fanout_config_info()
    logger.info(f"Fanout distribution: type={fanout_info['type']}")
    if fanout_info['type'] != 'static':
        logger.info(f"Fanout config details: {fanout_info}")

    # ========================================================================
    # Strategy Management
    # ========================================================================

    logger.info("=" * 70)
    logger.info("Strategy-based Testing Mode")
    logger.info(f"Will test {len(strategies)} strategies: {', '.join(strategies)}")
    logger.info("=" * 70)

    # Set default quantiles if not specified
    quantiles = config.quantiles if config.quantiles else [0.1, 0.25, 0.5, 0.75, 0.99]

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
        logger.info("\n" + "=" * 70)
        logger.info(f"Running experiment with strategy: {strategy_name}")
        logger.info("=" * 70)
        run_single_experiment(config, logger, strategy_name=strategy_name)

        logger.info(f"\nCompleted experiment for strategy: {strategy_name}")

    logger.info("\n" + "=" * 70)
    logger.info("All strategy experiments completed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
