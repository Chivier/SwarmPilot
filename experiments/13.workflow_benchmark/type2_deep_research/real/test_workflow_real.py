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

import math
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
    add_type2_args,
    parse_strategies,
    generate_strategy_comparison_table,
)
from type2_deep_research.config import DeepResearchConfig
from type2_deep_research.workflow_data import pre_generate_workflows
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


def run_single_experiment(config, logger, strategy_name=None, pre_generated_workflows=None):
    """Run a single experiment with the given configuration.

    Args:
        config: DeepResearchConfig instance
        logger: Logger instance
        strategy_name: Optional strategy name to use
        pre_generated_workflows: Pre-generated workflow data for reproducibility (required for real mode)

    Returns:
        Dict with strategy results containing task_metrics and workflow_metrics
    """

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
    logger.info(f"Max loops: {config.max_loops_count}")
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

    # Create A submitter first to populate workflow_states
    # Pass pre_generated_workflows for reproducibility with Exp07
    a_submitter = ATaskSubmitter(
        name="ASubmitter",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        pre_generated_workflows=pre_generated_workflows,
        scheduler_url=config.scheduler_a_url,
        qps=config.qps,
        duration=config.duration,
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

    # Pre-generate task IDs for receivers
    # IMPORTANT: Must be done after ATaskSubmitter init to have workflow_states populated
    # Use workflow_data.strategy to ensure consistency with submitters
    # Task IDs now include loop iteration: task-{type}-loop{N}-{strategy}-workflow-{num}
    # Generate IDs for ALL loop iterations per workflow
    a_task_ids = []
    b1_task_ids = []
    b2_task_ids = []
    merge_task_ids = []
    with state_lock:
        for workflow_id, workflow_data in workflow_states.items():
            workflow_num = workflow_id.split('-')[-1]
            wf_strategy = workflow_data.strategy
            max_loops = workflow_data.max_loops

            # Generate A task IDs (one per loop iteration per workflow)
            for loop_iter in range(1, max_loops + 1):
                a_task_ids.append(f"task-A-loop{loop_iter}-{wf_strategy}-workflow-{workflow_num}")

            # Generate B1/B2 task IDs for all loop iterations with their respective fanouts
            for loop_iter in range(1, max_loops + 1):
                loop_idx = loop_iter - 1
                fanout = workflow_data.loop_fanouts[loop_idx] if loop_idx < len(workflow_data.loop_fanouts) else workflow_data.fanout_count
                for j in range(fanout):
                    b1_task_ids.append(f"task-B1-loop{loop_iter}-{wf_strategy}-workflow-{workflow_num}-{j}")
                    b2_task_ids.append(f"task-B2-loop{loop_iter}-{wf_strategy}-workflow-{workflow_num}-{j}")

            # Generate Merge task IDs (one per loop iteration per workflow)
            for loop_iter in range(1, max_loops + 1):
                merge_task_ids.append(f"task-merge-loop{loop_iter}-{wf_strategy}-workflow-{workflow_num}")

    logger.info(f"Pre-generated {len(a_task_ids)} A, {len(b1_task_ids)} B1, "
                f"{len(b2_task_ids)} B2, {len(merge_task_ids)} Merge task IDs (loop-aware)")

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

    # Merge Receiver - Pass a_submitter for loop triggering
    # When merge completes and more loops needed, MergeReceiver will call
    # a_submitter.add_task() to start the next loop iteration
    merge_receiver = MergeTaskReceiver(
        name="MergeReceiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        a_submitter=a_submitter,  # For loop triggering
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

    # Calculate target workflows for early stopping based on SUBMISSION ORDER
    # We need:
    # 1. All warmup workflows (first num_warmup) to complete
    # 2. The first portion_stats fraction of non-warmup workflows to complete
    # This matches how MetricsCollector filters workflows for statistics
    num_warmup = config.num_warmup
    num_non_warmup = config.num_workflows - num_warmup
    target_non_warmup = math.ceil(num_non_warmup * config.portion_stats)

    # Generate the list of workflow IDs that must complete for metrics
    # Warmup workflows: workflow-0000 to workflow-(num_warmup-1)
    # Target non-warmup: workflow-num_warmup to workflow-(num_warmup + target_non_warmup - 1)
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
                completed = merge_receiver.completed_workflows
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
        task_types=["A", "B1", "B2", "merge"],
        portion_stats=config.portion_stats
    ))

    # Collect and return strategy results for comparison table
    task_types = ["A", "B1", "B2", "merge"]
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

    parser = create_base_parser(description="Deep Research Workflow - Real Cluster Mode")
    parser = add_type2_args(parser)

    # Add loop configuration argument
    parser.add_argument(
        "--max-loops-count",
        type=int,
        default=1,
        help="Maximum loop iterations per workflow (1 = no loop, default: 1). "
             "When > 1, each workflow loops A→B1→B2→Merge up to max_loops times. "
             "Actual loop count is sampled from [5-15], capped by this value."
    )

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
        max_loops_count=args.max_loops_count,
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
    logger.info(f"Max loops count: {config.max_loops_count}")

    # Log fanout distribution info
    fanout_info = config.get_fanout_config_info()
    logger.info(f"Fanout distribution: type={fanout_info['type']}")
    if fanout_info['type'] != 'static':
        logger.info(f"Fanout config details: {fanout_info}")

    # ========================================================================
    # Pre-generate Workflows (aligned with Exp07, seed=42)
    # ========================================================================
    # Pre-generate all workflow data ONCE before testing any strategies.
    # This ensures all strategies use IDENTICAL input data for fair comparison.
    # Uses seed=42 to match Exp07's default random seed.
    pre_generated = pre_generate_workflows(config, seed=42)
    logger.info(f"Pre-generated {len(pre_generated)} workflows from dataset.jsonl (seed=42)")
    if pre_generated:
        wf0 = pre_generated[0]
        logger.info(f"  Sample workflow[0]: max_loops={wf0.max_loops}, "
                    f"loop_fanouts={wf0.loop_fanouts}")

    # Log fanout distribution from pre-generated workflows
    pre_gen_fanouts = [w.fanout_count for w in pre_generated]
    logger.info(f"Fanout range: {min(pre_gen_fanouts)}-{max(pre_gen_fanouts)}, "
                f"avg={sum(pre_gen_fanouts)/len(pre_gen_fanouts):.1f}")

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
        experiment_results = run_single_experiment(
            config, logger,
            strategy_name=strategy_name,
            pre_generated_workflows=pre_generated
        )

        # Store results for comparison
        all_strategy_results[strategy_name] = experiment_results

        logger.info(f"\nCompleted experiment for strategy: {strategy_name}")

    # Generate strategy comparison table
    if len(all_strategy_results) > 1:
        comparison_table = generate_strategy_comparison_table(
            all_strategy_results,
            task_types=["A", "B1", "B2", "merge"]
        )
        print(comparison_table)
        logger.info("Strategy comparison table generated")

    logger.info("\n" + "=" * 70)
    logger.info("All strategy experiments completed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
