#!/usr/bin/env python3
"""
Deep Research Workflow - Simulation Mode

This script runs the Deep Research workflow (A→n×B1→n×B2→Merge) using sleep models.

Architecture:
- Thread 1 (A Submitter): Submit A tasks with Poisson arrivals
- Thread 2 (A Receiver): Receive A results → fan out to n B1 tasks
- Thread 3 (B1 Submitter): Submit B1 tasks
- Thread 4 (B1 Receiver): Receive B1 results → trigger corresponding B2 (1:1)
- Thread 5 (B2 Submitter): Submit B2 tasks
- Thread 6 (B2 Receiver): Receive B2 results → trigger Merge when all complete
- Thread 7 (Merge Submitter): Submit Merge tasks
- Thread 8 (Merge Receiver): Receive Merge results → mark workflow complete

Workflow Pattern:
- A: Initial query
- B1: n parallel tasks (fanout)
- B2: n parallel tasks (1:1 from B1)
- Merge: Aggregation (synchronization point)

Usage:
    python -m type2_deep_research.simulation.test_workflow_sim \\
        --num-workflows 50 --qps 2.0 --fanout 4 --strategies min_time,probabilistic --duration 300
"""

import random
import sys
import threading
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Dict, List, Any

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
        strategy_name: Optional strategy name (for logging purposes)
        pre_generated_workflows: Optional pre-generated workflow data for reproducibility.
                                 If provided, all strategies use identical workflow data.

    Returns:
        Dictionary of results
    """

    # Update config.strategy if strategy_name is provided
    # This is CRITICAL: ATaskSubmitter uses config.strategy to set workflow_data.strategy
    if strategy_name:
        config.strategy = strategy_name

    # Clear all tasks from schedulers before starting experiment
    # This is critical when running multiple experiments sequentially
    from common import clear_scheduler_tasks
    logger.info("Clearing all tasks from schedulers before experiment...")
    clear_scheduler_tasks(config.scheduler_a_url, logger)
    if config.scheduler_b_url and config.scheduler_b_url != config.scheduler_a_url:
        clear_scheduler_tasks(config.scheduler_b_url, logger)
    logger.info("Task clearing complete")

    logger.info("="*70)
    if strategy_name:
        logger.info(f"Deep Research Workflow Simulation - Strategy: {strategy_name}")
    else:
        logger.info("Deep Research Workflow Simulation")
    logger.info("="*70)
    logger.info(f"QPS: {config.qps}")
    logger.info(f"Duration: {config.duration}s")
    logger.info(f"Workflows: {config.num_workflows}")
    logger.info(f"Fanout count: {config.fanout_count}")
    logger.info(f"Max loops: {config.max_loops_count}")
    logger.info(f"Scheduler A: {config.scheduler_a_url}")
    logger.info(f"Scheduler B: {config.scheduler_b_url}")
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
    a_result_queue = Queue()           # A receiver → B1 submitter
    b1_result_queue = Queue()          # B1 receiver → B2 submitter
    b2_result_queue = Queue()          # Not used (B2 receiver checks completion)
    merge_trigger_queue = Queue()      # B2 receiver → Merge submitter

    # Metrics collector
    metrics = MetricsCollector(custom_logger=logger)

    # Rate limiter (shared across all submitters)
    rate_limiter = RateLimiter(rate=config.qps)

    # ========================================================================
    # Create Components
    # ========================================================================

    logger.info("Initializing components...")

    # A Submitter (Thread 1)
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
    )

    # B1 Submitter (Thread 3)
    b1_submitter = B1TaskSubmitter(
        name="B1Submitter",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        a_result_queue=a_result_queue,
        scheduler_url=config.scheduler_b_url,
        rate_limiter=None,  # B1 not rate limited - only A controls workflow arrival rate
        metrics=metrics,
    )

    # B2 Submitter (Thread 5)
    b2_submitter = B2TaskSubmitter(
        name="B2Submitter",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        b1_result_queue=b1_result_queue,
        scheduler_url=config.scheduler_b_url,
        rate_limiter=None,  # B2 not rate limited - only A controls workflow arrival rate
        metrics=metrics,
    )

    # Merge Submitter (Thread 7)
    merge_submitter = MergeTaskSubmitter(
        name="MergeSubmitter",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        merge_trigger_queue=merge_trigger_queue,
        scheduler_url=config.scheduler_a_url,
        rate_limiter=None,  # Merge not rate limited - only A controls workflow arrival rate
        metrics=metrics,
    )

    # ========================================================================
    # Pre-generate Task IDs for Receivers
    # ========================================================================

    # Task IDs now include loop iteration: task-{type}-loop{N}-{strategy}-workflow-{num}
    # Generate IDs for ALL loop iterations per workflow

    # A task IDs (one per loop iteration per workflow)
    a_task_ids = []
    with state_lock:
        for workflow_id, workflow_data in workflow_states.items():
            workflow_num = workflow_id.split('-')[-1]
            wf_strategy = workflow_data.strategy
            max_loops = workflow_data.max_loops
            for loop_iter in range(1, max_loops + 1):
                a_task_ids.append(f"task-A-loop{loop_iter}-{wf_strategy}-workflow-{workflow_num}")

    # B1/B2 task IDs - need to get from workflow states since each loop may have different fanout
    # Generate for all loop iterations with their respective fanouts
    b1_task_ids = []
    b2_task_ids = []
    with state_lock:
        for workflow_id, workflow_data in workflow_states.items():
            workflow_num = workflow_id.split('-')[-1]
            wf_strategy = workflow_data.strategy
            max_loops = workflow_data.max_loops

            for loop_iter in range(1, max_loops + 1):
                # Get fanout for this loop iteration
                loop_idx = loop_iter - 1
                fanout = workflow_data.loop_fanouts[loop_idx] if loop_idx < len(workflow_data.loop_fanouts) else workflow_data.fanout_count

                for j in range(fanout):
                    b1_task_ids.append(f"task-B1-loop{loop_iter}-{wf_strategy}-workflow-{workflow_num}-{j}")
                    b2_task_ids.append(f"task-B2-loop{loop_iter}-{wf_strategy}-workflow-{workflow_num}-{j}")

    # Merge task IDs (one per loop iteration per workflow)
    merge_task_ids = []
    with state_lock:
        for workflow_id, workflow_data in workflow_states.items():
            workflow_num = workflow_id.split('-')[-1]
            wf_strategy = workflow_data.strategy
            max_loops = workflow_data.max_loops
            for loop_iter in range(1, max_loops + 1):
                merge_task_ids.append(f"task-merge-loop{loop_iter}-{wf_strategy}-workflow-{workflow_num}")

    logger.info(f"Pre-generated {len(a_task_ids)} A, {len(b1_task_ids)} B1, "
                f"{len(b2_task_ids)} B2, {len(merge_task_ids)} Merge task IDs (loop-aware)")

    # ========================================================================
    # Create Receivers
    # ========================================================================

    # A Receiver (Thread 2)
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
    )

    # B1 Receiver (Thread 4)
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
    )

    # B2 Receiver (Thread 6)
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
    )

    # Merge Receiver (Thread 8)
    # Pass a_submitter for loop triggering - when merge completes and more loops needed,
    # MergeReceiver will call a_submitter.add_task() to start the next loop iteration
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
    )

    # ========================================================================
    # Start All Threads
    # ========================================================================

    logger.info("Starting all threads...")
    start_time = time.time()

    # Start receivers first (so they're ready to receive)
    a_receiver.start()
    b1_receiver.start()
    b2_receiver.start()
    merge_receiver.start()

    # Wait a moment for receivers to connect
    time.sleep(1.0)

    # Start submitters
    b1_submitter.start()
    b2_submitter.start()
    merge_submitter.start()
    a_submitter.start()  # Start A last (it drives the workflow)

    logger.info("All threads started successfully")

    # ========================================================================
    # Wait for Duration or Early Completion
    # ========================================================================

    logger.info(f"Running experiment for up to {config.duration} seconds...")

    # Calculate target workflows for early stopping based on SUBMISSION ORDER
    # We need:
    # 1. All warmup workflows (first num_warmup) to complete
    # 2. The first portion_stats fraction of non-warmup workflows to complete
    # This matches how MetricsCollector filters workflows for statistics
    import math
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

        # Early stopping: check if all TARGET workflows (by submission order) have completed
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
    b1_submitter.stop()
    b2_submitter.stop()
    merge_submitter.stop()

    # Wait for submitters to finish
    a_submitter.join(timeout=10)
    b1_submitter.join(timeout=10)
    b2_submitter.join(timeout=10)
    merge_submitter.join(timeout=10)

    # Stop receivers
    a_receiver.stop()
    b1_receiver.stop()
    b2_receiver.stop()
    merge_receiver.stop()

    # Wait for receivers to finish
    a_receiver.join(timeout=10)
    b1_receiver.join(timeout=10)
    b2_receiver.join(timeout=10)
    merge_receiver.join(timeout=10)

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

    # Print detailed metrics report with per-task-type metrics (Avg, P90, P99)
    report = metrics.generate_detailed_text_report(
        task_types=["A", "B1", "B2", "merge"],
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
    task_types = ["A", "B1", "B2", "merge"]
    task_metrics_result = {}
    for task_type in task_types:
        task_metrics_result[task_type] = metrics.get_task_metrics_by_type(
            task_type, exclude_warmup=True, portion_stats=config.portion_stats
        )

    # Get workflow metrics
    summary = metrics.get_summary(exclude_warmup=True, portion_stats=config.portion_stats)
    workflow_durations = []
    with metrics._workflow_lock:
        # Get included workflow IDs based on portion_stats
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

    parser = create_base_parser(description="Deep Research Workflow - Simulation Mode")
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

    # Create config with hardcoded simulation mode
    config = DeepResearchConfig(
        mode="simulation",  # Hardcoded for simulation
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
        max_sleep_time_seconds=args.max_sleep_time,
    )

    logger = configure_logging(level="INFO")

    logger.info(f"Mode: {config.mode} (hardcoded for simulation)")
    logger.info(f"Model A: {config.model_a_id}")
    logger.info(f"Model B: {config.model_b_id}")
    logger.info(f"Model Merge: {config.model_merge_id}")
    logger.info(f"Scheduler A: {config.scheduler_a_url}")
    logger.info(f"Scheduler B: {config.scheduler_b_url}")
    logger.info(f"Max sleep time: {config.max_sleep_time_seconds:.1f}s")
    logger.info(f"Max loops count: {config.max_loops_count}")

    # Log fanout distribution info
    fanout_info = config.get_fanout_config_info()
    logger.info(f"Fanout distribution: type={fanout_info['type']}")
    if fanout_info['type'] != 'static':
        logger.info(f"Fanout config details: {fanout_info}")

    # ========================================================================
    # Pre-generate Workflow Data (BEFORE strategy loop)
    # ========================================================================

    logger.info("="*70)
    logger.info("Pre-generating Workflow Data")
    logger.info("="*70)

    # Pre-generate all workflow data ONCE before testing any strategies.
    # This ensures all strategies use IDENTICAL input data for fair comparison.
    pre_generated_workflows = pre_generate_workflows(config, seed=42)

    logger.info(f"Pre-generated {len(pre_generated_workflows)} workflows")
    if pre_generated_workflows and pre_generated_workflows[0].a_sleep_time is not None:
        wf0 = pre_generated_workflows[0]
        logger.info(f"  Sample workflow[0]: max_loops={wf0.max_loops}, "
                    f"loop_fanouts={wf0.loop_fanouts}, "
                    f"sleep_times A={wf0.a_sleep_time:.3f}s, "
                    f"B1[0]={wf0.b1_sleep_times[0]:.3f}s, "
                    f"B2[0]={wf0.b2_sleep_times[0]:.3f}s, "
                    f"merge={wf0.merge_sleep_time:.3f}s")

    # ========================================================================
    # Strategy Management
    # ========================================================================

    logger.info("="*70)
    logger.info("Strategy-based Testing Mode")
    logger.info(f"Will test {len(strategies)} strategies: {', '.join(strategies)}")
    logger.info(f"Configuration: {config.num_workflows} workflows, QPS={config.qps}, Fanout={config.fanout_count}")
    logger.info("="*70)

    # Set default quantiles if not specified
    quantiles = config.quantiles if config.quantiles else [0.1, 0.25, 0.5, 0.75, 0.99]

    all_strategy_results = {}

    # Run experiment for each strategy
    for strategy_name in strategies:
        logger.info("\n" + "="*70)
        logger.info(f"Setting up strategy: {strategy_name}")
        logger.info("="*70)

        # Update config strategy
        config.strategy = strategy_name

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

        experiment_results = run_single_experiment(
            config, logger,
            strategy_name=strategy_name,
            pre_generated_workflows=pre_generated_workflows
        )

        # Store results for comparison
        all_strategy_results[strategy_name] = experiment_results

        logger.info(f"\nCompleted experiment for strategy: {strategy_name}")

        # Brief pause between strategies
        time.sleep(2.0)

    # Generate strategy comparison table
    if len(all_strategy_results) > 1:
        comparison_table = generate_strategy_comparison_table(
            all_strategy_results,
            task_types=["A", "B1", "B2", "merge"]
        )
        print(comparison_table)
        logger.info("Strategy comparison table generated")

    logger.info("\n" + "="*70)
    logger.info("All strategy experiments completed!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
