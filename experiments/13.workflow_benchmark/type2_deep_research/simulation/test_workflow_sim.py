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


def calculate_task_metrics_from_collector(collector: MetricsCollector, task_type: str, workflow_states: Dict, portion_stats: float = 1.0) -> Dict:
    """
    Calculate metrics for a specific task type from the collector.

    Args:
        collector: MetricsCollector instance
        task_type: Task type string ("A", "B1", "B2", "merge")
        workflow_states: Dictionary of workflow states to check for warmup status
        portion_stats: Portion of non-warmup workflows to include (0.0-1.0)

    Returns:
        Dictionary of metrics
    """
    # Filter tasks by type
    # Note: MetricsCollector stores task_type as passed during record_task_submit
    # The submitters use "A", "B1", "B2", "merge"

    all_tasks = [t for t in collector.task_metrics if t.task_type == task_type]

    # Get included workflow IDs based on portion_stats
    # Use collector's _non_warmup_workflow_order for submission order
    total_non_warmup = len(collector._non_warmup_workflow_order)
    num_to_include = int(total_non_warmup * portion_stats)
    included_workflow_ids = set(collector._non_warmup_workflow_order[:num_to_include])

    # Separate warmup, included, and excluded tasks
    actual_tasks = []
    warmup_tasks = []

    for t in all_tasks:
        # Check if workflow is warmup
        wf_state = workflow_states.get(t.workflow_id)
        is_warmup = False
        if wf_state:
            # DeepResearchWorkflowData has is_warmup attribute
            is_warmup = getattr(wf_state, 'is_warmup', False)

        if is_warmup:
            warmup_tasks.append(t)
        elif t.workflow_id in included_workflow_ids:
            # Only include tasks from workflows within portion_stats
            actual_tasks.append(t)
            
    # Calculate stats for actual tasks
    num_generated = len(actual_tasks)
    num_submitted = len(actual_tasks) # In this collector, we only record submitted tasks
    
    completed_tasks = [t for t in actual_tasks if t.complete_time is not None]
    num_completed = len(completed_tasks)
    
    failed_tasks = [t for t in actual_tasks if t.complete_time is not None and not t.success]
    num_failed = len(failed_tasks)
    
    # Calculate completion times
    completion_times = []
    for t in completed_tasks:
        if t.duration is not None:
            completion_times.append(t.duration)
            
    if completion_times:
        times_arr = np.array(completion_times)
        avg_completion = float(np.mean(times_arr))
        median_completion = float(np.median(times_arr))
        p95_completion = float(np.percentile(times_arr, 95))
        p99_completion = float(np.percentile(times_arr, 99))
    else:
        avg_completion = 0.0
        median_completion = 0.0
        p95_completion = 0.0
        p99_completion = 0.0
        
    return {
        "task_type": task_type,
        "num_generated": num_generated,
        "num_submitted": num_submitted,
        "num_completed": num_completed,
        "num_failed": num_failed,
        "num_warmup": len(warmup_tasks),
        "completion_times": completion_times,
        "avg_completion_time": avg_completion,
        "median_completion_time": median_completion,
        "p95_completion_time": p95_completion,
        "p99_completion_time": p99_completion
    }


def calculate_workflow_metrics_from_collector(collector: MetricsCollector, workflow_states: Dict, portion_stats: float = 1.0) -> Dict:
    """
    Calculate workflow-level metrics from the collector.

    Args:
        collector: MetricsCollector instance
        workflow_states: Dictionary of workflow states
        portion_stats: Portion of non-warmup workflows to include (0.0-1.0)

    Returns:
        Dictionary of workflow metrics
    """
    all_workflows = collector.workflow_metrics

    # Get included workflow IDs based on portion_stats
    total_non_warmup = len(collector._non_warmup_workflow_order)
    num_to_include = int(total_non_warmup * portion_stats)
    included_workflow_ids = set(collector._non_warmup_workflow_order[:num_to_include])

    actual_workflows = []
    warmup_workflows = []
    excluded_by_portion = 0

    for w in all_workflows:
        # Check if workflow is warmup
        wf_state = workflow_states.get(w.workflow_id)
        is_warmup = False
        if wf_state:
            is_warmup = getattr(wf_state, 'is_warmup', False)

        if is_warmup:
            warmup_workflows.append(w)
        elif w.workflow_id in included_workflow_ids:
            actual_workflows.append(w)
        else:
            excluded_by_portion += 1
            
    completed_workflows = [w for w in actual_workflows if w.end_time is not None]
    
    # Calculate workflow times
    workflow_times = []
    for w in completed_workflows:
        if w.total_duration is not None:
            workflow_times.append(w.total_duration)
            
    if workflow_times:
        times_arr = np.array(workflow_times)
        avg_workflow = float(np.mean(times_arr))
        median_workflow = float(np.median(times_arr))
        p95_workflow = float(np.percentile(times_arr, 95))
        p99_workflow = float(np.percentile(times_arr, 99))
    else:
        avg_workflow = 0.0
        median_workflow = 0.0
        p95_workflow = 0.0
        p99_workflow = 0.0
        
    # Calculate fanout stats
    fanout_values = []
    for w in completed_workflows:
        wf_state = workflow_states.get(w.workflow_id)
        if wf_state:
            fanout_values.append(getattr(wf_state, 'fanout_count', 0))
            
    fanout_distribution = {}
    avg_fanout = 0.0
    if fanout_values:
        unique, counts = np.unique(fanout_values, return_counts=True)
        fanout_distribution = {int(k): int(v) for k, v in zip(unique, counts)}
        avg_fanout = float(np.mean(fanout_values))
        
    return {
        "num_completed": len(completed_workflows),
        "num_warmup": len(warmup_workflows),
        "num_excluded": excluded_by_portion,  # Workflows excluded by portion_stats
        "workflow_times": workflow_times,
        "avg_workflow_time": avg_workflow,
        "median_workflow_time": median_workflow,
        "p50_workflow_time": median_workflow,
        "p95_workflow_time": p95_workflow,
        "p99_workflow_time": p99_workflow,
        "fanout_distribution": fanout_distribution,
        "avg_fanout": avg_fanout
    }


def print_metrics_summary(
    strategy: str,
    a_metrics: Dict,
    b1_metrics: Dict,
    b2_metrics: Dict,
    merge_metrics: Dict,
    wf_metrics: Dict
):
    """Print a summary of metrics."""
    print("\n" + "=" * 80)
    print(f"Results Summary: {strategy}")
    print("=" * 80)

    print("\nA Tasks:")
    print(f"  Generated:  {a_metrics['num_generated']} (excl. {a_metrics['num_warmup']} warmup)")
    print(f"  Submitted:  {a_metrics['num_submitted']}")
    print(f"  Completed:  {a_metrics['num_completed']}")
    print(f"  Failed:     {a_metrics['num_failed']}")
    if a_metrics['avg_completion_time'] > 0:
        print(f"  Avg time:   {a_metrics['avg_completion_time']:.2f}s")
        print(f"  Median:     {a_metrics['median_completion_time']:.2f}s")
        print(f"  P95:        {a_metrics['p95_completion_time']:.2f}s")

    print("\nB1 Tasks:")
    print(f"  Generated:  {b1_metrics['num_generated']} (excl. {b1_metrics['num_warmup']} warmup)")
    print(f"  Submitted:  {b1_metrics['num_submitted']}")
    print(f"  Completed:  {b1_metrics['num_completed']}")
    print(f"  Failed:     {b1_metrics['num_failed']}")
    if b1_metrics['avg_completion_time'] > 0:
        print(f"  Avg time:   {b1_metrics['avg_completion_time']:.2f}s")
        print(f"  Median:     {b1_metrics['median_completion_time']:.2f}s")
        print(f"  P95:        {b1_metrics['p95_completion_time']:.2f}s")

    print("\nB2 Tasks:")
    print(f"  Generated:  {b2_metrics['num_generated']} (excl. {b2_metrics['num_warmup']} warmup)")
    print(f"  Submitted:  {b2_metrics['num_submitted']}")
    print(f"  Completed:  {b2_metrics['num_completed']}")
    print(f"  Failed:     {b2_metrics['num_failed']}")
    if b2_metrics['avg_completion_time'] > 0:
        print(f"  Avg time:   {b2_metrics['avg_completion_time']:.2f}s")
        print(f"  Median:     {b2_metrics['median_completion_time']:.2f}s")
        print(f"  P95:        {b2_metrics['p95_completion_time']:.2f}s")

    print("\nMerge Tasks:")
    print(f"  Generated:  {merge_metrics['num_generated']} (excl. {merge_metrics['num_warmup']} warmup)")
    print(f"  Submitted:  {merge_metrics['num_submitted']}")
    print(f"  Completed:  {merge_metrics['num_completed']}")
    print(f"  Failed:     {merge_metrics['num_failed']}")
    if merge_metrics['avg_completion_time'] > 0:
        print(f"  Avg time:   {merge_metrics['avg_completion_time']:.2f}s")
        print(f"  Median:     {merge_metrics['median_completion_time']:.2f}s")
        print(f"  P95:        {merge_metrics['p95_completion_time']:.2f}s")

    print("\nWorkflows:")
    print(f"  Completed:  {wf_metrics['num_completed']} (excl. {wf_metrics['num_warmup']} warmup)")
    print(f"  Avg fanout: {wf_metrics['avg_fanout']:.1f} B tasks per A task")
    if wf_metrics['avg_workflow_time'] > 0:
        print(f"  Avg time:   {wf_metrics['avg_workflow_time']:.2f}s")
        print(f"  Median:     {wf_metrics['median_workflow_time']:.2f}s")
        print(f"  P95:        {wf_metrics['p95_workflow_time']:.2f}s")
        print(f"  P99:        {wf_metrics['p99_workflow_time']:.2f}s")

    print("\nFanout Distribution:")
    for fanout, count in sorted(wf_metrics['fanout_distribution'].items()):
        percentage = (count / wf_metrics['num_completed']) * 100 if wf_metrics['num_completed'] > 0 else 0
        print(f"  {fanout} B tasks: {count} workflows ({percentage:.1f}%)")

    print("=" * 80)


def run_single_experiment(config, logger, strategy_name=None):
    """Run a single experiment with the given configuration.

    Args:
        config: DeepResearchConfig instance
        logger: Logger instance
        strategy_name: Optional strategy name (for logging purposes)
        
    Returns:
        Dictionary of results
    """

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
        scheduler_url=config.scheduler_a_url,
        rate_limiter=rate_limiter,
        metrics=metrics,
        duration=config.duration,
    )

    # B1 Submitter (Thread 3)
    b1_submitter = B1TaskSubmitter(
        name="B1Submitter",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        a_result_queue=a_result_queue,
        scheduler_url=config.scheduler_b_url,
        rate_limiter=rate_limiter,
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
        rate_limiter=rate_limiter,
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
        rate_limiter=rate_limiter,
        metrics=metrics,
    )

    # ========================================================================
    # Pre-generate Task IDs for Receivers
    # ========================================================================

    # Get strategy name for task ID generation
    # Use strategy_name parameter if provided (for multi-strategy testing),
    # otherwise fall back to config.strategy
    strategy = strategy_name if strategy_name else getattr(config, 'strategy', 'probabilistic')

    # A task IDs (one per workflow)
    a_task_ids = [
        f"task-A-{strategy}-workflow-{i:04d}"
        for i in range(config.num_workflows)
    ]

    # B1/B2 task IDs - need to get from workflow states since each may have different fanout
    # Note: workflows are created during ATaskSubmitter.__init__ with distribution-based fanout
    b1_task_ids = []
    b2_task_ids = []
    with state_lock:
        for workflow_id, workflow_data in workflow_states.items():
            workflow_num = workflow_id.split('-')[-1]
            fanout = workflow_data.fanout_count
            for j in range(fanout):
                b1_task_ids.append(f"task-B1-{strategy}-workflow-{workflow_num}-{j}")
                b2_task_ids.append(f"task-B2-{strategy}-workflow-{workflow_num}-{j}")

    # Merge task IDs (one per workflow)
    merge_task_ids = [
        f"task-merge-{strategy}-workflow-{i:04d}"
        for i in range(config.num_workflows)
    ]

    logger.info(f"Pre-generated {len(a_task_ids)} A, {len(b1_task_ids)} B1, "
                f"{len(b2_task_ids)} B2, {len(merge_task_ids)} Merge task IDs")

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
    merge_receiver = MergeTaskReceiver(
        name="MergeReceiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
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
    # Wait for Duration
    # ========================================================================

    logger.info(f"Running experiment for {config.duration} seconds...")

    # Print progress every 10 seconds
    elapsed = 0
    while elapsed < config.duration:
        time.sleep(10)
        elapsed = time.time() - start_time

        # Calculate progress
        with metrics._workflow_lock:
            completed_workflows = len([w for w in metrics.workflow_metrics if w.end_time is not None])
            total_workflows = len(workflow_states)
            
        logger.info(
            f"Progress: {elapsed:.1f}s elapsed, "
            f"{completed_workflows}/{total_workflows} workflows complete "
            f"({100*completed_workflows/total_workflows if total_workflows > 0 else 0:.1f}%)"
        )
        
        # Stop if all workflows are complete
        if completed_workflows >= total_workflows and total_workflows > 0:
            logger.info("All workflows completed early")
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

    # Calculate metrics (with portion_stats filtering)
    a_metrics = calculate_task_metrics_from_collector(metrics, "A", workflow_states, config.portion_stats)
    b1_metrics = calculate_task_metrics_from_collector(metrics, "B1", workflow_states, config.portion_stats)
    b2_metrics = calculate_task_metrics_from_collector(metrics, "B2", workflow_states, config.portion_stats)
    merge_metrics = calculate_task_metrics_from_collector(metrics, "merge", workflow_states, config.portion_stats)
    wf_metrics = calculate_workflow_metrics_from_collector(metrics, workflow_states, config.portion_stats)

    # Print summary
    print_metrics_summary(strategy, a_metrics, b1_metrics, b2_metrics, merge_metrics, wf_metrics)
    
    # Calculate actual QPS
    actual_qps = 0.0
    # We can estimate actual QPS from A metrics
    if a_metrics['num_submitted'] > 0 and total_time > 0:
        actual_qps = a_metrics['num_submitted'] / total_time

    return {
        "strategy": strategy,
        "num_workflows": config.num_workflows,
        "target_qps": config.qps,
        "actual_qps": actual_qps,
        "a_tasks": a_metrics,
        "b1_tasks": b1_metrics,
        "b2_tasks": b2_metrics,
        "merge_tasks": merge_metrics,
        "workflows": wf_metrics,
        "submission_time": total_time
    }


def main():
    """Main entry point - handles strategy management and experiment orchestration."""

    # ========================================================================
    # Parse Command Line Arguments
    # ========================================================================

    parser = create_base_parser(description="Deep Research Workflow - Simulation Mode")
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
        strategies=strategies,
        portion_stats=args.portion_stats,
    )

    logger = configure_logging(level="INFO")

    logger.info(f"Mode: {config.mode} (hardcoded for simulation)")
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

    logger.info("="*70)
    logger.info("Strategy-based Testing Mode")
    logger.info(f"Will test {len(strategies)} strategies: {', '.join(strategies)}")
    logger.info(f"Configuration: {config.num_workflows} workflows, QPS={config.qps}, Fanout={config.fanout_count}")
    logger.info("="*70)

    # Set default quantiles if not specified
    quantiles = config.quantiles if config.quantiles else [0.1, 0.25, 0.5, 0.75, 0.99]

    all_results = []

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

        result = run_single_experiment(config, logger, strategy_name=strategy_name)
        all_results.append(result)

        logger.info(f"\nCompleted experiment for strategy: {strategy_name}")

        # Brief pause between strategies
        time.sleep(2.0)

    logger.info("\n" + "="*70)
    logger.info("All strategy experiments completed!")
    logger.info("="*70)

    # ========================================================================
    # Save Results
    # ========================================================================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/results_workflow_b1b2_{timestamp}.json"

    output_data = {
        "experiment": "type2_deep_research_simulation",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_workflows": config.num_workflows,
            "qps": config.qps,
            "fanout_count": config.fanout_count,
            "warmup": args.warmup,
            "seed": args.seed
        },
        "results": all_results
    }

    ensure_directory("results")

    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")

    # ========================================================================
    # Print Comparison Table
    # ========================================================================

    print("\n" + "=" * 100)
    print("Strategy Comparison")
    print("=" * 100)
    print(f"{'Strategy':<15} {'A Avg (s)':<12} {'B1 Avg (s)':<12} {'B2 Avg (s)':<12} {'WF Avg (s)':<12} {'WF P95 (s)':<12} {'Completed':<12}")
    print("-" * 100)

    for result in all_results:
        strategy = result['strategy']
        a_avg = result['a_tasks']['avg_completion_time']
        b1_avg = result['b1_tasks']['avg_completion_time']
        b2_avg = result['b2_tasks']['avg_completion_time']
        wf_avg = result['workflows']['avg_workflow_time']
        wf_p95 = result['workflows']['p95_workflow_time']
        wf_completed = result['workflows']['num_completed']

        print(f"{strategy:<15} {a_avg:<12.2f} {b1_avg:<12.2f} {b2_avg:<12.2f} {wf_avg:<12.2f} {wf_p95:<12.2f} {wf_completed:<12}")

    print("=" * 100)


if __name__ == "__main__":
    main()
