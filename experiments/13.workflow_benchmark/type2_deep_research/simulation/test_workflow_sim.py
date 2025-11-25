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
    python -m type2_deep_research.simulation.test_workflow_sim --help
"""

import sys
import threading
import time
import argparse
import json
import os
import numpy as np
from datetime import datetime
from dataclasses import asdict
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


def calculate_task_metrics_from_collector(collector: MetricsCollector, task_type: str, workflow_states: Dict) -> Dict:
    """
    Calculate metrics for a specific task type from the collector.
    
    Args:
        collector: MetricsCollector instance
        task_type: Task type string ("A", "B1", "B2", "merge")
        workflow_states: Dictionary of workflow states to check for warmup status
        
    Returns:
        Dictionary of metrics
    """
    # Filter tasks by type
    # Note: MetricsCollector stores task_type as passed during record_task_submit
    # The submitters use "A", "B1", "B2", "merge"
    
    all_tasks = [t for t in collector.task_metrics if t.task_type == task_type]
    
    # Separate warmup and actual tasks
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
        else:
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


def calculate_workflow_metrics_from_collector(collector: MetricsCollector, workflow_states: Dict) -> Dict:
    """
    Calculate workflow-level metrics from the collector.
    
    Args:
        collector: MetricsCollector instance
        workflow_states: Dictionary of workflow states
        
    Returns:
        Dictionary of workflow metrics
    """
    all_workflows = collector.workflow_metrics
    
    actual_workflows = []
    warmup_workflows = []
    
    for w in all_workflows:
        # Check if workflow is warmup
        wf_state = workflow_states.get(w.workflow_id)
        is_warmup = False
        if wf_state:
            is_warmup = getattr(wf_state, 'is_warmup', False)
            
        if is_warmup:
            warmup_workflows.append(w)
        else:
            actual_workflows.append(w)
            
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
        "num_excluded": 0, # Not strictly tracking excluded in this simple version
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

    # Calculate metrics
    a_metrics = calculate_task_metrics_from_collector(metrics, "A", workflow_states)
    b1_metrics = calculate_task_metrics_from_collector(metrics, "B1", workflow_states)
    b2_metrics = calculate_task_metrics_from_collector(metrics, "B2", workflow_states)
    merge_metrics = calculate_task_metrics_from_collector(metrics, "merge", workflow_states)
    wf_metrics = calculate_workflow_metrics_from_collector(metrics, workflow_states)

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
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Deep Research Workflow Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--num-workflows",
        type=int,
        default=100,
        help="Number of workflows to generate and execute per strategy"
    )

    parser.add_argument(
        "--qps",
        type=float,
        default=1.0,
        help="Target queries per second (QPS) for A task submission"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        choices=["min_time", "round_robin", "probabilistic", "random", "po2", "serverless"],
        help="Scheduling strategies to test (if not set, uses env var or default)"
    )

    parser.add_argument(
        "--warmup",
        type=float,
        default=0.1,
        help="Warmup task ratio (0.0-1.0)"
    )
    
    parser.add_argument(
        "--fanout",
        type=int,
        default=3,
        help="Fanout count (B tasks per A task)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=600,
        help="Maximum duration in seconds"
    )

    args = parser.parse_args()

    # ========================================================================
    # Configuration and Setup
    # ========================================================================

    # Initialize config from env first, then override with args
    config = DeepResearchConfig.from_env()

    # Ensure mode is set to simulation (overrides any environment variable)
    config.mode = "simulation"
    config.model_a_id = "sleep_model_a"
    config.model_b_id = "sleep_model_b"
    config.model_merge_id = "sleep_model_a"

    logger = configure_logging(level="INFO")
    logger.info(f"Mode: {config.mode} (forced for simulation test)")
    logger.info(f"Model A: {config.model_a_id}")
    logger.info(f"Model B: {config.model_b_id}")
    logger.info(f"Model Merge: {config.model_merge_id}")

    # Override with args
    config.num_workflows = args.num_workflows
    config.qps = args.qps
    config.fanout_count = args.fanout
    config.duration = args.duration
    
    # Calculate num_warmup based on ratio
    config.num_warmup = int(args.num_workflows * args.warmup)
    
    # Set strategies
    strategies = args.strategies
    if not strategies:
        strategies = config.strategies if config.strategies else ["probabilistic"]

    # ========================================================================
    # Strategy Management
    # ========================================================================

    logger.info("="*70)
    logger.info("Deep Research Workflow Benchmark")
    logger.info(f"Configuration: {config.num_workflows} workflows, QPS={config.qps}, Fanout={config.fanout_count}")
    logger.info(f"Strategies to test: {', '.join(strategies)}")
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

        # Setup this strategy (clear tasks and configure schedulers)
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
            "warmup_ratio": args.warmup,
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
