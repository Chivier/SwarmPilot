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
    # Set environment variables
    export QPS=1.0
    export DURATION=300
    export NUM_WORKFLOWS=600
    export FANOUT_COUNT=3

    # Run simulation
    python -m type2_deep_research.simulation.test_workflow_sim
"""

import sys
import threading
import time
from pathlib import Path
from queue import Queue

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


def run_single_experiment(config, logger, strategy_name=None):
    """Run a single experiment with the given configuration.

    Args:
        config: DeepResearchConfig instance
        logger: Logger instance
        strategy_name: Optional strategy name (for logging purposes)
    """

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
    metrics = MetricsCollector(logger=logger)

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

    # A Receiver (Thread 2)
    a_receiver = ATaskReceiver(
        name="AReceiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        b1_submitter=b1_submitter,
        a_result_queue=a_result_queue,
        scheduler_url=config.scheduler_a_url,
        model_id=config.model_a_id,
    )

    # B1 Receiver (Thread 4)
    b1_receiver = B1TaskReceiver(
        name="B1Receiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        b2_submitter=b2_submitter,
        b1_result_queue=b1_result_queue,
        scheduler_url=config.scheduler_b_url,
        model_id=config.model_b_id,
    )

    # B2 Receiver (Thread 6)
    b2_receiver = B2TaskReceiver(
        name="B2Receiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        merge_submitter=merge_submitter,
        merge_trigger_queue=merge_trigger_queue,
        scheduler_url=config.scheduler_b_url,
        model_id=config.model_b_id,
    )

    # Merge Receiver (Thread 8)
    merge_receiver = MergeTaskReceiver(
        name="MergeReceiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        scheduler_url=config.scheduler_a_url,
        model_id=config.model_merge_id,
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

        with state_lock:
            total_workflows = len(workflow_states)
            completed = sum(1 for w in workflow_states.values() if w.is_complete())
            all_b1_done = sum(1 for w in workflow_states.values() if w.all_b1_complete())
            all_b2_done = sum(1 for w in workflow_states.values() if w.all_b2_complete())

        logger.info(
            f"Progress: {elapsed:.1f}s elapsed, "
            f"{completed}/{total_workflows} workflows complete "
            f"({100*completed/total_workflows if total_workflows > 0 else 0:.1f}%), "
            f"B1 done: {all_b1_done}, B2 done: {all_b2_done}"
        )

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

    with state_lock:
        total_workflows = len(workflow_states)
        completed_workflows = sum(1 for w in workflow_states.values() if w.is_complete())
        total_b1_tasks = sum(len(w.b1_complete_times) for w in workflow_states.values())
        total_b2_tasks = sum(len(w.b2_complete_times) for w in workflow_states.values())

    logger.info(f"Total workflows: {total_workflows}")
    logger.info(f"Completed workflows: {completed_workflows}")
    logger.info(f"Completion rate: {100*completed_workflows/total_workflows if total_workflows > 0 else 0:.1f}%")
    logger.info(f"Total B1 tasks completed: {total_b1_tasks}")
    logger.info(f"Total B2 tasks completed: {total_b2_tasks}")
    logger.info(f"Avg B1 tasks/workflow: {total_b1_tasks/total_workflows if total_workflows > 0 else 0:.2f}")
    logger.info(f"Avg B2 tasks/workflow: {total_b2_tasks/total_workflows if total_workflows > 0 else 0:.2f}")

    logger.info("\nSubmitter Statistics:")
    logger.info(f"  A:     {a_submitter.submitted_count} submitted, {a_submitter.failed_count} failed")
    logger.info(f"  B1:    {b1_submitter.submitted_count} submitted, {b1_submitter.failed_count} failed")
    logger.info(f"  B2:    {b2_submitter.submitted_count} submitted, {b2_submitter.failed_count} failed")
    logger.info(f"  Merge: {merge_submitter.submitted_count} submitted, {merge_submitter.failed_count} failed")

    logger.info("\nReceiver Statistics:")
    logger.info(f"  A:     {a_receiver.received_count} received")
    logger.info(f"  B1:    {b1_receiver.received_count} received")
    logger.info(f"  B2:    {b2_receiver.received_count} received")
    logger.info(f"  Merge: {merge_receiver.received_count} received, {merge_receiver.completed_workflows} workflows completed")

    # ========================================================================
    # Export Metrics
    # ========================================================================

    output_dir = Path(config.output_dir)
    ensure_directory(str(output_dir))

    metrics_path = output_dir / config.metrics_file
    logger.info(f"\nExporting metrics to: {metrics_path}")

    # Export to JSON
    metrics.export_to_json(str(metrics_path))

    # Generate text report
    report = metrics.generate_text_report()
    print("\n" + report)

    logger.info("="*70)
    if strategy_name:
        logger.info(f"Simulation complete for strategy: {strategy_name}!")
    else:
        logger.info("Simulation complete!")
    logger.info("="*70)


def main():
    """Main entry point - handles strategy management and experiment orchestration."""

    # ========================================================================
    # Configuration and Setup
    # ========================================================================

    config = DeepResearchConfig.from_env()
    logger = configure_logging(level="INFO")

    # ========================================================================
    # Strategy Management
    # ========================================================================

    if config.strategies:
        logger.info("="*70)
        logger.info("Strategy-based Testing Mode")
        logger.info(f"Will test {len(config.strategies)} strategies: {', '.join(config.strategies)}")
        logger.info("="*70)

        # Set default quantiles if not specified
        quantiles = config.quantiles if config.quantiles else [0.1, 0.25, 0.5, 0.75, 0.99]

        # Run experiment for each strategy
        for strategy_name in config.strategies:
            logger.info("\n" + "="*70)
            logger.info(f"Setting up strategy: {strategy_name}")
            logger.info("="*70)

            # Setup this strategy (clear tasks and configure schedulers)
            strategy_results = setup_scheduler_strategies(
                strategies=[strategy_name],  # Only this strategy
                scheduler_a_url=config.scheduler_a_url,
                scheduler_b_url=config.scheduler_b_url,
                target_quantile=config.target_quantile,
                quantiles=quantiles,
                logger=logger
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
            run_single_experiment(config, logger, strategy_name=strategy_name)

            logger.info(f"\nCompleted experiment for strategy: {strategy_name}")

        logger.info("\n" + "="*70)
        logger.info("All strategy experiments completed!")
        logger.info("="*70)

    else:
        # No strategies specified - run single experiment with current scheduler configuration
        logger.info("Running single experiment (no strategy testing)")
        run_single_experiment(config, logger)


if __name__ == "__main__":
    main()
