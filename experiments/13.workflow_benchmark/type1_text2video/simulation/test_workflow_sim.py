#!/usr/bin/env python3
"""
Text2Video Workflow - Simulation Mode

This script runs the Text2Video workflow (A1→A2→B) using sleep models for simulation.

Architecture:
- Thread 1 (A1 Submitter): Submit A1 tasks with Poisson arrivals
- Thread 2 (A1 Receiver): Receive A1 results → trigger A2
- Thread 3 (A2 Submitter): Submit A2 tasks
- Thread 4 (A2 Receiver): Receive A2 results → trigger B
- Thread 5 (B Submitter): Submit B tasks with loop support
- Thread 6 (B Receiver): Receive B results → loop or complete

Workflow Pattern:
- A1: caption → positive prompt
- A2: positive prompt → negative prompt
- B: video generation (1-4 iterations per workflow)

Usage:
    # Set environment variables
    export QPS=2.0
    export DURATION=300
    export NUM_WORKFLOWS=600
    export MAX_B_LOOPS=4

    # Run simulation
    python -m type1_text2video.simulation.test_workflow_sim
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
from type1_text2video.config import Text2VideoConfig
from type1_text2video.workflow_data import load_captions
from type1_text2video.submitters import A1TaskSubmitter, A2TaskSubmitter, BTaskSubmitter
from type1_text2video.receivers import A1TaskReceiver, A2TaskReceiver, BTaskReceiver


def run_single_experiment(config, captions, logger, strategy_name=None):
    """Run a single experiment with the given configuration.

    Args:
        config: Text2VideoConfig instance
        captions: List of captions to use
        logger: Logger instance
        strategy_name: Optional strategy name (for logging purposes)
    """

    logger.info("="*70)
    if strategy_name:
        logger.info(f"Text2Video Workflow Simulation - Strategy: {strategy_name}")
    else:
        logger.info("Text2Video Workflow Simulation")
    logger.info("="*70)
    logger.info(f"QPS: {config.qps}")
    logger.info(f"Duration: {config.duration}s")
    logger.info(f"Workflows: {config.num_workflows}")
    logger.info(f"Max B loops: {config.max_b_loops}")
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
    a1_result_queue = Queue()  # A1 receiver → A2 submitter
    a2_result_queue = Queue()  # A2 receiver → B submitter

    # Metrics collector
    metrics = MetricsCollector(logger=logger)

    # Rate limiter (shared across all submitters)
    rate_limiter = RateLimiter(rate=config.qps)

    # ========================================================================
    # Create Components
    # ========================================================================

    logger.info("Initializing components...")

    # A1 Submitter (Thread 1)
    a1_submitter = A1TaskSubmitter(
        name="A1Submitter",
        captions=captions,
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        scheduler_url=config.scheduler_a_url,
        rate_limiter=rate_limiter,
        duration=config.duration,
    )

    # A2 Submitter (Thread 3)
    a2_submitter = A2TaskSubmitter(
        name="A2Submitter",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        a1_result_queue=a1_result_queue,
        scheduler_url=config.scheduler_a_url,
        rate_limiter=rate_limiter,
    )

    # B Submitter (Thread 5)
    b_submitter = BTaskSubmitter(
        name="BSubmitter",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        a2_result_queue=a2_result_queue,
        scheduler_url=config.scheduler_b_url,
        rate_limiter=rate_limiter,
    )

    # A1 Receiver (Thread 2)
    a1_receiver = A1TaskReceiver(
        name="A1Receiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        a2_submitter=a2_submitter,
        a1_result_queue=a1_result_queue,
        scheduler_url=config.scheduler_a_url,
        model_id=config.model_a_id,
    )

    # A2 Receiver (Thread 4)
    a2_receiver = A2TaskReceiver(
        name="A2Receiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        b_submitter=b_submitter,
        a2_result_queue=a2_result_queue,
        scheduler_url=config.scheduler_a_url,
        model_id=config.model_a_id,
    )

    # B Receiver (Thread 6)
    b_receiver = BTaskReceiver(
        name="BReceiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        b_submitter=b_submitter,
        scheduler_url=config.scheduler_b_url,
        model_id=config.model_b_id,
    )

    # ========================================================================
    # Start All Threads
    # ========================================================================

    logger.info("Starting all threads...")
    start_time = time.time()

    # Start receivers first (so they're ready to receive)
    a1_receiver.start()
    a2_receiver.start()
    b_receiver.start()

    # Wait a moment for receivers to connect
    time.sleep(1.0)

    # Start submitters
    a2_submitter.start()
    b_submitter.start()
    a1_submitter.start()  # Start A1 last (it drives the workflow)

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

        logger.info(
            f"Progress: {elapsed:.1f}s elapsed, "
            f"{completed}/{total_workflows} workflows complete "
            f"({100*completed/total_workflows if total_workflows > 0 else 0:.1f}%)"
        )

    # ========================================================================
    # Stop All Threads
    # ========================================================================

    logger.info("Stopping all threads...")

    # Stop submitters first (no new tasks)
    a1_submitter.stop()
    a2_submitter.stop()
    b_submitter.stop()

    # Wait for submitters to finish
    a1_submitter.join(timeout=10)
    a2_submitter.join(timeout=10)
    b_submitter.join(timeout=10)

    # Stop receivers
    a1_receiver.stop()
    a2_receiver.stop()
    b_receiver.stop()

    # Wait for receivers to finish
    a1_receiver.join(timeout=10)
    a2_receiver.join(timeout=10)
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

    with state_lock:
        total_workflows = len(workflow_states)
        completed_workflows = sum(1 for w in workflow_states.values() if w.is_complete())
        total_b_iterations = sum(len(w.b_complete_times) for w in workflow_states.values())

    logger.info(f"Total workflows: {total_workflows}")
    logger.info(f"Completed workflows: {completed_workflows}")
    logger.info(f"Completion rate: {100*completed_workflows/total_workflows if total_workflows > 0 else 0:.1f}%")
    logger.info(f"Total B iterations: {total_b_iterations}")
    logger.info(f"Avg B iterations/workflow: {total_b_iterations/total_workflows if total_workflows > 0 else 0:.2f}")

    logger.info("\nSubmitter Statistics:")
    logger.info(f"  A1: {a1_submitter.submitted_count} submitted, {a1_submitter.failed_count} failed")
    logger.info(f"  A2: {a2_submitter.submitted_count} submitted, {a2_submitter.failed_count} failed")
    logger.info(f"  B:  {b_submitter.submitted_count} submitted, {b_submitter.failed_count} failed")

    logger.info("\nReceiver Statistics:")
    logger.info(f"  A1: {a1_receiver.received_count} received")
    logger.info(f"  A2: {a2_receiver.received_count} received")
    logger.info(f"  B:  {b_receiver.received_count} received, {b_receiver.completed_workflows} workflows completed")

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

    config = Text2VideoConfig.from_env()
    logger = configure_logging(level="INFO")

    # Load captions
    project_root = Path(__file__).parent.parent.parent.parent.parent
    caption_path = project_root / config.caption_file

    logger.info(f"Loading captions from: {caption_path}")
    try:
        captions = load_captions(str(caption_path))
        logger.info(f"Loaded {len(captions)} captions")
    except FileNotFoundError:
        logger.error(f"Caption file not found: {caption_path}")
        logger.error("Please ensure captions_10k.json exists in project root")
        sys.exit(1)

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
            run_single_experiment(config, captions, logger, strategy_name=strategy_name)

            logger.info(f"\nCompleted experiment for strategy: {strategy_name}")

        logger.info("\n" + "="*70)
        logger.info("All strategy experiments completed!")
        logger.info("="*70)

    else:
        # No strategies specified - run single experiment with current scheduler configuration
        logger.info("Running single experiment (no strategy testing)")
        run_single_experiment(config, captions, logger)


if __name__ == "__main__":
    main()
