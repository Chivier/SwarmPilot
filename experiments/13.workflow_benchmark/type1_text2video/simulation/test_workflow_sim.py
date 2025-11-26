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
    python -m type1_text2video.simulation.test_workflow_sim \\
        --num-workflows 50 --qps 2.0 --strategies min_time,probabilistic --duration 300
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
    clear_scheduler_tasks,
    create_base_parser,
    add_type1_args,
    parse_strategies,
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
        strategy_name: Optional strategy name (for logging and task ID generation)
    """

    # Update config.strategy if strategy_name is provided
    if strategy_name:
        config.strategy = strategy_name
        logger.info(f"Using strategy: {strategy_name}")

    logger.info("="*70)
    if strategy_name:
        logger.info(f"Text2Video Workflow Simulation - Strategy: {strategy_name}")
    else:
        logger.info("Text2Video Workflow Simulation")
    logger.info("="*70)
    logger.info(f"QPS: {config.qps}")
    logger.info(f"Duration: {config.duration}s")
    logger.info(f"Workflows: {config.num_workflows}")
    logger.info(f"Max B loops config: {config.get_max_b_loops_config()}")
    logger.info(f"Frame count config: {config.get_frame_count_config()}")
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
    metrics = MetricsCollector(custom_logger=logger)

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
        metrics=metrics,
        duration=config.duration,
    )

    # Pre-generate task IDs for receivers to subscribe to
    # Must match the format used in submitters.py
    strategy = getattr(config, 'strategy', 'probabilistic')
    a1_task_ids = [f"task-A1-{strategy}-workflow-{i:04d}" for i in range(config.num_workflows)]
    a2_task_ids = [f"task-A2-{strategy}-workflow-{i:04d}" for i in range(config.num_workflows)]

    # Generate all B task IDs (including all loop iterations)
    # With distribution-based sampling, each workflow may have different max_b_loops,
    # so we use each workflow's actual max_b_loops from the pre-generated workflow data
    b_task_ids = []
    with state_lock:
        for workflow_id, workflow_data in workflow_states.items():
            workflow_num = workflow_id.split('-')[-1]
            for loop in range(1, workflow_data.max_b_loops + 1):
                b_task_ids.append(f"task-B{loop}-{strategy}-workflow-{workflow_num}")

    # A2 Submitter (Thread 3)
    a2_submitter = A2TaskSubmitter(
        name="A2Submitter",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        a1_result_queue=a1_result_queue,
        scheduler_url=config.scheduler_a_url,
        rate_limiter=rate_limiter,
        metrics=metrics,
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
        metrics=metrics,
    )

    # A1 Receiver (Thread 2)
    a1_receiver = A1TaskReceiver(
        name="A1Receiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        a2_submitter=a2_submitter,
        a1_result_queue=a1_result_queue,
        task_ids=a1_task_ids,
        scheduler_url=config.scheduler_a_url,
        model_id=config.model_a_id,
        metrics=metrics,
    )

    # A2 Receiver (Thread 4)
    a2_receiver = A2TaskReceiver(
        name="A2Receiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        b_submitter=b_submitter,
        a2_result_queue=a2_result_queue,
        task_ids=a2_task_ids,
        scheduler_url=config.scheduler_a_url,
        model_id=config.model_a_id,
        metrics=metrics,
    )

    # B Receiver (Thread 6)
    b_receiver = BTaskReceiver(
        name="BReceiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        b_submitter=b_submitter,
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

    # Export to JSON (with portion_stats filtering)
    metrics.export_to_json(str(metrics_path), portion_stats=config.portion_stats)

    # Generate text report (with portion_stats filtering)
    report = metrics.generate_text_report(portion_stats=config.portion_stats)
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
    # Parse Command Line Arguments
    # ========================================================================

    parser = create_base_parser(description="Text2Video Workflow - Simulation Mode")
    parser = add_type1_args(parser)
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
    config = Text2VideoConfig(
        mode="simulation",  # Hardcoded for simulation
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
    )

    logger = configure_logging(level="INFO")

    logger.info(f"Mode: {config.mode} (hardcoded for simulation)")
    logger.info(f"Model A: {config.model_a_id}")
    logger.info(f"Model B: {config.model_b_id}")
    logger.info(f"Scheduler A: {config.scheduler_a_url}")
    logger.info(f"Scheduler B: {config.scheduler_b_url}")

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

    logger.info("="*70)
    logger.info("Strategy-based Testing Mode")
    logger.info(f"Will test {len(strategies)} strategies: {', '.join(strategies)}")
    logger.info("="*70)

    # Set default quantiles if not specified
    quantiles = config.quantiles if config.quantiles else [0.1, 0.25, 0.5, 0.75, 0.99]

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
        run_single_experiment(config, captions, logger, strategy_name=strategy_name)

        logger.info(f"\nCompleted experiment for strategy: {strategy_name}")

    logger.info("\n" + "="*70)
    logger.info("All strategy experiments completed!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
