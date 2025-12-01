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
    add_type1_args,
    parse_strategies,
    generate_strategy_comparison_table,
)
from type1_text2video.config import Text2VideoConfig
from type1_text2video.workflow_data import load_captions, pre_generate_workflows
from type1_text2video.submitters import A1TaskSubmitter, A2TaskSubmitter, BTaskSubmitter
from type1_text2video.receivers import A1TaskReceiver, A2TaskReceiver, BTaskReceiver


def run_single_experiment(config, captions, logger, strategy_name=None, pre_generated_workflows=None):
    """Run a single experiment with the given configuration.

    Args:
        config: Text2VideoConfig instance
        captions: List of captions to use
        logger: Logger instance
        strategy_name: Optional strategy name (for logging and task ID generation)
        pre_generated_workflows: Optional pre-generated workflow data for reproducibility.
                                 If provided, all strategies use identical workflow data.

    Returns:
        Dict with strategy results containing task_metrics and workflow_metrics
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
    logger.info(f"QPS: {config.qps}, Duration: {config.duration}s, Workflows: {config.num_workflows}")
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
        pre_generated_workflows=pre_generated_workflows,
        scheduler_url=config.scheduler_a_url,
        qps=config.qps,
        duration=config.duration,
        rate_limiter=rate_limiter,
        metrics=metrics,
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
        rate_limiter=None,  # A2 not rate limited - only A1 controls workflow arrival rate
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
        rate_limiter=None,  # B not rate limited - only A1 controls workflow arrival rate
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
    if early_stop:
        logger.info(f"Stopped early: All {target_completion} workflows needed for metrics completed")

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

    metrics_path = output_dir / metrics_filename
    logger.info(f"\nExporting metrics to: {metrics_path}")

    # Export to JSON (with portion_stats filtering)
    metrics.export_to_json(str(metrics_path), portion_stats=config.portion_stats)

    # Generate detailed text report with per-task-type metrics (Avg, P90, P99)
    report = metrics.generate_detailed_text_report(
        task_types=["A1", "A2", "B"],
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
    task_types = ["A1", "A2", "B"]
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
        max_sleep_time_seconds=args.max_sleep_time,
    )

    logger = configure_logging(level="INFO")

    # ========================================================================
    # Display Configuration and Data Statistics
    # ========================================================================
    logger.info("="*70)
    logger.info("Configuration Summary")
    logger.info("="*70)
    logger.info(f"Mode: {config.mode}")
    logger.info(f"Model A (LLM): {config.model_a_id}")
    logger.info(f"Model B (T2VID): {config.model_b_id}")
    logger.info(f"Scheduler A: {config.scheduler_a_url}")
    logger.info(f"Scheduler B: {config.scheduler_b_url}")

    # Display data statistics from data_loader
    logger.info("-"*70)
    logger.info("Benchmark Data Statistics")
    logger.info("-"*70)
    data_stats = config.data_loader.get_statistics_summary()

    # LLM statistics
    llm = data_stats["llm"]
    logger.info(f"LLM Samples: {llm['sample_count']} samples")
    logger.info(f"  Runtime range: {llm['min_runtime_ms']:.1f}ms - {llm['max_runtime_ms']:.1f}ms")
    logger.info(f"  Average runtime: {llm['avg_runtime_ms']:.1f}ms ({llm['avg_runtime_ms']/1000:.2f}s)")

    # T2VID statistics
    t2vid = data_stats["t2vid"]
    logger.info(f"T2VID Samples: {t2vid['training_sample_count']} training samples")
    logger.info(f"  Regression model: runtime = {t2vid['regression_slope']:.2f} * frames + {t2vid['regression_intercept']:.2f}")
    logger.info(f"  Scale factor: {t2vid['scale_factor']:.4f}")
    logger.info(f"  Runtime range (scaled): {t2vid['min_runtime_ms']:.1f}ms - {t2vid['max_runtime_ms']:.1f}ms")
    logger.info(f"  Average runtime: {t2vid['avg_runtime_ms']:.1f}ms ({t2vid['avg_runtime_ms']/1000:.2f}s)")

    # Frame statistics
    frames = data_stats["frames"]
    logger.info(f"Frame Data: {frames['caption_count']} captions")
    logger.info(f"  Frame range: {frames['min_frames']} - {frames['max_frames']}")

    # Config statistics
    cfg = data_stats["config"]
    logger.info(f"Max sleep time: {cfg['max_sleep_time_ms']/1000:.1f}s")
    logger.info(f"Data sampling seed: {cfg['seed']}")

    logger.info("-"*70)
    logger.info("Distribution Configurations")
    logger.info("-"*70)
    logger.info(f"Max B loops: {config.get_max_b_loops_config()}")
    if config.frame_count_config is not None:
        logger.info(f"Frame count: from config ({config.get_frame_count_config()})")
    else:
        logger.info(f"Frame count: from dataset (captions_10k.jsonl)")
    logger.info("="*70)

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
    # Pre-generate Workflow Data (BEFORE strategy loop)
    # ========================================================================

    logger.info("="*70)
    logger.info("Pre-generating Workflow Data")
    logger.info("="*70)

    # Pre-generate all workflow data ONCE before testing any strategies.
    # This ensures all strategies use IDENTICAL input data for fair comparison.
    pre_generated_workflows = pre_generate_workflows(
        config, captions, seed=42,
        submission_order=args.submission_order
    )

    logger.info(f"Pre-generated {len(pre_generated_workflows)} workflows")

    # Log submission order info
    if args.submission_order in ("alternating-peaks", "interleaved-2", "interleaved-4"):
        from collections import Counter
        peak_counts = Counter(w.peak_index for w in pre_generated_workflows if w.peak_index is not None)
        logger.info(f"Using {args.submission_order} submission order")
        logger.info(f"  Peak distribution: {dict(sorted(peak_counts.items()))}")
        if args.submission_order.startswith("interleaved-"):
            num_splits = int(args.submission_order.split("-")[1])
            logger.info(f"  Interleaving with {num_splits} splits per peak")
    logger.info(f"  Sample workflow[0]: sleep_times A1={pre_generated_workflows[0].a1_sleep_time:.3f}s, "
                f"A2={pre_generated_workflows[0].a2_sleep_time:.3f}s, "
                f"B={pre_generated_workflows[0].b_sleep_time:.3f}s, "
                f"max_b_loops={pre_generated_workflows[0].max_b_loops}, "
                f"frame_count={pre_generated_workflows[0].frame_count}")

    # ========================================================================
    # Strategy Management
    # ========================================================================

    logger.info("="*70)
    logger.info("Strategy-based Testing Mode")
    logger.info(f"Will test {len(strategies)} strategies: {', '.join(strategies)}")
    logger.info("="*70)

    # Set default quantiles if not specified
    quantiles = config.quantiles if config.quantiles else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

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
        experiment_results = run_single_experiment(
            config, captions, logger,
            strategy_name=strategy_name,
            pre_generated_workflows=pre_generated_workflows
        )

        # Store results for comparison
        all_strategy_results[strategy_name] = experiment_results

        logger.info(f"\nCompleted experiment for strategy: {strategy_name}")

    # Generate strategy comparison table
    if len(all_strategy_results) > 1:
        comparison_table = generate_strategy_comparison_table(
            all_strategy_results,
            task_types=["A1", "A2", "B"]
        )
        print(comparison_table)
        logger.info("Strategy comparison table generated")

    logger.info("\n" + "="*70)
    logger.info("All strategy experiments completed!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
