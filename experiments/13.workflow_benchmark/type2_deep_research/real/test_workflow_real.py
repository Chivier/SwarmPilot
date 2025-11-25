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
    # Set mode to real
    export MODE=real
    export QPS=1.0
    export NUM_WORKFLOWS=100
    export FANOUT_COUNT=3

    # Run experiment
    python test_workflow_real.py
"""

import sys
import threading
import time
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common import (
    configure_logging,
    MetricsCollector,
    RateLimiter,
    ensure_directory
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
from queue import Queue


def main():
    # Load configuration (mode=real)
    config = DeepResearchConfig.from_env()

    # Ensure mode is set to real (overrides any environment variable)
    config.mode = "real"
    # Ensure model IDs are set for real mode
    config.model_a_id = "llm_service_small_model"
    config.model_b_id = "llm_service_small_model"
    config.model_merge_id = "llm_service_small_model"

    # Setup logging
    logger = configure_logging(level="INFO")
    logger.info(f"Mode: {config.mode} (forced for real test)")
    logger.info(f"Model A: {config.model_a_id}")
    logger.info(f"Model B: {config.model_b_id}")
    logger.info(f"Model Merge: {config.model_merge_id}")

    # Clear all tasks from schedulers before starting experiment
    from common import clear_scheduler_tasks
    logger.info("Clearing all tasks from schedulers before experiment...")
    clear_scheduler_tasks(config.scheduler_a_url, logger)
    if config.scheduler_b_url and config.scheduler_b_url != config.scheduler_a_url:
        clear_scheduler_tasks(config.scheduler_b_url, logger)
    logger.info("Task clearing complete")
    logger.info("=" * 80)
    logger.info("Deep Research Workflow - Real Cluster Mode")
    logger.info("=" * 80)
    logger.info(f"Mode: {config.mode}")
    logger.info(f"QPS: {config.qps}")
    logger.info(f"Duration: {config.duration}s")
    logger.info(f"Workflows: {config.num_workflows}")
    logger.info(f"Fanout count: {config.fanout_count}")
    logger.info(f"Model A: {config.model_a_id}")
    logger.info(f"Model B: {config.model_b_id}")
    logger.info(f"Model Merge: {config.model_merge_id}")
    logger.info(f"Scheduler A: {config.scheduler_a_url}")
    logger.info(f"Scheduler B: {config.scheduler_b_url}")
    logger.info(f"Max tokens: {config.max_tokens}")

    # Initialize components
    metrics = MetricsCollector(logger)
    rate_limiter = RateLimiter(rate=config.qps)
    workflow_states = {}
    state_lock = threading.Lock()

    # Create inter-thread queues
    a_result_queue = Queue()
    b1_result_queue = Queue()
    merge_trigger_queue = Queue()

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

            # Log progress
            with state_lock:
                completed = merge_receiver.completed_workflows
                total = config.num_workflows
                progress_pct = (completed / total * 100) if total > 0 else 0

                logger.info(
                    f"Progress: {completed}/{total} workflows "
                    f"({progress_pct:.1f}%) - "
                    f"Elapsed: {time.time() - start_time:.1f}s"
                )

                # Early exit if all workflows complete
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

    # Export metrics
    output_dir = ensure_directory(config.output_dir)
    metrics_file = output_dir / config.metrics_file
    metrics.export_to_json(str(metrics_file))
    logger.info(f"Metrics exported to: {metrics_file}")

    # Print metrics report
    print("\n" + metrics.generate_text_report())


if __name__ == "__main__":
    main()
