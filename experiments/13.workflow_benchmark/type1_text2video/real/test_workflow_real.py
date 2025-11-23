#!/usr/bin/env python3
"""
Text2Video Workflow - Real Cluster Mode

This script runs the Text2Video workflow (A1→A2→B) in real cluster mode
using actual model IDs and task inputs instead of sleep simulations.

Key differences from simulation mode:
- Uses real model inputs (sentence, max_tokens) instead of sleep_time
- Includes token estimation in metadata for performance prediction
- Supports predictor and planner integration
- Can be deployed across distributed schedulers

Architecture:
- Thread 1: A1 submitter (Poisson process, QPS-controlled)
- Thread 2: A1 receiver → triggers A2
- Thread 3: A2 submitter
- Thread 4: A2 receiver → triggers B
- Thread 5: B submitter
- Thread 6: B receiver (with loop control)

Usage:
    # Set mode to real
    export MODE=real
    export QPS=2.0
    export NUM_WORKFLOWS=100

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
from type1_text2video.config import Text2VideoConfig
from type1_text2video.workflow_data import load_captions
from type1_text2video.submitters import A1TaskSubmitter, A2TaskSubmitter, BTaskSubmitter
from type1_text2video.receivers import A1TaskReceiver, A2TaskReceiver, BTaskReceiver
from queue import Queue


def main():
    # Load configuration (mode=real)
    config = Text2VideoConfig.from_env()

    # Ensure mode is set to real
    if config.mode != "real":
        print(f"Warning: MODE should be 'real', but got '{config.mode}'. Setting to 'real'.")
        config.mode = "real"

    # Setup logging
    logger = configure_logging(level="INFO")
    logger.info("=" * 80)
    logger.info("Text2Video Workflow - Real Cluster Mode")
    logger.info("=" * 80)
    logger.info(f"Mode: {config.mode}")
    logger.info(f"QPS: {config.qps}")
    logger.info(f"Duration: {config.duration}s")
    logger.info(f"Workflows: {config.num_workflows}")
    logger.info(f"Max B loops: {config.max_b_loops}")
    logger.info(f"Model A: {config.model_a_id}")
    logger.info(f"Model B: {config.model_b_id}")
    logger.info(f"Scheduler A: {config.scheduler_a_url}")
    logger.info(f"Scheduler B: {config.scheduler_b_url}")
    logger.info(f"Max tokens: {config.max_tokens}")

    # Load captions
    caption_path = Path(__file__).parent.parent.parent.parent.parent / config.caption_file
    if not caption_path.exists():
        logger.warning(f"Caption file not found: {caption_path}")
        logger.warning("Using dummy captions")
        captions = [f"Sample caption {i}" for i in range(100)]
    else:
        captions = load_captions(str(caption_path))

    logger.info(f"Loaded {len(captions)} captions")

    # Initialize components
    metrics = MetricsCollector(logger)
    rate_limiter = RateLimiter(rate=config.qps)
    workflow_states = {}
    state_lock = threading.Lock()

    # Create inter-thread queues
    a1_result_queue = Queue()
    a2_result_queue = Queue()

    # Create submitters
    a1_submitter = A1TaskSubmitter(
        name="A1Submitter",
        captions=captions,
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        scheduler_url=config.scheduler_a_url,
        rate_limiter=rate_limiter,
        metrics=metrics,
        logger=logger
    )

    a2_submitter = A2TaskSubmitter(
        name="A2Submitter",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        a1_result_queue=a1_result_queue,
        scheduler_url=config.scheduler_a_url,
        rate_limiter=None,  # No rate limiting for A2 (triggered by A1)
        metrics=metrics,
        logger=logger
    )

    b_submitter = BTaskSubmitter(
        name="BSubmitter",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        a2_result_queue=a2_result_queue,
        scheduler_url=config.scheduler_b_url,
        rate_limiter=None,  # No rate limiting for B (triggered by A2)
        metrics=metrics,
        logger=logger
    )

    # Create receivers
    a1_receiver = A1TaskReceiver(
        name="A1Receiver",
        model_id=config.model_a_id,
        a2_submitter=a2_submitter,
        a1_result_queue=a1_result_queue,
        workflow_states=workflow_states,
        state_lock=state_lock,
        scheduler_url=config.scheduler_a_url,
        metrics=metrics,
        logger=logger
    )

    a2_receiver = A2TaskReceiver(
        name="A2Receiver",
        model_id=config.model_a_id,
        b_submitter=b_submitter,
        a2_result_queue=a2_result_queue,
        workflow_states=workflow_states,
        state_lock=state_lock,
        scheduler_url=config.scheduler_a_url,
        metrics=metrics,
        logger=logger
    )

    b_receiver = BTaskReceiver(
        name="BReceiver",
        model_id=config.model_b_id,
        b_submitter=b_submitter,
        workflow_states=workflow_states,
        state_lock=state_lock,
        scheduler_url=config.scheduler_b_url,
        metrics=metrics,
        logger=logger
    )

    # Start all threads
    logger.info("Starting experiment...")
    start_time = time.time()

    a1_submitter.start()
    a1_receiver.start()
    a2_submitter.start()
    a2_receiver.start()
    b_submitter.start()
    b_receiver.start()

    # Monitor progress
    try:
        while time.time() - start_time < config.duration:
            time.sleep(10)

            # Log progress
            with state_lock:
                completed = b_receiver.completed_workflows
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
    a1_submitter.stop()
    a1_receiver.stop()
    a2_submitter.stop()
    a2_receiver.stop()
    b_submitter.stop()
    b_receiver.stop()

    # Wait for threads to finish
    a1_submitter.join()
    a1_receiver.join()
    a2_submitter.join()
    a2_receiver.join()
    b_submitter.join()
    b_receiver.join()

    # Calculate final statistics
    elapsed_time = time.time() - start_time
    with state_lock:
        completed_workflows = b_receiver.completed_workflows

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
