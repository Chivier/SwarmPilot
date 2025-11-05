#!/usr/bin/env python3
"""
Continuous Request Mode Wrapper for Multi-Model Workflow Experiments.

This wrapper enables continuous request mode for any experiment (04-07) without
modifying the original experiment code. It handles:
1. Generating 2*num_workflows tasks
2. Marking first num_workflows as target for statistics
3. Running the experiment
4. Forcefully clearing schedulers after target workflows complete
5. Calculating and displaying continuous-mode-specific metrics
"""

import sys
import time
import logging
from typing import List, Dict, Optional
from pathlib import Path

# Import common utilities from unified experiment
from common import (
    force_clear_scheduler_tasks,
    calculate_makespan,
    print_continuous_mode_summary,
    calculate_task_metrics,
    calculate_workflow_metrics
)


logger = logging.getLogger("ContinuousWrapper")


def run_experiment_continuous(
    experiment_module,
    experiment_key: str,
    num_workflows: int,
    qps: float,
    gqps: Optional[float],
    warmup: float,
    seed: int,
    strategies: List[str]
):
    """
    Run an experiment in continuous request mode.

    The experiment itself handles:
    1. Generating 2*num_workflows tasks
    2. Marking first num_workflows as targets for statistics
    3. Tracking and calculating continuous-mode metrics
    4. Force clearing schedulers after completion

    Args:
        experiment_module: The imported experiment module (e.g., test_dynamic_workflow)
        experiment_key: Experiment identifier (e.g., "04-ocr")
        num_workflows: Target number of workflows to track
        qps: Target QPS for A task submission
        gqps: Global QPS limit
        warmup: Warmup ratio
        seed: Random seed
        strategies: List of scheduling strategies

    Returns:
        Dict with experiment results
    """
    logger.info("=" * 80)
    logger.info(f"CONTINUOUS REQUEST MODE: Experiment {experiment_key}")
    logger.info("=" * 80)
    logger.info(f"Target workflows to track: {num_workflows}")
    logger.info(f"Total workflows to submit: {2 * num_workflows}")
    logger.info(f"Warmup ratio: {warmup}")
    logger.info("=" * 80)

    try:
        # Call the experiment's main function with continuous_mode=True
        # The experiment will internally handle:
        # - Generating 2*num_workflows
        # - Marking first num_workflows as targets
        # - Calculating makespan
        # - Force clearing schedulers
        # - Printing continuous-mode-specific statistics
        experiment_module.main(
            num_workflows=num_workflows,  # Pass original num_workflows, experiment will double it
            qps_a=qps,
            gqps=gqps,
            warmup_ratio=warmup,
            seed=seed,
            strategies=strategies,
            continuous_mode=True  # Enable continuous mode
        )

        logger.info(f"Experiment completed successfully")

    except Exception as e:
        logger.error(f"Experiment execution failed: {e}")
        raise

    logger.info("=" * 80)
    logger.info("CONTINUOUS MODE COMPLETED")
    logger.info("=" * 80)

    return {
        "mode": "continuous",
        "experiment": experiment_key,
        "target_workflows": num_workflows,
        "total_workflows": 2 * num_workflows
    }


def wrap_experiment_for_continuous(
    experiment_module,
    experiment_key: str
):
    """
    Create a wrapper function that runs experiment in continuous mode.

    Args:
        experiment_module: The imported experiment module
        experiment_key: Experiment identifier

    Returns:
        Wrapper function
    """
    def wrapper(num_workflows: int, qps: float, gqps: Optional[float],
                warmup: float, seed: int, strategies: List[str]):
        return run_experiment_continuous(
            experiment_module=experiment_module,
            experiment_key=experiment_key,
            num_workflows=num_workflows,
            qps=qps,
            gqps=gqps,
            warmup=warmup,
            seed=seed,
            strategies=strategies
        )

    return wrapper
