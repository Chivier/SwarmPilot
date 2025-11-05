#!/usr/bin/env python3
"""
Unified Multi-Model Workflow Experiment Launcher

This is the unified entry point for experiments 04-07, allowing users to run
specific experiments or all experiments sequentially with a single command.

Experiment Mapping:
- 04-ocr:      Experiment 04 - 1→n parallel workflow (4 threads)
- 05-t2vid:    Experiment 05 - 1→n sequential workflow (4 threads)
- 06-dr_simple: Experiment 06 - 1→n→1 with merge (6 threads)
- 07-dr:       Experiment 07 - 1→n→n→1 with B1/B2 split (7 threads)

Usage Examples:
    # Run a specific experiment
    python unified_workflow.py --experiment 04-ocr --num-workflows 100 --qps 8.0
    python unified_workflow.py --experiment 05-t2vid --num-workflows 50

    # Run all experiments sequentially
    python unified_workflow.py --experiment all --num-workflows 100

    # Run with custom strategies
    python unified_workflow.py --experiment 06-dr_simple --strategies min_time round_robin
"""

import sys
import os
import argparse
import importlib.util
import logging
from pathlib import Path
from typing import List, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("UnifiedWorkflow")

# Experiment configuration
EXPERIMENT_MAP = {
    "04-ocr": {
        "name": "04-ocr (1→n parallel)",
        "dir": "04.multi_model_workflow_dynamic",
        "module": "test_dynamic_workflow",
        "description": "1-to-n parallel B task submission (4 threads)",
        "threads": 4
    },
    "05-t2vid": {
        "name": "05-t2vid (1→n sequential)",
        "dir": "05.multi_model_workflow_dynamic_parallel",
        "module": "test_dynamic_workflow",
        "description": "1-to-n sequential B task submission (4 threads)",
        "threads": 4
    },
    "06-dr_simple": {
        "name": "06-dr_simple (1→n→1 merge)",
        "dir": "06.multi_model_workflow_dynamic_merge",
        "module": "test_dynamic_workflow",
        "description": "1-to-n-to-1 with merge task (6 threads)",
        "threads": 6
    },
    "07-dr": {
        "name": "07-dr (1→n→n→1 B1/B2)",
        "dir": "07.multi_model_workflow_dynamic_merge_2",
        "module": "test_dynamic_workflow",
        "description": "1-to-n-to-n-to-1 with B1/B2 split and merge (7 threads)",
        "threads": 7
    }
}


def get_experiment_path(experiment_key: str) -> Path:
    """
    Get the absolute path to an experiment directory.

    Args:
        experiment_key: Experiment key (e.g., "04-ocr")

    Returns:
        Path object pointing to the experiment directory
    """
    # Get the experiments directory (parent of current script's directory)
    current_dir = Path(__file__).parent
    experiments_dir = current_dir.parent

    experiment_config = EXPERIMENT_MAP[experiment_key]
    experiment_dir = experiments_dir / experiment_config["dir"]

    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    return experiment_dir


def import_experiment_module(experiment_key: str):
    """
    Dynamically import the main module for a specific experiment.

    Args:
        experiment_key: Experiment key (e.g., "04-ocr")

    Returns:
        The imported module
    """
    experiment_config = EXPERIMENT_MAP[experiment_key]
    experiment_dir = get_experiment_path(experiment_key)
    module_path = experiment_dir / f"{experiment_config['module']}.py"

    if not module_path.exists():
        raise FileNotFoundError(f"Experiment module not found: {module_path}")

    # Import the module dynamically
    spec = importlib.util.spec_from_file_location(
        f"experiment_{experiment_key.replace('-', '_')}",
        module_path
    )
    module = importlib.util.module_from_spec(spec)

    # Add the experiment directory to sys.path so imports work
    sys.path.insert(0, str(experiment_dir))

    spec.loader.exec_module(module)

    return module


def run_experiment(
    experiment_key: str,
    num_workflows: int,
    qps: float,
    gqps: Optional[float],
    warmup: float,
    seed: int,
    strategies: List[str],
    continuous: bool = False
):
    """
    Run a specific experiment.

    Args:
        experiment_key: Experiment key (e.g., "04-ocr")
        num_workflows: Number of workflows per strategy
        qps: Target QPS for A task submission
        gqps: Global QPS limit for both A and B tasks (optional)
        warmup: Warmup task ratio
        seed: Random seed
        strategies: List of scheduling strategies to test
        continuous: Enable continuous request mode
    """
    experiment_config = EXPERIMENT_MAP[experiment_key]

    logger.info("=" * 80)
    logger.info(f"Starting Experiment: {experiment_config['name']}")
    logger.info(f"Description: {experiment_config['description']}")
    logger.info(f"Threads: {experiment_config['threads']}")
    if continuous:
        logger.info("Mode: CONTINUOUS REQUEST (submits 2*num_workflows, tracks first num_workflows)")
    logger.info("=" * 80)

    try:
        # Import the experiment module
        module = import_experiment_module(experiment_key)

        # Check if the module has a main function
        if not hasattr(module, 'main'):
            raise AttributeError(f"Experiment module does not have a 'main' function")

        if continuous:
            # Use continuous wrapper
            from continuous_wrapper import run_experiment_continuous
            run_experiment_continuous(
                experiment_module=module,
                experiment_key=experiment_key,
                num_workflows=num_workflows,
                qps=qps,
                gqps=gqps,
                warmup=warmup,
                seed=seed,
                strategies=strategies
            )
        else:
            # Call the experiment's main function with provided arguments
            module.main(
                num_workflows=num_workflows,
                qps_a=qps,
                gqps=gqps,
                warmup_ratio=warmup,
                seed=seed,
                strategies=strategies
            )

        logger.info(f"Experiment {experiment_config['name']} completed successfully")

    except Exception as e:
        logger.error(f"Error running experiment {experiment_config['name']}: {e}")
        raise


def run_all_experiments(
    num_workflows: int,
    qps: float,
    gqps: Optional[float],
    warmup: float,
    seed: int,
    strategies: List[str],
    continuous: bool = False
):
    """
    Run all experiments sequentially.

    Args:
        num_workflows: Number of workflows per strategy
        qps: Target QPS for A task submission
        gqps: Global QPS limit for both A and B tasks (optional)
        warmup: Warmup task ratio
        seed: Random seed
        strategies: List of scheduling strategies to test
        continuous: Enable continuous request mode
    """
    logger.info("=" * 80)
    logger.info("Running ALL Experiments (04-ocr, 05-t2vid, 06-dr_simple, 07-dr)")
    if continuous:
        logger.info("Mode: CONTINUOUS REQUEST")
    logger.info("=" * 80)

    experiment_keys = ["04-ocr", "05-t2vid", "06-dr_simple", "07-dr"]

    for i, experiment_key in enumerate(experiment_keys, 1):
        logger.info(f"\n[{i}/{len(experiment_keys)}] Running experiment: {experiment_key}")

        try:
            run_experiment(
                experiment_key=experiment_key,
                num_workflows=num_workflows,
                qps=qps,
                gqps=gqps,
                warmup=warmup,
                seed=seed,
                strategies=strategies,
                continuous=continuous
            )

            # Brief pause between experiments
            if i < len(experiment_keys):
                logger.info("Pausing 5 seconds before next experiment...")
                time.sleep(5.0)

        except Exception as e:
            logger.error(f"Failed to run experiment {experiment_key}: {e}")
            logger.info("Continuing with next experiment...")

    logger.info("\n" + "=" * 80)
    logger.info("All experiments completed!")
    logger.info("=" * 80)


def main():
    """Main entry point for unified workflow launcher."""
    parser = argparse.ArgumentParser(
        description="Unified Multi-Model Workflow Experiment Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiment Options:
  04-ocr       : 1→n parallel B task submission (4 threads)
  05-t2vid     : 1→n sequential B task submission (4 threads)
  06-dr_simple : 1→n→1 with merge task (6 threads)
  07-dr        : 1→n→n→1 with B1/B2 split and merge (7 threads)
  all          : Run all experiments sequentially

Examples:
  # Run experiment 04-ocr
  python unified_workflow.py --experiment 04-ocr --num-workflows 100

  # Run experiment 06-dr_simple with specific strategies
  python unified_workflow.py --experiment 06-dr_simple --strategies min_time round_robin

  # Run all experiments
  python unified_workflow.py --experiment all --num-workflows 50 --qps 10.0
        """
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=list(EXPERIMENT_MAP.keys()) + ["all"],
        help="Experiment to run (04-ocr, 05-t2vid, 06-dr_simple, 07-dr, or all)"
    )

    parser.add_argument(
        "--num-workflows",
        type=int,
        default=100,
        help="Number of workflows to generate and execute per strategy (default: 100)"
    )

    parser.add_argument(
        "--qps",
        type=float,
        default=8.0,
        help="Target queries per second (QPS) for A task submission (default: 8.0)"
    )

    parser.add_argument(
        "--gqps",
        type=float,
        default=None,
        help="Global QPS limit for both A and B task submissions (overrides --qps if set)"
    )

    parser.add_argument(
        "--warmup",
        type=float,
        default=0.0,
        help="Warmup task ratio (0.0-1.0). E.g., 0.2 means 20%% warmup tasks (default: 0.0)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["round_robin", "probabilistic", "min_time"],
        choices=["min_time", "round_robin", "probabilistic"],
        help="Scheduling strategies to test (default: all three)"
    )

    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Enable continuous request mode (submits 2*num_workflows, tracks first num_workflows, force clears schedulers)"
    )

    args = parser.parse_args()

    # Print configuration
    logger.info("Unified Multi-Model Workflow Experiment Launcher")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Experiment:    {args.experiment}")
    logger.info(f"  Workflows:     {args.num_workflows}")
    logger.info(f"  QPS:           {args.qps}")
    if args.gqps is not None:
        logger.info(f"  Global QPS:    {args.gqps}")
    logger.info(f"  Warmup Ratio:  {args.warmup}")
    logger.info(f"  Random Seed:   {args.seed}")
    logger.info(f"  Strategies:    {', '.join(args.strategies)}")
    if args.continuous:
        logger.info(f"  Mode:          CONTINUOUS (2x workflows, track first half)")
    logger.info("=" * 80)

    # Run the selected experiment(s)
    try:
        if args.experiment == "all":
            run_all_experiments(
                num_workflows=args.num_workflows,
                qps=args.qps,
                gqps=args.gqps,
                warmup=args.warmup,
                seed=args.seed,
                strategies=args.strategies,
                continuous=args.continuous
            )
        else:
            run_experiment(
                experiment_key=args.experiment,
                num_workflows=args.num_workflows,
                qps=args.qps,
                gqps=args.gqps,
                warmup=args.warmup,
                seed=args.seed,
                strategies=args.strategies,
                continuous=args.continuous
            )

    except KeyboardInterrupt:
        logger.warning("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nExperiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
