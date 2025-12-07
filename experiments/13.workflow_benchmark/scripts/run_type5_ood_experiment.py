#!/usr/bin/env python3
"""
Run Type5 OOD Recovery Experiment with configurable parameters.

This script provides a convenient way to run OOD recovery experiments
in both recovery mode and baseline mode for comparison.

Usage:
    # Run both modes for comparison
    python run_type5_ood_experiment.py --num-tasks 100 --qps 2.0

    # Run only recovery mode
    python run_type5_ood_experiment.py --num-tasks 100 --mode recovery

    # Run only baseline mode
    python run_type5_ood_experiment.py --num-tasks 100 --mode baseline
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_experiment(
    mode: str,
    num_tasks: int,
    qps: float,
    phase1_qps: float,
    phase23_qps: float,
    runtime_scale: float,
    duration: int,
    scheduler_url: str,
    predictor_url: str,
    model_id: str,
    output_dir: str,
    seed: int,
    phase2_transition_count: int = 10,
    phase1_count: int = 100,
    transition_on_submit: bool = True,
    phase23_distribution: str = "four_peak",
    skip_service_check: bool = False,
) -> dict:
    """
    Run a single OOD experiment.

    Args:
        mode: 'recovery' or 'baseline'
        num_tasks: Number of tasks to submit
        qps: Default queries per second
        phase1_qps: Phase 1 QPS (None = use qps)
        phase23_qps: Phase 2/3 QPS (None = use qps)
        runtime_scale: Global scaling factor for task runtime
        duration: Maximum duration in seconds
        scheduler_url: Scheduler URL
        predictor_url: Predictor URL
        model_id: Model ID for task submission
        output_dir: Output directory
        seed: Random seed for reproducibility
        phase2_transition_count: Number of Phase 2 tasks before Phase 3 transition
        phase1_count: Number of tasks in Phase 1
        transition_on_submit: If True, trigger transition on submission count
        phase23_distribution: Distribution mode for Phase 2/3 sleep times
        skip_service_check: If True, skip automatic service health checks

    Returns:
        dict with experiment results
    """
    # Build command
    cmd = [
        sys.executable, "-m", "type5_ood_recovery.simulation.test_ood_sim",
        "--num-tasks", str(num_tasks),
        "--qps", str(qps),
        "--runtime-scale", str(runtime_scale),
        "--duration", str(duration),
        "--scheduler-url", scheduler_url,
        "--predictor-url", predictor_url,
        "--model-id", model_id,
        "--output-dir", output_dir,
        "--seed", str(seed),
        "--phase2-transition-count", str(phase2_transition_count),
        "--phase1-count", str(phase1_count),
        "--phase23-distribution", phase23_distribution,
    ]
    
    # Add phase-specific QPS if provided
    if phase1_qps is not None:
        cmd.extend(["--phase1-qps", str(phase1_qps)])
    if phase23_qps is not None:
        cmd.extend(["--phase23-qps", str(phase23_qps)])

    # Add transition mode flag
    if not transition_on_submit:
        cmd.append("--transition-on-complete")

    if skip_service_check:
        cmd.append("--skip-service-check")

    if mode == "baseline":
        cmd.append("--no-recovery")

    print(f"\n{'=' * 60}")
    print(f"Running OOD Experiment - {mode.upper()} mode")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    # Run experiment
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time

    return {
        "mode": mode,
        "returncode": result.returncode,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Type5 OOD Recovery Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Experiment mode
    parser.add_argument(
        "--mode",
        choices=["recovery", "baseline", "both"],
        default="both",
        help="Experiment mode: recovery, baseline, or both (default: both)"
    )

    # Task parameters
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=100,
        help="Total number of tasks (default: 100)"
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=1.0,
        help="Target queries per second (default: 1.0)"
    )
    parser.add_argument(
        "--phase1-qps",
        type=float,
        default=None,
        help="Phase 1 QPS (default: same as --qps)"
    )
    parser.add_argument(
        "--phase23-qps",
        type=float,
        default=None,
        help="Phase 2/3 QPS (default: same as --qps)"
    )
    parser.add_argument(
        "--runtime-scale",
        type=float,
        default=1.0,
        help="Global scaling factor for task runtime (default: 1.0)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=600,
        help="Maximum experiment duration in seconds (default: 600)"
    )
    parser.add_argument(
        "--phase1-count",
        type=int,
        default=100,
        help="Number of tasks in Phase 1 (default: 100)"
    )

    # Phase transition configuration
    parser.add_argument(
        "--phase2-transition-count",
        type=int,
        default=10,
        help="Number of Phase 2 tasks before transitioning to Phase 3 (default: 10). "
             "Interpreted as completion count (default) or submission count with --transition-on-submit."
    )
    parser.add_argument(
        "--transition-on-submit",
        action="store_true",
        default=False,
        help="Trigger Phase 2→3 transition based on submission count. "
             "This avoids queue backlog but is less realistic."
    )
    parser.add_argument(
        "--transition-on-complete",
        action="store_true",
        default=True,
        help="Trigger Phase 2→3 transition based on completion count (default). "
             "Realistic: simulates OOD detection after tasks complete."
    )

    # Scheduler configuration
    parser.add_argument(
        "--scheduler-url",
        type=str,
        default="http://127.0.0.1:8100",
        help="Scheduler URL (default: http://127.0.0.1:8100)"
    )
    parser.add_argument(
        "--predictor-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Predictor URL (default: http://127.0.0.1:8000)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="sleep_model_a",
        help="Model ID for task submission (default: sleep_model_a)"
    )
    parser.add_argument(
        "--skip-service-check",
        action="store_true",
        help="Skip automatic service health checks (use when services are user-started)"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_ood",
        help="Output directory (default: output_ood)"
    )

    # Phase 2/3 distribution configuration
    parser.add_argument(
        "--phase23-distribution",
        type=str,
        default="four_peak",
        choices=["normal", "uniform", "peak_dependent", "four_peak"],
        help="Distribution mode for Phase 2/3 sleep times (default: four_peak)"
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42). "
             "IMPORTANT: Same seed ensures identical data for both modes in 'both' mode."
    )

    args = parser.parse_args()

    # Determine which modes to run
    if args.mode == "both":
        modes = ["baseline", "recovery"]  # Run baseline first, then recovery
    else:
        modes = [args.mode]

    # Print seed information for both mode
    if args.mode == "both":
        print(f"\n{'=' * 60}")
        print(f"Running BOTH modes with SAME seed={args.seed}")
        print(f"This ensures identical task data for fair comparison.")
        print(f"{'=' * 60}")
        print(f"\nData Consistency Guarantee:")
        print(f"  - Same base_sleep_time[i] for each task index")
        print(f"  - Same exp_runtime_base[i] for each task index")
        print(f"  - Same phase23_random_scale[i] for each task index")
        print(f"\nSleep Time Consistency:")
        print(f"  Phase 2/3 tasks in Recovery mode and Phase 2 tasks in Baseline mode")
        print(f"  will have IDENTICAL sleep_time values for the same task index.")
        print(f"  Formula: actual_sleep_time = base_sleep_time × phase23_random_scale")
        print(f"  (This formula is the same for both Phase 2 and Phase 3)")
        print(f"{'=' * 60}")

    # Run experiments
    # Determine transition mode: --transition-on-submit overrides default completion-based
    transition_on_submit = args.transition_on_submit

    results = []
    for mode in modes:
        result = run_experiment(
            mode=mode,
            num_tasks=args.num_tasks,
            qps=args.qps,
            phase1_qps=args.phase1_qps,
            phase23_qps=args.phase23_qps,
            runtime_scale=args.runtime_scale,
            duration=args.duration,
            scheduler_url=args.scheduler_url,
            predictor_url=args.predictor_url,
            model_id=args.model_id,
            output_dir=args.output_dir,
            seed=args.seed,
            phase2_transition_count=args.phase2_transition_count,
            transition_on_submit=transition_on_submit,
            phase1_count=args.phase1_count,
            phase23_distribution=args.phase23_distribution,
            skip_service_check=args.skip_service_check,
        )
        results.append(result)

        if result["returncode"] != 0:
            print(f"\nExperiment {mode} failed with return code {result['returncode']}")
        else:
            print(f"\nExperiment {mode} completed in {result['elapsed']:.1f}s")

    # Summary
    print(f"\n{'=' * 60}")
    print("Experiment Summary")
    print(f"{'=' * 60}")
    print(f"  Seed: {args.seed} (same for all experiments)")
    print(f"  Phase2 Transition Count: {args.phase2_transition_count}")
    for result in results:
        status = "SUCCESS" if result["returncode"] == 0 else "FAILED"
        print(f"  {result['mode'].upper()}: {status} ({result['elapsed']:.1f}s)")

    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - metrics_recovery.json (if recovery mode ran)")
    print(f"  - metrics_baseline.json (if baseline mode ran)")
    if args.mode == "both":
        print(f"\nData Consistency: Both experiments used seed={args.seed}")
        print(f"  → Same base_sleep_time, exp_runtime_base, phase23_random_scale")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
