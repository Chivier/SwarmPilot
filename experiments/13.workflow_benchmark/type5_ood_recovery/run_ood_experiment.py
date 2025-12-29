#!/usr/bin/env python3
"""
All-in-One OOD Recovery Experiment Runner

This script runs both Recovery and Baseline simulations, then performs
SLO violation analysis automatically.

Usage:
    # Run with default settings
    python run_ood_experiment.py

    # Run with custom parameters (same as standalone_sim.py)
    python run_ood_experiment.py \
        --num-instances 512 \
        --num-tasks 10000 \
        --phase1-count 500 \
        --phase1-qps 200.0 \
        --phase23-qps 150.0 \
        --phase23-distribution weighted_bimodal \
        --runtime-scale 0.05 \
        --phase23-bimodal-scale 3.0 \
        --phase23-small-peak-ratio 0.1 \
        --phase2-transition-count 300 \
        --seed 42 \
        --output-dir output_experiment

    # Skip SLO analysis
    python run_ood_experiment.py --skip-slo-analysis

    # Custom SLO thresholds
    python run_ood_experiment.py \
        --slo-min-threshold 1.0 \
        --slo-max-threshold 5.0 \
        --slo-step 0.2
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run OOD Recovery experiment (Recovery + Baseline + SLO Analysis)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # =========================================================================
    # standalone_sim.py arguments (copied for compatibility)
    # =========================================================================

    # Predictor settings
    parser.add_argument(
        "--predictor-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Predictor service URL (default: http://127.0.0.1:8000)",
    )

    # Instance settings
    parser.add_argument(
        "--num-instances",
        type=int,
        default=48,
        help="Number of simulated instances (default: 48)",
    )

    # Task settings
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=500,
        help="Number of tasks to simulate (default: 500)",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=2.0,
        help="Default task submission rate (queries per second, default: 2.0)",
    )
    parser.add_argument(
        "--phase1-qps",
        type=float,
        default=None,
        help="Phase 1 QPS (default: same as --qps)",
    )
    parser.add_argument(
        "--phase23-qps",
        type=float,
        default=None,
        help="Phase 2/3 QPS (default: same as --qps)",
    )

    # Scaling
    parser.add_argument(
        "--runtime-scale",
        type=float,
        default=1.0,
        help="Global scaling factor for task runtime (default: 1.0)",
    )

    # Phase settings
    parser.add_argument(
        "--phase1-count",
        type=int,
        default=100,
        help="Number of Phase 1 tasks (default: 100)",
    )
    parser.add_argument(
        "--phase2-transition-count",
        type=int,
        default=20,
        help="Phase 2 completions before transition (default: 20)",
    )
    parser.add_argument(
        "--phase23-distribution",
        type=str,
        default="weighted_bimodal",
        choices=["normal", "uniform", "peak_dependent", "four_peak", "weighted_bimodal"],
        help="Distribution for Phase 2/3 runtimes (default: weighted_bimodal)",
    )
    parser.add_argument(
        "--phase23-bimodal-scale",
        type=float,
        default=2.0,
        help="Scale factor for Phase 2/3 weighted bimodal samples (default: 2.0)",
    )
    parser.add_argument(
        "--phase23-small-peak-ratio",
        type=float,
        default=0.2,
        help="Ratio of small peak samples in Phase 2/3 (default: 0.2)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_experiment",
        help="Base output directory for results (default: output_experiment)",
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    # =========================================================================
    # SLO Analysis arguments
    # =========================================================================

    parser.add_argument(
        "--skip-slo-analysis",
        action="store_true",
        help="Skip SLO violation analysis after simulation",
    )
    parser.add_argument(
        "--slo-min-threshold",
        type=float,
        default=1.0,
        help="Minimum SLO ratio threshold (default: 1.0)",
    )
    parser.add_argument(
        "--slo-max-threshold",
        type=float,
        default=10.0,
        help="Maximum SLO ratio threshold (default: 10.0)",
    )
    parser.add_argument(
        "--slo-step",
        type=float,
        default=0.5,
        help="Step size for SLO threshold sweep (default: 0.5)",
    )

    return parser.parse_args()


def build_sim_command(args, is_recovery: bool) -> list:
    """Build command for standalone_sim.py."""
    script_dir = Path(__file__).parent
    sim_script = script_dir / "standalone_sim.py"

    # Base output directory
    base_output = Path(args.output_dir)
    if is_recovery:
        output_dir = base_output
    else:
        output_dir = base_output

    cmd = [
        sys.executable,
        str(sim_script),
        "--predictor-url", args.predictor_url,
        "--num-instances", str(args.num_instances),
        "--num-tasks", str(args.num_tasks),
        "--qps", str(args.qps),
        "--runtime-scale", str(args.runtime_scale),
        "--phase1-count", str(args.phase1_count),
        "--phase2-transition-count", str(args.phase2_transition_count),
        "--phase23-distribution", args.phase23_distribution,
        "--phase23-bimodal-scale", str(args.phase23_bimodal_scale),
        "--phase23-small-peak-ratio", str(args.phase23_small_peak_ratio),
        "--seed", str(args.seed),
        "--output-dir", str(output_dir),
    ]

    # Optional arguments
    if args.phase1_qps is not None:
        cmd.extend(["--phase1-qps", str(args.phase1_qps)])
    if args.phase23_qps is not None:
        cmd.extend(["--phase23-qps", str(args.phase23_qps)])
    if args.verbose:
        cmd.append("--verbose")

    # Baseline mode
    if not is_recovery:
        cmd.append("--no-recovery")

    return cmd


def run_simulation(args, is_recovery: bool) -> bool:
    """Run a single simulation."""
    mode = "Recovery" if is_recovery else "Baseline"
    print(f"\n{'='*60}")
    print(f"Running {mode} Simulation")
    print(f"{'='*60}\n")

    cmd = build_sim_command(args, is_recovery)

    if args.verbose:
        print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n{mode} simulation completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError: {mode} simulation failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\nError: Could not find standalone_sim.py")
        return False


def run_slo_analysis(args) -> bool:
    """Run SLO violation analysis."""
    print(f"\n{'='*60}")
    print("Running SLO Violation Analysis")
    print(f"{'='*60}\n")

    script_dir = Path(__file__).parent
    slo_script = script_dir / "plot_slo_violation.py"

    base_output = Path(args.output_dir)
    baseline_metrics = base_output / "baseline" / "metrics.json"
    recovery_metrics = base_output / "recovery" / "metrics.json"
    slo_output_dir = base_output / "slo_analysis"

    # Check if metrics files exist
    if not baseline_metrics.exists():
        print(f"Error: Baseline metrics not found: {baseline_metrics}")
        return False
    if not recovery_metrics.exists():
        print(f"Error: Recovery metrics not found: {recovery_metrics}")
        return False

    cmd = [
        sys.executable,
        str(slo_script),
        "--baseline", str(baseline_metrics),
        "--recovery", str(recovery_metrics),
        "--output-dir", str(slo_output_dir),
        "--min-threshold", str(args.slo_min_threshold),
        "--max-threshold", str(args.slo_max_threshold),
        "--step", str(args.slo_step),
    ]

    if args.verbose:
        print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\nSLO analysis completed. Results saved to: {slo_output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError: SLO analysis failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\nError: Could not find plot_slo_violation.py")
        return False


def print_summary(args):
    """Print experiment summary."""
    base_output = Path(args.output_dir)

    print(f"\n{'='*60}")
    print("Experiment Complete!")
    print(f"{'='*60}\n")

    print("Output Files:")
    print(f"  Recovery:  {base_output / 'recovery' / 'metrics.json'}")
    print(f"  Baseline:  {base_output / 'baseline' / 'metrics.json'}")

    if not args.skip_slo_analysis:
        slo_dir = base_output / "slo_analysis"
        print(f"  SLO Plots: {slo_dir}/slo_ratio_*.png")

    print(f"\nVisualization Files:")
    print(f"  Recovery Gantt:      {base_output / 'recovery' / 'gantt.png'}")
    print(f"  Baseline Gantt:      {base_output / 'baseline' / 'gantt.png'}")
    print(f"  Recovery Throughput: {base_output / 'recovery' / 'throughput.png'}")
    print(f"  Baseline Throughput: {base_output / 'baseline' / 'throughput.png'}")


def main():
    """Main entry point."""
    args = parse_args()

    print(f"\n{'#'*60}")
    print("# OOD Recovery Experiment - All-in-One Runner")
    print(f"{'#'*60}")
    print(f"\nConfiguration:")
    print(f"  Instances:    {args.num_instances}")
    print(f"  Tasks:        {args.num_tasks}")
    print(f"  Phase 1 QPS:  {args.phase1_qps or args.qps}")
    print(f"  Phase 2/3 QPS: {args.phase23_qps or args.qps}")
    print(f"  Distribution: {args.phase23_distribution}")
    print(f"  Output:       {args.output_dir}")

    # Step 1: Run Recovery simulation
    if not run_simulation(args, is_recovery=True):
        print("\nAborting: Recovery simulation failed.")
        sys.exit(1)

    # Step 2: Run Baseline simulation
    if not run_simulation(args, is_recovery=False):
        print("\nAborting: Baseline simulation failed.")
        sys.exit(1)

    # Step 3: Run SLO analysis (optional)
    if not args.skip_slo_analysis:
        if not run_slo_analysis(args):
            print("\nWarning: SLO analysis failed, but simulations completed.")

    # Print summary
    print_summary(args)


if __name__ == "__main__":
    main()
