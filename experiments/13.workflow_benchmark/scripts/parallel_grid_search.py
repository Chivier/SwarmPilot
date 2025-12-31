
import os
import sys
import json
import csv
import statistics
import subprocess
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict

# Path setup - derive paths relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent  # experiments/13.workflow_benchmark
PROJECT_ROOT = BENCHMARK_DIR.parent.parent  # swarmpilot-refresh

# Add benchmark directory to path for imports
sys.path.insert(0, str(BENCHMARK_DIR))

# Import plotting function
from type5_ood_recovery.plot_throughput import plot_recovery_vs_baseline

# Path to standalone simulation script
STANDALONE_SIM = BENCHMARK_DIR / "type5_ood_recovery" / "standalone_sim.py"

# Constants
N_INSTANCES = 48
BASE_MEAN_RUNTIME = 18.63 # From optimize_128.py
N_TRIALS = 3
MAX_WORKERS = 10

# Search Grid
TARGET_LOADS = [0.95, 1.0, 1.02, 1.05, 1.10, 1.15]
RUNTIME_SCALES = [0.2, 0.5, 1.0]


def load_full_metrics(metrics_file: Path) -> dict:
    """Load complete metrics from JSON file for plotting."""
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def get_metrics_from_file(metrics_file: Path) -> dict:
    """Read metrics file and extract key metrics for comparison."""
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        summary = metrics['summary']
        trend = metrics.get('throughput_trend', [])

        # Get real-time throughput
        realtime_throughput = summary.get('realtime_throughput_avg', None)
        if realtime_throughput is None and trend:
            rt_values = [s.get('realtime_throughput', s.get('instantaneous_throughput', 0)) for s in trend]
            realtime_throughput = statistics.mean(rt_values) if rt_values else 0.0
        elif realtime_throughput is None:
            realtime_throughput = 0.0

        return {
            "total_completed": summary.get('total_completed', 0),
            "duration_s": summary.get('experiment_duration_s', 0),
            "realtime_throughput": realtime_throughput,
            "average_throughput": summary.get('average_throughput', 0),
            "throughput_trend": trend,  # For time-based comparison
        }
    except Exception as e:
        return {
            "total_completed": 0,
            "duration_s": 0,
            "realtime_throughput": 0.0,
            "average_throughput": 0.0,
            "throughput_trend": [],
        }


def generate_comparison_plot(rec_metrics_path: Path, base_metrics_path: Path, output_path: Path, title: str):
    """Generate comparison plot for a Recovery/Baseline pair."""
    try:
        rec_metrics = load_full_metrics(rec_metrics_path)
        base_metrics = load_full_metrics(base_metrics_path)

        if rec_metrics and base_metrics:
            plot_recovery_vs_baseline(
                recovery_metrics=rec_metrics,
                baseline_metrics=base_metrics,
                output_path=str(output_path),
                title=title,
                smoothing_window=5
            )
            return True
    except Exception as e:
        print(f"Warning: Failed to generate comparison plot: {e}")
    return False

def calculate_qps(target_load: float, runtime_scale: float) -> float:
    """Calculate required QPS to achieve target load."""
    mean_runtime = BASE_MEAN_RUNTIME * runtime_scale
    return (target_load * N_INSTANCES) / mean_runtime

def run_simulation(qps: float, runtime_scale: float, recovery: bool, output_subdir: str, seed: int,
                   phase23_distribution: str = "weighted_bimodal") -> dict:
    """Run simulation and return throughput metrics (real-time and average)."""
    cmd = [
        "uv", "run", str(STANDALONE_SIM),
        "--num-instances", str(N_INSTANCES),
        "--num-tasks", "8000",
        "--phase1-count", "1000",
        "--phase1-qps", "10.0",
        "--phase23-qps", str(qps),
        "--phase23-distribution", phase23_distribution,
        "--runtime-scale", str(runtime_scale),
        "--seed", str(seed),
        "--phase2-transition-count", "100",
        "--output-dir", output_subdir
    ]
    if not recovery:
        cmd.append("--no-recovery")

    try:
        # Run with capture_output=True to avoid stdout noise in parallel
        # Execute from project root to ensure correct path resolution
        subprocess.run(cmd, check=True, capture_output=True, cwd=str(PROJECT_ROOT))
        # Parse metrics - standalone_sim.py creates recovery/ or baseline/ subdirectory
        subdir = "baseline" if not recovery else "recovery"
        metrics_file = Path(output_subdir) / subdir / "metrics.json"
        return get_metrics_from_file(metrics_file)
    except Exception as e:
        # Don't print stack trace to avoid noise
        return {
            "total_completed": 0,
            "duration_s": 0,
            "realtime_throughput": 0.0,
            "average_throughput": 0.0,
            "throughput_trend": [],
        }

def get_completed_at_time(trend: list, target_time: float) -> int:
    """
    Get number of tasks completed at a specific time point.
    Uses linear interpolation between trend data points.
    """
    if not trend:
        return 0

    # Find the data points bracketing the target time
    for i, point in enumerate(trend):
        if point["elapsed_s"] >= target_time:
            if i == 0:
                return point["total_completed"]
            # Linear interpolation
            prev = trend[i - 1]
            t0, c0 = prev["elapsed_s"], prev["total_completed"]
            t1, c1 = point["elapsed_s"], point["total_completed"]
            if t1 == t0:
                return c1
            ratio = (target_time - t0) / (t1 - t0)
            return int(c0 + ratio * (c1 - c0))

    # Target time is beyond the last data point
    return trend[-1]["total_completed"] if trend else 0


def analyze_existing_results(output_base: Path) -> dict:
    """Analyze existing results from disk without running simulations."""
    print("Analyzing existing results...")
    results_store = {}

    for scale in RUNTIME_SCALES:
        for load in TARGET_LOADS:
            for i in range(N_TRIALS):
                # Recovery - check both old path (metrics.json) and new path (recovery/metrics.json)
                rec_dir = output_base / f"S{scale}_L{load}_T{i}_rec"
                rec_metrics_file = rec_dir / "recovery" / "metrics.json"
                if not rec_metrics_file.exists():
                    rec_metrics_file = rec_dir / "metrics.json"  # fallback to old path
                if rec_metrics_file.exists():
                    metrics = get_metrics_from_file(rec_metrics_file)
                    if scale not in results_store: results_store[scale] = {}
                    if load not in results_store[scale]: results_store[scale][load] = {"rec": {}, "base": {}}
                    results_store[scale][load]["rec"][i] = metrics

                # Baseline - check both old path (metrics.json) and new path (baseline/metrics.json)
                base_dir = output_base / f"S{scale}_L{load}_T{i}_base"
                base_metrics_file = base_dir / "baseline" / "metrics.json"
                if not base_metrics_file.exists():
                    base_metrics_file = base_dir / "metrics.json"  # fallback to old path
                if base_metrics_file.exists():
                    metrics = get_metrics_from_file(base_metrics_file)
                    if scale not in results_store: results_store[scale] = {}
                    if load not in results_store[scale]: results_store[scale][load] = {"rec": {}, "base": {}}
                    results_store[scale][load]["base"][i] = metrics

    return results_store


def run_all_simulations(output_base: Path, phase23_distribution: str = "weighted_bimodal") -> dict:
    """Run all simulations in parallel and return results."""
    # Generate all tasks
    tasks = []
    configs = []

    for scale in RUNTIME_SCALES:
        for load in TARGET_LOADS:
            qps = calculate_qps(load, scale)
            configs.append((scale, load, qps))

            for i in range(N_TRIALS):
                seed = 42 + i
                # Recovery Task
                rec_dir = str(output_base / f"S{scale}_L{load}_T{i}_rec")
                tasks.append({
                    "func": run_simulation,
                    "args": (qps, scale, True, rec_dir, seed, phase23_distribution),
                    "meta": {"scale": scale, "load": load, "type": "rec", "trial": i}
                })
                # Baseline Task
                base_dir = str(output_base / f"S{scale}_L{load}_T{i}_base")
                tasks.append({
                    "func": run_simulation,
                    "args": (qps, scale, False, base_dir, seed, phase23_distribution),
                    "meta": {"scale": scale, "load": load, "type": "base", "trial": i}
                })

    print(f"Total simulations to run: {len(tasks)}")

    # Results storage
    results_store = {}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_meta = {
            executor.submit(t["func"], *t["args"]): t["meta"]
            for t in tasks
        }

        completed_count = 0
        for future in as_completed(future_to_meta):
            meta = future_to_meta[future]
            scale, load, type_, trial = meta["scale"], meta["load"], meta["type"], meta["trial"]

            try:
                metrics = future.result()
            except Exception:
                metrics = {"realtime_throughput": 0.0, "average_throughput": 0.0}

            # Store result
            if scale not in results_store: results_store[scale] = {}
            if load not in results_store[scale]: results_store[scale][load] = {"rec": {}, "base": {}}
            results_store[scale][load][type_][trial] = metrics

            completed_count += 1
            if completed_count % 10 == 0:
                print(f"Progress: {completed_count}/{len(tasks)} ({completed_count/len(tasks)*100:.1f}%)")

    return results_store


def main():
    parser = argparse.ArgumentParser(description="Parallel Grid Search for OOD Recovery")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze existing results without running simulations")
    parser.add_argument("--output-dir", type=str, default=str(SCRIPT_DIR / "grid_search_results"),
                        help="Output directory for results")
    parser.add_argument("--phase23-distribution", type=str, default="weighted_bimodal",
                        choices=["normal", "uniform", "peak_dependent", "four_peak", "weighted_bimodal"],
                        help="Distribution for Phase 2/3 runtimes (default: weighted_bimodal)")
    args = parser.parse_args()

    print(f"Starting Parallel Multi-Variable Grid Search (Workers={MAX_WORKERS})...")
    print(f"Phase 2/3 Distribution: {args.phase23_distribution}")
    start_time = time.time()

    output_base = Path(args.output_dir)
    output_base.mkdir(exist_ok=True)

    # Get results (either from existing files or by running simulations)
    if args.analyze_only:
        results_store = analyze_existing_results(output_base)
    else:
        results_store = run_all_simulations(output_base, args.phase23_distribution)

    # Limit float precision for neat output
    def fmt(f): return f"{f:.2f}"

    # Analyze Results
    final_output = []
    print("\n--- Summary Results ---")

    for scale in RUNTIME_SCALES:
        for load in TARGET_LOADS:
            if scale not in results_store or load not in results_store[scale]:
                continue

            data = results_store[scale][load]
            rec_metrics = [data["rec"][t] for t in range(N_TRIALS) if t in data["rec"]]
            base_metrics = [data["base"][t] for t in range(N_TRIALS) if t in data["base"]]

            if not rec_metrics or not base_metrics:
                continue

            # Calculate average throughput (total_completed / duration_s) for each trial
            rec_avg_throughput_list = []
            base_avg_throughput_list = []

            for m in rec_metrics:
                if m["duration_s"] > 0:
                    avg_tp = m["total_completed"] / m["duration_s"]
                else:
                    avg_tp = m["average_throughput"]
                rec_avg_throughput_list.append(avg_tp)

            for m in base_metrics:
                if m["duration_s"] > 0:
                    avg_tp = m["total_completed"] / m["duration_s"]
                else:
                    avg_tp = m["average_throughput"]
                base_avg_throughput_list.append(avg_tp)

            # Calculate mean across trials
            avg_rec_throughput = statistics.mean(rec_avg_throughput_list) if rec_avg_throughput_list else 0
            avg_base_throughput = statistics.mean(base_avg_throughput_list) if base_avg_throughput_list else 0
            throughput_diff = avg_rec_throughput - avg_base_throughput
            pct_diff = (throughput_diff / avg_base_throughput * 100) if avg_base_throughput > 0 else 0

            # Also calculate duration stats
            rec_durations = [m["duration_s"] for m in rec_metrics]
            base_durations = [m["duration_s"] for m in base_metrics]
            avg_rec_duration = statistics.mean(rec_durations) if rec_durations else 0
            avg_base_duration = statistics.mean(base_durations) if base_durations else 0

            qps = calculate_qps(load, scale)

            final_output.append({
                "Runtime_Scale": scale,
                "Target_Load": load,
                "QPS": qps,
                "Avg_Rec_Throughput": avg_rec_throughput,
                "Avg_Base_Throughput": avg_base_throughput,
                "Throughput_Diff": throughput_diff,
                "Throughput_Pct_Diff": pct_diff,
                "Avg_Rec_Duration": avg_rec_duration,
                "Avg_Base_Duration": avg_base_duration,
            })

            print(f"Scale={scale}, Load={load} (QPS={qps:.2f}) -> Improvement={pct_diff:.2f}% "
                  f"(Rec={avg_rec_throughput:.2f}, Base={avg_base_throughput:.2f})")

    # Save to CSV
    csv_path = output_base / "summary_parallel.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ["Runtime_Scale", "Target_Load", "QPS",
                      "Avg_Rec_Throughput", "Avg_Base_Throughput", "Throughput_Diff", "Throughput_Pct_Diff",
                      "Avg_Rec_Duration", "Avg_Base_Duration"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_output)

    # Generate comparison plots for each configuration
    print("\n--- Generating Comparison Plots ---")
    plot_count = 0
    for scale in RUNTIME_SCALES:
        for load in TARGET_LOADS:
            # Generate plot for trial 0 (representative trial)
            rec_dir = output_base / f"S{scale}_L{load}_T0_rec"
            base_dir = output_base / f"S{scale}_L{load}_T0_base"

            # Check both new and old path structures
            rec_metrics_file = rec_dir / "recovery" / "metrics.json"
            if not rec_metrics_file.exists():
                rec_metrics_file = rec_dir / "metrics.json"

            base_metrics_file = base_dir / "baseline" / "metrics.json"
            if not base_metrics_file.exists():
                base_metrics_file = base_dir / "metrics.json"

            if rec_metrics_file.exists() and base_metrics_file.exists():
                qps = calculate_qps(load, scale)
                plot_path = output_base / f"comparison_S{scale}_L{load}.png"
                title = f"Scale={scale}, Load={load} (QPS={qps:.2f})"

                if generate_comparison_plot(rec_metrics_file, base_metrics_file, plot_path, title):
                    plot_count += 1

    print(f"Generated {plot_count} comparison plots")

    # Find Best (based on average throughput improvement)
    if not final_output:
        print("\nNo results to analyze!")
        return
    best = max(final_output, key=lambda x: x['Throughput_Pct_Diff'])
    duration = time.time() - start_time

    print(f"\nSearch Complete in {duration:.2f}s")
    print(f"Results saved to {csv_path}")
    print("\n=== Best Configuration ===")
    print(f"Runtime Scale: {best['Runtime_Scale']}")
    print(f"Target Load: {best['Target_Load']}")
    print(f"QPS: {best['QPS']:.4f}")
    print(f"Throughput Improvement: {best['Throughput_Pct_Diff']:.2f}% ({best['Throughput_Diff']:+.2f} tasks/s)")

if __name__ == "__main__":
    main()
