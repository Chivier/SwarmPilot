
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def load_metrics(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_wait_times(metrics):
    """Extract wait times for Phase 2 and 3 (excluding warmup Phase 1)."""
    tasks = metrics.get('task_executions', [])
    wait_times = []
    completion_times = []
    
    for task in tasks:
        # Filter for OOD phases (2 and 3)
        if task.get('phase') not in [2, 3]:
            continue
            
        submit = task.get('submit_time')
        start = task.get('exec_start_time')
        complete = task.get('complete_time')
        
        if None in (submit, start, complete):
            continue
            
        wait_times.append(start - submit)
        completion_times.append(complete)
        
    return np.array(wait_times), np.sort(completion_times)

def plot_cdf(baseline_wait, recovery_wait, output_dir):
    plt.figure(figsize=(10, 6))
    
    # Baseline
    sorted_baseline = np.sort(baseline_wait)
    y_baseline = np.arange(1, len(sorted_baseline) + 1) / len(sorted_baseline)
    plt.plot(sorted_baseline, y_baseline, label='Baseline', color='#E94F37', linewidth=2)
    
    # Recovery
    sorted_recovery = np.sort(recovery_wait)
    y_recovery = np.arange(1, len(sorted_recovery) + 1) / len(sorted_recovery)
    plt.plot(sorted_recovery, y_recovery, label='Recovery', color='#2E86AB', linewidth=2)
    
    plt.xlabel('Wait Time (seconds)', fontsize=12)
    plt.ylabel('Cumulative Probability (CDF)', fontsize=12)
    plt.title('CDF of Wait Times (Phases 2 & 3)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Annotate P95
    p95_baseline = np.percentile(baseline_wait, 95)
    p95_recovery = np.percentile(recovery_wait, 95)
    
    plt.axvline(p95_baseline, color='#E94F37', linestyle=':', alpha=0.5)
    plt.axvline(p95_recovery, color='#2E86AB', linestyle=':', alpha=0.5)
    
    plt.text(p95_baseline + 2, 0.05, f'P95={p95_baseline:.1f}s', color='#E94F37')
    plt.text(p95_recovery - 15, 0.15, f'P95={p95_recovery:.1f}s', color='#2E86AB')

    output_path = os.path.join(output_dir, 'wait_time_cdf.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

def plot_box(baseline_wait, recovery_wait, output_dir):
    plt.figure(figsize=(8, 6))
    
    data = [baseline_wait, recovery_wait]
    labels = ['Baseline', 'Recovery']
    colors = ['#E94F37', '#2E86AB']
    
    box = plt.boxplot(data, labels=labels, patch_artist=True, showfliers=True)
    
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        
    plt.ylabel('Wait Time (seconds)', fontsize=12)
    plt.title('Wait Time Distribution (Phases 2 & 3)', fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    
    output_path = os.path.join(output_dir, 'wait_time_boxplot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

def plot_throughput(baseline_complete, recovery_complete, output_dir):
    plt.figure(figsize=(10, 6))
    
    # Normalize time to start at 0 for comparison phases
    # Assuming the experiment aligns roughly, but we want to show 'Time from Start of Experiment'
    # Since these are timestamps from the start, we can just plot them directly.
    
    # Note: completions are already sorted
    
    # Create x, y for step plot
    x_base = baseline_complete
    y_base = np.arange(1, len(x_base) + 1)
    
    x_rec = recovery_complete
    y_rec = np.arange(1, len(x_rec) + 1)
    
    plt.plot(x_base, y_base, label='Baseline', color='#E94F37', linewidth=2)
    plt.plot(x_rec, y_rec, label='Recovery', color='#2E86AB', linewidth=2)
    
    plt.xlabel('Experiment Time (seconds)', fontsize=12)
    plt.ylabel('Tasks Completed', fontsize=12)
    plt.title('Cumulative Task Completions', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Annotate total time
    max_time_base = x_base[-1]
    max_time_rec = x_rec[-1]
    
    plt.annotate(f'Total: {max_time_base:.1f}s', xy=(max_time_base, y_base[-1]), 
                 xytext=(max_time_base - 20, y_base[-1] - 500),
                 arrowprops=dict(arrowstyle='->', color='#E94F37'), color='#E94F37')
                 
    plt.annotate(f'Total: {max_time_rec:.1f}s', xy=(max_time_rec, y_rec[-1]), 
                 xytext=(max_time_rec - 20, y_rec[-1] + 500),
                 arrowprops=dict(arrowstyle='->', color='#2E86AB'), color='#2E86AB')

    output_path = os.path.join(output_dir, 'cumulative_completion.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

def main():
    base_dir = Path("/home/yanweiye/Projects/swarmpilot-refresh/experiments/13.workflow_benchmark/output_experiment")
    baseline_path = base_dir / "baseline/metrics.json"
    recovery_path = base_dir / "recovery/metrics.json"
    output_dir = base_dir / "advantage_analysis"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading metrics...")
    baseline = load_metrics(baseline_path)
    recovery = load_metrics(recovery_path)
    
    print("Extracting data...")
    base_wait, base_complete = extract_wait_times(baseline)
    rec_wait, rec_complete = extract_wait_times(recovery)
    
    print(f"Baseline tasks: {len(base_wait)}, Recovery tasks: {len(rec_wait)}")
    
    plot_cdf(base_wait, rec_wait, output_dir)
    plot_box(base_wait, rec_wait, output_dir)
    plot_throughput(base_complete, rec_complete, output_dir)
    
    print("Done!")

if __name__ == "__main__":
    main()
