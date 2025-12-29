
import json
import statistics
import sys
import math

def load_metrics(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_metrics(metrics, name):
    tasks = metrics.get('task_executions', [])
    
    phases = {}
    for task in tasks:
        phase = task.get('phase')
        if phase not in phases:
            phases[phase] = {'wait': [], 'exec': [], 'latency': [], 'ratio': []}
        
        submit = task.get('submit_time')
        start = task.get('exec_start_time')
        complete = task.get('complete_time')
        
        if None in (submit, start, complete):
            continue
            
        wait = start - submit
        execution = complete - start
        latency = complete - submit
        ratio = latency / execution if execution > 0 else 0
        
        phases[phase]['wait'].append(wait)
        phases[phase]['exec'].append(execution)
        phases[phase]['latency'].append(latency)
        phases[phase]['ratio'].append(ratio)

    print(f"--- Analysis for {name} ---")
    
    # helper for printing stats
    def print_stat(label, data):
        if not data:
            print(f"  {label}: N/A")
            return
        avg = statistics.mean(data)
        # simplistic percentile
        sorted_data = sorted(data)
        idx = int(len(data) * 0.95)
        p95 = sorted_data[idx] if idx < len(data) else sorted_data[-1]
        print(f"  {label}: Avg={avg:.4f}, P95={p95:.4f}")

    all_start = min(t['submit_time'] for t in tasks if t.get('submit_time') is not None)
    all_end = max(t['complete_time'] for t in tasks if t.get('complete_time') is not None)
    total_time = all_end - all_start
    print(f"Total Makespan: {total_time:.4f}s")

    for p in sorted(phases.keys()):
        print(f"Phase {p}:")
        count = len(phases[p]['wait'])
        print(f"  Count: {count}")
        print_stat("Wait Time", phases[p]['wait'])
        print_stat("Exec Time", phases[p]['exec'])
        print_stat("Latency  ", phases[p]['latency'])
        print_stat("SLO Ratio", phases[p]['ratio'])
        
        # Calculate violation rates for a few thresholds
        for thresh in [2.0, 3.0, 5.0]:
            violation_count = sum(1 for r in phases[p]['ratio'] if r > thresh)
            rate = (violation_count / count) * 100
            print(f"  SLO Violation (>{thresh}): {rate:.2f}%")

if __name__ == "__main__":
    baseline_path = "/home/yanweiye/Projects/swarmpilot-refresh/experiments/13.workflow_benchmark/output_experiment/baseline/metrics.json"
    recovery_path = "/home/yanweiye/Projects/swarmpilot-refresh/experiments/13.workflow_benchmark/output_experiment/recovery/metrics.json"
    
    try:
        baseline = load_metrics(baseline_path)
        analyze_metrics(baseline, "BASELINE")
    except Exception as e:
        print(f"Error loading baseline: {e}")

    print("\n")

    try:
        recovery = load_metrics(recovery_path)
        analyze_metrics(recovery, "RECOVERY")
    except Exception as e:
        print(f"Error loading recovery: {e}")
