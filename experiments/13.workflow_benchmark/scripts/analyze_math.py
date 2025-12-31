
import json
import statistics
import sys
import os
import random
import math

# Add project root to path
sys.path.append("/home/yanweiye/Projects/swarmpilot-refresh/experiments/13.workflow_benchmark")

from type5_ood_recovery.config import OODRecoveryConfig
from type5_ood_recovery.task_data import OODTaskData, TaskGenerator

def analyze():
    # 1. Phase 1 Base Mean
    base_mean = 68631.04 # From previous run
    print(f"Phase 1 Base Mean: {base_mean:.2f}ms")

    # 2. Simulate Phase 2/3 values
    config = OODRecoveryConfig()
    
    # Generate Phase 2 tasks
    print(f"\nPhase 2 Task Simulation (Wrong Prediction):")
    gen = TaskGenerator(seed=42)
    
    tasks = []
    print("Generating 20 tasks to check correlation...")
    for i in range(20):
        t = gen.generate_task(i)
        t.phase = 2
        t.calculate_times(config)
        tasks.append(t)
        print(f"T{i}: Actual={t.actual_sleep_time:.3f}s, Pred={t.exp_runtime_ms/1000:.3f}s, Diff={t.actual_sleep_time - t.exp_runtime_ms/1000:.3f}")

    # Calculate throughput impact
    # Capacity = 48 / Actual_Mean
    actuals = [t.actual_sleep_time for t in tasks]
    act_mean = statistics.mean(actuals)
    preds = [t.exp_runtime_ms/1000 for t in tasks]
    pred_mean = statistics.mean(preds)
    
    print(f"\nPhase 2 Stats:")
    print(f"Actual Mean: {act_mean:.3f}s")
    print(f"Predicted Mean: {pred_mean:.3f}s")
    
    # Basic Little's Law Calculation
    n_instances = 48
    qps = 20
    
    capacity_p1 = n_instances / (base_mean * config.phase1_scale / 1000)
    capacity_p2 = n_instances / act_mean
    
    print(f"\nTheoretical Capacity (instances={n_instances}):")
    print(f"Phase 1 (Mean={base_mean*config.phase1_scale/1000:.2f}s): {capacity_p1:.2f} QPS")
    print(f"Phase 2 (Mean={act_mean:.2f}s): {capacity_p2:.2f} QPS")
    print(f"Load (QPS={qps})")

if __name__ == "__main__":
    analyze()
