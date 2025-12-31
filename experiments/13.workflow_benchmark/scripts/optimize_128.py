
import statistics
import sys
import numpy as np

# Add project root to path
sys.path.append("/home/yanweiye/Projects/swarmpilot-refresh/experiments/13.workflow_benchmark")

from type5_ood_recovery.config import OODRecoveryConfig
from type5_ood_recovery.task_data import TaskGenerator

def optimize_128_instances():
    """
    Calculate optimal parameters for 128 instances to show MAX throughput degradation.
    Goal: Phase 2 (Wrong Prediction) << Phase 3 (Correct Prediction)
    Mechanism: 
    - Create extreme 'Head-of-Line Blocking' in Phase 2.
    - Many short predicted tasks assigned to one node, but one is actually long.
    - Blocking that node prevents processing of the short tasks.
    - Other nodes go idle (starvation) because they don't get tasks (load balancing failure).
    """
    
    # 1. Analyze "Four Peak" distribution properties
    config = OODRecoveryConfig(phase23_distribution="four_peak")
    gen = TaskGenerator(seed=42)
    
    tasks = []
    print("Generating 5000 tasks for statistical stability...")
    for i in range(5000):
        t = gen.generate_task(i)
        t.phase = 2
        t.calculate_times(config)
        tasks.append(t)

    actuals = [t.actual_sleep_time for t in tasks]
    preds = [t.exp_runtime_ms / 1000.0 for t in tasks]
    
    mean_act = statistics.mean(actuals)
    mean_pred = statistics.mean(preds)
    
    print(f"\n--- Distribution Stats (Phase 2/3) ---")
    print(f"Mean Actual: {mean_act:.2f}s")
    print(f"Mean Predicted: {mean_pred:.2f}s")
    
    # Analyze the inverse correlation / misleading nature
    # We want: Predicted SHORT -> Actual LONG (The "Blocker" task)
    # We want: Predicted LONG -> Actual SHORT (The "Ghost" capacity - scheduler avoids these)
    
    misleading_short = [t for t in tasks if t.exp_runtime_ms/1000 < 5.0 and t.actual_sleep_time > 15.0]
    misleading_long = [t for t in tasks if t.exp_runtime_ms/1000 > 10.0 and t.actual_sleep_time < 5.0]
    
    print(f"Blocking Tasks (Pred < 5s, Act > 15s): {len(misleading_short)/len(tasks)*100:.1f}%")
    print(f"Ghost Tasks (Pred > 10s, Act < 5s): {len(misleading_long)/len(tasks)*100:.1f}%")

    # 2. Capacity Calculation
    n_instances = 128
    capacity_ideal = n_instances / mean_act
    print(f"\n--- Capacity (128 Instances) ---")
    print(f"Ideal Capacity (Perfect Load Balancing): {capacity_ideal:.2f} QPS")
    
    # 3. Estimating "Broken" Capacity (Phase 2)
    # In Phase 2, scheduler sees "Predicted Mean" ~ 5s (assumed low).
    # It aggressively packs tasks. 
    # But effectively, if X% of nodes are blocked by "long" tasks that were predicted short,
    # the effective cluster size reduces.
    
    # Let's recommend a QPS that is:
    # 1. BELOW Ideal Capacity (so Phase 3 can clear queues)
    # 2. ABOVE Broken Capacity (so Phase 2 builds massive queues / fails)
    
    target_load_p = 0.85 # 85% of ideal capacity is usually safe for recovery
    rec_qps_p3 = capacity_ideal * target_load_p
    
    print(f"\n--- Recommendation ---")
    print(f"Phase 2/3 Recommended QPS: {rec_qps_p3:.2f}")
    
    # 4. Phase 1 - Keep it simple/stable
    # Phase 1 is scaled 0.1x.
    # Base Mean ~68s -> 6.8s.
    cap_p1 = n_instances / 6.86
    rec_qps_p1 = cap_p1 * 0.7 # 70% load
    print(f"Phase 1 Recommended QPS: {rec_qps_p1:.2f}")

if __name__ == "__main__":
    optimize_128_instances()
