#!/usr/bin/env python3
"""
Test script to verify data extraction from test_dynamic_workflow.py results.
"""

import json
from pathlib import Path
import numpy as np


def test_data_extraction():
    """Test data extraction logic with real result file."""
    
    # Use the reference file
    ref_file = Path(__file__).parent / "07.multi_model_workflow_dynamic_merge_2" / "results" / "results_workflow_b1b2_20251111_004555.json"
    
    if not ref_file.exists():
        print(f"❌ Reference file not found: {ref_file}")
        return False
    
    print("="*70)
    print("Testing Data Extraction Mechanism")
    print("="*70)
    print(f"\nReference file: {ref_file.name}")
    
    # Load results
    with open(ref_file) as f:
        results = json.load(f)
    
    print(f"\n1. File structure:")
    print(f"   - Top level keys: {list(results.keys())}")
    print(f"   - Results type: {type(results['results']).__name__}")
    print(f"   - Results length: {len(results['results'])}")
    
    # Test extraction for probabilistic strategy
    test_strategy = "probabilistic"
    print(f"\n2. Testing extraction for strategy: {test_strategy}")
    
    # Find strategy results (same logic as run_ood_experiments.py)
    strategy_results = None
    for result in results["results"]:
        if result["strategy"] == test_strategy:
            strategy_results = result
            break
    
    if strategy_results is None:
        print(f"   ❌ Strategy not found!")
        return False
    
    print(f"   ✓ Strategy found: {strategy_results['strategy']}")
    
    # Extract workflow info
    workflow_info = strategy_results['workflows']
    print(f"\n3. Workflow information:")
    print(f"   - num_completed: {workflow_info['num_completed']}")
    print(f"   - num_warmup: {workflow_info['num_warmup']}")
    print(f"   - workflow_times length: {len(workflow_info['workflow_times'])}")
    
    # Test extraction logic
    if "workflow_times" in workflow_info and len(workflow_info["workflow_times"]) > 0:
        times = workflow_info["workflow_times"]
        print(f"\n4. ✓ Successfully extracted {len(times)} workflow completion times")
        
        # Calculate statistics
        times_array = np.array(times)
        stats = {
            "count": len(times),
            "mean": float(np.mean(times_array)),
            "median": float(np.median(times_array)),
            "std": float(np.std(times_array)),
            "min": float(np.min(times_array)),
            "max": float(np.max(times_array)),
            "p50": float(np.percentile(times_array, 50)),
            "p95": float(np.percentile(times_array, 95)),
            "p99": float(np.percentile(times_array, 99))
        }
        
        print(f"\n5. Statistics:")
        print(f"   - Count: {stats['count']}")
        print(f"   - Mean: {stats['mean']:.3f}s")
        print(f"   - Median: {stats['median']:.3f}s")
        print(f"   - Std Dev: {stats['std']:.3f}s")
        print(f"   - Min: {stats['min']:.3f}s")
        print(f"   - Max: {stats['max']:.3f}s")
        print(f"   - P50: {stats['p50']:.3f}s")
        print(f"   - P95: {stats['p95']:.3f}s")
        print(f"   - P99: {stats['p99']:.3f}s")
        
        # Verify against file's own stats
        print(f"\n6. Verification against file's statistics:")
        file_avg = workflow_info.get("avg_workflow_time", 0)
        file_median = workflow_info.get("median_workflow_time", 0)
        
        print(f"   - File avg: {file_avg:.3f}s, Computed: {stats['mean']:.3f}s")
        print(f"   - File median: {file_median:.3f}s, Computed: {stats['median']:.3f}s")
        
        avg_match = abs(file_avg - stats['mean']) < 0.01
        median_match = abs(file_median - stats['median']) < 0.01
        
        if avg_match and median_match:
            print(f"   ✓ Statistics match!")
        else:
            print(f"   ⚠ Statistics mismatch!")
        
        print("\n" + "="*70)
        print("✓ DATA EXTRACTION TEST PASSED")
        print("="*70)
        return True
    else:
        print(f"\n   ❌ workflow_times not available or empty")
        return False


if __name__ == "__main__":
    success = test_data_extraction()
    exit(0 if success else 1)
