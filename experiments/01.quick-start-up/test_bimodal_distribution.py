#!/usr/bin/env python3
"""
Test script to verify the bimodal distribution generation.
"""
import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the main script
from test_scheduling_poisson import (
    generate_task_times,
    LEFT_PEAK_MIN, LEFT_PEAK_MAX, LEFT_PEAK_MEAN,
    RIGHT_PEAK_MIN, RIGHT_PEAK_MAX, RIGHT_PEAK_MEAN,
    PEAK_RATIO, TASK_MEAN, TASK_STD
)

def main():
    """Test the bimodal distribution generation."""
    num_tasks = 1000

    print("=" * 60)
    print("Bimodal Distribution Test")
    print("=" * 60)
    print(f"Generating {num_tasks} task times...")
    print()

    # Generate task times
    task_times = generate_task_times(num_tasks)

    # Analyze the distribution
    task_times_array = np.array(task_times)

    # Separate into left and right peaks based on threshold (5.0 seconds)
    threshold = 5.0
    left_peak_tasks = task_times_array[task_times_array < threshold]
    right_peak_tasks = task_times_array[task_times_array >= threshold]

    print(f"Configuration:")
    print(f"  Left peak:  [{LEFT_PEAK_MIN}, {LEFT_PEAK_MAX}]s, target mean={LEFT_PEAK_MEAN}s, ratio={PEAK_RATIO:.0%}")
    print(f"  Right peak: [{RIGHT_PEAK_MIN}, {RIGHT_PEAK_MAX}]s, target mean={RIGHT_PEAK_MEAN}s, ratio={1-PEAK_RATIO:.0%}")
    print(f"  Overall target: mean={TASK_MEAN:.3f}s, std={TASK_STD:.3f}s")
    print()

    print(f"Generated Distribution:")
    print(f"  Total tasks: {len(task_times)}")
    print(f"  Left peak tasks (<{threshold}s): {len(left_peak_tasks)} ({len(left_peak_tasks)/len(task_times):.1%})")
    print(f"    Range: [{left_peak_tasks.min():.3f}, {left_peak_tasks.max():.3f}]s")
    print(f"    Mean: {left_peak_tasks.mean():.3f}s")
    print(f"    Std: {left_peak_tasks.std():.3f}s")
    print()
    print(f"  Right peak tasks (>={threshold}s): {len(right_peak_tasks)} ({len(right_peak_tasks)/len(task_times):.1%})")
    print(f"    Range: [{right_peak_tasks.min():.3f}, {right_peak_tasks.max():.3f}]s")
    print(f"    Mean: {right_peak_tasks.mean():.3f}s")
    print(f"    Std: {right_peak_tasks.std():.3f}s")
    print()
    print(f"  Overall:")
    print(f"    Range: [{task_times_array.min():.3f}, {task_times_array.max():.3f}]s")
    print(f"    Mean: {task_times_array.mean():.3f}s")
    print(f"    Std: {task_times_array.std():.3f}s")
    print()

    # Verify the distribution is shuffled (check first 10 tasks)
    print(f"First 10 task times (should be mixed):")
    for i, time in enumerate(task_times[:10]):
        peak = "Left" if time < threshold else "Right"
        print(f"  Task {i}: {time:.3f}s ({peak} peak)")
    print()

    # Calculate histogram
    print("Histogram (by 1-second bins):")
    bins = np.arange(0, 11, 1)
    hist, _ = np.histogram(task_times_array, bins=bins)
    for i in range(len(hist)):
        bar = "█" * int(hist[i] / max(hist) * 50)
        print(f"  [{bins[i]:2.0f}-{bins[i+1]:2.0f}s): {hist[i]:4d} {bar}")

    print("=" * 60)
    print("✓ Distribution test completed")
    print("=" * 60)

if __name__ == "__main__":
    main()
