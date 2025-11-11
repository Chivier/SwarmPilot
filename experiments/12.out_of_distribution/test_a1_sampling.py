#!/usr/bin/env python3
"""
Test script for A1 sampling integration.

This script verifies that B1 tasks correctly use A1-sampled sleep_time values.
"""

from workload_generator import (
    generate_workflow_from_traces,
    generate_workflow_with_a1_b1_sampling,
    print_workflow_stats
)
import numpy as np


def test_a1_sampling():
    """Test A1 sampling function."""
    print("=" * 60)
    print("Testing A1 Sampling for B1 Tasks")
    print("=" * 60)

    # Generate standard workflow (baseline)
    print("\n1. Standard Workflow (B1 from dr_query):")
    print("-" * 60)
    standard_workflow, standard_config = generate_workflow_from_traces(
        num_workflows=100,
        seed=42
    )
    print_workflow_stats(standard_workflow)

    # Generate A1-sampled workflow
    print("\n2. A1-Sampled Workflow (B1 from dr_boot/A1):")
    print("-" * 60)
    a1_workflow, a1_config = generate_workflow_with_a1_b1_sampling(
        num_workflows=100,
        seed=42
    )
    print_workflow_stats(a1_workflow)

    # Compare distributions
    print("\n3. Distribution Comparison:")
    print("-" * 60)

    standard_b1 = np.array([t for workflow in standard_workflow.b1_times for t in workflow])
    a1_b1 = np.array([t for workflow in a1_workflow.b1_times for t in workflow])

    print(f"Standard B1 (from dr_query):")
    print(f"  Mean:   {standard_b1.mean():.3f}s")
    print(f"  Median: {np.median(standard_b1):.3f}s")
    print(f"  Std:    {standard_b1.std():.3f}s")
    print(f"  Min:    {standard_b1.min():.3f}s")
    print(f"  Max:    {standard_b1.max():.3f}s")

    print(f"\nA1-Sampled B1 (from dr_boot/A1):")
    print(f"  Mean:   {a1_b1.mean():.3f}s")
    print(f"  Median: {np.median(a1_b1):.3f}s")
    print(f"  Std:    {a1_b1.std():.3f}s")
    print(f"  Min:    {a1_b1.min():.3f}s")
    print(f"  Max:    {a1_b1.max():.3f}s")

    # A1 reference
    standard_a1 = np.array(standard_workflow.a1_times)
    print(f"\nA1 Reference (from dr_boot):")
    print(f"  Mean:   {standard_a1.mean():.3f}s")
    print(f"  Median: {np.median(standard_a1):.3f}s")
    print(f"  Std:    {standard_a1.std():.3f}s")
    print(f"  Min:    {standard_a1.min():.3f}s")
    print(f"  Max:    {standard_a1.max():.3f}s")

    print("\n4. Validation:")
    print("-" * 60)

    # Verify that A1-sampled B1 distribution is closer to A1 than to standard B1
    mean_diff_a1 = abs(a1_b1.mean() - standard_a1.mean())
    mean_diff_standard = abs(standard_b1.mean() - a1_b1.mean())

    print(f"A1-Sampled B1 mean difference from A1: {mean_diff_a1:.3f}s")
    print(f"A1-Sampled B1 mean difference from Standard B1: {mean_diff_standard:.3f}s")

    if mean_diff_a1 < mean_diff_standard:
        print("✓ A1-sampled B1 is closer to A1 distribution (EXPECTED)")
    else:
        print("✗ A1-sampled B1 is closer to standard B1 distribution (UNEXPECTED)")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_a1_sampling()
