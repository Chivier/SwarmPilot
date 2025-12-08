#!/usr/bin/env python3
"""
Test script to verify the 1:1 initial deployment ratio logic.

This script simulates the instance allocation logic without requiring
running services to verify that instances are correctly split 1:1
between Model A and Model B.
"""

from pydantic import BaseModel, Field
from typing import List


class InstanceInfo(BaseModel):
    """Information about a target instance."""
    endpoint: str = Field(..., description="Instance API endpoint")
    current_model: str = Field(..., description="Current model name")


def build_instance_info_test(instance_a_num, instance_b_num):
    """Test version of build_instance_info that doesn't require services."""
    instance_a_start_port = 8210
    instance_b_start_port = 8300

    total_instances = instance_a_num + instance_b_num

    # Deploy instances with 1:1 ratio (half Model A, half Model B)
    model_a_count = total_instances // 2
    model_b_count = total_instances - model_a_count

    print(f"Scheduler instances: A={instance_a_num}, B={instance_b_num}")
    print(f"Initial deployment ratio 1:1 → Model A={model_a_count}, Model B={model_b_count}")

    all_instances = []
    instance_idx = 0

    # Assign first half to Model A
    for i in range(instance_a_num):
        current_model = "sleep_model_a" if instance_idx < model_a_count else "sleep_model_b"
        all_instances.append(
            InstanceInfo(
                endpoint=f"http://localhost:{instance_a_start_port + i}",
                current_model=current_model
            )
        )
        instance_idx += 1

    # Assign second half to Model B (or Model A if still within model_a_count)
    for i in range(instance_b_num):
        current_model = "sleep_model_a" if instance_idx < model_a_count else "sleep_model_b"
        all_instances.append(
            InstanceInfo(
                endpoint=f"http://localhost:{instance_b_start_port + i}",
                current_model=current_model
            )
        )
        instance_idx += 1

    return all_instances, model_a_count, model_b_count


def test_1_1_ratio():
    """Test 1:1 deployment ratio with different instance configurations."""
    print("=" * 80)
    print("Testing 1:1 Initial Deployment Ratio")
    print("=" * 80)

    test_cases = [
        (5, 5, "Equal scheduler instances"),
        (3, 7, "More B scheduler instances"),
        (8, 2, "More A scheduler instances"),
        (4, 4, "Even total (8 instances)"),
        (3, 4, "Odd total (7 instances)"),
    ]

    all_passed = True

    for instance_a_num, instance_b_num, description in test_cases:
        print(f"\n{'='*80}")
        print(f"Test Case: {description}")
        print(f"{'='*80}")

        instances, model_a_count, model_b_count = build_instance_info_test(
            instance_a_num, instance_b_num
        )

        total = instance_a_num + instance_b_num
        expected_a = total // 2
        expected_b = total - expected_a

        # Count actual deployments
        actual_a = sum(1 for inst in instances if inst.current_model == "sleep_model_a")
        actual_b = sum(1 for inst in instances if inst.current_model == "sleep_model_b")

        print(f"\nExpected: Model A={expected_a}, Model B={expected_b}")
        print(f"Actual:   Model A={actual_a}, Model B={actual_b}")

        # Verify counts
        if actual_a == expected_a and actual_b == expected_b:
            print("✓ Instance counts match expected 1:1 ratio")
        else:
            print("✗ Instance counts DO NOT match expected ratio")
            all_passed = False

        # Verify ratio is approximately 1:1
        ratio = actual_b / actual_a if actual_a > 0 else 0
        print(f"Actual ratio (B:A): {ratio:.2f}:1")

        # For odd total instances, the difference can be at most 1
        # So the ratio can range from (n/2-1)/(n/2+1) to (n/2+1)/(n/2-1)
        # For 7 instances: 3/4 = 0.75 or 4/3 = 1.33, which is acceptable
        max_diff = abs(actual_a - actual_b)
        if max_diff <= 1:
            print("✓ Ratio is approximately 1:1 (difference ≤ 1)")
        else:
            print(f"✗ Ratio is NOT approximately 1:1 (difference = {max_diff})")
            all_passed = False

        # Print instance details
        print(f"\nInstance Details:")
        for i, inst in enumerate(instances):
            port = inst.endpoint.split(":")[-1]
            print(f"  [{i}] Port {port}: {inst.current_model}")

    # Final summary
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All Tests Passed!")
        print("\nThe 1:1 deployment ratio logic is working correctly.")
        print("Instances are evenly split between Model A and Model B.")
    else:
        print("✗ Some Tests Failed")
        print("\nPlease review the logic in build_instance_info()")

    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    import sys
    success = test_1_1_ratio()
    sys.exit(0 if success else 1)
