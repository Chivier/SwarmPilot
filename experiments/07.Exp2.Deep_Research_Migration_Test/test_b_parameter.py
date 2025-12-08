#!/usr/bin/env python3
"""
Test script to verify the updated B parameter works correctly.

This script validates that:
1. The new B parameter values are correctly formed
2. PlannerInput accepts the values
3. The calculated B matrix has the expected shape and values
"""

import sys
from pathlib import Path

# Import PlannerInput without triggering the perform_redeploy() call
# We'll read the redeply.py source and extract just the class definition
import importlib.util
spec = importlib.util.spec_from_file_location("redeply_module", "redeply.py")
if spec and spec.loader:
    # Create module but don't execute the module-level code
    # Instead, compile and exec only the class definitions
    with open("redeply.py", "r") as f:
        source = f.read()

    # Extract only the imports and class definitions, skip the function calls at the end
    lines = source.split('\n')
    # Find where perform_redeploy() is called
    filtered_lines = []
    for line in lines:
        if line.strip().startswith('perform_redeploy()'):
            break
        filtered_lines.append(line)

    filtered_source = '\n'.join(filtered_lines)

    # Create a module namespace
    namespace = {}
    exec(filtered_source, namespace)
    PlannerInput = namespace['PlannerInput']
else:
    print("Error: Could not load redeply.py")
    sys.exit(1)


def test_b_parameter():
    """Test the new B parameter calculation."""
    print("=" * 80)
    print("Testing Updated B Parameter")
    print("=" * 80)

    # Test with typical instance counts
    test_cases = [
        (5, 5, "Equal instances"),
        (3, 7, "More B instances"),
        (8, 2, "More A instances"),
    ]

    for instance_a_num, instance_b_num, description in test_cases:
        print(f"\nTest Case: {description}")
        print(f"  Instance A: {instance_a_num}, Instance B: {instance_b_num}")

        total_instances = instance_a_num + instance_b_num

        # Create B matrix with new values
        B = [[0.007571, 0.145008]] * total_instances

        print(f"  Total instances: {total_instances}")
        print(f"  B matrix shape: {len(B)} rows × {len(B[0])} columns")
        print(f"  B[0] = {B[0]}")

        # Verify all rows are identical
        assert all(row == [0.007571, 0.145008] for row in B), "All rows should be identical"

        # Create PlannerInput to validate
        try:
            planner_input = PlannerInput(
                M=total_instances,
                N=2,
                B=B,
                a=1,
                target=[1, 10],
                algorithm="simulated_annealing",
                objective_method="ratio_difference"
            )
            print(f"  ✓ PlannerInput validation passed")

            # Verify values
            assert planner_input.M == total_instances
            assert planner_input.N == 2
            assert len(planner_input.B) == total_instances
            assert all(len(row) == 2 for row in planner_input.B)
            print(f"  ✓ All assertions passed")

        except Exception as e:
            print(f"  ✗ Validation failed: {e}")
            return False

    # Test throughput ratio
    print("\n" + "=" * 80)
    print("Throughput Analysis")
    print("=" * 80)

    qps_model_a = 0.007571
    qps_model_b = 0.145008
    ratio = qps_model_b / qps_model_a

    print(f"\nModel A QPS: {qps_model_a:.6f} req/s")
    print(f"Model B QPS: {qps_model_b:.6f} req/s")
    print(f"Throughput Ratio (B/A): {ratio:.2f}:1")

    # Compare with target distribution
    target_ratio = 10 / 1
    print(f"\nTarget Distribution Ratio: {target_ratio:.2f}:1")
    print(f"Throughput Ratio: {ratio:.2f}:1")
    print(f"Difference: {abs(ratio - target_ratio):.2f}x")

    print("\nNote: The optimizer will handle the difference between target ratio")
    print("and throughput ratio by allocating instances appropriately.")

    # Verify values are positive
    assert qps_model_a > 0, "QPS for Model A should be positive"
    assert qps_model_b > 0, "QPS for Model B should be positive"
    assert qps_model_b > qps_model_a, "Model B should be faster than Model A"

    print("\n" + "=" * 80)
    print("✓ All Tests Passed!")
    print("=" * 80)

    print("\nThe updated B parameter is ready for use in experiments.")
    print("Next steps:")
    print("  1. Ensure services are running: ./start_all_services.sh")
    print("  2. Test with redeploy: python redeply.py")
    print("  3. Run full experiment: python test_dynamic_workflow.py --num-workflows 10")

    return True


if __name__ == "__main__":
    import sys
    success = test_b_parameter()
    sys.exit(0 if success else 1)
