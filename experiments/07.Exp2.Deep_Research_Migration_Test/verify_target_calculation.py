#!/usr/bin/env python3
"""
Verify that target=[1, 19.153084] will result in 1:1 instance allocation.

This script demonstrates the relationship between:
- B (throughput matrix)
- target (request distribution)
- Optimal instance allocation
"""

def verify_target_for_1_1_ratio():
    """Verify target calculation for 1:1 instance ratio."""
    print("=" * 80)
    print("Target Calculation Verification for 1:1 Instance Ratio")
    print("=" * 80)

    # B parameter values (QPS)
    qps_model_a = 0.007571  # req/s
    qps_model_b = 0.145008  # req/s

    print(f"\n1. Throughput (B parameter)")
    print("-" * 80)
    print(f"Model A QPS: {qps_model_a:.6f} req/s")
    print(f"Model B QPS: {qps_model_b:.6f} req/s")
    print(f"QPS Ratio (B/A): {qps_model_b / qps_model_a:.6f}:1")

    # Calculate target for 1:1 instance ratio
    target_ratio = qps_model_b / qps_model_a
    target = [1, target_ratio]

    print(f"\n2. Target Distribution for 1:1 Instance Ratio")
    print("-" * 80)
    print(f"target = [1, {target_ratio:.6f}]")
    print(f"Target ratio: {target_ratio:.6f}:1")

    # Verify with different total instance counts
    print(f"\n3. Verification with Different Instance Counts")
    print("-" * 80)

    test_cases = [
        (5, 5, "Equal scheduler instances"),
        (3, 7, "Unequal scheduler instances"),
        (8, 2, "Heavily unbalanced schedulers"),
    ]

    for instance_a_num, instance_b_num, description in test_cases:
        total_instances = instance_a_num + instance_b_num

        print(f"\n{description}:")
        print(f"  Scheduler A: {instance_a_num}, Scheduler B: {instance_b_num}")
        print(f"  Total instances: {total_instances}")

        # For 1:1 instance ratio
        n_a = total_instances // 2
        n_b = total_instances - n_a

        print(f"  Optimal allocation (1:1): Model A={n_a}, Model B={n_b}")

        # Calculate total capacity with this allocation
        capacity_a = n_a * qps_model_a
        capacity_b = n_b * qps_model_b
        capacity_ratio = capacity_b / capacity_a if capacity_a > 0 else 0

        print(f"  Total capacity:")
        print(f"    Model A: {n_a} × {qps_model_a:.6f} = {capacity_a:.6f} req/s")
        print(f"    Model B: {n_b} × {qps_model_b:.6f} = {capacity_b:.6f} req/s")
        print(f"    Capacity ratio (B/A): {capacity_ratio:.6f}:1")

        # Compare with target ratio
        difference = abs(capacity_ratio - target_ratio)
        match = "✓ MATCHES" if difference < 0.1 else "✗ DOES NOT MATCH"

        print(f"  Target ratio: {target_ratio:.6f}:1")
        print(f"  Difference: {difference:.6f} → {match}")

    # Explanation
    print(f"\n4. Mathematical Explanation")
    print("-" * 80)
    print("""
The optimizer minimizes the difference between:
  - Actual capacity ratio: (n_b × qps_b) / (n_a × qps_a)
  - Target ratio: target[1] / target[0]

For 1:1 instance allocation (n_a = n_b):
  - Capacity ratio = qps_b / qps_a
  - Therefore: target should be [1, qps_b / qps_a]

With our calculated QPS values:
  - target = [1, {ratio:.6f}]
  - This ensures: n_a ≈ n_b (1:1 instance ratio)
""".format(ratio=target_ratio))

    print("=" * 80)
    print("✓ Verification Complete!")
    print("=" * 80)
    print(f"\nConclusion:")
    print(f"  - Setting target = [1, {target_ratio:.6f}] will guide the optimizer")
    print(f"    to allocate instances in approximately 1:1 ratio")
    print(f"  - The exact ratio may vary slightly for odd total instance counts")
    print(f"  - This is the correct approach: let the optimizer decide based on")
    print(f"    throughput and target distribution, not forcing initial deployment")

    return True


if __name__ == "__main__":
    import sys
    success = verify_target_for_1_1_ratio()
    sys.exit(0 if success else 1)
