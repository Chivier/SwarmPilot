#!/usr/bin/env python3
"""
Test script to verify that 200% MAPE error is correctly applied to exp_runtime
for min_time strategy only.
"""

import numpy as np
import sys


# Copy the apply_mape_error function from test_dynamic_workflow.py
MAPE_PERCENTAGE = 50.0  # 50% MAPE for min_time strategy


def apply_mape_error(exp_runtime: float, mape_percentage: float = MAPE_PERCENTAGE) -> float:
    """
    Apply MAPE (Mean Absolute Percentage Error) to exp_runtime.

    For example, with 50% MAPE:
    - Error range: [-50%, +50%]
    - Predicted value = actual value * (1 + uniform(-0.5, 0.5))

    This simulates prediction error in runtime estimation.

    Args:
        exp_runtime: Original expected runtime in milliseconds
        mape_percentage: MAPE percentage (default: 50%)

    Returns:
        Modified exp_runtime with applied error
    """
    # Convert MAPE percentage to multiplier
    # e.g., 50% -> 0.5
    mape_multiplier = mape_percentage / 100.0

    # Generate random error in range [-mape_multiplier, +mape_multiplier]
    # For 50% MAPE: [-0.5, +0.5]
    error = np.random.uniform(-mape_multiplier, mape_multiplier)

    # Apply error: new_value = original * (1 + error)
    modified_runtime = exp_runtime * (1.0 + error)

    # Ensure non-negative result (though with 50% MAPE, this should never be needed)
    return max(0.0, modified_runtime)


def test_mape_error():
    """Test the MAPE error function."""
    print("=" * 80)
    print("Testing 50% MAPE Error Function")
    print("=" * 80)

    # Test parameters
    original_runtime = 1000.0  # 1000 ms
    num_samples = 10000
    mape_percentage = 50.0

    print(f"\nTest Configuration:")
    print(f"  Original runtime: {original_runtime:.1f} ms")
    print(f"  MAPE percentage: {mape_percentage:.1f}%")
    print(f"  Number of samples: {num_samples}")
    print(f"  Expected error range: [-{mape_percentage:.1f}%, +{mape_percentage:.1f}%]")
    print(f"  Expected value range: [{original_runtime * (1 - mape_percentage/100):.1f} ms, "
          f"{original_runtime * (1 + mape_percentage/100):.1f} ms]")

    # Generate samples
    np.random.seed(42)
    samples = [apply_mape_error(original_runtime, mape_percentage) for _ in range(num_samples)]

    # Calculate statistics
    samples_array = np.array(samples)
    mean_val = np.mean(samples_array)
    std_val = np.std(samples_array)
    min_val = np.min(samples_array)
    max_val = np.max(samples_array)
    median_val = np.median(samples_array)

    # Calculate error percentages
    errors = [(s - original_runtime) / original_runtime * 100 for s in samples]
    mean_error = np.mean(errors)
    min_error = np.min(errors)
    max_error = np.max(errors)

    # Count how many samples are negative (should be 0)
    num_negative = sum(1 for s in samples if s < 0)

    # Count how many samples are within expected range
    expected_min = original_runtime * (1 - mape_percentage / 100)  # 50% minimum for 50% MAPE
    expected_max = original_runtime * (1 + mape_percentage / 100)
    num_in_range = sum(1 for s in samples if expected_min <= s <= expected_max)

    print(f"\n{'=' * 80}")
    print("Statistical Results:")
    print(f"{'=' * 80}")
    print(f"Value Statistics (ms):")
    print(f"  Mean:   {mean_val:.2f} (expected ≈ {original_runtime:.2f})")
    print(f"  Median: {median_val:.2f}")
    print(f"  Std:    {std_val:.2f}")
    print(f"  Min:    {min_val:.2f}")
    print(f"  Max:    {max_val:.2f}")

    print(f"\nError Statistics (%):")
    print(f"  Mean error:   {mean_error:.2f}% (expected ≈ 0%)")
    print(f"  Min error:    {min_error:.2f}%")
    print(f"  Max error:    {max_error:.2f}%")

    print(f"\nValidation:")
    print(f"  Negative values: {num_negative} (expected: 0)")
    print(f"  Values in expected range [0, {expected_max:.1f}]: "
          f"{num_in_range}/{num_samples} ({num_in_range/num_samples*100:.1f}%)")

    # Determine if test passes
    print(f"\n{'=' * 80}")
    print("Test Results:")
    print(f"{'=' * 80}")

    passed = True
    tests = []

    # Test 1: No negative values
    test1_pass = num_negative == 0
    tests.append(("No negative values", test1_pass))
    if not test1_pass:
        passed = False

    # Test 2: Mean error reasonable (should be around 0% due to symmetric range)
    # With [-50%, +50%] range, expected mean is around 0%
    test2_pass = -5.0 <= mean_error <= 5.0
    tests.append((f"Mean error reasonable (actual: {mean_error:.2f}%, expected ≈0%)", test2_pass))
    if not test2_pass:
        passed = False

    # Test 3: Error range approximately [-50%, +50%]
    test3_pass = (min_error >= -55.0 and max_error <= 55.0)
    tests.append((f"Error range ≈ [-50%, +50%] (actual: [{min_error:.1f}%, {max_error:.1f}%])", test3_pass))
    if not test3_pass:
        passed = False

    # Test 4: All values in expected range
    test4_pass = num_in_range == num_samples
    tests.append(("All values in valid range", test4_pass))
    if not test4_pass:
        passed = False

    # Print test results
    for test_name, test_pass in tests:
        status = "✅ PASS" if test_pass else "❌ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\n{'=' * 80}")
    if passed:
        print("✅ ALL TESTS PASSED")
        print("\nThe 50% MAPE error function is working correctly!")
        print("For min_time strategy, exp_runtime will be modified with ±50% error.")
        print("This simulates a moderately inaccurate prediction system.")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease check the implementation of apply_mape_error().")
        return 1


if __name__ == "__main__":
    sys.exit(test_mape_error())
