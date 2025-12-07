#!/usr/bin/env python3
"""
Test script to verify JSON serialization fix for timeline tracker.

This script tests that the timeline tracker can now properly handle:
1. Numpy int64 types (from instance counts)
2. Special float values (inf, -inf, nan)
3. Normal values

Run this before restarting services to verify the fix works.
"""

import numpy as np
import json
import sys
from pathlib import Path

# Add planner src to path
planner_src = Path(__file__).parent.parent.parent / "planner" / "src"
sys.path.insert(0, str(planner_src))

from instance_timeline_tracker import InstanceTimelineTracker, compute_instance_counts


def test_numpy_types():
    """Test that numpy types are properly converted."""
    print("\n" + "=" * 80)
    print("Test 1: Numpy Type Conversion")
    print("=" * 80)

    # Create a temporary tracker
    tracker = InstanceTimelineTracker(output_path="/tmp/test_timeline.json")

    # Test with numpy int64 (common from numpy operations)
    instance_counts = {
        "model_a": np.int64(5),
        "model_b": np.int64(10)
    }

    changes_count = np.int64(2)
    target_distribution = [np.float64(8.5), np.float64(42.3)]
    score = np.float64(0.1234)

    try:
        tracker.record_migration(
            event_type="test_migration",
            instance_counts=instance_counts,
            changes_count=changes_count,
            success=True,
            target_distribution=target_distribution,
            score=score
        )
        print("✓ Successfully recorded migration with numpy types")

        # Verify JSON file is valid
        with open("/tmp/test_timeline.json", "r") as f:
            data = json.load(f)
            print("✓ Timeline JSON is valid and can be loaded")
            print(f"  Entries: {len(data['entries'])}")
            entry = data['entries'][0]
            print(f"  Instance counts: {entry['instance_counts']}")
            print(f"  Changes count: {entry['changes_count']} (type: {type(entry['changes_count']).__name__})")
            print(f"  Score: {entry['score']} (type: {type(entry['score']).__name__})")

        return True
    except Exception as e:
        print(f"✗ Failed to handle numpy types: {e}")
        return False


def test_special_float_values():
    """Test that special float values (inf, nan) are properly handled."""
    print("\n" + "=" * 80)
    print("Test 2: Special Float Values (inf, -inf, nan)")
    print("=" * 80)

    tracker = InstanceTimelineTracker(output_path="/tmp/test_timeline_special.json")

    test_cases = [
        ("inf", float('inf')),
        ("-inf", float('-inf')),
        ("nan", float('nan')),
        ("numpy inf", np.inf),
        ("numpy -inf", -np.inf),
        ("numpy nan", np.nan),
    ]

    all_passed = True
    for name, score_value in test_cases:
        try:
            tracker.clear()  # Clear before each test
            tracker.record_migration(
                event_type=f"test_{name}",
                instance_counts={"model": 5},
                changes_count=0,
                success=True,
                score=score_value
            )

            # Verify JSON is valid
            with open("/tmp/test_timeline_special.json", "r") as f:
                data = json.load(f)
                entry = data['entries'][0]
                stored_score = entry['score']

                if stored_score is None:
                    print(f"✓ {name}: Converted to None (was {score_value})")
                else:
                    print(f"✗ {name}: Expected None, got {stored_score}")
                    all_passed = False

        except Exception as e:
            print(f"✗ {name}: Failed with error: {e}")
            all_passed = False

    return all_passed


def test_normal_values():
    """Test that normal values still work correctly."""
    print("\n" + "=" * 80)
    print("Test 3: Normal Values")
    print("=" * 80)

    tracker = InstanceTimelineTracker(output_path="/tmp/test_timeline_normal.json")

    try:
        tracker.record_migration(
            event_type="normal_migration",
            instance_counts={"model_a": 3, "model_b": 7},
            changes_count=2,
            success=True,
            target_distribution=[10.5, 25.3],
            score=0.0234
        )

        with open("/tmp/test_timeline_normal.json", "r") as f:
            data = json.load(f)
            entry = data['entries'][0]

            # Verify all fields
            checks = [
                ("instance_counts", entry['instance_counts'] == {"model_a": 3, "model_b": 7}),
                ("changes_count", entry['changes_count'] == 2),
                ("target_distribution", entry['target_distribution'] == [10.5, 25.3]),
                ("score", abs(entry['score'] - 0.0234) < 0.0001),
            ]

            all_passed = True
            for field, passed in checks:
                if passed:
                    print(f"✓ {field}: Correct")
                else:
                    print(f"✗ {field}: Incorrect - {entry.get(field)}")
                    all_passed = False

            return all_passed

    except Exception as e:
        print(f"✗ Failed with normal values: {e}")
        return False


def cleanup():
    """Clean up test files."""
    import os
    test_files = [
        "/tmp/test_timeline.json",
        "/tmp/test_timeline_special.json",
        "/tmp/test_timeline_normal.json",
    ]
    for f in test_files:
        if os.path.exists(f):
            os.remove(f)


def main():
    print("\n" + "=" * 80)
    print("Timeline Tracker JSON Serialization Fix - Verification")
    print("=" * 80)
    print("\nThis test verifies that the timeline tracker can handle:")
    print("  1. Numpy int64/float64 types (common from optimization)")
    print("  2. Special float values (inf, -inf, nan)")
    print("  3. Normal Python types")

    try:
        results = []
        results.append(("Numpy types", test_numpy_types()))
        results.append(("Special float values", test_special_float_values()))
        results.append(("Normal values", test_normal_values()))

        # Print summary
        print("\n" + "=" * 80)
        print("Test Summary")
        print("=" * 80)

        all_passed = True
        for test_name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{status}: {test_name}")
            if not passed:
                all_passed = False

        print("=" * 80)

        if all_passed:
            print("\n✓ All tests passed! The JSON serialization fix is working correctly.")
            print("\nYou can now:")
            print("  1. Restart the Planner service to apply the fix")
            print("  2. Run experiments without timeline errors")
            print("\nTo restart Planner:")
            print("  ./stop_all_services.sh")
            print("  ./start_all_services.sh")
            return 0
        else:
            print("\n✗ Some tests failed. Please check the errors above.")
            return 1

    finally:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
