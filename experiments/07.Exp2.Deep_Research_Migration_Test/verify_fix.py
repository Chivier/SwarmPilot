#!/usr/bin/env python3
"""
Verification script to check if the data collection fix works correctly.

This script reads the raw result files and verifies that sleep_time and execution_time
are properly extracted for all task types (A, B1, B2, Merge).
"""

import json
import sys
from pathlib import Path


def verify_task_results(file_path: Path, task_type: str):
    """
    Verify that a task result file contains non-zero sleep_time values.

    Args:
        file_path: Path to the JSONL file containing task results
        task_type: Type of task (A1, B1, B2, Merge)

    Returns:
        True if verification passes, False otherwise
    """
    if not file_path.exists():
        print(f"❌ {task_type}: File not found: {file_path}")
        return False

    total_records = 0
    records_with_sleep_time = 0
    records_with_execution_time = 0
    total_sleep_time = 0.0
    total_execution_time = 0.0

    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            total_records += 1
            record = json.loads(line)

            # Check if sleep_time exists in result
            if "result" in record and "sleep_time" in record["result"]:
                sleep_time = record["result"]["sleep_time"]
                if sleep_time > 0:
                    records_with_sleep_time += 1
                    total_sleep_time += sleep_time

            # Check if execution_time exists
            if "execution_time" in record:
                execution_time = record["execution_time"]
                if execution_time > 0:
                    records_with_execution_time += 1
                    total_execution_time += execution_time

    if total_records == 0:
        print(f"⚠️  {task_type}: No records found in {file_path}")
        return False

    # Calculate success rate
    sleep_time_rate = (records_with_sleep_time / total_records) * 100
    execution_time_rate = (records_with_execution_time / total_records) * 100
    avg_sleep_time = total_sleep_time / records_with_sleep_time if records_with_sleep_time > 0 else 0
    avg_execution_time = total_execution_time / records_with_execution_time if records_with_execution_time > 0 else 0

    print(f"\n{task_type} Task Results ({file_path.name}):")
    print(f"  Total records: {total_records}")
    print(f"  Records with sleep_time > 0: {records_with_sleep_time} ({sleep_time_rate:.1f}%)")
    print(f"  Records with execution_time > 0: {records_with_execution_time} ({execution_time_rate:.1f}%)")
    print(f"  Average sleep_time: {avg_sleep_time:.3f}s")
    print(f"  Average execution_time: {avg_execution_time:.3f}s")

    # Verification passes if most records have both values
    if sleep_time_rate >= 95 and execution_time_rate >= 95:
        print(f"✅ {task_type}: Verification PASSED")
        return True
    else:
        print(f"❌ {task_type}: Verification FAILED (expected >= 95% success rate)")
        return False


def main():
    """Main verification logic."""
    print("=" * 80)
    print("Data Collection Fix Verification")
    print("=" * 80)

    # Check all three strategies
    strategies = ["probabilistic", "round_robin", "min_time"]
    all_passed = True

    for strategy in strategies:
        print(f"\n{'=' * 80}")
        print(f"Strategy: {strategy}")
        print(f"{'=' * 80}")

        results_dir = Path(f"raw_results_{strategy}")

        if not results_dir.exists():
            print(f"⚠️  Results directory not found: {results_dir}")
            print(f"   Skipping {strategy} strategy...")
            continue

        # Verify each task type
        task_files = [
            (results_dir / "a1_task_results.jsonl", "A1"),
            (results_dir / "b1_task_results.jsonl", "B1"),
            (results_dir / "b2_task_results.jsonl", "B2"),
            (results_dir / "merge_task_result.jsonl", "Merge"),
        ]

        strategy_passed = True
        for file_path, task_type in task_files:
            if not verify_task_results(file_path, task_type):
                strategy_passed = False
                all_passed = False

        if strategy_passed:
            print(f"\n✅ Strategy '{strategy}': All task types PASSED")
        else:
            print(f"\n❌ Strategy '{strategy}': Some task types FAILED")

    # Final summary
    print(f"\n{'=' * 80}")
    print("Final Summary")
    print(f"{'=' * 80}")

    if all_passed:
        print("✅ ALL VERIFICATIONS PASSED!")
        print("   The data collection fix is working correctly.")
        print("   All task types (A1, B1, B2, Merge) now properly record:")
        print("   - sleep_time (extracted from result)")
        print("   - execution_time (from WebSocket response)")
        return 0
    else:
        print("❌ SOME VERIFICATIONS FAILED")
        print("   Please check the output above for details.")
        print("   Note: If raw result files don't exist yet, run the experiment first.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
