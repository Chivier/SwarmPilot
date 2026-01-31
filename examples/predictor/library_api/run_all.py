"""Run all Library API examples in sequence.

This script runs all 8 examples for the Library API, tracking
success/failure for each. Each example runs in isolation with
its own storage directory.

Usage:
    uv run python -m examples.library_api.run_all
    # Or from predictor directory:
    uv run python examples/library_api/run_all.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path


# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import all example modules
from examples.library_api import ex01_train_without_preprocessor
from examples.library_api import ex02_train_with_preprocessor
from examples.library_api import ex03_predict_without_preprocessor
from examples.library_api import ex04_predict_with_preprocessor
from examples.library_api import ex05_error_handling
from examples.library_api import ex06_collect_update_same_chain
from examples.library_api import ex07_collect_retrain_new_chain
from examples.library_api import ex08_custom_preprocessor


EXAMPLES = [
    ("01", "Train without preprocessor", ex01_train_without_preprocessor),
    ("02", "Train with preprocessor", ex02_train_with_preprocessor),
    ("03", "Predict without preprocessor", ex03_predict_without_preprocessor),
    ("04", "Predict with preprocessor", ex04_predict_with_preprocessor),
    ("05", "Error handling", ex05_error_handling),
    ("06", "Collect & update (same chain)", ex06_collect_update_same_chain),
    ("07", "Collect & retrain (new chain)", ex07_collect_retrain_new_chain),
    ("08", "Custom preprocessor", ex08_custom_preprocessor),
]


def print_header() -> None:
    """Print header for all-in-one runner."""
    print("=" * 70)
    print("  V2 API Examples - Library API")
    print("  All-in-One Runner")
    print("=" * 70)
    print()
    print("  This script runs all 8 Library API examples in sequence.")
    print("  Each example uses isolated storage that is cleaned up after.")
    print()


def print_separator() -> None:
    """Print separator between examples."""
    print()
    print("-" * 70)
    print()


def print_summary(
    passed: int, failed: int, failures: list[tuple[str, str, str]]
) -> None:
    """Print final summary."""
    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print()

    total = passed + failed

    if failed == 0:
        print(f"  All {total} examples passed!")
    else:
        print(f"  Results: {passed}/{total} passed, {failed} failed")
        print()
        print("  Failed examples:")
        for num, name, error in failures:
            print(f"    [{num}] {name}")
            print(f"        Error: {error[:60]}...")

    print()
    print("=" * 70)


def main() -> int:
    """Run all Library API examples."""
    print_header()

    passed = 0
    failed = 0
    failures: list[tuple[str, str, str]] = []

    for num, name, module in EXAMPLES:
        print(f"[{num}] {name}")
        print("-" * 50)

        try:
            module.main()
            print("\n  Result: PASSED")
            passed += 1
        except Exception as e:
            error_msg = str(e)
            print("\n  Result: FAILED")
            print(f"  Error: {error_msg}")
            if "--verbose" in sys.argv or "-v" in sys.argv:
                traceback.print_exc()
            failed += 1
            failures.append((num, name, error_msg))

        print_separator()

    print_summary(passed, failed, failures)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
