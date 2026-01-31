#!/usr/bin/env python3
"""Dummy script B: simulates a workload that takes 20 seconds to complete.

This script represents model B with throughput capacity of 0.05 req/s (1/20).
Used for testing the planner's deployment optimization algorithm.
"""

import sys
import time


def main():
    """Execute the dummy workload - sleep for 20 seconds."""
    print("Script B starting - sleeping for 20 seconds...")
    time.sleep(20)
    print("Script B completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
