#!/usr/bin/env python3
"""Dummy script A: simulates a workload that takes 10 seconds to complete.

This script represents model A with throughput capacity of 0.1 req/s (1/10).
Used for testing the planner's deployment optimization algorithm.
"""

import sys
import time


def main():
    """Execute the dummy workload - sleep for 10 seconds."""
    print("Script A starting - sleeping for 10 seconds...")
    time.sleep(10)
    print("Script A completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
