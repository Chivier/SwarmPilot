#!/usr/bin/env python3
"""Test script to verify strategy API call with custom quantiles."""

import sys
sys.path.insert(0, '.')

from test_dynamic_workflow import set_scheduling_strategy
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

# Test URL (won't actually connect, just for demonstration)
test_url = "http://localhost:8100"

print("Testing strategy API payload generation...\n")

# Test probabilistic strategy
print("1. Testing probabilistic strategy:")
print("   This should include custom quantiles [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.99]")
print("   (Note: This will fail to connect, but we can see the payload in logs)\n")

# We can't actually call it without a running scheduler, but we can verify the code structure
import inspect
source = inspect.getsource(set_scheduling_strategy)
print("Function source code snippet:")
print("-" * 60)
if 'payload["quantiles"]' in source and '[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.99]' in source:
    print("✓ Custom quantiles found in code!")
    print("  Quantiles: [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.99]")
else:
    print("✗ Custom quantiles not found!")
    
print("-" * 60)
print("\nTest complete!")
