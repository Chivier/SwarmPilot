"""E2E PyLet Benchmark Test Suite.

This package contains a comprehensive end-to-end test for the SwarmPilot
PyLet-based architecture. It tests the full stack:

- Mock predictor service (returns sleep_time as predicted runtime)
- Scheduler (task scheduling and instance management)
- Planner (PyLet-based instance deployment)
- PyLet cluster (head + workers)
- Sleep model instances (deployed via PyLet)

Test Configuration:
- 3 models: sleep_model_a (4 instances), sleep_model_b (3), sleep_model_c (3)
- 10 total instances via PyLet
- QPS-based workload (default: 5 QPS for 60 seconds = 300 tasks)

Usage:
    # Run the full E2E benchmark
    python -m tests.integration.e2e_pylet_benchmark.run_e2e_pylet_benchmark

    # Or via pytest (requires --run-integration flag)
    pytest tests/integration/e2e_pylet_benchmark/ -v --run-integration
"""
