"""E2E Benchmark Test Suite.

This module provides pytest-based tests for the E2E benchmark.
It wraps the orchestrator to allow running via pytest with
configurable options.

Usage:
    # Run with default settings
    pytest tests/integration/e2e_pylet_benchmark/test_e2e_benchmark.py -v --run-e2e

    # Run with custom settings
    pytest tests/integration/e2e_pylet_benchmark/test_e2e_benchmark.py -v --run-e2e \
        --e2e-qps 10 --e2e-duration 30
"""

import asyncio
from pathlib import Path

import pytest

from .run_e2e_pylet_benchmark import (
    E2EBenchmarkOrchestrator,
    ServiceConfig,
    TestConfig,
)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_benchmark_full(
    e2e_qps: float,
    e2e_duration: float,
    e2e_output_dir: Path,
    log_dir: Path,
):
    """Run full E2E benchmark test.

    This test starts all services, deploys instances, runs workload,
    and generates a comprehensive report.
    """
    service_config = ServiceConfig(
        scheduler_port=18000,  # Use non-standard ports to avoid conflicts
        predictor_port=18002,
        instance_port_start=18100,
        log_dir=log_dir,
    )

    test_config = TestConfig(
        qps=e2e_qps,
        duration_seconds=e2e_duration,
        output_dir=e2e_output_dir,
    )

    orchestrator = E2EBenchmarkOrchestrator(service_config, test_config)
    result = await orchestrator.run()

    # Assertions
    assert result["success"], f"Benchmark failed: {result.get('error')}"

    summary = result["summary"]
    assert summary["tasks_submitted"] > 0, "No tasks were submitted"

    # Check success rate
    if summary["tasks_completed"] > 0:
        success_rate = summary["tasks_completed"] / summary["tasks_submitted"]
        assert success_rate >= 0.9, f"Low success rate: {success_rate:.2%}"

    # Check QPS
    expected_qps = e2e_qps
    actual_qps = summary["actual_qps"]
    qps_ratio = actual_qps / expected_qps
    assert qps_ratio >= 0.8, f"Low QPS achievement: {actual_qps:.2f}/{expected_qps}"

    # Report files should exist
    assert Path(result["report_json"]).exists(), "JSON report not generated"
    assert Path(result["report_md"]).exists(), "Markdown report not generated"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_benchmark_quick(log_dir: Path, tmp_path: Path):
    """Quick E2E benchmark test (short duration).

    This is a faster version of the full test for CI/CD pipelines.
    Uses 3 QPS for 10 seconds = 30 tasks.
    """
    service_config = ServiceConfig(
        scheduler_port=19000,
        predictor_port=19002,
        instance_port_start=19100,
        log_dir=log_dir,
    )

    test_config = TestConfig(
        qps=3.0,
        duration_seconds=10.0,
        model_distribution={
            "sleep_model_a": 2,
            "sleep_model_b": 1,
            "sleep_model_c": 1,
        },
        output_dir=tmp_path / "quick_benchmark_results",
    )

    orchestrator = E2EBenchmarkOrchestrator(service_config, test_config)
    result = await orchestrator.run()

    assert result["success"], f"Quick benchmark failed: {result.get('error')}"

    # Basic sanity checks
    summary = result["summary"]
    assert summary["tasks_submitted"] >= 20, "Too few tasks submitted"
    assert summary["tasks_completed"] > 0, "No tasks completed"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_benchmark_single_model(log_dir: Path, tmp_path: Path):
    """E2E benchmark test with single model.

    Tests the system with just one model type to isolate model-specific issues.
    """
    service_config = ServiceConfig(
        scheduler_port=20000,
        predictor_port=20002,
        instance_port_start=20100,
        log_dir=log_dir,
    )

    test_config = TestConfig(
        qps=2.0,
        duration_seconds=10.0,
        model_ids=["sleep_model_single"],
        model_distribution={
            "sleep_model_single": 3,
        },
        output_dir=tmp_path / "single_model_results",
    )

    orchestrator = E2EBenchmarkOrchestrator(service_config, test_config)
    result = await orchestrator.run()

    assert result["success"], f"Single model benchmark failed: {result.get('error')}"

    # All tasks should be for the single model
    summary = result["summary"]
    assert summary["tasks_submitted"] >= 15, "Too few tasks submitted"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_benchmark_high_qps(log_dir: Path, tmp_path: Path):
    """E2E benchmark test with high QPS.

    Tests system behavior under higher load.
    Uses 20 QPS for 10 seconds = 200 tasks.
    """
    service_config = ServiceConfig(
        scheduler_port=21000,
        predictor_port=21002,
        instance_port_start=21100,
        log_dir=log_dir,
    )

    test_config = TestConfig(
        qps=20.0,
        duration_seconds=10.0,
        model_distribution={
            "sleep_model_a": 4,
            "sleep_model_b": 3,
            "sleep_model_c": 3,
        },
        output_dir=tmp_path / "high_qps_results",
    )

    orchestrator = E2EBenchmarkOrchestrator(service_config, test_config)
    result = await orchestrator.run()

    assert result["success"], f"High QPS benchmark failed: {result.get('error')}"

    # Check we handled the high load
    summary = result["summary"]
    assert summary["tasks_submitted"] >= 150, "Too few tasks submitted"

    # Some degradation is acceptable under high load
    if summary["tasks_completed"] > 0:
        success_rate = summary["tasks_completed"] / summary["tasks_submitted"]
        assert success_rate >= 0.8, f"Too low success rate under load: {success_rate:.2%}"
