#!/usr/bin/env python3
"""
Quick functionality test for comparison experiment scripts.

This script verifies that:
1. Configuration generation works correctly
2. Result parsing works correctly (using sample data)
3. Analysis functions work correctly

Author: Claude Code
Date: 2025-11-03
"""

import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from run_comparison_experiments import ExperimentConfig
from analyze_comparison_results import ResultParser, ComparisonAnalyzer


def test_configuration_generation():
    """Test that all 6 configurations are generated correctly."""
    print("\n=== Testing Configuration Generation ===")

    strategies = ["min_time", "probabilistic", "round_robin"]
    migration_modes = [True, False]

    configs = []
    for strategy in strategies:
        for enable_migration in migration_modes:
            config = ExperimentConfig(
                strategy=strategy,
                enable_migration=enable_migration,
                num_workflows_per_phase=100,
                qps=2.0,
                total_instances=16,
                instance_start_port=8210,
            )
            configs.append(config)

    print(f"✓ Generated {len(configs)} configurations")

    # Verify we have exactly 6
    assert len(configs) == 6, f"Expected 6 configs, got {len(configs)}"

    # Verify all combinations exist
    expected_names = {
        "min_time_migration", "min_time_static",
        "probabilistic_migration", "probabilistic_static",
        "round_robin_migration", "round_robin_static",
    }

    actual_names = {config.get_name() for config in configs}
    assert actual_names == expected_names, f"Config names mismatch: {actual_names}"

    print("✓ All 6 configurations are correct:")
    for config in configs:
        print(f"  - {config.get_name()}")

    return True


def test_result_parser():
    """Test result parsing with sample data."""
    print("\n=== Testing Result Parser ===")

    # Create sample result data
    sample_result = {
        "experiment": "08_dynamic_migration",
        "timestamp": "2025-01-03T12:00:00",
        "configuration": {
            "strategy": "min_time",
            "migration_enabled": True,
            "qps": 2.0,
            "num_workflows_per_phase": 100,
        },
        "phases": [
            {
                "phase_id": 1,
                "fanout": 3,
                "num_workflows": 100,
                "completed_workflows": 100,
                "phase_duration_s": 50.0,
                "latency_stats": {
                    "avg_ms": 15000.0,
                    "min_ms": 10000.0,
                    "max_ms": 20000.0,
                    "p50_ms": 15000.0,
                    "p90_ms": 18000.0,
                    "p99_ms": 19500.0,
                },
                "all_latencies_ms": [15000.0] * 100,
            }
        ],
        "migrations": [
            {
                "total_migrations": 2,
                "completed": 2,
                "failed": 0,
                "total_duration_ms": 3000.0,
                "avg_drain_duration_ms": 1500.0,
            }
        ],
    }

    # Write to temp file
    temp_file = Path("/tmp/test_exp08_result.json")
    with open(temp_file, 'w') as f:
        json.dump(sample_result, f)

    # Parse it
    parsed = ResultParser.parse_file(temp_file)

    # Verify parsing
    assert parsed is not None, "Parsing failed"
    assert parsed.strategy == "min_time", f"Strategy mismatch: {parsed.strategy}"
    assert parsed.migration_enabled == True, "Migration enabled mismatch"
    assert len(parsed.phases) == 1, f"Expected 1 phase, got {len(parsed.phases)}"
    assert len(parsed.migrations) == 1, f"Expected 1 migration, got {len(parsed.migrations)}"

    print(f"✓ Successfully parsed result file")
    print(f"  - Strategy: {parsed.strategy}")
    print(f"  - Migration enabled: {parsed.migration_enabled}")
    print(f"  - Phases: {len(parsed.phases)}")
    print(f"  - Migrations: {len(parsed.migrations)}")

    # Cleanup
    temp_file.unlink()

    return True


def test_analyzer():
    """Test analyzer with sample experiments."""
    print("\n=== Testing Analyzer ===")

    # Create sample experiments
    from analyze_comparison_results import ExperimentData, PhaseStats, MigrationStats

    experiments = []

    # Create 2 sample experiments (migration vs static for min_time)
    for migration_enabled in [True, False]:
        exp = ExperimentData(
            strategy="min_time",
            migration_enabled=migration_enabled,
            timestamp="2025-01-03T12:00:00",
            num_workflows_per_phase=100,
            qps=2.0,
        )

        # Add phases
        for phase_id, fanout in [(1, 3), (2, 8), (3, 1)]:
            phase = PhaseStats(
                phase_id=phase_id,
                fanout=fanout,
                num_workflows=100,
                completed_workflows=100,
                phase_duration_s=50.0,
                avg_ms=15000.0 if migration_enabled else 18000.0,  # Migration is better
                min_ms=10000.0,
                max_ms=20000.0,
                p50_ms=15000.0,
                p90_ms=18000.0,
                p99_ms=19500.0,
                std_dev_ms=2000.0,
                coefficient_of_variation=13.3,
            )
            exp.phases.append(phase)

        # Add migrations (only for migration-enabled)
        if migration_enabled:
            for i in range(2):
                migration = MigrationStats(
                    from_phase=i + 1,
                    to_phase=i + 2,
                    total_migrations=2,
                    completed=2,
                    failed=0,
                    total_duration_ms=3000.0,
                )
                exp.migrations.append(migration)

        experiments.append(exp)

    # Create analyzer
    analyzer = ComparisonAnalyzer(experiments)

    # Test latency comparison
    latency = analyzer.analyze_latency_comparison()
    assert "by_phase" in latency, "Missing by_phase in latency comparison"
    assert "overall_summary" in latency, "Missing overall_summary in latency comparison"
    print(f"✓ Latency comparison analysis works")

    # Test migration overhead
    overhead = analyzer.analyze_migration_overhead()
    assert "by_experiment" in overhead, "Missing by_experiment in overhead"
    print(f"✓ Migration overhead analysis works")

    # Test throughput
    throughput = analyzer.analyze_throughput()
    assert "by_experiment" in throughput, "Missing by_experiment in throughput"
    print(f"✓ Throughput analysis works")

    # Test load balancing
    load_balance = analyzer.analyze_load_balancing()
    assert "by_experiment" in load_balance, "Missing by_experiment in load_balance"
    print(f"✓ Load balancing analysis works")

    # Test migration impact
    impact = analyzer.compare_migration_impact()
    assert "by_strategy" in impact, "Missing by_strategy in impact"
    assert "min_time" in impact["by_strategy"], "Missing min_time in impact"
    print(f"✓ Migration impact comparison works")

    # Get overall improvement
    min_time_impact = impact["by_strategy"]["min_time"]
    improvement = min_time_impact["overall_improvement_percent"]
    print(f"  - Migration improvement: {improvement:.1f}%")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Testing Comparison Experiment Scripts")
    print("="*70)

    tests = [
        ("Configuration Generation", test_configuration_generation),
        ("Result Parser", test_result_parser),
        ("Analyzer", test_analyzer),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n✗ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*70)

    if failed == 0:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
