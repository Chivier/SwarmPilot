#!/usr/bin/env python3
"""
Analyze Comparison Results

This script analyzes results from all 6 experiment configurations and generates
comprehensive comparison reports including:
- Workflow latency statistics
- Migration overhead analysis
- Throughput comparison
- Load balancing effectiveness

Author: Claude Code
Date: 2025-11-03
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PhaseStats:
    """Statistics for a single phase."""
    phase_id: int
    fanout: int
    num_workflows: int
    completed_workflows: int
    phase_duration_s: float
    avg_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    std_dev_ms: float = 0.0
    coefficient_of_variation: float = 0.0


@dataclass
class MigrationStats:
    """Statistics for a migration event."""
    from_phase: int
    to_phase: int
    total_migrations: int
    completed: int
    failed: int
    total_duration_ms: float
    avg_drain_duration_ms: Optional[float] = None
    avg_total_duration_ms: Optional[float] = None


@dataclass
class ExperimentData:
    """Parsed data from a single experiment."""
    strategy: str
    migration_enabled: bool
    timestamp: str
    num_workflows_per_phase: int
    qps: float

    phases: List[PhaseStats] = field(default_factory=list)
    migrations: List[MigrationStats] = field(default_factory=list)

    def get_name(self) -> str:
        """Get experiment name."""
        mode = "migration" if self.migration_enabled else "static"
        return f"{self.strategy}_{mode}"


# ============================================================================
# Result Parser
# ============================================================================

class ResultParser:
    """Parses experiment result JSON files."""

    @staticmethod
    def parse_file(filepath: Path) -> Optional[ExperimentData]:
        """Parse a single result JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            config = data.get("configuration", {})
            experiment = ExperimentData(
                strategy=config.get("strategy", "unknown"),
                migration_enabled=config.get("migration_enabled", False),
                timestamp=data.get("timestamp", "unknown"),
                num_workflows_per_phase=config.get("num_workflows_per_phase", 0),
                qps=config.get("qps", 0.0),
            )

            # Parse phases
            for phase_data in data.get("phases", []):
                latency_stats = phase_data.get("latency_stats", {})
                all_latencies = phase_data.get("all_latencies_ms", [])

                # Calculate additional statistics
                std_dev = np.std(all_latencies) if all_latencies else 0.0
                avg = latency_stats.get("avg_ms", 0.0)
                cv = (std_dev / avg * 100) if avg > 0 else 0.0

                phase_stats = PhaseStats(
                    phase_id=phase_data.get("phase_id", 0),
                    fanout=phase_data.get("fanout", 0),
                    num_workflows=phase_data.get("num_workflows", 0),
                    completed_workflows=phase_data.get("completed_workflows", 0),
                    phase_duration_s=phase_data.get("phase_duration_s", 0.0),
                    avg_ms=latency_stats.get("avg_ms", 0.0),
                    min_ms=latency_stats.get("min_ms", 0.0),
                    max_ms=latency_stats.get("max_ms", 0.0),
                    p50_ms=latency_stats.get("p50_ms", 0.0),
                    p90_ms=latency_stats.get("p90_ms", 0.0),
                    p99_ms=latency_stats.get("p99_ms", 0.0),
                    std_dev_ms=std_dev,
                    coefficient_of_variation=cv,
                )
                experiment.phases.append(phase_stats)

            # Parse migrations
            for idx, migration_data in enumerate(data.get("migrations", [])):
                migration_stats = MigrationStats(
                    from_phase=idx + 1,
                    to_phase=idx + 2,
                    total_migrations=migration_data.get("total_migrations", 0),
                    completed=migration_data.get("completed", 0),
                    failed=migration_data.get("failed", 0),
                    total_duration_ms=migration_data.get("total_duration_ms", 0.0),
                    avg_drain_duration_ms=migration_data.get("avg_drain_duration_ms"),
                    avg_total_duration_ms=migration_data.get("avg_total_duration_ms"),
                )
                experiment.migrations.append(migration_stats)

            return experiment

        except Exception as e:
            logger.error(f"Failed to parse {filepath}: {e}")
            return None


# ============================================================================
# Comparison Analyzer
# ============================================================================

class ComparisonAnalyzer:
    """Analyzes and compares multiple experiment results."""

    def __init__(self, experiments: List[ExperimentData]):
        self.experiments = experiments

        # Group experiments by strategy and mode
        self.by_strategy: Dict[str, List[ExperimentData]] = defaultdict(list)
        self.by_mode: Dict[bool, List[ExperimentData]] = defaultdict(list)

        for exp in experiments:
            self.by_strategy[exp.strategy].append(exp)
            self.by_mode[exp.migration_enabled].append(exp)

    def analyze_latency_comparison(self) -> Dict:
        """Compare workflow latency across all configurations."""
        logger.info("Analyzing workflow latency comparison...")

        comparison = {
            "by_phase": {},
            "overall_summary": []
        }

        # Analyze each phase separately
        for phase_id in [1, 2, 3]:
            phase_comparison = []

            for exp in self.experiments:
                phase_stats = next((p for p in exp.phases if p.phase_id == phase_id), None)
                if phase_stats:
                    phase_comparison.append({
                        "name": exp.get_name(),
                        "strategy": exp.strategy,
                        "migration_enabled": exp.migration_enabled,
                        "avg_ms": phase_stats.avg_ms,
                        "p50_ms": phase_stats.p50_ms,
                        "p90_ms": phase_stats.p90_ms,
                        "p99_ms": phase_stats.p99_ms,
                        "min_ms": phase_stats.min_ms,
                        "max_ms": phase_stats.max_ms,
                        "std_dev_ms": phase_stats.std_dev_ms,
                        "cv_percent": phase_stats.coefficient_of_variation,
                    })

            comparison["by_phase"][f"phase_{phase_id}"] = phase_comparison

        # Overall summary (average across all phases)
        for exp in self.experiments:
            if exp.phases:
                avg_latency = np.mean([p.avg_ms for p in exp.phases])
                avg_p90 = np.mean([p.p90_ms for p in exp.phases])
                avg_p99 = np.mean([p.p99_ms for p in exp.phases])

                comparison["overall_summary"].append({
                    "name": exp.get_name(),
                    "strategy": exp.strategy,
                    "migration_enabled": exp.migration_enabled,
                    "avg_latency_ms": avg_latency,
                    "avg_p90_ms": avg_p90,
                    "avg_p99_ms": avg_p99,
                })

        return comparison

    def analyze_migration_overhead(self) -> Dict:
        """Analyze migration overhead for migration-enabled experiments."""
        logger.info("Analyzing migration overhead...")

        overhead_analysis = {
            "by_experiment": [],
            "summary": {}
        }

        migration_exps = [exp for exp in self.experiments if exp.migration_enabled]

        for exp in migration_exps:
            exp_data = {
                "name": exp.get_name(),
                "strategy": exp.strategy,
                "migrations": []
            }

            total_migration_time = 0.0
            for migration in exp.migrations:
                migration_data = {
                    "from_phase": migration.from_phase,
                    "to_phase": migration.to_phase,
                    "total_migrations": migration.total_migrations,
                    "duration_ms": migration.total_duration_ms,
                    "avg_drain_ms": migration.avg_drain_duration_ms,
                    "avg_total_ms": migration.avg_total_duration_ms,
                }
                exp_data["migrations"].append(migration_data)
                total_migration_time += migration.total_duration_ms

            exp_data["total_migration_time_ms"] = total_migration_time
            overhead_analysis["by_experiment"].append(exp_data)

        # Summary statistics
        if migration_exps:
            all_migration_times = [sum(m.total_duration_ms for m in exp.migrations) for exp in migration_exps]
            overhead_analysis["summary"] = {
                "avg_total_migration_time_ms": np.mean(all_migration_times),
                "min_total_migration_time_ms": np.min(all_migration_times),
                "max_total_migration_time_ms": np.max(all_migration_times),
            }

        return overhead_analysis

    def analyze_throughput(self) -> Dict:
        """Analyze throughput metrics."""
        logger.info("Analyzing throughput...")

        throughput_analysis = {
            "by_experiment": [],
            "by_phase": {}
        }

        for exp in self.experiments:
            exp_data = {
                "name": exp.get_name(),
                "strategy": exp.strategy,
                "migration_enabled": exp.migration_enabled,
                "phases": []
            }

            for phase in exp.phases:
                completion_rate = (phase.completed_workflows / phase.num_workflows * 100) if phase.num_workflows > 0 else 0.0
                actual_qps = phase.completed_workflows / phase.phase_duration_s if phase.phase_duration_s > 0 else 0.0

                phase_data = {
                    "phase_id": phase.phase_id,
                    "completion_rate_percent": completion_rate,
                    "actual_qps": actual_qps,
                    "phase_duration_s": phase.phase_duration_s,
                }
                exp_data["phases"].append(phase_data)

            throughput_analysis["by_experiment"].append(exp_data)

        return throughput_analysis

    def analyze_load_balancing(self) -> Dict:
        """Analyze load balancing effectiveness using latency distribution."""
        logger.info("Analyzing load balancing effectiveness...")

        load_balance_analysis = {
            "by_experiment": []
        }

        for exp in self.experiments:
            exp_data = {
                "name": exp.get_name(),
                "strategy": exp.strategy,
                "migration_enabled": exp.migration_enabled,
                "phases": []
            }

            for phase in exp.phases:
                phase_data = {
                    "phase_id": phase.phase_id,
                    "std_dev_ms": phase.std_dev_ms,
                    "coefficient_of_variation": phase.coefficient_of_variation,
                    "latency_range_ms": phase.max_ms - phase.min_ms,
                }
                exp_data["phases"].append(phase_data)

            # Calculate average CV across phases
            avg_cv = np.mean([p.coefficient_of_variation for p in exp.phases]) if exp.phases else 0.0
            exp_data["avg_cv_percent"] = avg_cv

            load_balance_analysis["by_experiment"].append(exp_data)

        return load_balance_analysis

    def compare_migration_impact(self) -> Dict:
        """Compare impact of migration vs static for each strategy."""
        logger.info("Comparing migration impact...")

        impact_analysis = {
            "by_strategy": {}
        }

        for strategy in self.by_strategy.keys():
            strategy_exps = self.by_strategy[strategy]

            migration_exp = next((e for e in strategy_exps if e.migration_enabled), None)
            static_exp = next((e for e in strategy_exps if not e.migration_enabled), None)

            if migration_exp and static_exp:
                # Compare each phase
                phase_comparisons = []
                for phase_id in [1, 2, 3]:
                    migration_phase = next((p for p in migration_exp.phases if p.phase_id == phase_id), None)
                    static_phase = next((p for p in static_exp.phases if p.phase_id == phase_id), None)

                    if migration_phase and static_phase:
                        improvement_percent = ((static_phase.avg_ms - migration_phase.avg_ms) / static_phase.avg_ms * 100) if static_phase.avg_ms > 0 else 0.0

                        phase_comparisons.append({
                            "phase_id": phase_id,
                            "migration_avg_ms": migration_phase.avg_ms,
                            "static_avg_ms": static_phase.avg_ms,
                            "improvement_percent": improvement_percent,
                            "migration_p90_ms": migration_phase.p90_ms,
                            "static_p90_ms": static_phase.p90_ms,
                        })

                # Overall comparison
                migration_avg = np.mean([p.avg_ms for p in migration_exp.phases]) if migration_exp.phases else 0.0
                static_avg = np.mean([p.avg_ms for p in static_exp.phases]) if static_exp.phases else 0.0
                overall_improvement = ((static_avg - migration_avg) / static_avg * 100) if static_avg > 0 else 0.0

                impact_analysis["by_strategy"][strategy] = {
                    "phases": phase_comparisons,
                    "overall_migration_avg_ms": migration_avg,
                    "overall_static_avg_ms": static_avg,
                    "overall_improvement_percent": overall_improvement,
                }

        return impact_analysis

    def generate_analysis(self) -> Dict:
        """Generate complete analysis."""
        logger.info(f"\nAnalyzing {len(self.experiments)} experiments...")

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "num_experiments": len(self.experiments),
            "experiments_analyzed": [exp.get_name() for exp in self.experiments],
            "latency_comparison": self.analyze_latency_comparison(),
            "migration_overhead": self.analyze_migration_overhead(),
            "throughput": self.analyze_throughput(),
            "load_balancing": self.analyze_load_balancing(),
            "migration_impact": self.compare_migration_impact(),
        }

        return analysis


# ============================================================================
# Report Generator
# ============================================================================

class ReportGenerator:
    """Generates formatted reports from analysis."""

    @staticmethod
    def generate_markdown_report(analysis: Dict, output_file: Path):
        """Generate a Markdown report."""
        logger.info(f"Generating Markdown report: {output_file}")

        with open(output_file, 'w') as f:
            f.write("# Experiment 08: Comparison Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Experiments Analyzed:** {analysis['num_experiments']}\n\n")

            # Table of contents
            f.write("## Table of Contents\n\n")
            f.write("1. [Workflow Latency Comparison](#workflow-latency-comparison)\n")
            f.write("2. [Migration Overhead Analysis](#migration-overhead-analysis)\n")
            f.write("3. [Throughput Comparison](#throughput-comparison)\n")
            f.write("4. [Load Balancing Effectiveness](#load-balancing-effectiveness)\n")
            f.write("5. [Migration Impact](#migration-impact)\n\n")

            f.write("---\n\n")

            # 1. Latency Comparison
            f.write("## Workflow Latency Comparison\n\n")

            for phase_key, phase_data in analysis["latency_comparison"]["by_phase"].items():
                phase_id = phase_key.split("_")[1]
                f.write(f"### Phase {phase_id}\n\n")

                f.write("| Configuration | Strategy | Migration | Avg (ms) | P50 (ms) | P90 (ms) | P99 (ms) | Std Dev | CV (%) |\n")
                f.write("|---------------|----------|-----------|----------|----------|----------|----------|---------|--------|\n")

                for exp in sorted(phase_data, key=lambda x: x["avg_ms"]):
                    mode = "✓" if exp["migration_enabled"] else "✗"
                    f.write(f"| {exp['name']:25s} | {exp['strategy']:12s} | {mode:9s} | "
                           f"{exp['avg_ms']:8.1f} | {exp['p50_ms']:8.1f} | {exp['p90_ms']:8.1f} | "
                           f"{exp['p99_ms']:8.1f} | {exp['std_dev_ms']:7.1f} | {exp['cv_percent']:6.1f} |\n")

                f.write("\n")

            # Overall summary
            f.write("### Overall Summary (Average Across All Phases)\n\n")
            f.write("| Configuration | Strategy | Migration | Avg Latency (ms) | Avg P90 (ms) | Avg P99 (ms) |\n")
            f.write("|---------------|----------|-----------|------------------|--------------|-------------|\n")

            for exp in sorted(analysis["latency_comparison"]["overall_summary"], key=lambda x: x["avg_latency_ms"]):
                mode = "✓" if exp["migration_enabled"] else "✗"
                f.write(f"| {exp['name']:25s} | {exp['strategy']:12s} | {mode:9s} | "
                       f"{exp['avg_latency_ms']:16.1f} | {exp['avg_p90_ms']:12.1f} | {exp['avg_p99_ms']:11.1f} |\n")

            f.write("\n---\n\n")

            # 2. Migration Overhead
            f.write("## Migration Overhead Analysis\n\n")

            if analysis["migration_overhead"]["by_experiment"]:
                f.write("### Migration Time by Experiment\n\n")
                f.write("| Configuration | Strategy | Migration 1→2 (ms) | Migration 2→3 (ms) | Total (ms) |\n")
                f.write("|---------------|----------|--------------------|--------------------|-----------|\n")

                for exp in analysis["migration_overhead"]["by_experiment"]:
                    m1 = exp["migrations"][0]["duration_ms"] if len(exp["migrations"]) > 0 else 0
                    m2 = exp["migrations"][1]["duration_ms"] if len(exp["migrations"]) > 1 else 0
                    total = exp["total_migration_time_ms"]
                    f.write(f"| {exp['name']:25s} | {exp['strategy']:12s} | {m1:18.1f} | {m2:18.1f} | {total:9.1f} |\n")

                f.write("\n")

                summary = analysis["migration_overhead"]["summary"]
                f.write(f"**Average Total Migration Time:** {summary['avg_total_migration_time_ms']:.1f} ms\n\n")
                f.write(f"**Min Total Migration Time:** {summary['min_total_migration_time_ms']:.1f} ms\n\n")
                f.write(f"**Max Total Migration Time:** {summary['max_total_migration_time_ms']:.1f} ms\n\n")
            else:
                f.write("*No migration data available (all experiments ran in static mode)*\n\n")

            f.write("---\n\n")

            # 3. Throughput
            f.write("## Throughput Comparison\n\n")

            for phase_id in [1, 2, 3]:
                f.write(f"### Phase {phase_id}\n\n")
                f.write("| Configuration | Strategy | Migration | Completion (%) | Actual QPS | Duration (s) |\n")
                f.write("|---------------|----------|-----------|----------------|------------|-------------|\n")

                for exp in analysis["throughput"]["by_experiment"]:
                    phase = next((p for p in exp["phases"] if p["phase_id"] == phase_id), None)
                    if phase:
                        mode = "✓" if exp["migration_enabled"] else "✗"
                        f.write(f"| {exp['name']:25s} | {exp['strategy']:12s} | {mode:9s} | "
                               f"{phase['completion_rate_percent']:14.1f} | {phase['actual_qps']:10.2f} | {phase['phase_duration_s']:11.1f} |\n")

                f.write("\n")

            f.write("---\n\n")

            # 4. Load Balancing
            f.write("## Load Balancing Effectiveness\n\n")
            f.write("*Lower Coefficient of Variation (CV) indicates better load balancing*\n\n")

            f.write("| Configuration | Strategy | Migration | Avg CV (%) | Phase 1 CV | Phase 2 CV | Phase 3 CV |\n")
            f.write("|---------------|----------|-----------|------------|------------|------------|------------|\n")

            for exp in sorted(analysis["load_balancing"]["by_experiment"], key=lambda x: x["avg_cv_percent"]):
                mode = "✓" if exp["migration_enabled"] else "✗"
                cv1 = exp["phases"][0]["coefficient_of_variation"] if len(exp["phases"]) > 0 else 0
                cv2 = exp["phases"][1]["coefficient_of_variation"] if len(exp["phases"]) > 1 else 0
                cv3 = exp["phases"][2]["coefficient_of_variation"] if len(exp["phases"]) > 2 else 0

                f.write(f"| {exp['name']:25s} | {exp['strategy']:12s} | {mode:9s} | "
                       f"{exp['avg_cv_percent']:10.1f} | {cv1:10.1f} | {cv2:10.1f} | {cv3:10.1f} |\n")

            f.write("\n---\n\n")

            # 5. Migration Impact
            f.write("## Migration Impact\n\n")
            f.write("*Comparison of migration vs static for each strategy. Positive improvement means migration performed better.*\n\n")

            for strategy, data in analysis["migration_impact"]["by_strategy"].items():
                f.write(f"### Strategy: {strategy}\n\n")

                f.write("| Phase | Migration Avg (ms) | Static Avg (ms) | Improvement (%) |\n")
                f.write("|-------|--------------------|-----------------|-----------------|\n")

                for phase in data["phases"]:
                    f.write(f"| {phase['phase_id']:5d} | {phase['migration_avg_ms']:18.1f} | "
                           f"{phase['static_avg_ms']:15.1f} | {phase['improvement_percent']:15.1f} |\n")

                f.write(f"\n**Overall Improvement:** {data['overall_improvement_percent']:.1f}%\n\n")

            f.write("---\n\n")
            f.write("*End of Report*\n")

        logger.info(f"✓ Markdown report generated: {output_file}")

    @staticmethod
    def generate_json_report(analysis: Dict, output_file: Path):
        """Generate JSON report."""
        logger.info(f"Generating JSON report: {output_file}")

        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"✓ JSON report generated: {output_file}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare experiment results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing result JSON files (default: results)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="exp08_*.json",
        help="Glob pattern for result files (default: exp08_*.json)"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="comparison_analysis",
        help="Prefix for output files (default: comparison_analysis)"
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Find result files
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    result_files = list(results_dir.glob(args.pattern))
    if not result_files:
        logger.error(f"No result files found matching pattern: {args.pattern}")
        sys.exit(1)

    logger.info(f"Found {len(result_files)} result files")

    # Parse all experiments
    experiments = []
    for filepath in result_files:
        logger.info(f"Parsing: {filepath.name}")
        exp = ResultParser.parse_file(filepath)
        if exp:
            experiments.append(exp)

    if not experiments:
        logger.error("No experiments successfully parsed")
        sys.exit(1)

    logger.info(f"Successfully parsed {len(experiments)} experiments")

    # Analyze
    analyzer = ComparisonAnalyzer(experiments)
    analysis = analyzer.generate_analysis()

    # Generate reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_output = results_dir / f"{args.output_prefix}_{timestamp}.json"
    md_output = results_dir / f"{args.output_prefix}_{timestamp}.md"

    ReportGenerator.generate_json_report(analysis, json_output)
    ReportGenerator.generate_markdown_report(analysis, md_output)

    logger.info("\n✅ Analysis complete!")
    logger.info(f"  JSON report: {json_output}")
    logger.info(f"  Markdown report: {md_output}")


if __name__ == "__main__":
    main()
