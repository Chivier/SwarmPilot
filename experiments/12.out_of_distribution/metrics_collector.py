#!/usr/bin/env python3
"""
Metrics Collection System for Out-of-Distribution Experiments.

Collects and analyzes:
- Workflow completion latencies (mean, median, P95, P99)
- Makespan (total experiment duration)
- Statistical comparisons between baseline and comparison tests
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


@dataclass
class LatencyMetrics:
    """Latency statistics for workflows."""
    count: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float


@dataclass
class ExperimentMetrics:
    """Complete metrics for an experiment run."""
    experiment_name: str
    experiment_description: str
    timestamp: str

    # Timing
    makespan_seconds: float
    start_time: str
    end_time: str

    # Workflow latencies
    workflow_latencies: LatencyMetrics

    # Raw data
    workflow_completion_times: List[float]

    # Configuration
    num_workflows: int
    strategy: str


class MetricsCollector:
    """
    Collector for experiment metrics.

    Tracks workflow completion times and computes statistical metrics.
    """

    def __init__(self, experiment_name: str, experiment_description: str):
        """
        Initialize metrics collector.

        Args:
            experiment_name: Name of the experiment
            experiment_description: Description of experiment configuration
        """
        self.experiment_name = experiment_name
        self.experiment_description = experiment_description
        self.workflow_completion_times: List[float] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start_collection(self):
        """Mark the start of experiment data collection."""
        self.start_time = time.time()

    def end_collection(self):
        """Mark the end of experiment data collection."""
        self.end_time = time.time()

    def record_workflow_completion(self, latency_seconds: float):
        """
        Record a workflow completion time.

        Args:
            latency_seconds: Workflow completion latency in seconds
        """
        self.workflow_completion_times.append(latency_seconds)

    def compute_latency_metrics(self) -> LatencyMetrics:
        """
        Compute statistical metrics from collected workflow latencies.

        Returns:
            LatencyMetrics object with all statistics
        """
        if not self.workflow_completion_times:
            raise ValueError("No workflow completion times recorded")

        times = np.array(self.workflow_completion_times)

        return LatencyMetrics(
            count=len(times),
            mean=float(np.mean(times)),
            median=float(np.median(times)),
            std=float(np.std(times)),
            min=float(np.min(times)),
            max=float(np.max(times)),
            p50=float(np.percentile(times, 50)),
            p95=float(np.percentile(times, 95)),
            p99=float(np.percentile(times, 99))
        )

    def get_makespan(self) -> float:
        """
        Get experiment makespan (total duration).

        Returns:
            Makespan in seconds
        """
        if self.start_time is None or self.end_time is None:
            raise ValueError("Experiment timing not complete")
        return self.end_time - self.start_time

    def generate_metrics_report(
        self,
        num_workflows: int,
        strategy: str
    ) -> ExperimentMetrics:
        """
        Generate complete metrics report.

        Args:
            num_workflows: Number of workflows in experiment
            strategy: Scheduling strategy used

        Returns:
            ExperimentMetrics object with all collected data
        """
        if self.start_time is None or self.end_time is None:
            raise ValueError("Experiment timing not complete")

        latency_metrics = self.compute_latency_metrics()
        makespan = self.get_makespan()

        return ExperimentMetrics(
            experiment_name=self.experiment_name,
            experiment_description=self.experiment_description,
            timestamp=datetime.now().isoformat(),
            makespan_seconds=makespan,
            start_time=datetime.fromtimestamp(self.start_time).isoformat(),
            end_time=datetime.fromtimestamp(self.end_time).isoformat(),
            workflow_latencies=latency_metrics,
            workflow_completion_times=self.workflow_completion_times,
            num_workflows=num_workflows,
            strategy=strategy
        )

    def save_metrics(
        self,
        metrics: ExperimentMetrics,
        output_dir: Path
    ):
        """
        Save metrics to JSON file.

        Args:
            metrics: Metrics to save
            output_dir: Directory to save metrics file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{metrics.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)

        print(f"Metrics saved to: {filepath}")


class MetricsComparator:
    """Compare metrics between baseline and comparison experiments."""

    @staticmethod
    def compare_experiments(
        baseline: ExperimentMetrics,
        comparison: ExperimentMetrics
    ) -> Dict:
        """
        Compare two experiment results.

        Args:
            baseline: Baseline experiment metrics
            comparison: Comparison experiment metrics

        Returns:
            Dictionary with comparison statistics
        """
        baseline_latency = baseline.workflow_latencies
        comparison_latency = comparison.workflow_latencies

        # Calculate improvements/degradations
        mean_diff = comparison_latency.mean - baseline_latency.mean
        mean_pct_change = (mean_diff / baseline_latency.mean) * 100

        median_diff = comparison_latency.median - baseline_latency.median
        median_pct_change = (median_diff / baseline_latency.median) * 100

        p95_diff = comparison_latency.p95 - baseline_latency.p95
        p95_pct_change = (p95_diff / baseline_latency.p95) * 100

        p99_diff = comparison_latency.p99 - baseline_latency.p99
        p99_pct_change = (p99_diff / baseline_latency.p99) * 100

        makespan_diff = comparison.makespan_seconds - baseline.makespan_seconds
        makespan_pct_change = (makespan_diff / baseline.makespan_seconds) * 100

        return {
            "baseline_name": baseline.experiment_name,
            "comparison_name": comparison.experiment_name,
            "latency_comparison": {
                "mean": {
                    "baseline": baseline_latency.mean,
                    "comparison": comparison_latency.mean,
                    "diff": mean_diff,
                    "pct_change": mean_pct_change
                },
                "median": {
                    "baseline": baseline_latency.median,
                    "comparison": comparison_latency.median,
                    "diff": median_diff,
                    "pct_change": median_pct_change
                },
                "p95": {
                    "baseline": baseline_latency.p95,
                    "comparison": comparison_latency.p95,
                    "diff": p95_diff,
                    "pct_change": p95_pct_change
                },
                "p99": {
                    "baseline": baseline_latency.p99,
                    "comparison": comparison_latency.p99,
                    "diff": p99_diff,
                    "pct_change": p99_pct_change
                }
            },
            "makespan_comparison": {
                "baseline": baseline.makespan_seconds,
                "comparison": comparison.makespan_seconds,
                "diff": makespan_diff,
                "pct_change": makespan_pct_change
            }
        }

    @staticmethod
    def statistical_significance_test(
        baseline: ExperimentMetrics,
        comparison: ExperimentMetrics
    ) -> Dict:
        """
        Perform statistical significance tests between baseline and comparison.

        Uses t-test for normal distributions and Mann-Whitney U test as fallback.

        Args:
            baseline: Baseline experiment metrics
            comparison: Comparison experiment metrics

        Returns:
            Dictionary with test results including p-values and effect sizes
        """
        baseline_times = np.array(baseline.workflow_completion_times)
        comparison_times = np.array(comparison.workflow_completion_times)

        # T-test (assumes normal distribution)
        t_statistic, t_pvalue = stats.ttest_ind(baseline_times, comparison_times)

        # Mann-Whitney U test (non-parametric)
        u_statistic, u_pvalue = stats.mannwhitneyu(
            baseline_times,
            comparison_times,
            alternative='two-sided'
        )

        # Cohen's d effect size
        pooled_std = np.sqrt(
            ((len(baseline_times) - 1) * np.var(baseline_times, ddof=1) +
             (len(comparison_times) - 1) * np.var(comparison_times, ddof=1)) /
            (len(baseline_times) + len(comparison_times) - 2)
        )
        cohens_d = (np.mean(baseline_times) - np.mean(comparison_times)) / pooled_std

        # Confidence interval for mean difference
        mean_diff = np.mean(comparison_times) - np.mean(baseline_times)
        se_diff = np.sqrt(
            np.var(baseline_times, ddof=1) / len(baseline_times) +
            np.var(comparison_times, ddof=1) / len(comparison_times)
        )
        ci_95 = stats.t.interval(
            0.95,
            len(baseline_times) + len(comparison_times) - 2,
            loc=mean_diff,
            scale=se_diff
        )

        return {
            "t_test": {
                "statistic": float(t_statistic),
                "p_value": float(t_pvalue),
                "significant": t_pvalue < 0.05
            },
            "mann_whitney_u": {
                "statistic": float(u_statistic),
                "p_value": float(u_pvalue),
                "significant": u_pvalue < 0.05
            },
            "effect_size": {
                "cohens_d": float(cohens_d),
                "interpretation": (
                    "large" if abs(cohens_d) >= 0.8 else
                    "medium" if abs(cohens_d) >= 0.5 else
                    "small"
                )
            },
            "confidence_interval_95": {
                "lower": float(ci_95[0]),
                "upper": float(ci_95[1]),
                "mean_difference": float(mean_diff)
            }
        }

    @staticmethod
    def generate_visualization(
        baseline: ExperimentMetrics,
        comparison: ExperimentMetrics,
        output_dir: Path
    ):
        """
        Generate visualization plots comparing baseline and comparison experiments.

        Args:
            baseline: Baseline experiment metrics
            comparison: Comparison experiment metrics
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        baseline_times = np.array(baseline.workflow_completion_times)
        comparison_times = np.array(comparison.workflow_completion_times)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Comparison: {baseline.experiment_name} vs {comparison.experiment_name}",
                    fontsize=14, fontweight='bold')

        # 1. Box plot comparison
        ax1 = axes[0, 0]
        ax1.boxplot([baseline_times, comparison_times],
                   labels=['Baseline', 'Comparison'])
        ax1.set_ylabel('Latency (seconds)')
        ax1.set_title('Workflow Latency Distribution')
        ax1.grid(True, alpha=0.3)

        # 2. Histogram comparison
        ax2 = axes[0, 1]
        ax2.hist(baseline_times, bins=30, alpha=0.5, label='Baseline', color='blue')
        ax2.hist(comparison_times, bins=30, alpha=0.5, label='Comparison', color='orange')
        ax2.set_xlabel('Latency (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Latency Histogram')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Violin plot
        ax3 = axes[1, 0]
        parts = ax3.violinplot([baseline_times, comparison_times],
                              positions=[1, 2],
                              showmeans=True,
                              showmedians=True)
        ax3.set_xticks([1, 2])
        ax3.set_xticklabels(['Baseline', 'Comparison'])
        ax3.set_ylabel('Latency (seconds)')
        ax3.set_title('Violin Plot')
        ax3.grid(True, alpha=0.3)

        # 4. Percentile comparison
        ax4 = axes[1, 1]
        percentiles = [50, 75, 90, 95, 99]
        baseline_percentiles = [np.percentile(baseline_times, p) for p in percentiles]
        comparison_percentiles = [np.percentile(comparison_times, p) for p in percentiles]

        x = np.arange(len(percentiles))
        width = 0.35
        ax4.bar(x - width/2, baseline_percentiles, width, label='Baseline', color='blue', alpha=0.7)
        ax4.bar(x + width/2, comparison_percentiles, width, label='Comparison', color='orange', alpha=0.7)
        ax4.set_xlabel('Percentile')
        ax4.set_ylabel('Latency (seconds)')
        ax4.set_title('Percentile Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'P{p}' for p in percentiles])
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_filename = f"comparison_{baseline.experiment_name}_vs_{comparison.experiment_name}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Visualization saved: {plot_path}")

    @staticmethod
    def print_comparison(comparison: Dict):
        """Print comparison results in human-readable format."""
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPARISON")
        print("=" * 60)
        print(f"Baseline:   {comparison['baseline_name']}")
        print(f"Comparison: {comparison['comparison_name']}")
        print()

        print("Workflow Latency Comparison:")
        print("-" * 60)

        for metric in ["mean", "median", "p95", "p99"]:
            data = comparison["latency_comparison"][metric]
            sign = "+" if data["pct_change"] > 0 else ""
            print(f"{metric.upper()}:")
            print(f"  Baseline:   {data['baseline']:.3f}s")
            print(f"  Comparison: {data['comparison']:.3f}s")
            print(f"  Difference: {sign}{data['diff']:.3f}s ({sign}{data['pct_change']:.2f}%)")
            print()

        print("Makespan Comparison:")
        print("-" * 60)
        data = comparison["makespan_comparison"]
        sign = "+" if data["pct_change"] > 0 else ""
        print(f"  Baseline:   {data['baseline']:.3f}s")
        print(f"  Comparison: {data['comparison']:.3f}s")
        print(f"  Difference: {sign}{data['diff']:.3f}s ({sign}{data['pct_change']:.2f}%)")
        print("=" * 60)


if __name__ == "__main__":
    # Demo usage
    print("Metrics Collector Demo")
    print("=" * 60)

    # Simulate baseline experiment
    print("\nSimulating baseline experiment...")
    baseline_collector = MetricsCollector(
        "exp1_baseline",
        "Baseline: B1 sleep_time from A1"
    )

    baseline_collector.start_collection()
    # Simulate some workflow completions
    np.random.seed(42)
    for _ in range(100):
        latency = np.random.normal(10.0, 2.0)  # Mean 10s, std 2s
        baseline_collector.record_workflow_completion(latency)
    time.sleep(0.1)  # Simulate experiment duration
    baseline_collector.end_collection()

    baseline_metrics = baseline_collector.generate_metrics_report(
        num_workflows=100,
        strategy="probabilistic"
    )

    print(f"\nBaseline Metrics:")
    print(f"  Mean latency: {baseline_metrics.workflow_latencies.mean:.3f}s")
    print(f"  P95 latency:  {baseline_metrics.workflow_latencies.p95:.3f}s")
    print(f"  Makespan:     {baseline_metrics.makespan_seconds:.3f}s")

    # Simulate comparison experiment
    print("\nSimulating comparison experiment...")
    comparison_collector = MetricsCollector(
        "exp1_comparison",
        "Comparison: B1 exp_runtime = sleep_time"
    )

    comparison_collector.start_collection()
    # Simulate some workflow completions (slightly better performance)
    for _ in range(100):
        latency = np.random.normal(9.0, 1.8)  # Mean 9s, std 1.8s
        comparison_collector.record_workflow_completion(latency)
    time.sleep(0.09)  # Simulate shorter experiment duration
    comparison_collector.end_collection()

    comparison_metrics = comparison_collector.generate_metrics_report(
        num_workflows=100,
        strategy="probabilistic"
    )

    print(f"\nComparison Metrics:")
    print(f"  Mean latency: {comparison_metrics.workflow_latencies.mean:.3f}s")
    print(f"  P95 latency:  {comparison_metrics.workflow_latencies.p95:.3f}s")
    print(f"  Makespan:     {comparison_metrics.makespan_seconds:.3f}s")

    # Compare experiments
    comparison = MetricsComparator.compare_experiments(
        baseline_metrics,
        comparison_metrics
    )
    MetricsComparator.print_comparison(comparison)
