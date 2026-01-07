"""Report Generator for E2E Testing.

This module generates comprehensive test reports in JSON and Markdown formats.
It includes system configuration, execution metrics, latency statistics,
error analysis, and problem identification.

Features:
- JSON and Markdown report formats
- Latency percentile calculations
- Error analysis and categorization
- Problem detection and recommendations
- Log file location tracking

Usage:
    from report_generator import ReportGenerator, TestReport

    generator = ReportGenerator(
        workload_result=workload_result,
        task_results=task_results,
        config=test_config,
        log_paths=log_paths,
    )
    report = generator.generate_report()
    report.to_json("report.json")
    report.to_markdown("report.md")
"""

import json
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from .workload_generator import WorkloadResult, SubmissionResult


@dataclass
class LatencyStats:
    """Latency statistics in milliseconds."""

    p50: float
    p90: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float
    stddev: float
    count: int

    @classmethod
    def from_samples(cls, samples: list[float]) -> "LatencyStats":
        """Calculate statistics from a list of samples.

        Args:
            samples: List of latency values in ms

        Returns:
            LatencyStats instance
        """
        if not samples:
            return cls(
                p50=0, p90=0, p95=0, p99=0,
                mean=0, min=0, max=0, stddev=0, count=0,
            )

        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        def percentile(p: float) -> float:
            k = (n - 1) * p
            f = int(k)
            c = f + 1 if f + 1 < n else f
            return sorted_samples[f] + (sorted_samples[c] - sorted_samples[f]) * (k - f)

        return cls(
            p50=percentile(0.5),
            p90=percentile(0.9),
            p95=percentile(0.95),
            p99=percentile(0.99),
            mean=statistics.mean(samples),
            min=min(samples),
            max=max(samples),
            stddev=statistics.stdev(samples) if len(samples) > 1 else 0,
            count=n,
        )


@dataclass
class SystemConfig:
    """System configuration for the test."""

    scheduler_url: str
    planner_url: str
    predictor_url: str
    pylet_head_url: str
    num_workers: int
    scheduling_strategy: str
    model_distribution: dict[str, int]
    total_instances: int


@dataclass
class TestParams:
    """Test parameters."""

    target_qps: float
    duration_seconds: float
    total_tasks: int
    sleep_time_range: tuple[float, float]
    model_ids: list[str]


@dataclass
class ExecutionMetrics:
    """Execution metrics from the test."""

    # Submission metrics
    tasks_submitted: int
    successful_submissions: int
    failed_submissions: int
    submission_success_rate: float

    # Completion metrics
    tasks_completed: int
    tasks_failed: int
    tasks_pending: int
    completion_success_rate: float

    # Timing metrics
    actual_duration_seconds: float
    actual_qps: float

    # Per-model breakdown
    model_stats: dict[str, dict[str, int]]


@dataclass
class ErrorInfo:
    """Information about an error occurrence."""

    task_id: str
    model_id: str
    error_type: str
    error_message: str
    timestamp: float | None = None
    instance_id: str | None = None


@dataclass
class TestReport:
    """Comprehensive test report."""

    # Metadata
    report_name: str
    report_version: str
    generated_at: str

    # Configuration
    system_config: SystemConfig
    test_params: TestParams

    # Metrics
    execution_metrics: ExecutionMetrics

    # Latency
    submission_latency: LatencyStats
    execution_latency: LatencyStats | None

    # Errors
    submission_errors: list[ErrorInfo]
    execution_errors: list[ErrorInfo]

    # Problems
    problems: list[str]
    recommendations: list[str]

    # Logs
    log_files: dict[str, str]

    # Raw data (optional)
    raw_submission_results: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return asdict(self)

    def to_json(self, path: str | Path | None = None) -> str:
        """Generate JSON report.

        Args:
            path: Optional path to write JSON file

        Returns:
            JSON string
        """
        report_dict = self.to_dict()

        # Custom JSON encoder for non-serializable types
        def default(obj):
            if isinstance(obj, Path):
                return str(obj)
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        json_str = json.dumps(report_dict, indent=2, default=default)

        if path:
            Path(path).write_text(json_str)
            logger.info(f"JSON report written to {path}")

        return json_str

    def to_markdown(self, path: str | Path | None = None) -> str:
        """Generate Markdown report.

        Args:
            path: Optional path to write Markdown file

        Returns:
            Markdown string
        """
        md_lines = [
            f"# {self.report_name}",
            "",
            f"**Generated:** {self.generated_at}",
            f"**Version:** {self.report_version}",
            "",
            "---",
            "",
            "## System Configuration",
            "",
            "| Component | Value |",
            "|-----------|-------|",
            f"| Scheduler URL | `{self.system_config.scheduler_url}` |",
            f"| Planner URL | `{self.system_config.planner_url}` |",
            f"| Predictor URL | `{self.system_config.predictor_url}` |",
            f"| PyLet Head URL | `{self.system_config.pylet_head_url}` |",
            f"| Workers | {self.system_config.num_workers} |",
            f"| Strategy | {self.system_config.scheduling_strategy} |",
            f"| Total Instances | {self.system_config.total_instances} |",
            "",
            "### Model Distribution",
            "",
            "| Model | Instances |",
            "|-------|-----------|",
        ]

        for model, count in self.system_config.model_distribution.items():
            md_lines.append(f"| {model} | {count} |")

        md_lines.extend([
            "",
            "---",
            "",
            "## Test Parameters",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Target QPS | {self.test_params.target_qps} |",
            f"| Duration | {self.test_params.duration_seconds}s |",
            f"| Total Tasks | {self.test_params.total_tasks} |",
            f"| Sleep Time Range | {self.test_params.sleep_time_range[0]}s - {self.test_params.sleep_time_range[1]}s |",
            "",
            "---",
            "",
            "## Execution Metrics",
            "",
            "### Submissions",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Submitted | {self.execution_metrics.tasks_submitted} |",
            f"| Successful | {self.execution_metrics.successful_submissions} |",
            f"| Failed | {self.execution_metrics.failed_submissions} |",
            f"| Success Rate | {self.execution_metrics.submission_success_rate:.2%} |",
            "",
            "### Completions",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Completed | {self.execution_metrics.tasks_completed} |",
            f"| Failed | {self.execution_metrics.tasks_failed} |",
            f"| Pending | {self.execution_metrics.tasks_pending} |",
            f"| Success Rate | {self.execution_metrics.completion_success_rate:.2%} |",
            "",
            "### Throughput",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Actual Duration | {self.execution_metrics.actual_duration_seconds:.2f}s |",
            f"| Actual QPS | {self.execution_metrics.actual_qps:.2f} |",
            "",
            "---",
            "",
            "## Latency Statistics (ms)",
            "",
            "### Submission Latency",
            "",
            "| Percentile | Value |",
            "|------------|-------|",
            f"| p50 | {self.submission_latency.p50:.2f} |",
            f"| p90 | {self.submission_latency.p90:.2f} |",
            f"| p95 | {self.submission_latency.p95:.2f} |",
            f"| p99 | {self.submission_latency.p99:.2f} |",
            f"| mean | {self.submission_latency.mean:.2f} |",
            f"| min | {self.submission_latency.min:.2f} |",
            f"| max | {self.submission_latency.max:.2f} |",
            f"| stddev | {self.submission_latency.stddev:.2f} |",
            "",
        ])

        if self.execution_latency:
            md_lines.extend([
                "### Execution Latency",
                "",
                "| Percentile | Value |",
                "|------------|-------|",
                f"| p50 | {self.execution_latency.p50:.2f} |",
                f"| p90 | {self.execution_latency.p90:.2f} |",
                f"| p95 | {self.execution_latency.p95:.2f} |",
                f"| p99 | {self.execution_latency.p99:.2f} |",
                f"| mean | {self.execution_latency.mean:.2f} |",
                f"| min | {self.execution_latency.min:.2f} |",
                f"| max | {self.execution_latency.max:.2f} |",
                f"| stddev | {self.execution_latency.stddev:.2f} |",
                "",
            ])

        md_lines.extend([
            "---",
            "",
            "## Error Analysis",
            "",
        ])

        if self.submission_errors:
            md_lines.extend([
                "### Submission Errors",
                "",
                "| Task ID | Model | Error Type | Message |",
                "|---------|-------|------------|---------|",
            ])
            for err in self.submission_errors[:20]:  # Limit to first 20
                md_lines.append(
                    f"| {err.task_id} | {err.model_id} | {err.error_type} | {err.error_message[:50]}... |"
                )
            if len(self.submission_errors) > 20:
                md_lines.append(f"| ... | ... | ... | ({len(self.submission_errors) - 20} more errors) |")
            md_lines.append("")
        else:
            md_lines.extend(["*No submission errors*", ""])

        if self.execution_errors:
            md_lines.extend([
                "### Execution Errors",
                "",
                "| Task ID | Model | Instance | Error |",
                "|---------|-------|----------|-------|",
            ])
            for err in self.execution_errors[:20]:
                md_lines.append(
                    f"| {err.task_id} | {err.model_id} | {err.instance_id or 'N/A'} | {err.error_message[:50]}... |"
                )
            if len(self.execution_errors) > 20:
                md_lines.append(f"| ... | ... | ... | ({len(self.execution_errors) - 20} more errors) |")
            md_lines.append("")
        else:
            md_lines.extend(["*No execution errors*", ""])

        md_lines.extend([
            "---",
            "",
            "## Problems Identified",
            "",
        ])

        if self.problems:
            for i, problem in enumerate(self.problems, 1):
                md_lines.append(f"{i}. {problem}")
            md_lines.append("")
        else:
            md_lines.extend(["*No problems identified*", ""])

        if self.recommendations:
            md_lines.extend([
                "### Recommendations",
                "",
            ])
            for rec in self.recommendations:
                md_lines.append(f"- {rec}")
            md_lines.append("")

        md_lines.extend([
            "---",
            "",
            "## Log Files",
            "",
            "| Component | Path |",
            "|-----------|------|",
        ])

        for component, log_path in self.log_files.items():
            md_lines.append(f"| {component} | `{log_path}` |")

        md_lines.extend([
            "",
            "---",
            "",
            "*Report generated by E2E PyLet Benchmark*",
        ])

        md_str = "\n".join(md_lines)

        if path:
            Path(path).write_text(md_str)
            logger.info(f"Markdown report written to {path}")

        return md_str


class ReportGenerator:
    """Generate test reports from workload and task results."""

    def __init__(
        self,
        workload_result: WorkloadResult,
        task_results: dict[str, dict[str, Any]],
        system_config: SystemConfig,
        test_params: TestParams,
        log_paths: dict[str, str | Path],
    ):
        """Initialize report generator.

        Args:
            workload_result: Result from workload generator
            task_results: Dict of task_id -> task info from scheduler
            system_config: System configuration
            test_params: Test parameters
            log_paths: Dict of component -> log file path
        """
        self.workload_result = workload_result
        self.task_results = task_results
        self.system_config = system_config
        self.test_params = test_params
        self.log_paths = {k: str(v) for k, v in log_paths.items()}

    def generate_report(self, include_raw: bool = False) -> TestReport:
        """Generate comprehensive test report.

        Args:
            include_raw: Whether to include raw submission results

        Returns:
            TestReport instance
        """
        logger.info("Generating test report...")

        # Calculate latency stats
        submission_latencies = self.workload_result.get_submission_latencies()
        submission_latency = LatencyStats.from_samples(submission_latencies)

        # Calculate execution latencies if available
        execution_times = []
        for task_info in self.task_results.values():
            if task_info.get("status") == "completed":
                exec_time = task_info.get("execution_time_ms")
                if exec_time is not None:
                    execution_times.append(exec_time)

        execution_latency = (
            LatencyStats.from_samples(execution_times) if execution_times else None
        )

        # Calculate completion metrics
        completed = sum(1 for t in self.task_results.values() if t.get("status") == "completed")
        failed = sum(1 for t in self.task_results.values() if t.get("status") == "failed")
        pending = len(self.workload_result.submission_results) - len(self.task_results)

        # Build execution metrics
        execution_metrics = ExecutionMetrics(
            tasks_submitted=self.workload_result.total_tasks,
            successful_submissions=self.workload_result.successful_submissions,
            failed_submissions=self.workload_result.failed_submissions,
            submission_success_rate=self.workload_result.submission_success_rate(),
            tasks_completed=completed,
            tasks_failed=failed,
            tasks_pending=max(0, pending),
            completion_success_rate=completed / max(1, completed + failed),
            actual_duration_seconds=self.workload_result.actual_duration,
            actual_qps=self.workload_result.actual_qps,
            model_stats=self.workload_result.model_stats,
        )

        # Collect errors
        submission_errors = self._collect_submission_errors()
        execution_errors = self._collect_execution_errors()

        # Identify problems
        problems = self._identify_problems(execution_metrics)
        recommendations = self._generate_recommendations(problems)

        # Build raw results if requested
        raw_results = None
        if include_raw:
            raw_results = [
                {
                    "task_id": r.task_id,
                    "model_id": r.model_id,
                    "sleep_time": r.sleep_time,
                    "submitted_at": r.submitted_at,
                    "response_time_ms": r.response_time_ms,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.workload_result.submission_results
            ]

        report = TestReport(
            report_name="E2E PyLet Benchmark Report",
            report_version="1.0.0",
            generated_at=datetime.now().isoformat(),
            system_config=self.system_config,
            test_params=self.test_params,
            execution_metrics=execution_metrics,
            submission_latency=submission_latency,
            execution_latency=execution_latency,
            submission_errors=submission_errors,
            execution_errors=execution_errors,
            problems=problems,
            recommendations=recommendations,
            log_files=self.log_paths,
            raw_submission_results=raw_results,
        )

        logger.success("Test report generated successfully")
        return report

    def _collect_submission_errors(self) -> list[ErrorInfo]:
        """Collect submission errors from workload result."""
        errors = []
        for result in self.workload_result.get_failed_submissions():
            error_type = "unknown"
            if result.error:
                if "timeout" in result.error.lower():
                    error_type = "timeout"
                elif "http" in result.error.lower():
                    error_type = "http_error"
                elif "connection" in result.error.lower():
                    error_type = "connection_error"

            errors.append(ErrorInfo(
                task_id=result.task_id,
                model_id=result.model_id,
                error_type=error_type,
                error_message=result.error or "Unknown error",
                timestamp=result.submitted_at,
            ))
        return errors

    def _collect_execution_errors(self) -> list[ErrorInfo]:
        """Collect execution errors from task results."""
        errors = []
        for task_id, task_info in self.task_results.items():
            if task_info.get("status") == "failed":
                errors.append(ErrorInfo(
                    task_id=task_id,
                    model_id=task_info.get("model_id", "unknown"),
                    error_type="execution_failure",
                    error_message=task_info.get("error", "Unknown error"),
                    instance_id=task_info.get("assigned_instance"),
                ))
        return errors

    def _identify_problems(self, metrics: ExecutionMetrics) -> list[str]:
        """Identify problems based on metrics."""
        problems = []

        # Check submission success rate
        if metrics.submission_success_rate < 0.99:
            problems.append(
                f"Low submission success rate: {metrics.submission_success_rate:.2%} "
                f"({metrics.failed_submissions} failures)"
            )

        # Check completion success rate
        if metrics.completion_success_rate < 0.99 and metrics.tasks_completed > 0:
            problems.append(
                f"Low completion success rate: {metrics.completion_success_rate:.2%} "
                f"({metrics.tasks_failed} failures)"
            )

        # Check pending tasks
        if metrics.tasks_pending > 0:
            problems.append(
                f"Tasks still pending after test: {metrics.tasks_pending}"
            )

        # Check QPS achievement
        qps_ratio = metrics.actual_qps / self.test_params.target_qps
        if qps_ratio < 0.9:
            problems.append(
                f"Failed to achieve target QPS: {metrics.actual_qps:.2f} "
                f"(target: {self.test_params.target_qps}, ratio: {qps_ratio:.2%})"
            )

        # Check for model-specific failures
        for model, stats in metrics.model_stats.items():
            if stats["failed"] > 0:
                fail_rate = stats["failed"] / max(1, stats["submitted"])
                if fail_rate > 0.05:
                    problems.append(
                        f"High failure rate for {model}: {fail_rate:.2%} "
                        f"({stats['failed']}/{stats['submitted']})"
                    )

        return problems

    def _generate_recommendations(self, problems: list[str]) -> list[str]:
        """Generate recommendations based on identified problems."""
        recommendations = []

        for problem in problems:
            if "submission success rate" in problem.lower():
                recommendations.append(
                    "Check scheduler capacity and connection limits"
                )
            elif "completion success rate" in problem.lower():
                recommendations.append(
                    "Review instance health and execution logs"
                )
            elif "pending" in problem.lower():
                recommendations.append(
                    "Increase test duration or reduce QPS to allow completion"
                )
            elif "qps" in problem.lower():
                recommendations.append(
                    "Check network bandwidth and scheduler processing capacity"
                )
            elif "failure rate" in problem.lower():
                recommendations.append(
                    "Investigate model-specific issues in instance logs"
                )

        return recommendations
