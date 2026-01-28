"""Multi-Scheduler Report Generator for E2E Testing.

This module generates comprehensive test reports with multi-scheduler
specific metrics, including per-scheduler statistics and routing verification.

Features:
- Per-scheduler latency breakdown
- Routing correctness verification
- Multi-scheduler system configuration
- JSON and Markdown output formats

Usage:
    from report_generator import (
        MultiSchedulerReportGenerator,
        MultiSchedulerSystemConfig,
        MultiSchedulerTestParams,
    )

    generator = MultiSchedulerReportGenerator(
        workload_result=workload_result,
        task_results=task_results,
        system_config=system_config,
        test_params=test_params,
        log_paths=log_paths,
    )
    report = generator.generate_report()
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from .workload_generator import (
    MultiSchedulerWorkloadResult,
)


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
    def from_samples(cls, samples: list[float]) -> LatencyStats:
        """Calculate statistics from a list of samples.

        Args:
            samples: List of latency values in ms.

        Returns:
            LatencyStats instance.
        """
        if not samples:
            return cls(
                p50=0,
                p90=0,
                p95=0,
                p99=0,
                mean=0,
                min=0,
                max=0,
                stddev=0,
                count=0,
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
class MultiSchedulerSystemConfig:
    """System configuration for multi-scheduler test."""

    planner_url: str
    scheduler_urls: dict[str, str]
    pylet_head_url: str
    num_workers: int
    model_distribution: dict[str, int]
    total_instances: int
    total_schedulers: int


@dataclass
class MultiSchedulerTestParams:
    """Test parameters for multi-scheduler test."""

    target_qps: float
    duration_seconds: float
    total_tasks: int
    sleep_time_range: tuple[float, float]
    model_ids: list[str]
    qps_distribution: dict[str, float]


@dataclass
class MultiSchedulerExecutionMetrics:
    """Execution metrics from multi-scheduler test."""

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
    model_stats: dict[str, dict[str, Any]]


@dataclass
class SchedulerStats:
    """Statistics for a single scheduler."""

    model_id: str
    scheduler_url: str
    tasks_submitted: int
    tasks_successful: int
    tasks_failed: int
    target_qps: float
    actual_qps: float
    latency: LatencyStats
    instance_count: int


@dataclass
class ErrorInfo:
    """Information about an error occurrence."""

    task_id: str
    model_id: str
    scheduler_url: str
    error_type: str
    error_message: str
    timestamp: float | None = None
    instance_id: str | None = None


@dataclass
class MultiSchedulerTestReport:
    """Comprehensive test report for multi-scheduler experiments."""

    # Metadata
    report_name: str
    report_version: str
    generated_at: str

    # Configuration
    system_config: MultiSchedulerSystemConfig
    test_params: MultiSchedulerTestParams

    # Metrics
    execution_metrics: MultiSchedulerExecutionMetrics

    # Latency
    submission_latency: LatencyStats
    execution_latency: LatencyStats | None

    # Per-scheduler breakdown
    scheduler_stats: list[SchedulerStats]

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
            path: Optional path to write JSON file.

        Returns:
            JSON string.
        """
        report_dict = self.to_dict()

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
            path: Optional path to write Markdown file.

        Returns:
            Markdown string.
        """
        md_lines = [
            f"# {self.report_name}",
            "",
            f"**Generated:** {self.generated_at}",
            f"**Version:** {self.report_version}",
            "",
            "---",
            "",
            "## Multi-Scheduler Architecture",
            "",
            f"This experiment validates the multi-scheduler architecture with "
            f"**{len(self.scheduler_stats)} schedulers** and "
            f"**{self.system_config.total_instances} instances**.",
            "",
            "### Scheduler Configuration",
            "",
            "| Model | Scheduler URL | Instances | Target QPS |",
            "|-------|---------------|-----------|------------|",
        ]

        for ss in self.scheduler_stats:
            md_lines.append(
                f"| {ss.model_id} | `{ss.scheduler_url}` | "
                f"{ss.instance_count} | {ss.target_qps:.2f} |"
            )

        md_lines.extend(
            [
                "",
                "---",
                "",
                "## System Configuration",
                "",
                "| Component | Value |",
                "|-----------|-------|",
                f"| Planner URL | `{self.system_config.planner_url}` |",
                f"| PyLet Head | `{self.system_config.pylet_head_url}` |",
                f"| Workers | {self.system_config.num_workers} |",
                f"| Total Instances | {self.system_config.total_instances} |",
                f"| Total Schedulers | {self.system_config.total_schedulers} |",
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
                f"| Sleep Time Range | {self.test_params.sleep_time_range[0]}s - "
                f"{self.test_params.sleep_time_range[1]}s |",
                "",
                "### QPS Distribution",
                "",
                "| Model | QPS |",
                "|-------|-----|",
            ]
        )

        for model_id, qps in self.test_params.qps_distribution.items():
            md_lines.append(f"| {model_id} | {qps:.2f} |")

        md_lines.extend(
            [
                "",
                "---",
                "",
                "## Execution Metrics",
                "",
                "### Aggregate Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Total Submitted | {self.execution_metrics.tasks_submitted} |",
                f"| Successful | {self.execution_metrics.successful_submissions} |",
                f"| Failed | {self.execution_metrics.failed_submissions} |",
                f"| Submission Success Rate | {self.execution_metrics.submission_success_rate:.2%} |",
                f"| Tasks Completed | {self.execution_metrics.tasks_completed} |",
                f"| Tasks Failed | {self.execution_metrics.tasks_failed} |",
                f"| Tasks Pending | {self.execution_metrics.tasks_pending} |",
                f"| Completion Success Rate | {self.execution_metrics.completion_success_rate:.2%} |",
                f"| Actual Duration | {self.execution_metrics.actual_duration_seconds:.2f}s |",
                f"| Actual QPS | {self.execution_metrics.actual_qps:.2f} |",
                "",
                "### Per-Scheduler Metrics",
                "",
                "| Model | Submitted | Successful | Failed | QPS | p50 | p99 |",
                "|-------|-----------|------------|--------|-----|-----|-----|",
            ]
        )

        for ss in self.scheduler_stats:
            md_lines.append(
                f"| {ss.model_id} | {ss.tasks_submitted} | {ss.tasks_successful} | "
                f"{ss.tasks_failed} | {ss.actual_qps:.2f} | "
                f"{ss.latency.p50:.1f}ms | {ss.latency.p99:.1f}ms |"
            )

        md_lines.extend(
            [
                "",
                "---",
                "",
                "## Latency Statistics (ms)",
                "",
                "### Aggregate Submission Latency",
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
            ]
        )

        if self.execution_latency and self.execution_latency.count > 0:
            md_lines.extend(
                [
                    "### Execution Latency",
                    "",
                    "| Percentile | Value |",
                    "|------------|-------|",
                    f"| p50 | {self.execution_latency.p50:.2f} |",
                    f"| p90 | {self.execution_latency.p90:.2f} |",
                    f"| p95 | {self.execution_latency.p95:.2f} |",
                    f"| p99 | {self.execution_latency.p99:.2f} |",
                    f"| mean | {self.execution_latency.mean:.2f} |",
                    "",
                ]
            )

        md_lines.extend(
            [
                "---",
                "",
                "## Error Analysis",
                "",
            ]
        )

        if self.submission_errors:
            md_lines.extend(
                [
                    "### Submission Errors",
                    "",
                    "| Task ID | Model | Scheduler | Error |",
                    "|---------|-------|-----------|-------|",
                ]
            )
            for err in self.submission_errors[:20]:
                md_lines.append(
                    f"| {err.task_id[:30]}... | {err.model_id} | "
                    f"{err.scheduler_url} | {err.error_message[:40]}... |"
                )
            if len(self.submission_errors) > 20:
                md_lines.append(
                    f"| ... | ... | ... | ({len(self.submission_errors) - 20} more) |"
                )
            md_lines.append("")
        else:
            md_lines.extend(["*No submission errors*", ""])

        if self.execution_errors:
            md_lines.extend(
                [
                    "### Execution Errors",
                    "",
                    "| Task ID | Model | Instance | Error |",
                    "|---------|-------|----------|-------|",
                ]
            )
            for err in self.execution_errors[:20]:
                md_lines.append(
                    f"| {err.task_id[:30]}... | {err.model_id} | "
                    f"{err.instance_id or 'N/A'} | {err.error_message[:40]}... |"
                )
            md_lines.append("")
        else:
            md_lines.extend(["*No execution errors*", ""])

        md_lines.extend(
            [
                "---",
                "",
                "## Problems Identified",
                "",
            ]
        )

        if self.problems:
            for i, problem in enumerate(self.problems, 1):
                md_lines.append(f"{i}. {problem}")
            md_lines.append("")
        else:
            md_lines.extend(["*No problems identified*", ""])

        if self.recommendations:
            md_lines.extend(
                [
                    "### Recommendations",
                    "",
                ]
            )
            for rec in self.recommendations:
                md_lines.append(f"- {rec}")
            md_lines.append("")

        md_lines.extend(
            [
                "---",
                "",
                "## Log Files",
                "",
                "| Component | Path |",
                "|-----------|------|",
            ]
        )

        for component, log_path in self.log_files.items():
            md_lines.append(f"| {component} | `{log_path}` |")

        md_lines.extend(
            [
                "",
                "---",
                "",
                "*Report generated by E2E Multi-Scheduler Experiment Framework*",
            ]
        )

        md_str = "\n".join(md_lines)

        if path:
            Path(path).write_text(md_str)
            logger.info(f"Markdown report written to {path}")

        return md_str


class MultiSchedulerReportGenerator:
    """Generate test reports for multi-scheduler experiments."""

    def __init__(
        self,
        workload_result: MultiSchedulerWorkloadResult,
        task_results: dict[str, dict[str, Any]],
        system_config: MultiSchedulerSystemConfig,
        test_params: MultiSchedulerTestParams,
        log_paths: dict[str, str | Path],
    ):
        """Initialize report generator.

        Args:
            workload_result: Result from workload generator.
            task_results: Dict of task_id -> task info from schedulers.
            system_config: System configuration.
            test_params: Test parameters.
            log_paths: Dict of component -> log file path.
        """
        self.workload_result = workload_result
        self.task_results = task_results
        self.system_config = system_config
        self.test_params = test_params
        self.log_paths = {k: str(v) for k, v in log_paths.items()}

    def generate_report(self, include_raw: bool = False) -> MultiSchedulerTestReport:
        """Generate comprehensive test report.

        Args:
            include_raw: Whether to include raw submission results.

        Returns:
            MultiSchedulerTestReport instance.
        """
        logger.info("Generating multi-scheduler test report...")

        # Calculate aggregate latency stats
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
        completed = sum(
            1 for t in self.task_results.values() if t.get("status") == "completed"
        )
        failed = sum(
            1 for t in self.task_results.values() if t.get("status") == "failed"
        )
        pending = self.workload_result.successful_submissions - len(self.task_results)

        # Build execution metrics
        execution_metrics = MultiSchedulerExecutionMetrics(
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

        # Build per-scheduler stats
        scheduler_stats = self._build_scheduler_stats()

        # Collect errors
        submission_errors = self._collect_submission_errors()
        execution_errors = self._collect_execution_errors()

        # Identify problems
        problems = self._identify_problems(execution_metrics, scheduler_stats)
        recommendations = self._generate_recommendations(problems)

        # Build raw results if requested
        raw_results = None
        if include_raw:
            raw_results = [
                {
                    "task_id": r.task_id,
                    "model_id": r.model_id,
                    "scheduler_url": r.scheduler_url,
                    "sleep_time": r.sleep_time,
                    "submitted_at": r.submitted_at,
                    "response_time_ms": r.response_time_ms,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.workload_result.submission_results
            ]

        report = MultiSchedulerTestReport(
            report_name="E2E Multi-Scheduler Experiment Report",
            report_version="1.0.0",
            generated_at=datetime.now().isoformat(),
            system_config=self.system_config,
            test_params=self.test_params,
            execution_metrics=execution_metrics,
            submission_latency=submission_latency,
            execution_latency=execution_latency,
            scheduler_stats=scheduler_stats,
            submission_errors=submission_errors,
            execution_errors=execution_errors,
            problems=problems,
            recommendations=recommendations,
            log_files=self.log_paths,
            raw_submission_results=raw_results,
        )

        logger.success("Multi-scheduler test report generated successfully")
        return report

    def _build_scheduler_stats(self) -> list[SchedulerStats]:
        """Build per-scheduler statistics.

        Returns:
            List of SchedulerStats.
        """
        stats = []

        for model_id in self.test_params.model_ids:
            model_stats = self.workload_result.model_stats.get(model_id, {})
            latencies = self.workload_result.get_model_latencies(model_id)

            submitted = model_stats.get("submitted", 0)
            successful = model_stats.get("successful", 0)
            duration = self.workload_result.actual_duration

            stats.append(
                SchedulerStats(
                    model_id=model_id,
                    scheduler_url=self.system_config.scheduler_urls.get(model_id, ""),
                    tasks_submitted=submitted,
                    tasks_successful=successful,
                    tasks_failed=submitted - successful,
                    target_qps=self.test_params.qps_distribution.get(model_id, 0),
                    actual_qps=successful / duration if duration > 0 else 0,
                    latency=LatencyStats.from_samples(latencies),
                    instance_count=self.system_config.model_distribution.get(
                        model_id, 0
                    ),
                )
            )

        return stats

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

            errors.append(
                ErrorInfo(
                    task_id=result.task_id,
                    model_id=result.model_id,
                    scheduler_url=result.scheduler_url,
                    error_type=error_type,
                    error_message=result.error or "Unknown error",
                    timestamp=result.submitted_at,
                )
            )
        return errors

    def _collect_execution_errors(self) -> list[ErrorInfo]:
        """Collect execution errors from task results."""
        errors = []
        for task_id, task_info in self.task_results.items():
            if task_info.get("status") == "failed":
                errors.append(
                    ErrorInfo(
                        task_id=task_id,
                        model_id=task_info.get("model_id", "unknown"),
                        scheduler_url="",  # Not tracked in task results
                        error_type="execution_failure",
                        error_message=task_info.get("error", "Unknown error"),
                        instance_id=task_info.get("assigned_instance"),
                    )
                )
        return errors

    def _identify_problems(
        self,
        metrics: MultiSchedulerExecutionMetrics,
        scheduler_stats: list[SchedulerStats],
    ) -> list[str]:
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
            problems.append(f"Tasks still pending after test: {metrics.tasks_pending}")

        # Check QPS achievement
        qps_ratio = metrics.actual_qps / self.test_params.target_qps
        if qps_ratio < 0.9:
            problems.append(
                f"Failed to achieve target QPS: {metrics.actual_qps:.2f} "
                f"(target: {self.test_params.target_qps}, ratio: {qps_ratio:.2%})"
            )

        # Check per-scheduler issues
        for ss in scheduler_stats:
            if ss.tasks_failed > 0:
                fail_rate = ss.tasks_failed / max(1, ss.tasks_submitted)
                if fail_rate > 0.05:
                    problems.append(
                        f"High failure rate for scheduler {ss.model_id}: "
                        f"{fail_rate:.2%} ({ss.tasks_failed}/{ss.tasks_submitted})"
                    )

            # Check per-scheduler QPS
            if ss.target_qps > 0:
                scheduler_qps_ratio = ss.actual_qps / ss.target_qps
                if scheduler_qps_ratio < 0.8:
                    problems.append(
                        f"Scheduler {ss.model_id} underperforming: "
                        f"{ss.actual_qps:.2f}/{ss.target_qps:.2f} QPS "
                        f"({scheduler_qps_ratio:.2%})"
                    )

        return problems

    def _generate_recommendations(self, problems: list[str]) -> list[str]:
        """Generate recommendations based on identified problems."""
        recommendations = []

        for problem in problems:
            if "submission success rate" in problem.lower():
                recommendations.append("Check scheduler capacity and connection limits")
            elif "completion success rate" in problem.lower():
                recommendations.append("Review instance health and execution logs")
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
                    "Investigate model-specific issues in scheduler logs"
                )
            elif "underperforming" in problem.lower():
                recommendations.append(
                    "Check scheduler-specific bottlenecks and instance counts"
                )

        return list(set(recommendations))  # Deduplicate
