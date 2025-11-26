"""
Metrics collection system for workflow experiments.

Provides comprehensive tracking of task-level, workflow-level, and system-level
performance metrics with thread-safe recording and statistical analysis.
"""

import csv
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
from loguru import logger as loguru_logger


@dataclass
class TaskMetrics:
    """Metrics for a single task execution."""
    task_id: str
    workflow_id: str
    task_type: str  # "A1", "A2", "B", "B1", "B2", "Merge", etc.

    submit_time: float
    complete_time: Optional[float] = None
    duration: Optional[float] = None  # Auto-calculated
    success: bool = False

    # Optional fields
    execution_time_ms: Optional[float] = None  # Reported by scheduler
    assigned_instance: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self):
        """Calculate duration if both times are available."""
        if self.complete_time and self.submit_time:
            self.duration = self.complete_time - self.submit_time


@dataclass
class WorkflowMetrics:
    """Metrics for a complete workflow execution."""
    workflow_id: str
    workflow_type: str  # "text2video" or "deep_research"

    start_time: float
    end_time: Optional[float] = None
    total_duration: Optional[float] = None  # Auto-calculated

    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0

    # Additional fields
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Calculate total duration if both times are available."""
        if self.end_time and self.start_time:
            self.total_duration = self.end_time - self.start_time


class MetricsCollector:
    """
    Thread-safe metrics collector for workflow experiments.

    Collects task-level and workflow-level metrics with real-time statistics
    calculation and multiple export formats.
    """

    def __init__(self, custom_logger: Optional[Any] = None):
        """
        Initialize metrics collector.

        Args:
            custom_logger: Optional custom logger (defaults to loguru logger)
        """
        self.logger = custom_logger or loguru_logger.bind(component="MetricsCollector")

        # Thread-safe storage
        # Note: Use RLock (reentrant lock) to allow nested locking within the same thread.
        # This is necessary because methods like export_to_json() call get_summary()
        # while already holding the locks.
        self.task_metrics: List[TaskMetrics] = []
        self.workflow_metrics: List[WorkflowMetrics] = []
        self._task_lock = threading.RLock()
        self._workflow_lock = threading.RLock()

        # Quick lookup by ID
        self._task_by_id: Dict[str, TaskMetrics] = {}
        self._workflow_by_id: Dict[str, WorkflowMetrics] = {}

        # Warmup workflow tracking
        self._warmup_workflow_ids: set = set()

        # Non-warmup workflow submission order (for portion filtering)
        # List of workflow_ids in submission order, excluding warmup
        self._non_warmup_workflow_order: List[str] = []

        # Collection start time
        self.collection_start_time = time.time()

    def record_task_submit(self,
                          task_id: str,
                          workflow_id: str,
                          task_type: str,
                          assigned_instance: Optional[str] = None) -> TaskMetrics:
        """
        Record task submission.

        Args:
            task_id: Task identifier
            workflow_id: Workflow identifier
            task_type: Type of task (A1, A2, B, etc.)
            assigned_instance: Instance assigned to task

        Returns:
            TaskMetrics object
        """
        metric = TaskMetrics(
            task_id=task_id,
            workflow_id=workflow_id,
            task_type=task_type,
            submit_time=time.time(),
            assigned_instance=assigned_instance
        )

        with self._task_lock:
            self.task_metrics.append(metric)
            self._task_by_id[task_id] = metric

        return metric

    def record_task_complete(self,
                            task_id: str,
                            success: bool = True,
                            execution_time_ms: Optional[float] = None,
                            error: Optional[str] = None):
        """
        Record task completion.

        Args:
            task_id: Task identifier
            success: Whether task completed successfully
            execution_time_ms: Execution time reported by scheduler
            error: Error message if failed
        """
        with self._task_lock:
            metric = self._task_by_id.get(task_id)
            if not metric:
                self.logger.warning(f"Task {task_id} not found in metrics")
                return

            metric.complete_time = time.time()
            metric.success = success
            metric.execution_time_ms = execution_time_ms
            metric.error = error

            # Recalculate duration
            if metric.submit_time:
                metric.duration = metric.complete_time - metric.submit_time

    def mark_workflow_as_warmup(self, workflow_id: str):
        """
        Mark a workflow as warmup (excluded from statistics).

        Args:
            workflow_id: Workflow identifier to mark as warmup
        """
        with self._workflow_lock:
            self._warmup_workflow_ids.add(workflow_id)

    def is_warmup_workflow(self, workflow_id: str) -> bool:
        """
        Check if a workflow is marked as warmup.

        Args:
            workflow_id: Workflow identifier

        Returns:
            True if workflow is warmup, False otherwise
        """
        return workflow_id in self._warmup_workflow_ids

    def record_workflow_start(self,
                             workflow_id: str,
                             workflow_type: str,
                             metadata: Optional[Dict] = None,
                             is_warmup: bool = False) -> WorkflowMetrics:
        """
        Record workflow start.

        Args:
            workflow_id: Workflow identifier
            workflow_type: Type of workflow
            metadata: Additional metadata
            is_warmup: Whether this is a warmup workflow (excluded from stats)

        Returns:
            WorkflowMetrics object
        """
        metric = WorkflowMetrics(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            start_time=time.time(),
            metadata=metadata or {}
        )

        with self._workflow_lock:
            self.workflow_metrics.append(metric)
            self._workflow_by_id[workflow_id] = metric
            if is_warmup:
                self._warmup_workflow_ids.add(workflow_id)
            else:
                # Track non-warmup workflow submission order
                self._non_warmup_workflow_order.append(workflow_id)

        return metric

    def record_workflow_complete(self,
                                 workflow_id: str,
                                 successful_tasks: int,
                                 failed_tasks: int):
        """
        Record workflow completion.

        Args:
            workflow_id: Workflow identifier
            successful_tasks: Number of successful tasks
            failed_tasks: Number of failed tasks
        """
        with self._workflow_lock:
            metric = self._workflow_by_id.get(workflow_id)
            if not metric:
                self.logger.warning(f"Workflow {workflow_id} not found in metrics")
                return

            metric.end_time = time.time()
            metric.successful_tasks = successful_tasks
            metric.failed_tasks = failed_tasks
            metric.total_tasks = successful_tasks + failed_tasks

            # Recalculate duration
            if metric.start_time:
                metric.total_duration = metric.end_time - metric.start_time

    def get_summary(self, exclude_warmup: bool = True, portion_stats: float = 1.0) -> Dict:
        """
        Calculate comprehensive summary statistics.

        Args:
            exclude_warmup: If True, exclude warmup workflows from statistics
            portion_stats: Portion of non-warmup workflows to include (0.0-1.0).
                          Only the first (portion_stats * num_non_warmup_workflows) workflows
                          are included in statistics.

        Returns:
            Dictionary with summary statistics
        """
        with self._task_lock, self._workflow_lock:
            # Calculate counts
            warmup_workflow_count = len(self._warmup_workflow_ids)
            warmup_task_count = len([t for t in self.task_metrics
                                    if t.workflow_id in self._warmup_workflow_ids])

            # Determine which workflow IDs to include in statistics
            if exclude_warmup:
                # Get the portion of non-warmup workflows based on submission order
                total_non_warmup = len(self._non_warmup_workflow_order)
                num_to_include = int(total_non_warmup * portion_stats)
                included_workflow_ids = set(self._non_warmup_workflow_order[:num_to_include])
                excluded_by_portion = total_non_warmup - num_to_include
            else:
                # Include all workflows
                included_workflow_ids = set(w.workflow_id for w in self.workflow_metrics)
                excluded_by_portion = 0

            # Filter tasks and workflows based on included IDs
            all_tasks = [t for t in self.task_metrics
                        if t.workflow_id in included_workflow_ids]
            all_workflows = [w for w in self.workflow_metrics
                           if w.workflow_id in included_workflow_ids]

            # Task statistics
            completed_tasks = [t for t in all_tasks if t.complete_time]
            successful_tasks = [t for t in completed_tasks if t.success]
            failed_tasks = [t for t in completed_tasks if not t.success]

            durations = [t.duration for t in completed_tasks if t.duration]

            # Workflow statistics
            completed_workflows = [w for w in all_workflows if w.end_time]
            workflow_durations = [w.total_duration for w in completed_workflows if w.total_duration]

            # Calculate percentiles if we have data
            p50 = p95 = p99 = 0.0
            if durations:
                p50 = float(np.percentile(durations, 50))
                p95 = float(np.percentile(durations, 95))
                p99 = float(np.percentile(durations, 99))

            # Calculate QPS (based on included tasks only)
            elapsed = time.time() - self.collection_start_time
            actual_qps = len(all_tasks) / elapsed if elapsed > 0 else 0

            return {
                'collection_duration': elapsed,
                'warmup_stats': {
                    'warmup_workflows': warmup_workflow_count,
                    'warmup_tasks': warmup_task_count,
                    'excluded_from_stats': exclude_warmup,
                },
                'portion_stats': {
                    'portion': portion_stats,
                    'total_non_warmup_workflows': len(self._non_warmup_workflow_order),
                    'included_workflows': len(included_workflow_ids),
                    'excluded_by_portion': excluded_by_portion,
                },
                'task_stats': {
                    'total_submitted': len(all_tasks),
                    'total_completed': len(completed_tasks),
                    'successful': len(successful_tasks),
                    'failed': len(failed_tasks),
                    'success_rate': len(successful_tasks) / len(completed_tasks) if completed_tasks else 0,
                    'actual_qps': actual_qps,
                },
                'task_latency': {
                    'p50': p50,
                    'p95': p95,
                    'p99': p99,
                    'mean': float(np.mean(durations)) if durations else 0,
                    'std': float(np.std(durations)) if durations else 0,
                },
                'workflow_stats': {
                    'total_started': len(all_workflows),
                    'total_completed': len(completed_workflows),
                    'completion_rate': len(completed_workflows) / len(all_workflows) if all_workflows else 0,
                },
                'workflow_duration': {
                    'mean': float(np.mean(workflow_durations)) if workflow_durations else 0,
                    'std': float(np.std(workflow_durations)) if workflow_durations else 0,
                    'min': float(np.min(workflow_durations)) if workflow_durations else 0,
                    'max': float(np.max(workflow_durations)) if workflow_durations else 0,
                }
            }

    def export_to_json(self, filepath: Union[str, Path],
                       exclude_warmup: bool = True,
                       portion_stats: float = 1.0):
        """
        Export all metrics to JSON file.

        Args:
            filepath: Output file path
            exclude_warmup: If True, exclude warmup workflows from summary statistics
            portion_stats: Portion of non-warmup workflows to include in summary (0.0-1.0)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with self._task_lock, self._workflow_lock:
            data = {
                'summary': self.get_summary(exclude_warmup=exclude_warmup, portion_stats=portion_stats),
                'tasks': [
                    {
                        'task_id': t.task_id,
                        'workflow_id': t.workflow_id,
                        'task_type': t.task_type,
                        'submit_time': t.submit_time,
                        'complete_time': t.complete_time,
                        'duration': t.duration,
                        'success': t.success,
                        'execution_time_ms': t.execution_time_ms,
                        'assigned_instance': t.assigned_instance,
                        'error': t.error,
                    }
                    for t in self.task_metrics
                ],
                'workflows': [
                    {
                        'workflow_id': w.workflow_id,
                        'workflow_type': w.workflow_type,
                        'start_time': w.start_time,
                        'end_time': w.end_time,
                        'total_duration': w.total_duration,
                        'total_tasks': w.total_tasks,
                        'successful_tasks': w.successful_tasks,
                        'failed_tasks': w.failed_tasks,
                        'metadata': w.metadata,
                    }
                    for w in self.workflow_metrics
                ]
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Metrics exported to {filepath}")

    def export_tasks_to_csv(self, filepath: Union[str, Path]):
        """
        Export task metrics to CSV file.

        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with self._task_lock:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'task_id', 'workflow_id', 'task_type', 'submit_time',
                    'complete_time', 'duration', 'success', 'execution_time_ms',
                    'assigned_instance', 'error'
                ])

                for t in self.task_metrics:
                    writer.writerow([
                        t.task_id, t.workflow_id, t.task_type, t.submit_time,
                        t.complete_time, t.duration, t.success, t.execution_time_ms,
                        t.assigned_instance, t.error
                    ])

        self.logger.info(f"Task metrics exported to {filepath}")

    def generate_text_report(self, filepath: Optional[Union[str, Path]] = None,
                               exclude_warmup: bool = True,
                               portion_stats: float = 1.0) -> str:
        """
        Generate human-readable text report.

        Args:
            filepath: Optional output file path
            exclude_warmup: If True, exclude warmup workflows from statistics
            portion_stats: Portion of non-warmup workflows to include (0.0-1.0)

        Returns:
            Report as string
        """
        summary = self.get_summary(exclude_warmup=exclude_warmup, portion_stats=portion_stats)

        report_lines = [
            "=" * 80,
            "WORKFLOW EXPERIMENT METRICS REPORT",
            "=" * 80,
            "",
            f"Collection Duration: {summary['collection_duration']:.2f}s",
        ]

        # Add warmup info if applicable
        warmup_stats = summary.get('warmup_stats', {})
        if warmup_stats.get('warmup_workflows', 0) > 0:
            report_lines.extend([
                "",
                "WARMUP (excluded from statistics):",
                f"  Warmup Workflows: {warmup_stats['warmup_workflows']}",
                f"  Warmup Tasks: {warmup_stats['warmup_tasks']}",
            ])

        # Add portion filtering info if applicable
        portion_info = summary.get('portion_stats', {})
        if portion_info.get('portion', 1.0) < 1.0:
            report_lines.extend([
                "",
                "PORTION FILTERING:",
                f"  Portion: {portion_info['portion']*100:.1f}%",
                f"  Total Non-Warmup Workflows: {portion_info['total_non_warmup_workflows']}",
                f"  Included in Stats: {portion_info['included_workflows']}",
                f"  Excluded by Portion: {portion_info['excluded_by_portion']}",
            ])

        report_lines.extend([
            "",
            "TASK STATISTICS:",
            f"  Total Submitted: {summary['task_stats']['total_submitted']}",
            f"  Total Completed: {summary['task_stats']['total_completed']}",
            f"  Successful: {summary['task_stats']['successful']}",
            f"  Failed: {summary['task_stats']['failed']}",
            f"  Success Rate: {summary['task_stats']['success_rate']*100:.2f}%",
            f"  Actual QPS: {summary['task_stats']['actual_qps']:.2f}",
            "",
            "TASK LATENCY (seconds):",
            f"  P50: {summary['task_latency']['p50']:.3f}",
            f"  P95: {summary['task_latency']['p95']:.3f}",
            f"  P99: {summary['task_latency']['p99']:.3f}",
            f"  Mean: {summary['task_latency']['mean']:.3f}",
            f"  Std Dev: {summary['task_latency']['std']:.3f}",
            "",
            "WORKFLOW STATISTICS:",
            f"  Total Started: {summary['workflow_stats']['total_started']}",
            f"  Total Completed: {summary['workflow_stats']['total_completed']}",
            f"  Completion Rate: {summary['workflow_stats']['completion_rate']*100:.2f}%",
            "",
            "WORKFLOW DURATION (seconds):",
            f"  Mean: {summary['workflow_duration']['mean']:.3f}",
            f"  Std Dev: {summary['workflow_duration']['std']:.3f}",
            f"  Min: {summary['workflow_duration']['min']:.3f}",
            f"  Max: {summary['workflow_duration']['max']:.3f}",
            "",
            "=" * 80,
        ])

        report = "\n".join(report_lines)

        if filepath:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(report)
            self.logger.info(f"Text report exported to {filepath}")

        return report
