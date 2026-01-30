#!/usr/bin/env python3
"""Generate workload for LLM Cluster example (Multi-Scheduler).

This script generates tasks with QPS ratios 5:1:3 (llm_fast:llm_medium:llm_slow),
queries the planner for the scheduler map, and submits tasks directly to the correct
per-model scheduler.

Usage:
    python examples/llm_cluster/generate_workload.py [options]

Options:
    --planner-url     Planner URL for scheduler discovery (default: http://localhost:8003)
    --total-qps       Target queries per second (default: 10.0)
    --duration        Test duration in seconds (default: 60.0)
    --wait-timeout    Timeout for task completion in seconds (default: 600.0)

Example:
    python examples/llm_cluster/generate_workload.py --total-qps 15 --duration 120

PYLET-036: Per-Model Schedulers + Correct Startup Sequence
"""

import argparse
import asyncio
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console()


class SchedulerRouter:
    """Routes tasks to the correct per-model scheduler.

    Queries the planner's /v1/scheduler/list endpoint to build a mapping
    from model_id to scheduler_url.

    Attributes:
        planner_url: Base URL of the planner service.
        scheduler_map: Mapping from model_id to scheduler URL.
    """

    def __init__(self, planner_url: str) -> None:
        """Initialize the scheduler router.

        Args:
            planner_url: Base URL of the planner service.
        """
        self.planner_url = planner_url.rstrip("/")
        self.scheduler_map: dict[str, str] = {}

    def refresh(self) -> None:
        """Refresh the scheduler map from the planner.

        Raises:
            RuntimeError: If the planner is unreachable or returns
                no schedulers.
        """
        try:
            response = httpx.get(
                f"{self.planner_url}/v1/scheduler/list", timeout=5.0
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to get scheduler list: HTTP {response.status_code}"
                )
            data = response.json()
            self.scheduler_map = {
                s["model_id"]: s["scheduler_url"]
                for s in data.get("schedulers", [])
            }
            if not self.scheduler_map:
                raise RuntimeError("No schedulers registered with planner")
        except httpx.RequestError as e:
            raise RuntimeError(f"Cannot reach planner: {e}") from e

    def get_scheduler_url(self, model_id: str) -> str:
        """Get the scheduler URL for a model.

        Args:
            model_id: Model identifier.

        Returns:
            Scheduler URL.

        Raises:
            KeyError: If no scheduler is registered for the model.
        """
        if model_id not in self.scheduler_map:
            raise KeyError(
                f"No scheduler registered for model: {model_id}. "
                f"Available: {list(self.scheduler_map.keys())}"
            )
        return self.scheduler_map[model_id]


@dataclass
class WorkloadConfig:
    """Configuration for LLM workload generation."""

    planner_url: str = "http://localhost:8003"
    total_qps: float = 10.0
    duration_seconds: float = 60.0
    wait_timeout_seconds: float = 600.0

    # Model configuration (3 models, QPS ratio 5:1:3)
    model_ids: list[str] = field(
        default_factory=lambda: ["llm_fast", "llm_medium", "llm_slow"]
    )
    qps_ratios: list[float] = field(
        default_factory=lambda: [5.0, 1.0, 3.0]
    )

    # Sample prompts for realistic workload
    sample_prompts: list[str] = field(
        default_factory=lambda: [
            "Explain quantum computing in simple terms.",
            "Write a short poem about artificial intelligence.",
            "What are the key differences between Python and JavaScript?",
            "Summarize the benefits of cloud computing.",
            "How does machine learning work?",
            "Describe the future of renewable energy.",
            "What is the importance of data privacy?",
            "Explain the concept of blockchain.",
            "What are best practices for software testing?",
            "How can we improve user experience in web applications?",
        ]
    )

    def get_model_qps(self) -> dict[str, float]:
        """Calculate QPS for each model based on ratios."""
        total_ratio = sum(self.qps_ratios)
        return {
            model_id: (ratio / total_ratio) * self.total_qps
            for model_id, ratio in zip(self.model_ids, self.qps_ratios)
        }


@dataclass
class TaskStats:
    """Statistics for a completed task."""

    task_id: str
    model_id: str
    submit_time: float
    complete_time: float | None = None
    status: str = "pending"
    error: str | None = None
    execution_time_ms: float | None = None

    @property
    def latency_ms(self) -> float | None:
        """Task latency in milliseconds."""
        if self.complete_time is None:
            return None
        return (self.complete_time - self.submit_time) * 1000


@dataclass
class WorkloadStats:
    """Aggregate statistics for workload run."""

    tasks: dict[str, TaskStats] = field(default_factory=dict)
    start_time: float | None = None
    end_time: float | None = None

    def add_task(self, task: TaskStats) -> None:
        """Add a task to statistics."""
        self.tasks[task.task_id] = task

    def mark_complete(
        self,
        task_id: str,
        status: str,
        execution_time_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        """Mark a task as complete."""
        if task_id in self.tasks:
            self.tasks[task_id].complete_time = time.time()
            self.tasks[task_id].status = status
            self.tasks[task_id].error = error
            self.tasks[task_id].execution_time_ms = execution_time_ms

    def get_model_stats(self, model_id: str) -> dict[str, Any]:
        """Get statistics for a specific model."""
        model_tasks = [t for t in self.tasks.values() if t.model_id == model_id]
        completed = [t for t in model_tasks if t.status == "completed"]
        latencies = [t.latency_ms for t in completed if t.latency_ms is not None]

        execution_times = [
            t.execution_time_ms
            for t in completed
            if t.execution_time_ms is not None
        ]

        return {
            "total": len(model_tasks),
            "completed": len(completed),
            "failed": len([t for t in model_tasks if t.status == "failed"]),
            "pending": len([t for t in model_tasks if t.status == "pending"]),
            "avg_submit_latency_ms": (
                sum(latencies) / len(latencies) if latencies else 0
            ),
            "min_submit_latency_ms": min(latencies) if latencies else 0,
            "max_submit_latency_ms": max(latencies) if latencies else 0,
            "avg_execution_time_ms": (
                sum(execution_times) / len(execution_times)
                if execution_times
                else 0
            ),
        }

    def summary(self) -> dict[str, Any]:
        """Get overall summary statistics."""
        total = len(self.tasks)
        completed = len([t for t in self.tasks.values() if t.status == "completed"])
        failed = len([t for t in self.tasks.values() if t.status == "failed"])
        pending = len([t for t in self.tasks.values() if t.status == "pending"])

        all_latencies = [
            t.latency_ms
            for t in self.tasks.values()
            if t.status == "completed" and t.latency_ms is not None
        ]

        duration = (self.end_time - self.start_time) if self.end_time else 0
        actual_qps = completed / duration if duration > 0 else 0

        return {
            "total_tasks": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "duration_s": duration,
            "actual_qps": actual_qps,
            "avg_submit_latency_ms": (
                sum(all_latencies) / len(all_latencies) if all_latencies else 0
            ),
        }


async def submit_task(
    client: httpx.AsyncClient,
    scheduler_url: str,
    task_id: str,
    model_id: str,
    prompt: str,
) -> bool:
    """Submit a single task to the model's scheduler.

    Args:
        client: HTTP client.
        scheduler_url: Scheduler base URL for this model.
        task_id: Unique task identifier.
        model_id: Model to route the task to.
        prompt: User prompt for the task.

    Returns:
        True if task was accepted, False otherwise.
    """
    task_input = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 256,
    }

    payload = {
        "model_id": model_id,
        "task_id": task_id,
        "task_input": task_input,
        "metadata": {
            "source": "llm_workload_generator",
            "timestamp": time.time(),
            "exp_runtime": 100.0,
            "path": "inference",
        },
    }

    try:
        response = await client.post(
            f"{scheduler_url}/v1/task/submit",
            json=payload,
            timeout=30.0,
        )
        return response.status_code == 200

    except Exception as e:
        console.print(f"[red]Failed to submit {task_id}: {e}[/red]")
        return False


async def poll_task_status(
    client: httpx.AsyncClient,
    scheduler_url: str,
    task_id: str,
) -> tuple[str, str | None, float | None]:
    """Poll task status from the model's scheduler.

    Args:
        client: HTTP client.
        scheduler_url: Scheduler base URL for this model.
        task_id: Task identifier.

    Returns:
        Tuple of (status, error_message, execution_time_ms).
    """
    try:
        response = await client.get(
            f"{scheduler_url}/v1/task/info",
            params={"task_id": task_id},
            timeout=5.0,
        )
        if response.status_code == 200:
            data = response.json()
            task = data.get("task", {})
            return (
                task.get("status", "unknown"),
                task.get("error"),
                task.get("execution_time_ms"),
            )
        return "unknown", None, None
    except Exception:
        return "unknown", None, None


async def run_workload(
    config: WorkloadConfig,
    router: SchedulerRouter,
) -> WorkloadStats:
    """Run the workload generator.

    Args:
        config: Workload configuration.
        router: Scheduler router for per-model task routing.

    Returns:
        WorkloadStats with results.
    """
    import random

    stats = WorkloadStats()
    stats.start_time = time.time()

    model_qps = config.get_model_qps()

    # Calculate tasks per model
    total_expected_tasks = int(config.total_qps * config.duration_seconds)

    # Calculate how many tasks per model
    tasks_per_model = {}
    total_ratio = sum(config.qps_ratios)
    for model_id, ratio in zip(config.model_ids, config.qps_ratios):
        tasks_per_model[model_id] = int(
            total_expected_tasks * (ratio / total_ratio)
        )

    # Adjust for rounding
    total_allocated = sum(tasks_per_model.values())
    if total_allocated < total_expected_tasks:
        tasks_per_model[config.model_ids[0]] += (
            total_expected_tasks - total_allocated
        )

    # Build task queue with interleaved order
    task_queue: list[tuple[str, str]] = []
    task_counters = {model_id: 0 for model_id in config.model_ids}

    for i in range(total_expected_tasks):
        # Select model based on QPS ratios using weighted random choice
        weights = [
            config.qps_ratios[j] / total_ratio
            for j in range(len(config.model_ids))
        ]
        model_id = random.choices(config.model_ids, weights=weights)[0]
        if task_counters[model_id] < tasks_per_model[model_id]:
            task_id = f"llm-{uuid.uuid4().hex[:8]}-{i:06d}"
            task_queue.append((task_id, model_id))
            task_counters[model_id] += 1

    console.print(
        f"\n[bold cyan]Workload Configuration:[/bold cyan]"
        f"\n  Total QPS: {config.total_qps:.2f}"
        f"\n  Duration: {config.duration_seconds:.1f}s"
        f"\n  Expected Tasks: {total_expected_tasks}"
        f"\n  QPS Ratio: 5:1:3 (fast:medium:slow)"
    )
    for model_id, qps in model_qps.items():
        console.print(
            f"    {model_id}: {qps:.2f} QPS ({tasks_per_model[model_id]} tasks)"
        )
    console.print("")

    # Create progress display
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )

    submit_task_id = progress.add_task(
        "Submitting tasks", total=total_expected_tasks
    )
    complete_task_id = progress.add_task(
        "Completed tasks", total=total_expected_tasks
    )

    interval = 1.0 / config.total_qps

    async with httpx.AsyncClient() as client:
        with Live(progress, console=console, refresh_per_second=10):
            # Phase 1: Submit tasks to per-model schedulers
            next_submit_time = time.time()
            for task_id, model_id in task_queue:
                task_stat = TaskStats(
                    task_id=task_id,
                    model_id=model_id,
                    submit_time=time.time(),
                )
                stats.add_task(task_stat)

                # Wait until next submission time
                current_time = time.time()
                if current_time < next_submit_time:
                    await asyncio.sleep(next_submit_time - current_time)

                # Get random prompt
                prompt = random.choice(config.sample_prompts)

                # Route to the correct per-model scheduler
                scheduler_url = router.get_scheduler_url(model_id)
                success = await submit_task(
                    client, scheduler_url, task_id, model_id, prompt
                )
                if not success:
                    stats.mark_complete(task_id, "failed", error="Submit failed")
                    progress.update(complete_task_id, advance=1)

                progress.update(submit_task_id, advance=1)
                next_submit_time += interval

            progress.update(submit_task_id, description="[green]Tasks submitted")

            # Phase 2: Poll for completion from per-model schedulers
            pending_tasks = [
                (t.task_id, t.model_id)
                for t in stats.tasks.values()
                if t.status == "pending"
            ]

            poll_start = time.time()
            poll_timeout = config.wait_timeout_seconds

            while pending_tasks and (time.time() - poll_start) < poll_timeout:
                still_pending = []

                for task_id, model_id in pending_tasks:
                    scheduler_url = router.get_scheduler_url(model_id)
                    poll_status, error, exec_time = await poll_task_status(
                        client, scheduler_url, task_id
                    )

                    if poll_status in ("completed", "failed"):
                        stats.mark_complete(
                            task_id, poll_status, exec_time, error
                        )
                        progress.update(complete_task_id, advance=1)
                    else:
                        still_pending.append((task_id, model_id))

                pending_tasks = still_pending
                if pending_tasks:
                    await asyncio.sleep(0.5)

            # Mark remaining as timeout
            for task_id, _ in pending_tasks:
                stats.mark_complete(
                    task_id, "failed", error="Timeout"
                )
                progress.update(complete_task_id, advance=1)

    stats.end_time = time.time()
    return stats


def print_results(stats: WorkloadStats, config: WorkloadConfig) -> None:
    """Print formatted results."""
    summary = stats.summary()

    # Overall summary table
    console.print("\n[bold cyan]Overall Results:[/bold cyan]")
    overall_table = Table(show_header=False, box=None)
    overall_table.add_column("Metric", style="dim")
    overall_table.add_column("Value", style="bold")

    overall_table.add_row("Total Tasks", str(summary["total_tasks"]))
    overall_table.add_row("Completed", f"[green]{summary['completed']}[/green]")
    overall_table.add_row("Failed", f"[red]{summary['failed']}[/red]")
    overall_table.add_row("Pending", str(summary["pending"]))
    overall_table.add_row("Duration", f"{summary['duration_s']:.2f}s")
    overall_table.add_row("Actual QPS", f"{summary['actual_qps']:.2f}")
    overall_table.add_row(
        "Avg Submit Latency", f"{summary['avg_submit_latency_ms']:.2f}ms"
    )

    console.print(overall_table)

    # Per-model statistics
    console.print("\n[bold cyan]Per-Model Statistics:[/bold cyan]")
    model_table = Table()
    model_table.add_column("Model")
    model_table.add_column("Total", justify="right")
    model_table.add_column("Completed", justify="right")
    model_table.add_column("Failed", justify="right")
    model_table.add_column("Avg Submit Latency", justify="right")
    model_table.add_column("Avg Execution Time", justify="right")

    for model_id in config.model_ids:
        model_stats = stats.get_model_stats(model_id)
        model_table.add_row(
            model_id,
            str(model_stats["total"]),
            f"[green]{model_stats['completed']}[/green]",
            (
                f"[red]{model_stats['failed']}[/red]"
                if model_stats["failed"] > 0
                else "0"
            ),
            f"{model_stats['avg_submit_latency_ms']:.2f}ms",
            f"{model_stats['avg_execution_time_ms']:.2f}ms",
        )

    console.print(model_table)

    # Traffic analysis
    console.print("\n[bold cyan]Traffic Analysis:[/bold cyan]")
    for model_id in config.model_ids:
        model_stats = stats.get_model_stats(model_id)
        console.print(f"  {model_id}: {model_stats['total']} tasks")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate workload for LLM Cluster (Multi-Scheduler)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--planner-url",
        type=str,
        default="http://localhost:8003",
        help="Planner URL for scheduler discovery",
    )
    parser.add_argument(
        "--total-qps",
        type=float,
        default=10.0,
        help="Target queries per second",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Test duration in seconds",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=600.0,
        help="Timeout for task completion in seconds",
    )

    args = parser.parse_args()

    console.print(
        "[bold blue]╔════════════════════════════════════════════════════════╗[/bold blue]"
    )
    console.print(
        "[bold blue]║     LLM Cluster - Workload Generator (Multi)          ║[/bold blue]"
    )
    console.print(
        "[bold blue]╚════════════════════════════════════════════════════════╝[/bold blue]"
    )

    # Discover schedulers from planner
    console.print("\nDiscovering schedulers from planner...")
    router = SchedulerRouter(args.planner_url)
    try:
        router.refresh()
        console.print(
            f"[green]✓ Found {len(router.scheduler_map)} schedulers[/green]"
        )
        for model_id, url in router.scheduler_map.items():
            console.print(f"  {model_id}: {url}")
    except RuntimeError as e:
        console.print(f"[red]Failed to discover schedulers: {e}[/red]")
        sys.exit(1)

    # Health-check each scheduler
    console.print("\nChecking scheduler health...")
    for model_id, scheduler_url in router.scheduler_map.items():
        try:
            response = httpx.get(
                f"{scheduler_url}/v1/health", timeout=5.0
            )
            if response.status_code != 200:
                console.print(
                    f"[red]Scheduler for {model_id} not healthy "
                    f"at {scheduler_url}[/red]"
                )
                sys.exit(1)
            console.print(f"[green]✓ {model_id} scheduler is healthy[/green]")
        except Exception as e:
            console.print(
                f"[red]Cannot connect to {model_id} scheduler: {e}[/red]"
            )
            sys.exit(1)

    # Check instances are registered in each scheduler
    console.print("Checking registered instances...")
    total_instances = 0
    for model_id, scheduler_url in router.scheduler_map.items():
        try:
            response = httpx.get(
                f"{scheduler_url}/v1/instance/list", timeout=5.0
            )
            if response.status_code == 200:
                instances = response.json().get("instances", [])
                total_instances += len(instances)
                console.print(
                    f"  {model_id}: {len(instances)} instances"
                )
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not check {model_id} instances: "
                f"{e}[/yellow]"
            )

    if total_instances == 0:
        console.print(
            "[red]No instances registered. "
            "Run deploy_model.sh first.[/red]"
        )
        sys.exit(1)

    console.print(
        f"[green]✓ {total_instances} total instances registered[/green]"
    )

    # Create config and run workload
    config = WorkloadConfig(
        planner_url=args.planner_url,
        total_qps=args.total_qps,
        duration_seconds=args.duration,
        wait_timeout_seconds=args.wait_timeout,
    )

    stats = asyncio.run(run_workload(config, router))

    # Print results
    print_results(stats, config)


if __name__ == "__main__":
    main()
