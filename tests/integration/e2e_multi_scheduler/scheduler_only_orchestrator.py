"""Single-Scheduler Orchestrator for Isolation Testing.

This module provides a simplified orchestrator that tests scheduler behavior
in isolation without planner or PyLet complexity.

Architecture:
    - Single scheduler (no planner registration)
    - Mock LLM instances launched as direct subprocesses
    - All instances register with the single scheduler
    - Uses llm_slow latency characteristics (~2000ms gamma)

The key simplification is that instances are launched directly as subprocesses
with SCHEDULER_URL set, which triggers automatic registration. No PyLet or
planner coordination is required.

Usage:
    from tests.integration.e2e_multi_scheduler import (
        SchedulerOnlyConfig,
        SchedulerOnlyOrchestrator,
    )

    config = SchedulerOnlyConfig(num_instances=4)
    orchestrator = SchedulerOnlyOrchestrator(config)
    result = await orchestrator.run()
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from .experiments.scheduler_only_config import SchedulerOnlyConfig
from .workload_generator import (
    MultiSchedulerWorkloadConfig,
    MultiSchedulerWorkloadGenerator,
    wait_for_multi_scheduler_tasks,
)
from .report_generator import (
    MultiSchedulerReportGenerator,
    MultiSchedulerSystemConfig,
    MultiSchedulerTestParams,
)


# Project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class SchedulerOnlyOrchestrator:
    """Orchestrates single-scheduler isolation experiments.

    This orchestrator manages a simplified experiment lifecycle:
    1. Start scheduler (standalone, no planner registration)
    2. Launch mock LLM instances as subprocesses
    3. Wait for instance registration with scheduler
    4. Run workload against single scheduler
    5. Generate report

    Attributes:
        config: Scheduler-only experiment configuration.
        processes: Dict of process name to subprocess.Popen.
    """

    def __init__(self, config: SchedulerOnlyConfig):
        """Initialize orchestrator.

        Args:
            config: Scheduler-only experiment configuration.
        """
        self.config = config
        self.processes: dict[str, subprocess.Popen] = {}
        self._cleanup_done = False
        self._log_files: dict[str, Any] = {}

    async def run(self) -> dict[str, Any]:
        """Run the complete experiment.

        Returns:
            Dict with results including report paths and summary.
        """
        try:
            # Setup
            self._setup_logging()
            self._setup_directories()

            # Start scheduler (standalone, no planner)
            await self._start_scheduler()
            await self._wait_for_scheduler()

            # Launch instances as subprocesses
            await self._start_instances()
            await self._wait_for_instances()

            # Run workload
            workload_result = await self._run_workload()

            # Wait for task completion
            task_results = await self._wait_for_completion(workload_result)

            # Generate report
            report = await self._generate_report(workload_result, task_results)

            return {
                "success": True,
                "report_json": str(self.config.output_dir / "report.json"),
                "report_md": str(self.config.output_dir / "report.md"),
                "problems": report.problems,
                "summary": {
                    "tasks_submitted": workload_result.total_tasks,
                    "tasks_completed": report.execution_metrics.tasks_completed,
                    "tasks_failed": report.execution_metrics.tasks_failed,
                    "actual_qps": workload_result.actual_qps,
                    "instances": self.config.num_instances,
                },
            }

        except Exception as e:
            logger.exception(f"Experiment failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

        finally:
            await self._cleanup()

    def _setup_logging(self) -> None:
        """Configure logging."""
        logger.remove()
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        )
        log_file = self.config.log_dir / "orchestrator.log"
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        )
        logger.info(f"Experiment: {self.config.name}")
        logger.info(f"Model: {self.config.model_id}")
        logger.info(f"Instances: {self.config.num_instances}")
        logger.info(f"Target QPS: {self.config.total_qps}")

    def _setup_directories(self) -> None:
        """Create necessary directories."""
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Log directory: {self.config.log_dir}")
        logger.info(f"Output directory: {self.config.output_dir}")

    async def _start_scheduler(self) -> None:
        """Start scheduler service in standalone mode.

        The scheduler is started without planner registration by setting
        empty values for SCHEDULER_MODEL_ID and PLANNER_REGISTRATION_URL.
        This allows the scheduler to operate independently.
        """
        logger.info("Starting scheduler in standalone mode...")

        log_file = open(self.config.log_dir / "scheduler.log", "w")
        self._log_files["scheduler"] = log_file

        scheduler_dir = PROJECT_ROOT / "scheduler"
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath = (
            f"{scheduler_dir}:{existing_pythonpath}"
            if existing_pythonpath
            else str(scheduler_dir)
        )

        # Critical: Empty values disable planner registration
        env = {
            **os.environ,
            "PYTHONPATH": pythonpath,
            # Scheduler server config
            "SCHEDULER_PORT": str(self.config.scheduler_port),
            "SCHEDULER_HOST": "0.0.0.0",
            # Empty values disable planner registration
            "SCHEDULER_MODEL_ID": "",
            "PLANNER_REGISTRATION_URL": "",
            "SCHEDULER_SELF_URL": "",
            # Use library mode for predictor (in-process)
            "PREDICTOR_MODE": "library",
            # Scheduling config
            "SCHEDULING_STRATEGY": "probabilistic",
            "SCHEDULER_LOGURU_LEVEL": "INFO",
            "PYTHONUNBUFFERED": "1",
        }

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "src.api:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(self.config.scheduler_port),
                "--log-level",
                "info",
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=scheduler_dir,
        )

        self.processes["scheduler"] = proc
        logger.info(
            f"Scheduler started (PID: {proc.pid}, port: {self.config.scheduler_port})"
        )

    async def _wait_for_scheduler(self) -> None:
        """Wait for scheduler to be healthy."""
        logger.info("Waiting for scheduler to be healthy...")

        async with httpx.AsyncClient(timeout=10.0) as client:
            for attempt in range(30):
                try:
                    response = await client.get(f"{self.config.scheduler_url}/health")
                    if response.status_code == 200:
                        logger.success("Scheduler is healthy")
                        return
                except Exception:
                    pass

                if attempt == 29:
                    raise RuntimeError("Scheduler failed to become healthy")

                await asyncio.sleep(1)

    async def _start_instances(self) -> None:
        """Start mock LLM instances as subprocesses.

        Each instance is launched with:
        - PORT: Unique port for HTTP server
        - MODEL_ID: Model identifier (determines latency distribution)
        - INSTANCE_ID: Unique instance identifier
        - SCHEDULER_URL: Scheduler endpoint for registration

        The mock_vllm_server.py automatically registers with the scheduler
        on startup when SCHEDULER_URL is set.
        """
        logger.info(f"Starting {self.config.num_instances} mock LLM instances...")

        mock_server_path = (
            PROJECT_ROOT
            / "tests"
            / "integration"
            / "e2e_llm_cluster"
            / "mock_vllm_server.py"
        )

        for i in range(self.config.num_instances):
            port = self.config.instance_port_start + i
            instance_id = self.config.get_instance_id(i)

            log_file = open(self.config.log_dir / f"instance_{i}.log", "w")
            self._log_files[f"instance_{i}"] = log_file

            env = {
                **os.environ,
                "PORT": str(port),
                "MODEL_ID": self.config.model_id,
                "INSTANCE_ID": instance_id,
                "SCHEDULER_URL": self.config.scheduler_url,
                "LOG_LEVEL": "INFO",
                "PYTHONUNBUFFERED": "1",
            }

            proc = subprocess.Popen(
                [sys.executable, str(mock_server_path)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )

            self.processes[f"instance_{i}"] = proc
            logger.info(
                f"Instance {instance_id} started (PID: {proc.pid}, port: {port})"
            )

            # Brief delay between instance starts
            await asyncio.sleep(0.5)

        # Wait for instances to initialize
        await asyncio.sleep(2)

    async def _wait_for_instances(self) -> None:
        """Wait for all instances to register with scheduler."""
        logger.info("Waiting for instances to register with scheduler...")

        async with httpx.AsyncClient(timeout=10.0) as client:
            for attempt in range(30):
                try:
                    response = await client.get(
                        f"{self.config.scheduler_url}/instance/list"
                    )

                    if response.status_code == 200:
                        data = response.json()
                        instances = data.get("instances", [])
                        active_count = len(
                            [
                                i
                                for i in instances
                                if i.get("status", "").lower() == "active"
                            ]
                        )

                        if active_count >= self.config.num_instances:
                            logger.success(
                                f"All {active_count} instances registered and ACTIVE"
                            )
                            return

                        logger.debug(
                            f"Waiting for instances: {active_count}/{self.config.num_instances} ACTIVE"
                        )

                except Exception as e:
                    logger.debug(f"Error checking instances: {e}")

                if attempt == 29:
                    # Log final state before failing
                    try:
                        response = await client.get(
                            f"{self.config.scheduler_url}/instance/list"
                        )
                        logger.error(f"Instance list: {response.json()}")
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Only {active_count}/{self.config.num_instances} "
                        "instances registered"
                    )

                await asyncio.sleep(1)

    async def _run_workload(self):
        """Run workload against single scheduler.

        Uses the existing MultiSchedulerWorkloadGenerator with a single
        model/scheduler mapping.
        """
        logger.info(
            f"Running workload: qps={self.config.total_qps}, "
            f"duration={self.config.duration_seconds}s"
        )

        # Create workload config with single model
        config = MultiSchedulerWorkloadConfig(
            scheduler_urls={self.config.model_id: self.config.scheduler_url},
            model_qps={self.config.model_id: self.config.total_qps},
            duration_seconds=self.config.duration_seconds,
            # For llm_slow, we don't use explicit sleep_time - the model's
            # latency distribution handles it. But we need some value for
            # the task_input, so use a small range.
            sleep_time_range=(0.1, 0.5),
            task_id_prefix="scheduler-only",
        )

        generator = MultiSchedulerWorkloadGenerator(config)
        return await generator.generate_workload()

    async def _wait_for_completion(self, workload_result) -> dict[str, dict[str, Any]]:
        """Wait for all tasks to complete."""
        # Group task IDs by model (single model in this case)
        model_task_ids: dict[str, list[str]] = {self.config.model_id: []}

        for result in workload_result.submission_results:
            if result.success:
                model_task_ids[self.config.model_id].append(result.task_id)

        total_tasks = len(model_task_ids[self.config.model_id])
        if total_tasks == 0:
            logger.warning("No tasks to wait for")
            return {}

        logger.info(f"Waiting for {total_tasks} tasks to complete...")

        return await wait_for_multi_scheduler_tasks(
            scheduler_urls={self.config.model_id: self.config.scheduler_url},
            model_task_ids=model_task_ids,
            timeout_seconds=self.config.task_completion_timeout,
        )

    async def _generate_report(self, workload_result, task_results):
        """Generate test report."""
        logger.info("Generating test report...")

        # Collect log paths
        log_paths = {
            "orchestrator": str(self.config.log_dir / "orchestrator.log"),
            "scheduler": str(self.config.log_dir / "scheduler.log"),
        }
        for i in range(self.config.num_instances):
            log_paths[f"instance_{i}"] = str(self.config.log_dir / f"instance_{i}.log")

        # Build system config (adapted for single-scheduler)
        system_config = MultiSchedulerSystemConfig(
            planner_url="N/A (standalone mode)",
            scheduler_urls={self.config.model_id: self.config.scheduler_url},
            pylet_head_url="N/A (direct subprocess)",
            num_workers=self.config.num_instances,
            model_distribution={self.config.model_id: self.config.num_instances},
            total_instances=self.config.num_instances,
            total_schedulers=1,
        )

        # Build test params
        test_params = MultiSchedulerTestParams(
            target_qps=self.config.total_qps,
            duration_seconds=self.config.duration_seconds,
            total_tasks=self.config.expected_tasks,
            sleep_time_range=(0.1, 0.5),
            model_ids=[self.config.model_id],
            qps_distribution={self.config.model_id: self.config.total_qps},
        )

        # Generate report
        generator = MultiSchedulerReportGenerator(
            workload_result=workload_result,
            task_results=task_results,
            system_config=system_config,
            test_params=test_params,
            log_paths=log_paths,
        )

        report = generator.generate_report(include_raw=True)

        # Write reports
        report.to_json(self.config.output_dir / "report.json")
        report.to_markdown(self.config.output_dir / "report.md")

        # Print summary
        self._print_summary(report)

        return report

    def _print_summary(self, report) -> None:
        """Print test summary to console."""
        logger.info("=" * 60)
        logger.info("SCHEDULER ISOLATION EXPERIMENT COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Experiment: {self.config.name}")
        logger.info(f"Model: {self.config.model_id}")
        logger.info(f"Instances: {self.config.num_instances}")
        logger.info(f"Tasks submitted: {report.execution_metrics.tasks_submitted}")
        logger.info(f"Tasks completed: {report.execution_metrics.tasks_completed}")
        logger.info(f"Tasks failed: {report.execution_metrics.tasks_failed}")
        logger.info(f"Actual QPS: {report.execution_metrics.actual_qps:.2f}")
        logger.info(f"Submission p50: {report.submission_latency.p50:.2f}ms")
        logger.info(f"Submission p99: {report.submission_latency.p99:.2f}ms")

        if report.problems:
            logger.warning("Problems identified:")
            for problem in report.problems:
                logger.warning(f"  - {problem}")
        else:
            logger.success("No problems identified")

        logger.info(f"Report: {self.config.output_dir / 'report.md'}")
        logger.info("=" * 60)

    async def _cleanup(self) -> None:
        """Stop all processes and close resources."""
        if self._cleanup_done:
            return

        self._cleanup_done = True
        logger.info("Cleaning up processes...")

        # Stop processes in reverse order (instances first, then scheduler)
        process_names = list(self.processes.keys())[::-1]

        for name in process_names:
            proc = self.processes[name]
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
                logger.debug(f"Stopped {name}")
            except Exception as e:
                logger.debug(f"Error stopping {name}: {e}")

        # Close log files
        for name, log_file in self._log_files.items():
            try:
                log_file.close()
            except Exception:
                pass

        logger.info("Cleanup complete")
