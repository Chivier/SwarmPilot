"""Multi-Scheduler Orchestrator for E2E experiments.

This module provides the main orchestrator that:
1. Starts PyLet cluster (head + workers)
2. Starts planner with PyLet integration
3. Starts one scheduler per model (with planner registration)
4. Deploys instances via planner /deploy_manually
5. Runs multi-scheduler workload
6. Generates comprehensive reports

Usage:
    from tests.integration.e2e_multi_scheduler import (
        ExperimentConfig,
        MultiSchedulerOrchestrator,
    )

    config = ExperimentConfig(...)
    orchestrator = MultiSchedulerOrchestrator(config)
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

# Project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

from .config import ExperimentConfig  # noqa: E402
from .workload_generator import (  # noqa: E402
    MultiSchedulerWorkloadConfig,
    MultiSchedulerWorkloadGenerator,
    wait_for_multi_scheduler_tasks,
)
from .report_generator import (  # noqa: E402
    MultiSchedulerReportGenerator,
    MultiSchedulerSystemConfig,
    MultiSchedulerTestParams,
)


class MultiSchedulerOrchestrator:
    """Orchestrates multi-scheduler E2E experiments.

    This class manages the complete lifecycle of an experiment:
    - Starting/stopping all services
    - Deploying instances via planner
    - Running workload across multiple schedulers
    - Generating reports

    Attributes:
        config: Experiment configuration.
        processes: Dict of process name to subprocess.Popen.
    """

    def __init__(self, config: ExperimentConfig):
        """Initialize orchestrator.

        Args:
            config: Experiment configuration.
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

            # Start services
            await self._start_planner()
            await self._start_schedulers()
            await self._wait_for_services()

            # Deploy instances via planner
            await self._deploy_instances()

            # Verify deployment
            await self._verify_deployment()

            # Run workload
            workload_result = await self._run_workload()

            # Wait for completion
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
                    "schedulers": len(self.config.models),
                    "instances": self.config.total_instances,
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
        logger.info(f"Models: {self.config.model_ids}")
        logger.info(f"Total instances: {self.config.total_instances}")

    def _setup_directories(self) -> None:
        """Create necessary directories."""
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Log directory: {self.config.log_dir}")
        logger.info(f"Output directory: {self.config.output_dir}")

    async def _start_planner(self) -> None:
        """Start planner service with PyLet integration."""
        logger.info("Starting planner service...")

        log_file = open(self.config.log_dir / "planner.log", "w")
        self._log_files["planner"] = log_file

        # Planner needs PYTHONPATH to include planner directory
        planner_dir = PROJECT_ROOT / "planner"
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath = (
            f"{planner_dir}:{existing_pythonpath}"
            if existing_pythonpath
            else str(planner_dir)
        )

        env = {
            **os.environ,
            "PYTHONPATH": pythonpath,
            "PLANNER_PORT": str(self.config.planner.port),
            "PYLET_ENABLED": "true",
            "PYLET_HEAD_URL": f"localhost:{self.config.cluster.pylet_head_port}",
            "PYLET_BACKEND": self.config.planner.pylet_backend,
            "PYLET_GPU_COUNT": str(self.config.cluster.gpu_per_worker),
            "PYLET_CPU_COUNT": str(self.config.cluster.cpu_per_worker),
            "PYLET_DEPLOY_TIMEOUT": str(self.config.planner.pylet_deploy_timeout),
            "PYLET_DRAIN_TIMEOUT": str(self.config.planner.pylet_drain_timeout),
            "PYLET_REUSE_CLUSTER": (
                "true" if self.config.cluster.reuse_cluster else "false"
            ),
            "PLANNER_LOGURU_LEVEL": "INFO",
            "PYTHONUNBUFFERED": "1",
        }

        if self.config.planner.pylet_custom_command:
            env["PYLET_CUSTOM_COMMAND"] = self.config.planner.pylet_custom_command

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "src.api:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(self.config.planner.port),
                "--log-level",
                "info",
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=planner_dir,
        )

        self.processes["planner"] = proc
        logger.info(
            f"Planner started (PID: {proc.pid}, port: {self.config.planner.port})"
        )

        # Wait for startup
        await asyncio.sleep(3)

    async def _start_schedulers(self) -> None:
        """Start one scheduler per model type."""
        logger.info(f"Starting {len(self.config.models)} schedulers...")

        scheduler_dir = PROJECT_ROOT / "scheduler"
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath = (
            f"{scheduler_dir}:{existing_pythonpath}"
            if existing_pythonpath
            else str(scheduler_dir)
        )

        for model in self.config.models:
            log_file = open(
                self.config.log_dir / f"scheduler_{model.model_id}.log", "w"
            )
            self._log_files[f"scheduler_{model.model_id}"] = log_file

            # Critical env vars for scheduler registration (from scheduler/src/config.py)
            env = {
                **os.environ,
                "PYTHONPATH": pythonpath,
                # Scheduler server config
                "SCHEDULER_PORT": str(model.scheduler_port),
                "SCHEDULER_HOST": "0.0.0.0",
                # Planner registration config (PlannerRegistrationConfig)
                "SCHEDULER_MODEL_ID": model.model_id,
                "PLANNER_REGISTRATION_URL": self.config.planner.planner_url,
                "SCHEDULER_SELF_URL": model.scheduler_url,
                # Use library mode for predictor (no external service)
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
                    str(model.scheduler_port),
                    "--log-level",
                    "info",
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=scheduler_dir,
            )

            self.processes[f"scheduler_{model.model_id}"] = proc
            logger.info(
                f"Scheduler for {model.model_id} started "
                f"(PID: {proc.pid}, port: {model.scheduler_port})"
            )

            # Brief delay between scheduler starts
            await asyncio.sleep(1)

        # Wait for all schedulers to initialize
        await asyncio.sleep(2)

    async def _wait_for_services(self) -> None:
        """Wait for all services to be healthy."""
        logger.info("Waiting for services to be healthy...")

        # Build list of services to check
        services = [
            ("planner", f"{self.config.planner.planner_url}/health"),
        ]
        for model in self.config.models:
            services.append(
                (f"scheduler_{model.model_id}", f"{model.scheduler_url}/health")
            )

        async with httpx.AsyncClient(timeout=10.0) as client:
            for name, url in services:
                for attempt in range(30):
                    try:
                        response = await client.get(url)
                        if response.status_code == 200:
                            logger.success(f"{name} is healthy")
                            break
                    except Exception:
                        pass

                    if attempt == 29:
                        raise RuntimeError(f"{name} failed to become healthy")

                    await asyncio.sleep(1)

        # Verify scheduler registration with planner
        logger.info("Verifying scheduler registration with planner...")
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{self.config.planner.planner_url}/scheduler/list"
            )
            if response.status_code != 200:
                raise RuntimeError(f"Failed to get scheduler list: {response.text}")

            data = response.json()
            registered = data.get("total", 0)
            expected = len(self.config.models)

            if registered < expected:
                # Wait and retry
                for _ in range(10):
                    await asyncio.sleep(2)
                    response = await client.get(
                        f"{self.config.planner.planner_url}/scheduler/list"
                    )
                    data = response.json()
                    registered = data.get("total", 0)
                    if registered >= expected:
                        break

            if registered < expected:
                raise RuntimeError(
                    f"Only {registered}/{expected} schedulers registered with planner"
                )

            logger.success(f"All {registered} schedulers registered with planner")

    async def _deploy_instances(self) -> None:
        """Deploy instances via planner's /deploy_manually endpoint."""
        logger.info(f"Deploying {self.config.total_instances} instances via planner...")

        target_state = self.config.model_distribution

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.config.planner.planner_url}/deploy_manually",
                json={
                    "target_state": target_state,
                    "wait_for_ready": True,
                },
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Deployment failed: HTTP {response.status_code}: {response.text}"
                )

            data = response.json()
            if not data.get("success"):
                raise RuntimeError(f"Deployment failed: {data.get('error')}")

            added = data.get("added_count", 0)
            active = len(data.get("active_instances", []))
            logger.success(f"Deployment complete: added={added}, active={active}")

    async def _verify_deployment(self) -> None:
        """Verify instances are registered with their schedulers."""
        logger.info("Verifying instance registration with schedulers...")

        async with httpx.AsyncClient(timeout=10.0) as client:
            for model in self.config.models:
                # Check instance count on this scheduler
                response = await client.get(f"{model.scheduler_url}/instance/list")

                if response.status_code != 200:
                    logger.warning(
                        f"Failed to get instance list for {model.model_id}: "
                        f"{response.status_code}"
                    )
                    continue

                data = response.json()
                instances = data.get("instances", [])
                count = len(instances)

                if count < model.instance_count:
                    # Wait and retry
                    for _ in range(20):
                        await asyncio.sleep(2)
                        response = await client.get(
                            f"{model.scheduler_url}/instance/list"
                        )
                        if response.status_code == 200:
                            data = response.json()
                            instances = data.get("instances", [])
                            count = len(instances)
                            if count >= model.instance_count:
                                break

                if count >= model.instance_count:
                    logger.success(
                        f"Scheduler {model.model_id}: {count} instances ready"
                    )
                else:
                    logger.warning(
                        f"Scheduler {model.model_id}: only {count}/{model.instance_count} "
                        "instances registered"
                    )

    async def _run_workload(self):
        """Run multi-scheduler workload."""
        qps_distribution = self.config.get_qps_distribution()

        logger.info(
            f"Running workload: total_qps={self.config.total_qps}, "
            f"duration={self.config.duration_seconds}s, "
            f"models={self.config.model_ids}"
        )
        logger.info(f"QPS distribution: {qps_distribution}")

        config = MultiSchedulerWorkloadConfig(
            scheduler_urls=self.config.scheduler_urls,
            model_qps=qps_distribution,
            duration_seconds=self.config.duration_seconds,
            sleep_time_range=self.config.sleep_time_range,
        )

        generator = MultiSchedulerWorkloadGenerator(config)
        return await generator.generate_workload()

    async def _wait_for_completion(self, workload_result) -> dict[str, dict[str, Any]]:
        """Wait for all tasks to complete."""
        # Group task IDs by model for efficient polling
        model_task_ids: dict[str, list[str]] = {m: [] for m in self.config.model_ids}

        for result in workload_result.submission_results:
            if result.success:
                model_task_ids[result.model_id].append(result.task_id)

        total_tasks = sum(len(ids) for ids in model_task_ids.values())
        if total_tasks == 0:
            logger.warning("No tasks to wait for")
            return {}

        return await wait_for_multi_scheduler_tasks(
            scheduler_urls=self.config.scheduler_urls,
            model_task_ids=model_task_ids,
            timeout_seconds=self.config.task_completion_timeout,
        )

    async def _generate_report(self, workload_result, task_results):
        """Generate test report."""
        logger.info("Generating test report...")

        # Collect log paths
        log_paths = {
            "orchestrator": str(self.config.log_dir / "orchestrator.log"),
            "planner": str(self.config.log_dir / "planner.log"),
        }
        for model in self.config.models:
            log_paths[f"scheduler_{model.model_id}"] = str(
                self.config.log_dir / f"scheduler_{model.model_id}.log"
            )

        # Build system config
        system_config = MultiSchedulerSystemConfig(
            planner_url=self.config.planner.planner_url,
            scheduler_urls=self.config.scheduler_urls,
            pylet_head_url=f"localhost:{self.config.cluster.pylet_head_port}",
            num_workers=self.config.cluster.num_workers,
            model_distribution=self.config.model_distribution,
            total_instances=self.config.total_instances,
            total_schedulers=len(self.config.models),
        )

        # Build test params
        test_params = MultiSchedulerTestParams(
            target_qps=self.config.total_qps,
            duration_seconds=self.config.duration_seconds,
            total_tasks=int(self.config.total_qps * self.config.duration_seconds),
            sleep_time_range=self.config.sleep_time_range,
            model_ids=self.config.model_ids,
            qps_distribution=self.config.get_qps_distribution(),
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
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Experiment: {self.config.name}")
        logger.info(f"Schedulers: {len(self.config.models)}")
        logger.info(f"Instances: {self.config.total_instances}")
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

        # Stop processes in reverse order (instances first, then schedulers, then planner)
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
