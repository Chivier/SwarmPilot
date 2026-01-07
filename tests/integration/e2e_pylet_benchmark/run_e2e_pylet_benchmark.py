#!/usr/bin/env python3
"""E2E PyLet Benchmark Test Orchestration.

This script orchestrates the complete end-to-end test:
1. Start PyLet cluster (head + workers)
2. Start mock predictor server
3. Start scheduler (pointing to mock predictor)
4. Deploy sleep model instances via scheduler registration
5. Run QPS workload test
6. Collect results and generate report
7. Clean up all processes

Usage:
    # Run with default settings (5 QPS, 60 seconds)
    python -m tests.integration.e2e_pylet_benchmark.run_e2e_pylet_benchmark

    # Run with custom settings
    python -m tests.integration.e2e_pylet_benchmark.run_e2e_pylet_benchmark \
        --qps 10 --duration 120 --output-dir /tmp/benchmark_results

Environment Variables:
    E2E_LOG_DIR: Directory for logs (default: /tmp/e2e_pylet_benchmark_logs)
    E2E_OUTPUT_DIR: Directory for reports (default: ./e2e_benchmark_results)
"""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.integration.e2e_pylet_benchmark.workload_generator import (
    WorkloadConfig,
    WorkloadGenerator,
    wait_for_task_completion,
)
from tests.integration.e2e_pylet_benchmark.log_collector import LogCollector
from tests.integration.e2e_pylet_benchmark.report_generator import (
    ReportGenerator,
    SystemConfig,
    TestParams,
)


@dataclass
class ServiceConfig:
    """Configuration for test services."""

    scheduler_port: int = 8000
    predictor_port: int = 8002
    instance_port_start: int = 8100
    instance_port_gap: int = 1
    log_dir: Path = field(default_factory=lambda: Path("/tmp/e2e_pylet_benchmark_logs"))

    def instance_port(self, index: int) -> int:
        """Get port for instance at index."""
        return self.instance_port_start + index * self.instance_port_gap


@dataclass
class TestConfig:
    """Configuration for the test."""

    qps: float = 5.0
    duration_seconds: float = 60.0
    model_ids: list[str] = field(
        default_factory=lambda: ["sleep_model_a", "sleep_model_b", "sleep_model_c"]
    )
    model_distribution: dict[str, int] = field(default_factory=lambda: {
        "sleep_model_a": 4,
        "sleep_model_b": 3,
        "sleep_model_c": 3,
    })
    sleep_time_range: tuple[float, float] = (0.1, 1.0)
    task_completion_timeout: float = 300.0
    output_dir: Path = field(default_factory=lambda: Path("./e2e_benchmark_results"))

    @property
    def total_instances(self) -> int:
        """Total number of instances to deploy."""
        return sum(self.model_distribution.values())


class E2EBenchmarkOrchestrator:
    """Orchestrates the complete E2E benchmark test."""

    def __init__(self, service_config: ServiceConfig, test_config: TestConfig):
        """Initialize orchestrator.

        Args:
            service_config: Service configuration
            test_config: Test configuration
        """
        self.service_config = service_config
        self.test_config = test_config
        self.processes: dict[str, subprocess.Popen] = {}
        self._cleanup_done = False

    async def run(self) -> dict[str, Any]:
        """Run the complete benchmark.

        Returns:
            Dict with report path and summary
        """
        try:
            # Setup
            self._setup_logging()
            self._setup_directories()

            # Start services
            await self._start_predictor()
            await self._start_scheduler()
            await self._wait_for_services()

            # Deploy instances
            await self._deploy_instances()

            # Run workload
            workload_result = await self._run_workload()

            # Wait for task completion
            task_results = await self._wait_for_completion(workload_result)

            # Generate report
            report = await self._generate_report(workload_result, task_results)

            return {
                "success": True,
                "report_json": str(self.test_config.output_dir / "report.json"),
                "report_md": str(self.test_config.output_dir / "report.md"),
                "problems": report.problems,
                "summary": {
                    "tasks_submitted": workload_result.total_tasks,
                    "tasks_completed": report.execution_metrics.tasks_completed,
                    "tasks_failed": report.execution_metrics.tasks_failed,
                    "actual_qps": workload_result.actual_qps,
                },
            }

        except Exception as e:
            logger.exception(f"Benchmark failed: {e}")
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
        logger.add(
            self.service_config.log_dir / "orchestrator.log",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        )

    def _setup_directories(self) -> None:
        """Create necessary directories."""
        self.service_config.log_dir.mkdir(parents=True, exist_ok=True)
        self.test_config.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Log directory: {self.service_config.log_dir}")
        logger.info(f"Output directory: {self.test_config.output_dir}")

    async def _start_predictor(self) -> None:
        """Start mock predictor server."""
        logger.info("Starting mock predictor server...")

        log_file = open(self.service_config.log_dir / "predictor.log", "w")

        env = {
            **os.environ,
            "PREDICTOR_PORT": str(self.service_config.predictor_port),
            "PYTHONUNBUFFERED": "1",
        }

        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "tests.integration.e2e_pylet_benchmark.mock_predictor_server:app",
                "--host", "0.0.0.0",
                "--port", str(self.service_config.predictor_port),
                "--log-level", "info",
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=PROJECT_ROOT,
        )

        self.processes["predictor"] = proc
        logger.info(f"Predictor started (PID: {proc.pid}, port: {self.service_config.predictor_port})")

        # Wait for startup
        await asyncio.sleep(2)

    async def _start_scheduler(self) -> None:
        """Start scheduler server."""
        logger.info("Starting scheduler server...")

        log_file = open(self.service_config.log_dir / "scheduler.log", "w")

        # Add scheduler directory to PYTHONPATH so its internal imports work
        # The scheduler uses `from src.xxx` imports which require the scheduler
        # directory to be in the Python path
        scheduler_dir = PROJECT_ROOT / "scheduler"
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath = f"{scheduler_dir}:{existing_pythonpath}" if existing_pythonpath else str(scheduler_dir)

        env = {
            **os.environ,
            "PYTHONPATH": pythonpath,
            "SCHEDULER_PORT": str(self.service_config.scheduler_port),
            "PREDICTOR_URL": f"http://localhost:{self.service_config.predictor_port}",
            "SCHEDULING_STRATEGY": "probabilistic",
            "SCHEDULER_LOGURU_LEVEL": "INFO",
            "PYTHONUNBUFFERED": "1",
        }

        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "src.api:app",
                "--host", "0.0.0.0",
                "--port", str(self.service_config.scheduler_port),
                "--log-level", "info",
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=scheduler_dir,
        )

        self.processes["scheduler"] = proc
        logger.info(f"Scheduler started (PID: {proc.pid}, port: {self.service_config.scheduler_port})")

        # Wait for startup
        await asyncio.sleep(3)

    async def _wait_for_services(self) -> None:
        """Wait for all services to be healthy."""
        logger.info("Waiting for services to be healthy...")

        services = [
            ("predictor", f"http://localhost:{self.service_config.predictor_port}/health"),
            ("scheduler", f"http://localhost:{self.service_config.scheduler_port}/health"),
        ]

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

    async def _deploy_instances(self) -> None:
        """Deploy sleep model instances by registering directly with scheduler."""
        logger.info(f"Deploying {self.test_config.total_instances} instances...")

        instance_index = 0
        scheduler_url = f"http://localhost:{self.service_config.scheduler_port}"

        for model_id, count in self.test_config.model_distribution.items():
            for i in range(count):
                instance_id = f"{model_id}-{i:03d}"
                port = self.service_config.instance_port(instance_index)

                # Start instance process
                await self._start_instance(instance_id, model_id, port)

                # Register with scheduler
                await self._register_instance(scheduler_url, instance_id, model_id, port)

                instance_index += 1

        logger.success(f"Deployed {instance_index} instances")

    async def _start_instance(self, instance_id: str, model_id: str, port: int) -> None:
        """Start a sleep model instance.

        Args:
            instance_id: Instance identifier
            model_id: Model identifier
            port: Port to run on
        """
        log_file = open(self.service_config.log_dir / f"instance_{instance_id}.log", "w")

        env = {
            **os.environ,
            "PORT": str(port),
            "MODEL_ID": model_id,
            "INSTANCE_ID": instance_id,
            "SCHEDULER_URL": "",  # Don't auto-register, we'll do it manually
            "LOG_LEVEL": "INFO",
            "PYTHONUNBUFFERED": "1",
        }

        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).parent / "pylet_sleep_model.py"),
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=PROJECT_ROOT,
        )

        self.processes[f"instance_{instance_id}"] = proc
        logger.debug(f"Started instance {instance_id} (PID: {proc.pid}, port: {port})")

        # Brief wait for startup
        await asyncio.sleep(0.5)

    async def _register_instance(
        self, scheduler_url: str, instance_id: str, model_id: str, port: int
    ) -> None:
        """Register instance with scheduler.

        Args:
            scheduler_url: Scheduler URL
            instance_id: Instance identifier
            model_id: Model identifier
            port: Instance port
        """
        # Wait for instance to be ready
        instance_url = f"http://localhost:{port}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            for attempt in range(10):
                try:
                    response = await client.get(f"{instance_url}/health")
                    if response.status_code == 200:
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.5)
            else:
                raise RuntimeError(f"Instance {instance_id} failed to start")

            # Register with scheduler
            registration_data = {
                "instance_id": instance_id,
                "model_id": model_id,
                "endpoint": instance_url,
                "platform_info": {
                    "software_name": "python",
                    "software_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                    "hardware_name": "cpu",
                },
            }

            response = await client.post(
                f"{scheduler_url}/instance/register",
                json=registration_data,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to register {instance_id}: {response.text}"
                )

            logger.debug(f"Registered instance {instance_id}")

    async def _run_workload(self):
        """Run QPS-based workload."""
        logger.info(
            f"Running workload: {self.test_config.qps} QPS for "
            f"{self.test_config.duration_seconds}s "
            f"({int(self.test_config.qps * self.test_config.duration_seconds)} tasks)"
        )

        config = WorkloadConfig(
            scheduler_url=f"http://localhost:{self.service_config.scheduler_port}",
            qps=self.test_config.qps,
            duration_seconds=self.test_config.duration_seconds,
            model_ids=self.test_config.model_ids,
            sleep_time_range=self.test_config.sleep_time_range,
        )

        generator = WorkloadGenerator(config)
        return await generator.generate_workload()

    async def _wait_for_completion(self, workload_result) -> dict[str, dict[str, Any]]:
        """Wait for all tasks to complete."""
        # Get task IDs from successful submissions
        task_ids = [
            r.task_id
            for r in workload_result.submission_results
            if r.success
        ]

        if not task_ids:
            logger.warning("No tasks to wait for")
            return {}

        return await wait_for_task_completion(
            scheduler_url=f"http://localhost:{self.service_config.scheduler_port}",
            task_ids=task_ids,
            timeout_seconds=self.test_config.task_completion_timeout,
        )

    async def _generate_report(self, workload_result, task_results):
        """Generate test report."""
        logger.info("Generating test report...")

        # Collect logs
        log_collector = LogCollector(self.service_config.log_dir)

        log_sources = {
            "orchestrator": str(self.service_config.log_dir / "orchestrator.log"),
            "predictor": str(self.service_config.log_dir / "predictor.log"),
            "scheduler": str(self.service_config.log_dir / "scheduler.log"),
        }

        # Add instance logs
        for name in self.processes:
            if name.startswith("instance_"):
                instance_id = name[9:]
                log_sources[name] = str(
                    self.service_config.log_dir / f"instance_{instance_id}.log"
                )

        log_paths = log_collector.collect_logs(log_sources)

        # Build system config
        system_config = SystemConfig(
            scheduler_url=f"http://localhost:{self.service_config.scheduler_port}",
            planner_url="N/A (direct registration)",
            predictor_url=f"http://localhost:{self.service_config.predictor_port}",
            pylet_head_url="N/A (standalone instances)",
            num_workers=self.test_config.total_instances,
            scheduling_strategy="probabilistic",
            model_distribution=self.test_config.model_distribution,
            total_instances=self.test_config.total_instances,
        )

        # Build test params
        test_params = TestParams(
            target_qps=self.test_config.qps,
            duration_seconds=self.test_config.duration_seconds,
            total_tasks=int(self.test_config.qps * self.test_config.duration_seconds),
            sleep_time_range=self.test_config.sleep_time_range,
            model_ids=self.test_config.model_ids,
        )

        # Generate report
        generator = ReportGenerator(
            workload_result=workload_result,
            task_results=task_results,
            system_config=system_config,
            test_params=test_params,
            log_paths=log_paths,
        )

        report = generator.generate_report(include_raw=True)

        # Write reports
        report.to_json(self.test_config.output_dir / "report.json")
        report.to_markdown(self.test_config.output_dir / "report.md")

        # Print summary
        logger.info("=" * 60)
        logger.info("BENCHMARK COMPLETE")
        logger.info("=" * 60)
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

        logger.info(f"Report: {self.test_config.output_dir / 'report.md'}")
        logger.info("=" * 60)

        return report

    async def _cleanup(self) -> None:
        """Stop all processes."""
        if self._cleanup_done:
            return

        self._cleanup_done = True
        logger.info("Cleaning up processes...")

        for name, proc in self.processes.items():
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

        logger.info("Cleanup complete")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="E2E PyLet Benchmark Test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--qps", type=float, default=5.0,
        help="Target queries per second",
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Test duration in seconds",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./e2e_benchmark_results",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--log-dir", type=str, default="/tmp/e2e_pylet_benchmark_logs",
        help="Directory for log files",
    )
    parser.add_argument(
        "--scheduler-port", type=int, default=8000,
        help="Scheduler port",
    )
    parser.add_argument(
        "--predictor-port", type=int, default=8002,
        help="Predictor port",
    )

    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    service_config = ServiceConfig(
        scheduler_port=args.scheduler_port,
        predictor_port=args.predictor_port,
        log_dir=Path(args.log_dir),
    )

    test_config = TestConfig(
        qps=args.qps,
        duration_seconds=args.duration,
        output_dir=Path(args.output_dir),
    )

    orchestrator = E2EBenchmarkOrchestrator(service_config, test_config)

    # Handle signals
    def signal_handler(signum, frame):
        logger.warning(f"Received signal {signum}, shutting down...")
        asyncio.create_task(orchestrator._cleanup())
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    result = await orchestrator.run()

    if result["success"]:
        logger.success("Benchmark completed successfully")
        return 0
    else:
        logger.error(f"Benchmark failed: {result.get('error')}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
