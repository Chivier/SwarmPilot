#!/usr/bin/env python3
"""E2E LLM Cluster Test Orchestration.

This script orchestrates a 32-worker PyLet cluster serving 3 LLM models:
1. Start PyLet cluster (head + 32 workers, 1 GPU limit each)
2. Start scheduler + predictor + planner
3. Use planner to compute optimal deployment based on QPS and runtime ratios
4. Deploy mock vLLM instances via PyLet
5. Run QPS-based workload through scheduler
6. Collect results in OpenAI-compatible format
7. Generate comprehensive report

Model Configuration:
- Runtime ratio: 1:5:20 (llm_fast:llm_medium:llm_slow)
- QPS ratio: 5:1:3 (llm_fast:llm_medium:llm_slow)

Usage:
    python -m tests.integration.e2e_llm_cluster.run_e2e_llm_cluster

    # With custom parameters
    python -m tests.integration.e2e_llm_cluster.run_e2e_llm_cluster \
        --total-qps 15 --duration 120 --workers 32

Environment Variables:
    E2E_LLM_LOG_DIR: Directory for logs (default: /tmp/e2e_llm_cluster_logs)
    E2E_LLM_OUTPUT_DIR: Directory for reports (default: ./e2e_llm_cluster_results)
"""

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.integration.e2e_llm_cluster.llm_workload_generator import (
    LLMWorkloadConfig,
    LLMWorkloadGenerator,
    wait_for_llm_task_completion,
)


@dataclass
class ClusterConfig:
    """Configuration for PyLet cluster."""

    num_workers: int = 32
    gpu_per_worker: int = 1  # Limit each worker to 1 GPU
    pylet_head_port: int = 15400
    pylet_worker_port_start: int = 15500


@dataclass
class ServiceConfig:
    """Configuration for services."""

    scheduler_port: int = 8000
    predictor_port: int = 8002
    planner_port: int = 8003
    instance_port_start: int = 8100
    log_dir: Path = field(
        default_factory=lambda: Path("/tmp/e2e_llm_cluster_logs")
    )


@dataclass
class TestConfig:
    """Test configuration."""

    # QPS configuration
    total_qps: float = 10.0
    duration_seconds: float = 60.0

    # Model configuration (3 LLM models)
    model_ids: list[str] = field(
        default_factory=lambda: ["llm_fast", "llm_medium", "llm_slow"]
    )

    # Runtime ratio 1:5:20 - used for capacity calculation
    runtime_ratios: list[float] = field(
        default_factory=lambda: [1.0, 5.0, 20.0]
    )

    # QPS ratio 5:1:3
    qps_ratios: list[float] = field(
        default_factory=lambda: [5.0, 1.0, 3.0]
    )

    task_completion_timeout: float = 600.0
    output_dir: Path = field(
        default_factory=lambda: Path("./e2e_llm_cluster_results")
    )

    def get_capacity_matrix(self, num_workers: int) -> list[list[float]]:
        """Build capacity matrix B for optimizer.

        B[i][j] = 1/runtime_ratio for model j (normalized capacity per worker)
        This means faster models have higher capacity (can handle more requests).
        """
        # Normalize runtime ratios (inverse for capacity)
        max_runtime = max(self.runtime_ratios)
        capacities = [max_runtime / r for r in self.runtime_ratios]

        # Each worker can run any model, but with different capacities
        return [capacities for _ in range(num_workers)]

    def get_target_distribution(self) -> list[float]:
        """Get target QPS distribution (normalized)."""
        total = sum(self.qps_ratios)
        return [q / total for q in self.qps_ratios]


class E2ELLMClusterOrchestrator:
    """Orchestrates the 32-worker LLM cluster E2E test."""

    def __init__(
        self,
        cluster_config: ClusterConfig,
        service_config: ServiceConfig,
        test_config: TestConfig,
    ):
        """Initialize orchestrator."""
        self.cluster_config = cluster_config
        self.service_config = service_config
        self.test_config = test_config
        self.processes: dict[str, subprocess.Popen] = {}
        self._cleanup_done = False

    async def run(self) -> dict[str, Any]:
        """Run the complete E2E test.

        Returns:
            Dict with test results and report paths
        """
        try:
            # Setup
            self._setup_logging()
            self._setup_directories()

            # Start all services
            await self._start_pylet_cluster()
            await self._start_predictor()
            await self._start_scheduler()
            await self._start_planner()
            await self._wait_for_services()

            # Compute and apply deployment via planner
            deployment_result = await self._compute_and_deploy()

            # Verify all instances are ready to accept requests
            await self._verify_instances_ready(deployment_result)

            # Run workload
            workload_result = await self._run_workload()

            # Wait for completion and collect results
            task_results = await self._wait_for_completion(workload_result)

            # Generate report
            report = self._generate_report(
                deployment_result, workload_result, task_results
            )

            return {
                "success": True,
                "report_json": str(self.test_config.output_dir / "report.json"),
                "summary": report,
            }

        except Exception as e:
            logger.exception(f"E2E test failed: {e}")
            return {"success": False, "error": str(e)}

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

    async def _start_pylet_cluster(self) -> None:
        """Start PyLet head node and workers."""
        logger.info(
            f"Starting PyLet cluster: 1 head + {self.cluster_config.num_workers} workers"
        )

        # Clean up stale PyLet state
        pylet_state_dir = Path.home() / ".pylet"
        if pylet_state_dir.exists():
            logger.info("Cleaning up stale PyLet state...")
            for db_file in pylet_state_dir.glob("pylet.db*"):
                try:
                    db_file.unlink()
                    logger.debug(f"Removed {db_file.name}")
                except Exception as e:
                    logger.debug(f"Could not remove {db_file}: {e}")
            # Also clean state and workers directories
            for subdir in ["state", "workers"]:
                subdir_path = pylet_state_dir / subdir
                if subdir_path.exists():
                    try:
                        shutil.rmtree(subdir_path)
                        logger.debug(f"Removed {subdir} directory")
                    except Exception as e:
                        logger.debug(f"Could not remove {subdir}: {e}")

        # Python code to start PyLet head
        head_code = f'''
import pylet
import sys
print("Starting PyLet head on port {self.cluster_config.pylet_head_port}...")
sys.stdout.flush()
pylet.start(port={self.cluster_config.pylet_head_port}, block=True)
'''
        head_log = open(self.service_config.log_dir / "pylet_head.log", "w")
        head_proc = subprocess.Popen(
            [sys.executable, "-c", head_code],
            stdout=head_log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )
        self.processes["pylet_head"] = head_proc
        logger.info(
            f"PyLet head started (PID: {head_proc.pid}, "
            f"port: {self.cluster_config.pylet_head_port})"
        )

        # Wait for head to be ready
        await asyncio.sleep(3)

        # Start workers with 200-port gap for instance ports
        WORKER_PORT_GAP = 200
        for i in range(self.cluster_config.num_workers):
            worker_http_port = self.cluster_config.pylet_worker_port_start + i * WORKER_PORT_GAP
            instance_port_start = worker_http_port + 1
            instance_port_end = worker_http_port + 100

            # Python code to start PyLet worker
            worker_code = f'''
import pylet
import time
import sys
print("Starting PyLet worker {i} on port {worker_http_port}...")
sys.stdout.flush()
time.sleep(1)
pylet.start(
    address="http://localhost:{self.cluster_config.pylet_head_port}",
    port={worker_http_port},
    cpu=1,
    gpu={self.cluster_config.gpu_per_worker},
    memory=1024,
    block=True
)
'''
            worker_env = {
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYLET_WORKER_HTTP_PORT": str(worker_http_port),
                "PYLET_WORKER_PORT_MIN": str(instance_port_start),
                "PYLET_WORKER_PORT_MAX": str(instance_port_end),
            }

            worker_log = open(
                self.service_config.log_dir / f"pylet_worker_{i:02d}.log", "w"
            )

            worker_proc = subprocess.Popen(
                [sys.executable, "-c", worker_code],
                stdout=worker_log,
                stderr=subprocess.STDOUT,
                env=worker_env,
                start_new_session=True,
            )
            self.processes[f"pylet_worker_{i:02d}"] = worker_proc

            # Brief delay between workers
            if i < self.cluster_config.num_workers - 1:
                await asyncio.sleep(0.3)

        logger.success(
            f"Started {self.cluster_config.num_workers} PyLet workers"
        )

        # Wait for workers to register
        await asyncio.sleep(5)

        # Verify workers connected
        await self._verify_pylet_cluster()

    async def _verify_pylet_cluster(self) -> None:
        """Verify PyLet cluster is operational."""
        logger.info("Verifying PyLet cluster...")

        pylet_head_url = f"http://localhost:{self.cluster_config.pylet_head_port}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            # Check worker count using /workers endpoint
            try:
                response = await client.get(f"{pylet_head_url}/workers")
                if response.status_code == 200:
                    workers = response.json()
                    if isinstance(workers, list):
                        logger.info(f"PyLet cluster: {len(workers)} workers connected")
                        if len(workers) < self.cluster_config.num_workers:
                            logger.warning(
                                f"Expected {self.cluster_config.num_workers} workers, "
                                f"got {len(workers)}"
                            )
                    else:
                        logger.debug(f"Workers response: {workers}")
            except Exception as e:
                logger.warning(f"Could not verify workers: {e}")

    async def _start_predictor(self) -> None:
        """Start mock predictor server."""
        logger.info("Starting predictor server...")

        predictor_log = open(self.service_config.log_dir / "predictor.log", "w")

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
            stdout=predictor_log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=PROJECT_ROOT,
        )

        self.processes["predictor"] = proc
        logger.info(
            f"Predictor started (PID: {proc.pid}, "
            f"port: {self.service_config.predictor_port})"
        )

        await asyncio.sleep(2)

    async def _start_scheduler(self) -> None:
        """Start scheduler server."""
        logger.info("Starting scheduler server...")

        scheduler_log = open(self.service_config.log_dir / "scheduler.log", "w")
        scheduler_dir = PROJECT_ROOT / "scheduler"

        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath = (
            f"{scheduler_dir}:{existing_pythonpath}"
            if existing_pythonpath
            else str(scheduler_dir)
        )

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
            stdout=scheduler_log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=scheduler_dir,
        )

        self.processes["scheduler"] = proc
        logger.info(
            f"Scheduler started (PID: {proc.pid}, "
            f"port: {self.service_config.scheduler_port})"
        )

        await asyncio.sleep(3)

    async def _start_planner(self) -> None:
        """Start planner server with PyLet integration."""
        logger.info("Starting planner server...")

        planner_log = open(self.service_config.log_dir / "planner.log", "w")
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
            "PLANNER_PORT": str(self.service_config.planner_port),
            "SCHEDULER_URL": f"http://localhost:{self.service_config.scheduler_port}",
            "PYLET_ENABLED": "true",
            "PYLET_HEAD_URL": f"http://localhost:{self.cluster_config.pylet_head_port}",
            "PYLET_DEFAULT_GPU_COUNT": str(self.cluster_config.gpu_per_worker),
            "PYLET_REUSE_CLUSTER": "true",
            # Custom command to run mock vLLM server
            "PYLET_CUSTOM_COMMAND": (
                f"{sys.executable} "
                f"{PROJECT_ROOT}/tests/integration/e2e_llm_cluster/mock_vllm_server.py"
            ),
            "PLANNER_LOGURU_LEVEL": "INFO",
            "PYTHONUNBUFFERED": "1",
        }

        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "src.api:app",
                "--host", "0.0.0.0",
                "--port", str(self.service_config.planner_port),
                "--log-level", "info",
            ],
            stdout=planner_log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=planner_dir,
        )

        self.processes["planner"] = proc
        logger.info(
            f"Planner started (PID: {proc.pid}, "
            f"port: {self.service_config.planner_port})"
        )

        await asyncio.sleep(3)

    async def _wait_for_services(self) -> None:
        """Wait for all services to be healthy."""
        logger.info("Waiting for services to be healthy...")

        # Services with standard /health endpoint
        services = [
            ("predictor", f"http://localhost:{self.service_config.predictor_port}/health"),
            ("scheduler", f"http://localhost:{self.service_config.scheduler_port}/health"),
            ("planner", f"http://localhost:{self.service_config.planner_port}/health"),
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

            # PyLet head uses /workers endpoint instead of /health
            pylet_url = f"http://localhost:{self.cluster_config.pylet_head_port}/workers"
            for attempt in range(30):
                try:
                    response = await client.get(pylet_url)
                    if response.status_code == 200:
                        workers = response.json()
                        logger.success(f"pylet_head is healthy ({len(workers)} workers)")
                        break
                except Exception:
                    pass

                if attempt == 29:
                    raise RuntimeError("pylet_head failed to become healthy")

                await asyncio.sleep(1)

    async def _compute_and_deploy(self) -> dict[str, Any]:
        """Compute instance distribution and deploy via planner.

        Uses the /pylet/deploy endpoint with a pre-calculated distribution:
        - instance_count = QPS_ratio * runtime_ratio (normalized to fit workers)

        For QPS ratio 5:1:3 and runtime ratio 1:5:20:
        - llm_fast: 5 * 1 = 5 units
        - llm_medium: 1 * 5 = 5 units
        - llm_slow: 3 * 20 = 60 units
        Total: 70 units -> distribute N workers proportionally
        """
        logger.info("Computing instance distribution...")

        # Calculate capacity-weighted distribution
        # capacity_needed = QPS_ratio * runtime_ratio
        capacity_units = [
            qps * runtime
            for qps, runtime in zip(
                self.test_config.qps_ratios,
                self.test_config.runtime_ratios
            )
        ]
        total_units = sum(capacity_units)

        # Distribute workers proportionally (ensure at least 1 per model)
        n_workers = self.cluster_config.num_workers
        n_models = len(self.test_config.model_ids)

        # First pass: proportional allocation
        raw_allocation = [
            (units / total_units) * n_workers
            for units in capacity_units
        ]

        # Second pass: ensure minimum of 1 per model and round
        target_state: dict[str, int] = {}
        allocated = 0
        for i, model_id in enumerate(self.test_config.model_ids):
            count = max(1, int(round(raw_allocation[i])))
            target_state[model_id] = count
            allocated += count

        # Adjust if over-allocated
        while allocated > n_workers:
            # Remove from the model with most instances
            max_model = max(target_state, key=lambda m: target_state[m])
            if target_state[max_model] > 1:
                target_state[max_model] -= 1
                allocated -= 1

        # Adjust if under-allocated
        while allocated < n_workers:
            # Add to the model with highest capacity need
            max_idx = capacity_units.index(max(capacity_units))
            max_model = self.test_config.model_ids[max_idx]
            target_state[max_model] += 1
            allocated += 1

        logger.info(f"Target distribution: {target_state}")
        logger.info(f"Capacity units: {dict(zip(self.test_config.model_ids, capacity_units))}")

        # Deploy using /pylet/deploy endpoint
        planner_url = f"http://localhost:{self.service_config.planner_port}"

        deploy_request = {
            "target_state": target_state,
            "wait_for_ready": True,
            "register_with_scheduler": True,
        }

        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{planner_url}/pylet/deploy",
                json=deploy_request,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Deployment failed: HTTP {response.status_code}: {response.text}"
                )

            result = response.json()

        if not result.get("success"):
            raise RuntimeError(f"Deployment failed: {result.get('error')}")

        # Log deployment summary
        active_instances = result.get("active_instances", [])
        model_counts: dict[str, int] = {}
        for inst in active_instances:
            model_id = inst.get("model_id", "unknown")
            model_counts[model_id] = model_counts.get(model_id, 0) + 1

        logger.success(f"Deployment complete: {len(active_instances)} instances")
        for model_id, count in model_counts.items():
            logger.info(f"  {model_id}: {count} instances")

        return {
            "target_state": target_state,
            "capacity_units": dict(zip(self.test_config.model_ids, capacity_units)),
            "active_instances": active_instances,
            "model_distribution": model_counts,
        }

    async def _verify_instances_ready(
        self, deployment_result: dict[str, Any], timeout: float = 60.0
    ) -> None:
        """Verify all deployed instances are ready to accept HTTP requests.

        This ensures instances are fully initialized before workload starts,
        preventing ConnectError on initial tasks.
        """
        active_instances = deployment_result.get("active_instances", [])
        if not active_instances:
            logger.warning("No instances to verify")
            return

        # Extract endpoints from active instances
        endpoints: list[tuple[str, str]] = []
        for inst in active_instances:
            endpoint = inst.get("endpoint")
            instance_id = inst.get("instance_id", "unknown")
            if endpoint:
                # Ensure endpoint has http:// prefix
                if not endpoint.startswith("http"):
                    endpoint = f"http://{endpoint}"
                endpoints.append((instance_id, endpoint))

        if not endpoints:
            logger.warning("No endpoints found in active instances")
            return

        logger.info(f"Verifying {len(endpoints)} instances are ready...")

        # Verify each instance can respond to health checks
        start_time = time.time()
        ready_instances: set[str] = set()
        failed_instances: dict[str, str] = {}

        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.time() - start_time < timeout:
                pending = [
                    (iid, ep) for iid, ep in endpoints
                    if iid not in ready_instances
                ]

                if not pending:
                    break

                # Check all pending instances
                for instance_id, endpoint in pending:
                    try:
                        response = await client.get(f"{endpoint}/health")
                        if response.status_code == 200:
                            ready_instances.add(instance_id)
                            logger.debug(f"Instance {instance_id} ready at {endpoint}")
                    except httpx.ConnectError as e:
                        failed_instances[instance_id] = f"ConnectError: {e}"
                    except httpx.TimeoutException:
                        failed_instances[instance_id] = "Timeout"
                    except Exception as e:
                        failed_instances[instance_id] = str(e)

                remaining = len(endpoints) - len(ready_instances)
                if remaining > 0:
                    logger.debug(
                        f"Waiting for {remaining} instances... "
                        f"({len(ready_instances)}/{len(endpoints)} ready)"
                    )
                    await asyncio.sleep(1.0)

        # Log results
        elapsed = time.time() - start_time
        if len(ready_instances) == len(endpoints):
            logger.success(
                f"All {len(endpoints)} instances ready in {elapsed:.1f}s"
            )
        else:
            not_ready = [iid for iid, _ in endpoints if iid not in ready_instances]
            logger.warning(
                f"Only {len(ready_instances)}/{len(endpoints)} instances ready "
                f"after {elapsed:.1f}s. Not ready: {not_ready[:5]}..."
            )

            # Log first few failure reasons
            for iid in not_ready[:3]:
                reason = failed_instances.get(iid, "Unknown")
                logger.warning(f"  {iid}: {reason}")

    async def _run_workload(self):
        """Run LLM workload with QPS ratios."""
        expected_tasks = int(
            self.test_config.total_qps * self.test_config.duration_seconds
        )

        logger.info(
            f"Running LLM workload: {self.test_config.total_qps} QPS for "
            f"{self.test_config.duration_seconds}s ({expected_tasks} tasks)"
        )

        config = LLMWorkloadConfig(
            scheduler_url=f"http://localhost:{self.service_config.scheduler_port}",
            total_qps=self.test_config.total_qps,
            duration_seconds=self.test_config.duration_seconds,
            model_ids=self.test_config.model_ids,
            qps_ratios=self.test_config.qps_ratios,
        )

        generator = LLMWorkloadGenerator(config)
        return await generator.generate_workload()

    async def _wait_for_completion(self, workload_result) -> dict[str, dict[str, Any]]:
        """Wait for all tasks to complete."""
        task_ids = [
            r.task_id for r in workload_result.submission_results if r.success
        ]

        if not task_ids:
            logger.warning("No successful task submissions to wait for")
            return {}

        logger.info(f"Waiting for {len(task_ids)} tasks to complete...")

        return await wait_for_llm_task_completion(
            scheduler_url=f"http://localhost:{self.service_config.scheduler_port}",
            task_ids=task_ids,
            timeout_seconds=self.test_config.task_completion_timeout,
        )

    def _generate_report(
        self,
        deployment_result: dict[str, Any],
        workload_result,
        task_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate and save test report."""
        logger.info("Generating test report...")

        # Calculate statistics
        completed = sum(
            1 for r in task_results.values()
            if r.get("status") == "completed"
        )
        failed = sum(
            1 for r in task_results.values()
            if r.get("status") == "failed"
        )
        timeout = sum(
            1 for r in task_results.values()
            if r.get("status") == "timeout"
        )

        # Task distribution by model
        tasks_by_model = workload_result.tasks_by_model()

        # Extract execution times from completed tasks
        execution_times: dict[str, list[float]] = {
            model_id: [] for model_id in self.test_config.model_ids
        }

        for task_id, result in task_results.items():
            if result.get("status") == "completed":
                exec_time = result.get("execution_time_ms", 0)
                # Find model from submission
                for sub in workload_result.submission_results:
                    if sub.task_id == task_id:
                        execution_times[sub.model_id].append(exec_time)
                        break

        # Calculate percentiles
        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]

        execution_stats = {}
        for model_id, times in execution_times.items():
            if times:
                execution_stats[model_id] = {
                    "count": len(times),
                    "p50_ms": percentile(times, 50),
                    "p90_ms": percentile(times, 90),
                    "p99_ms": percentile(times, 99),
                    "avg_ms": sum(times) / len(times),
                }

        report = {
            "test_config": {
                "total_qps": self.test_config.total_qps,
                "duration_seconds": self.test_config.duration_seconds,
                "model_ids": self.test_config.model_ids,
                "runtime_ratios": self.test_config.runtime_ratios,
                "qps_ratios": self.test_config.qps_ratios,
            },
            "cluster_config": {
                "num_workers": self.cluster_config.num_workers,
                "gpu_per_worker": self.cluster_config.gpu_per_worker,
            },
            "deployment": {
                "model_distribution": deployment_result.get("model_distribution"),
                "target_state": deployment_result.get("target_state"),
                "capacity_units": deployment_result.get("capacity_units"),
            },
            "workload": {
                "tasks_submitted": workload_result.total_tasks,
                "tasks_successful_submit": workload_result.successful_tasks,
                "tasks_failed_submit": workload_result.failed_tasks,
                "actual_qps": workload_result.actual_qps,
                "tasks_by_model": tasks_by_model,
            },
            "execution": {
                "tasks_completed": completed,
                "tasks_failed": failed,
                "tasks_timeout": timeout,
                "success_rate": completed / max(workload_result.successful_tasks, 1),
                "execution_stats": execution_stats,
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save report
        report_path = self.test_config.output_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        logger.info("=" * 60)
        logger.info("E2E LLM CLUSTER TEST COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Workers: {self.cluster_config.num_workers}")
        logger.info(f"Deployment: {deployment_result.get('model_distribution')}")
        logger.info(f"Tasks submitted: {workload_result.total_tasks}")
        logger.info(f"Tasks completed: {completed}")
        logger.info(f"Tasks failed: {failed + timeout}")
        logger.info(f"Actual QPS: {workload_result.actual_qps:.2f}")

        if execution_stats:
            logger.info("Execution times (p50):")
            for model_id, stats in execution_stats.items():
                logger.info(f"  {model_id}: {stats['p50_ms']:.2f}ms")

        logger.info(f"Report: {report_path}")
        logger.info("=" * 60)

        return report

    async def _cleanup(self) -> None:
        """Stop all processes."""
        if self._cleanup_done:
            return

        self._cleanup_done = True
        logger.info("Cleaning up processes...")

        # Terminate all via planner first
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(
                    f"http://localhost:{self.service_config.planner_port}/pylet/terminate-all"
                )
        except Exception:
            pass

        # Stop all processes (including process groups for PyLet)
        for name, proc in self.processes.items():
            try:
                # For PyLet processes, kill the entire process group
                if name.startswith("pylet_"):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        logger.debug(f"Terminated process group {name}")
                    except (ProcessLookupError, PermissionError):
                        pass
                else:
                    proc.terminate()

                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
                logger.debug(f"Stopped {name}")
            except Exception as e:
                logger.debug(f"Error stopping {name}: {e}")

        self.processes.clear()
        logger.info("Cleanup complete")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="E2E LLM Cluster Test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--total-qps", type=float, default=10.0,
        help="Total queries per second across all models",
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Test duration in seconds",
    )
    parser.add_argument(
        "--workers", type=int, default=32,
        help="Number of PyLet workers",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./e2e_llm_cluster_results",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--log-dir", type=str, default="/tmp/e2e_llm_cluster_logs",
        help="Directory for log files",
    )

    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    cluster_config = ClusterConfig(num_workers=args.workers)
    service_config = ServiceConfig(log_dir=Path(args.log_dir))
    test_config = TestConfig(
        total_qps=args.total_qps,
        duration_seconds=args.duration,
        output_dir=Path(args.output_dir),
    )

    orchestrator = E2ELLMClusterOrchestrator(
        cluster_config=cluster_config,
        service_config=service_config,
        test_config=test_config,
    )

    # Handle signals
    def signal_handler(signum, frame):
        logger.warning(f"Received signal {signum}, shutting down...")
        asyncio.create_task(orchestrator._cleanup())
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    result = await orchestrator.run()

    if result["success"]:
        logger.success("E2E test completed successfully")
        return 0
    else:
        logger.error(f"E2E test failed: {result.get('error')}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
