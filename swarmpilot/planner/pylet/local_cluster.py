"""Local PyLet cluster manager for embedded deployment.

Starts a PyLet head node and worker(s) as subprocesses, allowing the
Planner to manage a local cluster without external setup.  Based on
the ``PyLetCluster`` pattern in ``scripts/run_pylet_integration_tests.py``.

Enable via ``PYLET_LOCAL_MODE=true``.  The Planner lifespan starts the
cluster before ``create_pylet_service()`` and stops it on shutdown.

Example:
    cluster = LocalPyLetCluster(port=5100, num_workers=1, gpu_per_worker=4)
    cluster.start()
    # ... Planner operates ...
    cluster.stop()
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

# Inline Python script executed in a subprocess to start the head node.
_HEAD_SCRIPT = """\
import pylet
import sys

print("Starting PyLet head on port {port}...", flush=True)
pylet.start(port={port}, block=True)
"""

# Inline Python script executed in a subprocess to start a worker node.
_WORKER_SCRIPT = """\
import os
import time
import pylet

worker_http_port = os.environ.get("PYLET_WORKER_HTTP_PORT", "15599")
print(
    f"Starting PyLet worker (http_port={{worker_http_port}}) "
    f"connecting to localhost:{head_port}...",
    flush=True,
)

time.sleep(1)

pylet.start(
    address="http://localhost:{head_port}",
    port=int(worker_http_port),
    cpu={cpu},
    gpu={gpu},
    memory={memory},
    block=True,
)
"""


class LocalPyLetCluster:
    """Manages a local PyLet cluster (head + workers) as subprocesses.

    Args:
        port: Head node port.
        num_workers: Number of worker processes to start.
        cpu_per_worker: CPU cores advertised per worker.
        gpu_per_worker: GPUs advertised per worker.
        worker_port_start: First worker HTTP port.
        worker_port_gap: Port gap between consecutive workers.
        memory_per_worker: Memory (MB) advertised per worker.
    """

    def __init__(
        self,
        port: int = 5100,
        num_workers: int = 1,
        cpu_per_worker: int = 8,
        gpu_per_worker: int = 4,
        worker_port_start: int = 5300,
        worker_port_gap: int = 200,
        memory_per_worker: int = 65536,
    ) -> None:
        self.port = port
        self.num_workers = num_workers
        self.cpu_per_worker = cpu_per_worker
        self.gpu_per_worker = gpu_per_worker
        self.worker_port_start = worker_port_start
        self.worker_port_gap = worker_port_gap
        self.memory_per_worker = memory_per_worker

        self._python = sys.executable
        self._processes: list[subprocess.Popen] = []

    @property
    def head_url(self) -> str:
        """Return the head node URL."""
        return f"http://localhost:{self.port}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the local PyLet cluster.

        Returns:
            True if the cluster started and is verified, False otherwise.
        """
        logger.info(
            f"Starting local PyLet cluster "
            f"(head={self.port}, workers={self.num_workers}, "
            f"gpu/worker={self.gpu_per_worker})"
        )

        self._cleanup_stale_state()

        # -- Head node --
        if not self._start_head():
            return False

        # -- Workers --
        for i in range(self.num_workers):
            if not self._start_worker(i):
                logger.warning(f"Worker {i} may have failed to start")

        # Give workers time to register with the head.
        logger.info("Waiting for workers to connect...")
        time.sleep(5)

        if not self._verify():
            logger.error("Local PyLet cluster verification failed")
            self.stop()
            return False

        logger.success(f"Local PyLet cluster ready at {self.head_url}")
        return True

    def stop(self) -> None:
        """Stop all cluster subprocesses."""
        logger.info("Stopping local PyLet cluster...")

        for proc in self._processes:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                logger.debug(f"Terminated process group {proc.pid}")
            except (ProcessLookupError, PermissionError):
                pass

        time.sleep(1)

        for proc in self._processes:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                proc.kill()

        self._processes.clear()
        logger.info("Local PyLet cluster stopped")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cleanup_stale_state(self) -> None:
        """Remove stale PyLet database files to avoid 500 errors."""
        pylet_dir = Path.home() / ".pylet"
        if not pylet_dir.exists():
            return
        for db_file in pylet_dir.glob("pylet.db*"):
            try:
                db_file.unlink()
                logger.debug(f"Removed stale PyLet state: {db_file.name}")
            except OSError as exc:
                logger.warning(f"Could not remove {db_file}: {exc}")

    def _start_head(self) -> bool:
        """Start the head node subprocess.

        Returns:
            True if the head process is alive after a brief wait.
        """
        code = _HEAD_SCRIPT.format(port=self.port)
        proc = subprocess.Popen(
            [self._python, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self._processes.append(proc)

        # Wait for head to bind its port.
        time.sleep(3)

        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            logger.error(
                f"PyLet head failed to start: "
                f"stdout={stdout.decode()}, stderr={stderr.decode()}"
            )
            return False

        logger.info(f"PyLet head started (PID {proc.pid})")
        return True

    def _start_worker(self, index: int) -> bool:
        """Start a single worker subprocess.

        Args:
            index: Worker index (0-based).

        Returns:
            True if the worker process is alive after a brief wait.
        """
        worker_http_port = self.worker_port_start + index * self.worker_port_gap
        instance_port_start = worker_http_port + 1
        instance_port_end = worker_http_port + 100

        code = _WORKER_SCRIPT.format(
            head_port=self.port,
            cpu=self.cpu_per_worker,
            gpu=self.gpu_per_worker,
            memory=self.memory_per_worker,
        )

        env = {
            **os.environ,
            "PYLET_WORKER_HTTP_PORT": str(worker_http_port),
            "PYLET_WORKER_PORT_MIN": str(instance_port_start),
            "PYLET_WORKER_PORT_MAX": str(instance_port_end),
        }

        proc = subprocess.Popen(
            [self._python, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=env,
        )
        self._processes.append(proc)
        time.sleep(0.5)

        if proc.poll() is not None:
            logger.warning(
                f"Worker {index} (port {worker_http_port}) "
                f"may have failed to start"
            )
            return False

        logger.info(
            f"Worker {index} started (PID {proc.pid}, port {worker_http_port})"
        )
        return True

    def _verify(self) -> bool:
        """Verify the cluster is operational.

        Runs a short verification script in a subprocess that connects
        to the head and checks for available workers.

        Returns:
            True if at least one worker is connected.
        """
        verify_code = f"""\
import httpx
import pylet
from loguru import logger

try:
    resp = httpx.delete("http://localhost:{self.port}/api/workers/offline")
except Exception as exc:
    logger.debug(f"Cleanup failed for offline worker: {{exc}}")

pylet.init("http://localhost:{self.port}")
workers = pylet.workers()
print(f"Connected workers: {{len(workers)}}")
for w in workers:
    print(f"  - {{w.id}}")
if len(workers) == 0:
    exit(1)
"""
        result = subprocess.run(
            [self._python, "-c", verify_code],
            capture_output=True,
            text=True,
        )
        logger.info(result.stdout.strip())
        if result.returncode != 0:
            logger.error(f"Verification stderr: {result.stderr}")
            return False
        return "Connected workers: 0" not in result.stdout
