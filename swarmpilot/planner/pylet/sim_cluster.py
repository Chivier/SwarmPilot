"""Simulated PyLet cluster: a real PyLet head with simulated workers.

``PYLET_SIM_MODE=true`` makes the Planner start this instead of
:class:`~swarmpilot.planner.pylet.local_cluster.LocalPyLetCluster`. The
head is the real ``pylet`` head; every node of a cluster spec is a
simulated worker run by the Rust ``swarmpilot-sim up`` binary. When the
head assigns an instance, the sim worker starts a virtual engine (vLLM,
SGLang, diffusion or generic, on the node's GPU type) instead of running
the command, so the Planner, the Scheduler and PyLet's own placement all
run unchanged without GPUs.

The head keeps its state in a fresh temporary directory (its ``HOME`` and
``PYLET_DATA_DIR``), so every sim cluster starts empty and never touches
``~/.pylet``.

Example:
    cluster = SimPyLetCluster(port=5100, time_scale=0.1)
    cluster.start()  # default spec: 102 nodes, 288 GPUs
    # ... Planner operates ...
    cluster.stop()
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from loguru import logger

from swarmpilot import sim

_HEAD_SCRIPT = """\
import pylet
pylet.start(port={port}, block=True)
"""


class SimPyLetCluster:
    """A real PyLet head plus ``swarmpilot-sim up`` workers.

    Args:
        port: Head port.
        spec: Cluster spec file; defaults to the 256-instance spec.
        time_scale: Wall seconds per simulated second (0.1 runs 10x
            faster, cold starts included).
        port_start: First simulated node's worker port; node ``i`` uses
            ``port_start + i * port_gap`` and the ports above it.
        port_gap: Ports per node.
        control_port: Port of the simulator's control API.
        binary: Path to ``swarmpilot-sim``; found automatically if None.
        ready_timeout: Seconds to wait for every node to register.
    """

    def __init__(
        self,
        port: int = 5100,
        spec: str | Path | None = None,
        time_scale: float = 1.0,
        port_start: int = 21000,
        port_gap: int = 100,
        control_port: int = 20999,
        binary: str | Path | None = None,
        ready_timeout: float = 60.0,
    ) -> None:
        self.port = port
        self.spec = Path(spec) if spec else sim.DEFAULT_SPEC
        self.time_scale = time_scale
        self.port_start = port_start
        self.port_gap = port_gap
        self.control_port = control_port
        self.binary = Path(binary) if binary else sim.find_binary()
        self.ready_timeout = ready_timeout
        self.nodes = 0
        self._processes: list[subprocess.Popen] = []
        self._data_dir: tempfile.TemporaryDirectory | None = None

    @property
    def head_url(self) -> str:
        """Return the head URL."""
        return f"http://localhost:{self.port}"

    @property
    def control_url(self) -> str:
        """Return the simulator control API URL."""
        return f"http://127.0.0.1:{self.control_port}"

    def start(self) -> bool:
        """Start the head and the simulated workers.

        Returns:
            True once every node has registered with the head.
        """
        if self.binary is None:
            logger.error(
                "PYLET_SIM_MODE needs the swarmpilot-sim binary: run "
                "`cargo build --release --manifest-path rust/Cargo.toml` "
                "or set SWARMPILOT_SIM_BIN"
            )
            return False
        self._data_dir = tempfile.TemporaryDirectory(prefix="pylet-sim-")
        logs = Path(self._data_dir.name)
        # The head keeps its database at ~/.pylet/pylet.db whatever
        # PYLET_DATA_DIR says, so HOME points at the temp directory too.
        env = {
            **os.environ,
            "HOME": self._data_dir.name,
            "PYLET_DATA_DIR": self._data_dir.name,
        }
        # Output goes to files, not pipes: nobody drains a pipe while the
        # cluster runs, and a full pipe would block the head or the sim.
        head = self._spawn(
            [sys.executable, "-c", _HEAD_SCRIPT.format(port=self.port)],
            logs / "head.log",
            env,
        )
        if not self._wait(lambda c: c.get(f"{self.head_url}/workers"), 30):
            logger.error(f"PyLet head did not come up: {_tail(head)}")
            self.stop()
            return False

        workers = self._spawn(
            [
                str(self.binary),
                "--catalog",
                str(sim.DATA_DIR),
                "up",
                "--head",
                self.head_url,
                "--spec",
                str(self.spec),
                "--time-scale",
                str(self.time_scale),
                "--port-start",
                str(self.port_start),
                "--port-gap",
                str(self.port_gap),
                "--control-port",
                str(self.control_port),
            ],
            logs / "sim.log",
            None,
        )
        self.nodes = self._read_node_count(workers)
        if not self.nodes:
            logger.error(f"swarmpilot-sim up failed: {_tail(workers)}")
            self.stop()
            return False

        def all_registered(c: httpx.Client) -> httpx.Response:
            r = c.get(f"{self.head_url}/workers")
            online = [w for w in r.json() if w.get("status") == "ONLINE"]
            if len(online) < self.nodes:
                raise httpx.HTTPError("not all nodes registered")
            return r

        if not self._wait(all_registered, self.ready_timeout):
            logger.error(f"Not all {self.nodes} simulated nodes registered")
            self.stop()
            return False
        logger.success(
            f"Simulated PyLet cluster ready at {self.head_url}: "
            f"{self.nodes} nodes from {self.spec.name}, "
            f"time scale {self.time_scale}, control {self.control_url}"
        )
        return True

    def stop(self) -> None:
        """Stop the workers and the head."""
        for proc in reversed(self._processes):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        for proc in self._processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
        self._processes.clear()
        if self._data_dir is not None:
            self._data_dir.cleanup()
            self._data_dir = None

    @property
    def log_dir(self) -> Path | None:
        """Return the directory holding ``head.log`` and ``sim.log``."""
        return Path(self._data_dir.name) if self._data_dir else None

    def _spawn(
        self, argv: list[str], log: Path, env: dict[str, str] | None
    ) -> subprocess.Popen:
        with open(log, "wb") as out:
            proc = subprocess.Popen(
                argv,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
        proc.log_path = log  # type: ignore[attr-defined]
        self._processes.append(proc)
        return proc

    def _read_node_count(self, proc: subprocess.Popen) -> int:
        """Wait for the ``{"event": "up"}`` line and return its node count."""
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            for line in proc.log_path.read_text().splitlines():  # type: ignore[attr-defined]
                if line.startswith("{"):
                    with contextlib.suppress(ValueError, KeyError, TypeError):
                        event = json.loads(line)
                        if event.get("event") == "up":
                            return int(event["nodes"])
            if proc.poll() is not None:
                return 0
            time.sleep(0.1)
        return 0

    def _wait(self, probe, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with httpx.Client(timeout=5) as c:
            while time.monotonic() < deadline:
                if any(p.poll() is not None for p in self._processes):
                    return False
                try:
                    if probe(c).is_success:
                        return True
                except (httpx.HTTPError, ValueError):
                    pass
                time.sleep(0.3)
        return False


def _tail(proc: subprocess.Popen, lines: int = 20) -> str:
    path = getattr(proc, "log_path", None)
    if path is None or not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
