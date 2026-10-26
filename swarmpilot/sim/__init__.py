"""SwarmPilot cluster simulator: test and benchmark without GPUs.

The simulator is written in Rust (``rust/crates/sim-core`` and the
``swarmpilot-sim`` binary in ``rust/crates/sim-cli``). This package is its
Python face: it ships the catalogs and cluster specs the binary reads, finds
the binary, and runs it.

Data files:
    ``data/gpus.yaml``: datasheet numbers per GPU model.
    ``data/models.yaml``: LLM shapes and diffusion/generic latency anchors.
    ``data/engines.yaml``: per-engine efficiency factors and limits.
    ``specs/default-256.yaml``: 9 hardware configurations, 288 GPUs, 256
    engine instances.

Example:
    from swarmpilot import sim

    summary = sim.plan()  # the default 256-instance spec
    row = sim.bench(
        engine="vllm", model="llama-3.1-8b", gpu="H100-SXM-80GB", qps=4
    )
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
SPECS_DIR = PACKAGE_DIR / "specs"
DEFAULT_SPEC = SPECS_DIR / "default-256.yaml"

_BIN_NAME = "swarmpilot-sim"
_REPO_RUST = PACKAGE_DIR.parent.parent / "rust" / "target"


class SimulatorNotFoundError(RuntimeError):
    """The ``swarmpilot-sim`` binary is not built or not on PATH."""


def find_binary() -> Path | None:
    """Locate the ``swarmpilot-sim`` binary.

    Looks at ``SWARMPILOT_SIM_BIN``, then ``PATH``, then a source
    checkout's ``rust/target/release`` and ``rust/target/debug``.

    Returns:
        The binary's path, or None when it is not built.
    """
    env = os.getenv("SWARMPILOT_SIM_BIN")
    if env:
        return Path(env) if Path(env).is_file() else None
    found = shutil.which(_BIN_NAME)
    if found:
        return Path(found)
    for profile in ("release", "debug"):
        candidate = _REPO_RUST / profile / _BIN_NAME
        if candidate.is_file():
            return candidate
    return None


def run_cli(*args: str, timeout: float = 600.0) -> Any:
    """Run ``swarmpilot-sim`` with ``args`` and parse its JSON output.

    The binary reads the catalogs in this package (``--catalog``), so
    edits to ``data/*.yaml`` take effect without a rebuild.

    Args:
        *args: Command-line arguments, e.g. ``"plan"``.
        timeout: Seconds before the run is killed.

    Returns:
        The decoded JSON value.

    Raises:
        SimulatorNotFoundError: The binary is not built.
        RuntimeError: The binary exited with an error.
    """
    binary = find_binary()
    if binary is None:
        raise SimulatorNotFoundError(
            "swarmpilot-sim is not built; run "
            "`cargo build --release --manifest-path rust/Cargo.toml` "
            "or set SWARMPILOT_SIM_BIN"
        )
    proc = subprocess.run(
        [str(binary), "--catalog", str(DATA_DIR), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"exit {proc.returncode}")
    return json.loads(proc.stdout)


def plan(spec: str | Path | None = None, instances: bool = False) -> Any:
    """Validate a cluster spec and return its placement.

    Args:
        spec: Spec file; defaults to ``specs/default-256.yaml``.
        instances: Return every placed instance instead of the totals.

    Returns:
        The plan summary, or the list of instances.
    """
    args = ["plan", str(spec or DEFAULT_SPEC)]
    if instances:
        args.append("--instances")
    return run_cli(*args)


def profile(engine: str, model: str, gpu: str, tp: int = 1) -> dict:
    """Return the resolved performance profile for one configuration."""
    return run_cli(
        "profile",
        "--engine",
        engine,
        "--model",
        model,
        "--gpu",
        gpu,
        "--tp",
        str(tp),
    )


def bench(
    engine: str,
    model: str,
    gpu: str,
    tp: int = 1,
    qps: float = 0.0,
    requests: int = 200,
    seed: int = 0,
    slo_ms: float | None = None,
    extra: list[str] | None = None,
) -> dict:
    """Replay a request stream through one simulated engine.

    Args:
        engine: ``vllm``, ``sglang``, ``diffusers`` or ``generic``.
        model: Catalog name or alias, e.g. ``llama-3.1-8b``.
        gpu: GPU catalog name, e.g. ``H100-SXM-80GB``.
        tp: Tensor-parallel degree.
        qps: Poisson arrival rate; 0 sends everything at once.
        requests: Number of requests.
        seed: Workload seed.
        slo_ms: End-to-end SLO for the miss rate.
        extra: More ``bench`` flags, e.g. ``["--prompt", "32"]``.

    Returns:
        The summary row (throughput, TTFT, TPOT, ITL, end-to-end).
    """
    args = [
        "bench",
        "--engine",
        engine,
        "--model",
        model,
        "--gpu",
        gpu,
        "--tp",
        str(tp),
        "--qps",
        str(qps),
        "--requests",
        str(requests),
        "--seed",
        str(seed),
    ]
    if slo_ms is not None:
        args += ["--slo-ms", str(slo_ms)]
    return run_cli(*args, *(extra or []))
