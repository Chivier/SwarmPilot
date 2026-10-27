"""Hardware names for PyLet workers in a mixed cluster.

PyLet workers report GPU counts but not GPU types, so the Planner learns
each worker's hardware from a map. Every instance is registered with the
hardware of the worker it runs on, which keeps predictor models for
different GPUs apart (predictor keys include ``hardware_name``), and a
deployment can ask for one kind of hardware.

The map comes from ``PLANNER_HARDWARE_MAP``: a JSON object, or the path of
a JSON or YAML file holding one, whose keys are shell-style patterns
matched against the worker id first and then its host, in file order:

    {"gpu-h100-*": "H100-SXM-80GB", "10.0.2.*": "A100-40GB"}

In sim mode the Planner fills the map from the simulator's node list.
Workers that match nothing get ``PLANNER_HARDWARE_NAME`` (default
``"gpu"``), which is what every instance got before.
"""

from __future__ import annotations

import os
import threading
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

DEFAULT_HARDWARE = "gpu"


def load_hardware_map(raw: str | None) -> dict[str, str]:
    """Parse ``PLANNER_HARDWARE_MAP``.

    Args:
        raw: A JSON object, a path to a JSON or YAML file, or None.

    Returns:
        Pattern to hardware name, in the order given.

    Raises:
        ValueError: If the value is not a mapping of strings.
    """
    if not raw or not raw.strip():
        return {}
    text = raw.strip()
    if not text.startswith("{"):
        path = Path(text).expanduser()
        if not path.is_file():
            raise ValueError(f"PLANNER_HARDWARE_MAP file not found: {text}")
        text = path.read_text()
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"PLANNER_HARDWARE_MAP is not valid: {e}") from e
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise ValueError(
            "PLANNER_HARDWARE_MAP must map worker patterns to hardware names"
        )
    return dict(data)


def matches(hardware: str, selector: str) -> bool:
    """Whether a hardware name matches a selector (a pattern, any case)."""
    return fnmatchcase(hardware.lower(), selector.lower())


class HardwareResolver:
    """Finds the hardware name of PyLet workers.

    Thread-safe: deployments and the instance watcher run on different
    threads.

    Args:
        mapping: Pattern to hardware name, matched against worker id, then
            host. Defaults to ``PLANNER_HARDWARE_MAP``.
        default: Name for workers no pattern matches. Defaults to
            ``PLANNER_HARDWARE_NAME`` or ``"gpu"``.
    """

    def __init__(
        self,
        mapping: dict[str, str] | None = None,
        default: str | None = None,
    ) -> None:
        self._mapping = (
            dict(mapping)
            if mapping is not None
            else load_hardware_map(os.getenv("PLANNER_HARDWARE_MAP"))
        )
        self.default = (
            default or os.getenv("PLANNER_HARDWARE_NAME") or DEFAULT_HARDWARE
        )
        self._hosts: dict[str, str] = {}
        self._lock = threading.Lock()

    @property
    def heterogeneous(self) -> bool:
        """Whether any worker can resolve to a non-default name."""
        return bool(self._mapping)

    def update(self, mapping: dict[str, str]) -> None:
        """Add exact or pattern entries; they take precedence over older ones.

        Args:
            mapping: Pattern (or exact worker id) to hardware name.
        """
        with self._lock:
            rest = {k: v for k, v in self._mapping.items() if k not in mapping}
            self._mapping = {**mapping, **rest}
        logger.info(f"Hardware map: {len(mapping)} entries added")

    def observe_workers(self, workers: list[Any]) -> None:
        """Remember worker hosts, so host patterns resolve by worker id.

        Args:
            workers: ``pylet.workers()`` results (objects with ``id`` and
                ``host``) or dicts with ``worker_id`` and ``host``.
        """
        hosts = {}
        for w in workers:
            if isinstance(w, dict):
                wid, host = w.get("worker_id"), w.get("host")
            else:
                wid, host = getattr(w, "id", None), getattr(w, "host", None)
            if wid and host:
                hosts[wid] = host
        with self._lock:
            self._hosts.update(hosts)

    def resolve(self, worker_id: str | None, host: str | None = None) -> str:
        """Return the hardware name of a worker.

        Args:
            worker_id: PyLet worker id, if known.
            host: Worker host; looked up from observed workers if None.

        Returns:
            The matched hardware name, or the default.
        """
        with self._lock:
            if host is None and worker_id:
                host = self._hosts.get(worker_id)
            for pattern, name in self._mapping.items():
                if worker_id and fnmatchcase(worker_id, pattern):
                    return name
            if host:
                for pattern, name in self._mapping.items():
                    if fnmatchcase(host, pattern):
                        return name
        return self.default

    def describe(self) -> dict[str, Any]:
        """Return the map and default, for status endpoints and logs."""
        with self._lock:
            return {"default": self.default, "map": dict(self._mapping)}


def sim_hardware_map(control_url: str, timeout: float = 5.0) -> dict[str, str]:
    """Read node id to GPU type from the simulator's control API.

    Args:
        control_url: ``swarmpilot-sim`` control API base URL.
        timeout: Request timeout in seconds.

    Returns:
        Worker id to GPU name, or an empty map if the API is unreachable.
    """
    import httpx

    try:
        nodes = httpx.get(f"{control_url}/nodes", timeout=timeout).json()
    except (httpx.HTTPError, ValueError) as e:
        logger.warning(f"Could not read simulated nodes: {e}")
        return {}
    return {n["id"]: n["gpu"] for n in nodes if n.get("id") and n.get("gpu")}
