#!/usr/bin/env python3
"""Parse cluster.yaml and output config values for shell scripts.

Usage:
    python3 _config.py <key>

Keys:
    head_node           Head node IP
    planner_port        Planner listen port
    planner_url         Full Planner URL (http://head:port)
    pylet_head_url      PyLet head URL
    pylet_backend       PyLet backend (vllm/sglang)
    pylet_gpu           GPUs per instance
    pylet_cpu           CPUs per instance
    pylet_timeout       Deploy timeout (seconds)
    model_count         Number of models
    model_ids           Space-separated model IDs
    model_id.<N>        Model ID at index N (0-based)
    scheduler_port.<N>  Scheduler port at index N
    replicas.<N>        Replica count at index N
    node_count          Number of nodes
    node_host.<N>       Node host at index N
    node_gpus.<N>       Node GPU count at index N
    models_json         JSON dict {model_id: replicas} for deploy API
    scheduler_ports     Space-separated scheduler ports
    pylet_head_port     PyLet head listen port
    pylet_worker_port   PyLet worker port start
    pylet_worker_cpu    CPU cores per worker
    pylet_worker_gpu    GPUs per worker
    pylet_worker_mem    RAM (MB) per worker
"""

import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit(
        "Error: PyYAML not installed. Run: uv pip install pyyaml"
    )

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "cluster.yaml"


def load_config() -> dict:
    """Load and return the cluster config dict."""
    if not CONFIG_PATH.exists():
        sys.exit(f"Error: {CONFIG_PATH} not found")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["cluster"]


def get_value(cfg: dict, key: str) -> str:
    """Resolve a dotted key to a string value."""
    models = cfg.get("models", [])
    nodes = cfg.get("nodes", [])
    pylet = cfg.get("pylet", {})
    planner = cfg.get("planner", {})

    head = cfg["head_node"]
    planner_port = planner.get("port", 8002)

    lookup: dict[str, object] = {
        "head_node": head,
        "planner_port": planner_port,
        "planner_url": f"http://{head}:{planner_port}",
        "pylet_head_url": pylet.get("head_url", ""),
        "pylet_backend": pylet.get("backend", "vllm"),
        "pylet_gpu": pylet.get("gpu_per_instance", 1),
        "pylet_cpu": pylet.get("cpu_per_instance", 1),
        "pylet_timeout": pylet.get("deploy_timeout", 600),
        "pylet_head_port": pylet.get("head_port", 5100),
        "pylet_worker_port": pylet.get("worker", {}).get(
            "port_start", 5300
        ),
        "pylet_worker_cpu": pylet.get("worker", {}).get("cpu", 8),
        "pylet_worker_gpu": pylet.get("worker", {}).get("gpu", 8),
        "pylet_worker_mem": pylet.get("worker", {}).get(
            "memory", 65536
        ),
        "model_count": len(models),
        "model_ids": " ".join(m["model_id"] for m in models),
        "scheduler_ports": " ".join(
            str(m["scheduler_port"]) for m in models
        ),
        "node_count": len(nodes),
        "models_json": json.dumps(
            {m["model_id"]: m["replicas"] for m in models}
        ),
    }

    # Indexed access: model_id.0, scheduler_port.1, etc.
    if "." in key:
        prefix, idx_str = key.rsplit(".", 1)
        try:
            idx = int(idx_str)
        except ValueError:
            sys.exit(f"Error: invalid index in key '{key}'")

        if prefix == "model_id" and idx < len(models):
            return str(models[idx]["model_id"])
        elif prefix == "scheduler_port" and idx < len(models):
            return str(models[idx]["scheduler_port"])
        elif prefix == "replicas" and idx < len(models):
            return str(models[idx]["replicas"])
        elif prefix == "node_host" and idx < len(nodes):
            return str(nodes[idx]["host"])
        elif prefix == "node_gpus" and idx < len(nodes):
            return str(nodes[idx]["gpus"])
        else:
            sys.exit(f"Error: key '{key}' out of range")

    if key in lookup:
        return str(lookup[key])

    sys.exit(f"Error: unknown key '{key}'")


def main() -> None:
    """Entry point: print the requested config value."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cfg = load_config()
    print(get_value(cfg, sys.argv[1]))


if __name__ == "__main__":
    main()
