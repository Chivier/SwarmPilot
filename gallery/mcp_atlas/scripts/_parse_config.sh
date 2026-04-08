#!/bin/bash
# Shared config parser for all SwarmPilot benchmark scripts.
# Source this file: source "$(dirname "$0")/_parse_config.sh" [config_path]
#
# Exports:
#   PLANNER_PORT, SCHEDULER_LARGE_PORT, SCHEDULER_SMALL_PORT,
#   LARGE_MODEL_ID, SMALL_MODEL_ID, SCHEDULING_ALGORITHM,
#   SWARMPILOT_ROOT, BENCHMARK_SCRIPT_DIR, BENCHMARK_PROJECT_ROOT,
#   LOG_DIR, VENV_PYTHON,
#   GPU_PER_INSTANCE_LARGE, GPU_PER_INSTANCE_SMALL,
#   REPLICAS_LARGE, REPLICAS_SMALL,
#   PYLET_LOCAL_PORT, PYLET_LOCAL_WORKER_PORT_START,
#   PYLET_LOCAL_NUM_WORKERS, PYLET_LOCAL_GPU_PER_WORKER,
#   PYLET_LOCAL_CPU_PER_WORKER, PYLET_DEPLOY_TIMEOUT

# scripts/ → mcp_atlas/ → gallery/ → project_root
BENCHMARK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_PROJECT_ROOT="$(cd "$BENCHMARK_SCRIPT_DIR/../../.." && pwd)"
_CONFIG_PATH="${1:-$BENCHMARK_SCRIPT_DIR/../config.example.yaml}"
CLUSTER_TAG="${CLUSTER_TAG:-gallery-real}"
LOG_DIR="/tmp/swarmpilot_benchmark_real/${CLUSTER_TAG}"

# Colors (available to all scripts)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Venv python
VENV_PYTHON="$BENCHMARK_PROJECT_ROOT/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="python3"
fi

# ── Read all config values in a single python call ────────────────
_ALL_CONFIG=$("$VENV_PYTHON" -c "
import yaml, os, sys

path = '$_CONFIG_PATH'
if not os.path.isfile(path):
    print(f'ERROR: Config file not found: {path}', file=sys.stderr)
    sys.exit(1)

with open(path) as f:
    c = yaml.safe_load(f) or {}

# Cascading SwarmPilot root detection:
#   1. \$SWARMPILOT_ROOT env var   (cross-machine, set in shell profile)
#   2. config swarmpilot_root     (per-project, ~ expanded)
#   3. BENCHMARK_PROJECT_ROOT itself (script lives inside swarmpilot project)
project_root = '$BENCHMARK_PROJECT_ROOT'
env_root = os.environ.get('SWARMPILOT_ROOT', '').strip()
cfg_root = c.get('swarmpilot_root', '')

candidates = []
if env_root:
    candidates.append(('env \$SWARMPILOT_ROOT', env_root))
if cfg_root:
    candidates.append(('config swarmpilot_root', os.path.expanduser(cfg_root)))
candidates.append(('project root (script is inside swarmpilot)', project_root))

sp = None
for label, p in candidates:
    resolved = os.path.realpath(p)
    if os.path.isdir(resolved) and os.path.isfile(os.path.join(resolved, 'pyproject.toml')):
        sp = resolved
        break

if sp is None:
    print('ERROR: Cannot find SwarmPilot project directory.', file=sys.stderr)
    print('  Searched:', file=sys.stderr)
    for label, p in candidates:
        print(f'    [{label}] {p}', file=sys.stderr)
    print('', file=sys.stderr)
    print('  Fix: set env var SWARMPILOT_ROOT, or edit swarmpilot_root in', file=sys.stderr)
    print(f'       {path}', file=sys.stderr)
    sys.exit(1)

# Emit key=value pairs for bash eval
print(f'PLANNER_PORT={c.get(\"planner_port\", 8002)}')
print(f'SCHEDULER_LARGE_PORT={c.get(\"scheduler_large_port\", 8010)}')
print(f'SCHEDULER_SMALL_PORT={c.get(\"scheduler_small_port\", 8020)}')
print(f'SCHEDULING_ALGORITHM={c.get(\"scheduling_algorithm\", \"probabilistic\")}')
print(f'LARGE_MODEL_ID={c.get(\"large_model_id\", \"Qwen/Qwen3-Next-80B-A3B-Instruct\")}')
print(f'SMALL_MODEL_ID={c.get(\"small_model_id\", \"Qwen/Qwen3-VL-8B-Instruct\")}')
print(f'GPU_PER_INSTANCE_LARGE={c.get(\"gpu_per_instance_large\", 4)}')
print(f'GPU_PER_INSTANCE_SMALL={c.get(\"gpu_per_instance_small\", 1)}')
print(f'REPLICAS_LARGE={c.get(\"replicas_large\", 2)}')
print(f'REPLICAS_SMALL={c.get(\"replicas_small\", 2)}')
print(f'PYLET_LOCAL_PORT={c.get(\"pylet_local_port\", 5100)}')
print(f'PYLET_LOCAL_WORKER_PORT_START={c.get(\"pylet_local_worker_port_start\", 5300)}')
print(f'PYLET_LOCAL_NUM_WORKERS={c.get(\"pylet_local_num_workers\", 1)}')
print(f'PYLET_LOCAL_GPU_PER_WORKER={c.get(\"pylet_local_gpu_per_worker\", 8)}')
print(f'PYLET_LOCAL_CPU_PER_WORKER={c.get(\"pylet_local_cpu_per_worker\", 16)}')
print(f'PYLET_DEPLOY_TIMEOUT={c.get(\"pylet_deploy_timeout\", 600)}')
print(f'SWARMPILOT_ROOT={sp}')
") || exit 1

eval "$_ALL_CONFIG"
