#!/bin/bash
# Predictor Training Playground — Start Cluster
# Usage: ./examples/predictor_training_playground/start_qwen_cluster.sh
#
# Starts Planner (with local PyLet cluster) + Scheduler for
# runtime data collection and predictor training.
#
# The Scheduler starts without a fixed model ID. The Planner assigns
# the model dynamically on the first serve() call via /v1/model/reassign.
#
# Flow:
#   1. Dummy Health Server (satisfies Planner's scheduler health check)
#   2. Planner with PYLET_LOCAL_MODE (auto-starts PyLet head+worker)
#   3. Scheduler (registers with Planner, model assigned on first deploy)

set -e

# ── Configuration ────────────────────────────────────────────────
MODEL_ID=${MODEL_ID:-"Qwen/Qwen3-Next-80B-A3B-Instruct"}
PLANNER_PORT=${PLANNER_PORT:-8002}
SCHEDULER_PORT=${SCHEDULER_PORT:-8000}
DUMMY_HEALTH_PORT=${DUMMY_HEALTH_PORT:-9999}

# Instance settings
GPU_PER_INSTANCE=${GPU_PER_INSTANCE:-4}     # tensor-parallel-size
REPLICAS=${REPLICAS:-1}                     # single replica

# Local PyLet cluster settings
PYLET_LOCAL_PORT=${PYLET_LOCAL_PORT:-5100}
PYLET_LOCAL_WORKER_PORT=${PYLET_LOCAL_WORKER_PORT_START:-5300}
PYLET_LOCAL_GPU=${PYLET_LOCAL_GPU_PER_WORKER:-4}
PYLET_LOCAL_CPU=${PYLET_LOCAL_CPU_PER_WORKER:-8}

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="/tmp/qwen_cluster"

# ── Colors ───────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}${BOLD}║   Predictor Training — Cluster Startup            ║${NC}"
echo -e "${BLUE}${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Planner:  :$PLANNER_PORT (local PyLet on :$PYLET_LOCAL_PORT)"
echo "  Scheduler::$SCHEDULER_PORT (round_robin, model assigned on deploy)"
echo ""

# ── Helpers ──────────────────────────────────────────────────────
check_port() {
    local port=$1
    local name=$2
    if lsof -i:"$port" > /dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Port $port ($name) is already in use.${NC}"
        echo "Run ./examples/predictor_training_playground/stop_qwen_cluster.sh first."
        return 1
    fi
    return 0
}

echo "Checking ports..."
check_port $DUMMY_HEALTH_PORT "Dummy Health" || exit 1
check_port $PLANNER_PORT "Planner" || exit 1
check_port $PYLET_LOCAL_PORT "PyLet Head" || exit 1
check_port $PYLET_LOCAL_WORKER_PORT "PyLet Worker" || exit 1
check_port $SCHEDULER_PORT "Scheduler" || exit 1
echo -e "${GREEN}All ports available${NC}"
echo ""

# ── Pre-flight: ensure pylet is installed ────────────────────────
if ! uv run python -c "import pylet" 2>/dev/null; then
    echo -e "${YELLOW}pylet not found in venv, installing...${NC}"
    uv pip install pylet
    if ! uv run python -c "import pylet" 2>/dev/null; then
        echo -e "${RED}Error: Failed to install pylet${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}pylet available${NC}"
echo ""

mkdir -p "$LOG_DIR"

# ── [1/3] Dummy Health Server ────────────────────────────────────
# PyLet init needs a reachable SCHEDULER_URL at planner boot.
echo -e "${BLUE}[1/3] Starting Dummy Health Server on port $DUMMY_HEALTH_PORT...${NC}"

python3 -c "
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/health', '/v1/health'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{\"status\": \"ok\"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

server = HTTPServer(('localhost', $DUMMY_HEALTH_PORT), HealthHandler)
print('Dummy health server started', flush=True)
sys.stdout.flush()
server.serve_forever()
" > "$LOG_DIR/dummy_health.log" 2>&1 &

DUMMY_PID=$!
echo $DUMMY_PID > "$LOG_DIR/dummy_health.pid"
sleep 1
if ! kill -0 $DUMMY_PID 2>/dev/null; then
    echo -e "${RED}Error: Dummy Health Server failed to start${NC}"
    exit 1
fi
echo -e "${GREEN}Dummy Health Server started (PID: $DUMMY_PID)${NC}"

# ── [2/3] Planner (with local PyLet) ────────────────────────────
echo -e "${BLUE}[2/3] Starting Planner on port $PLANNER_PORT (local PyLet mode)...${NC}"
cd "$PROJECT_ROOT"

PYLET_ENABLED="true" \
    PYLET_LOCAL_MODE="true" \
    PYLET_LOCAL_PORT="$PYLET_LOCAL_PORT" \
    PYLET_LOCAL_NUM_WORKERS="1" \
    PYLET_LOCAL_GPU_PER_WORKER="$PYLET_LOCAL_GPU" \
    PYLET_LOCAL_CPU_PER_WORKER="$PYLET_LOCAL_CPU" \
    PYLET_LOCAL_WORKER_PORT_START="$PYLET_LOCAL_WORKER_PORT" \
    PYLET_BACKEND="vllm" \
    PYLET_GPU_COUNT="$GPU_PER_INSTANCE" \
    PYLET_DEPLOY_TIMEOUT="600" \
    SCHEDULER_URL="http://localhost:$DUMMY_HEALTH_PORT" \
    uv run splanner start --port $PLANNER_PORT \
    > "$LOG_DIR/planner.log" 2>&1 &
PLANNER_PID=$!
echo $PLANNER_PID > "$LOG_DIR/planner.pid"

echo "Waiting for Planner health check..."
for attempt in $(seq 1 60); do
    if curl -s "http://localhost:$PLANNER_PORT/v1/health" > /dev/null 2>&1; then
        echo -e "${GREEN}Planner started (PID: $PLANNER_PID)${NC}"
        break
    fi
    if [ "$attempt" -eq 60 ]; then
        echo -e "${RED}Error: Planner failed to become healthy. Check $LOG_DIR/planner.log${NC}"
        exit 1
    fi
    sleep 1
done

# Stop dummy health server — no longer needed
if kill -0 $DUMMY_PID 2>/dev/null; then
    kill $DUMMY_PID 2>/dev/null || true
    sleep 1
    kill -0 $DUMMY_PID 2>/dev/null && kill -9 $DUMMY_PID 2>/dev/null
fi
echo -e "${GREEN}Dummy Health Server stopped${NC}"
echo ""

# ── [3/3] Scheduler ─────────────────────────────────────────────
echo -e "${BLUE}[3/3] Starting Scheduler on port $SCHEDULER_PORT (round_robin)...${NC}"
cd "$PROJECT_ROOT"

SCHEDULING_STRATEGY="round_robin" \
    TRAINING_ENABLE_AUTO="false" \
    PROXY_ENABLED="true" \
    PROXY_TIMEOUT="600.0" \
    PLANNER_REGISTRATION_URL="http://localhost:$PLANNER_PORT" \
    SCHEDULER_SELF_URL="http://localhost:$SCHEDULER_PORT" \
    uv run sscheduler start --port $SCHEDULER_PORT \
    > "$LOG_DIR/scheduler.log" 2>&1 &
SCHEDULER_PID=$!
echo $SCHEDULER_PID > "$LOG_DIR/scheduler.pid"

sleep 2
if ! kill -0 $SCHEDULER_PID 2>/dev/null; then
    echo -e "${RED}Error: Scheduler failed to start. Check $LOG_DIR/scheduler.log${NC}"
    exit 1
fi
echo -e "${GREEN}Scheduler started (PID: $SCHEDULER_PID)${NC}"

# ── Verify ───────────────────────────────────────────────────────
echo ""
echo "Verifying scheduler registration with Planner..."
sleep 2
SCHEDULERS=$(curl -s "http://localhost:$PLANNER_PORT/v1/schedulers" 2>/dev/null || echo '{}')
echo "Registered schedulers: $SCHEDULERS"

echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║   Cluster Ready                                  ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Planner:   http://localhost:$PLANNER_PORT"
echo "  Scheduler: http://localhost:$SCHEDULER_PORT"
echo "  PyLet:     http://localhost:$PYLET_LOCAL_PORT"
echo "  Strategy:  round_robin"
echo "  Model:     (assigned on first deploy)"
echo "  Logs:      $LOG_DIR/"
echo ""
echo -e "${YELLOW}Next:${NC} uv run python examples/predictor_training_playground/collect_and_train_qwen.py --train"
