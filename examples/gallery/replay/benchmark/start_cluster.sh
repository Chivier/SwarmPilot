#!/bin/bash
# SwarmPilot Real-Model Benchmark - Start Cluster
# Usage: bash examples/gallery/replay/benchmark/start_cluster.sh [config.yaml]
#
# Starts Planner (with local PyLet cluster for vLLM) + 2 per-model Schedulers.
# No model instances are deployed here — run deploy_models.sh afterwards.
#
# Flow:
#   [1/4] Dummy Health Server  (satisfies Planner's scheduler health check at boot)
#   [2/4] Planner              (local PyLet mode, vLLM backend)
#   [3/4] Scheduler-Large      (LARGE_MODEL_ID, registers with Planner)
#   [4/4] Scheduler-Small      (SMALL_MODEL_ID, registers with Planner)

set -e

source "$(dirname "$0")/_parse_config.sh" "$1"

DUMMY_HEALTH_PORT=$((PLANNER_PORT + 7000))

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   SwarmPilot Real-Model Benchmark — Cluster Startup ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Config:          $_CONFIG_PATH"
echo "  SwarmPilot:      $SWARMPILOT_ROOT"
echo "  Algorithm:       $SCHEDULING_ALGORITHM"
echo "  Large model:     $LARGE_MODEL_ID → :$SCHEDULER_LARGE_PORT"
echo "  Small model:     $SMALL_MODEL_ID → :$SCHEDULER_SMALL_PORT"
echo "  Planner:         :$PLANNER_PORT  (PyLet head :$PYLET_LOCAL_PORT)"
echo "  Dummy health:    :$DUMMY_HEALTH_PORT (transient)"
echo "  Logs:            $LOG_DIR/"
echo ""

# ── Port check ────────────────────────────────────────────────────
check_port() {
    local port=$1 name=$2
    if lsof -i:"$port" > /dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Port $port ($name) is already in use.${NC}"
        echo "Run: bash examples/gallery/replay/benchmark/stop_cluster.sh"
        return 1
    fi
    return 0
}

echo "Checking ports..."
check_port "$PLANNER_PORT"         "Planner"        || exit 1
check_port "$SCHEDULER_LARGE_PORT" "Scheduler-Large" || exit 1
check_port "$SCHEDULER_SMALL_PORT" "Scheduler-Small" || exit 1
check_port "$DUMMY_HEALTH_PORT"    "Dummy Health"    || exit 1
check_port "$PYLET_LOCAL_PORT"     "PyLet Head"      || exit 1
echo -e "${GREEN}All ports available${NC}"
echo ""

# ── Pre-flight: pylet must be importable ─────────────────────────
echo "Checking pylet installation..."
cd "$SWARMPILOT_ROOT"
if ! uv run python -c "import pylet" 2>/dev/null; then
    echo -e "${YELLOW}pylet not found in venv, installing...${NC}"
    uv pip install pylet
    if ! uv run python -c "import pylet" 2>/dev/null; then
        echo -e "${RED}Error: Failed to install pylet. Install manually and retry.${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}pylet available${NC}"
echo ""

mkdir -p "$LOG_DIR"

# ── [1/4] Dummy Health Server ────────────────────────────────────
# PyLet init inside Planner requires a reachable SCHEDULER_URL at boot time.
# We stand up a minimal HTTP server that answers /health and /v1/health,
# then tear it down once Planner is healthy.
echo -e "${BLUE}[1/4] Starting Dummy Health Server on :$DUMMY_HEALTH_PORT...${NC}"

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
    echo -e "${RED}Error: Dummy Health Server failed to start. See $LOG_DIR/dummy_health.log${NC}"
    exit 1
fi
echo -e "${GREEN}Dummy Health Server started (PID: $DUMMY_PID)${NC}"
echo ""

# ── [2/4] Planner (local PyLet + vLLM) ──────────────────────────
echo -e "${BLUE}[2/4] Starting Planner on :$PLANNER_PORT (local PyLet mode, vLLM backend)...${NC}"
cd "$SWARMPILOT_ROOT"

PYLET_ENABLED="true" \
    PYLET_LOCAL_MODE="true" \
    PYLET_BACKEND="vllm" \
    PYLET_LOCAL_PORT="$PYLET_LOCAL_PORT" \
    PYLET_LOCAL_NUM_WORKERS="$PYLET_LOCAL_NUM_WORKERS" \
    PYLET_LOCAL_GPU_PER_WORKER="$PYLET_LOCAL_GPU_PER_WORKER" \
    PYLET_LOCAL_CPU_PER_WORKER="$PYLET_LOCAL_CPU_PER_WORKER" \
    PYLET_LOCAL_WORKER_PORT_START="$PYLET_LOCAL_WORKER_PORT_START" \
    PYLET_DEPLOY_TIMEOUT="$PYLET_DEPLOY_TIMEOUT" \
    SCHEDULER_URL="http://localhost:$DUMMY_HEALTH_PORT" \
    uv run splanner start --port "$PLANNER_PORT" \
    > "$LOG_DIR/planner.log" 2>&1 &

PLANNER_PID=$!
echo $PLANNER_PID > "$LOG_DIR/planner.pid"

echo "  Waiting for Planner health (up to 60s)..."
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

# Tear down dummy — Planner is up, real schedulers register next
if kill -0 $DUMMY_PID 2>/dev/null; then
    kill $DUMMY_PID 2>/dev/null || true
    sleep 1
    kill -0 $DUMMY_PID 2>/dev/null && kill -9 $DUMMY_PID 2>/dev/null || true
fi
echo -e "${GREEN}Dummy Health Server stopped${NC}"
echo ""

# ── [3/4] Scheduler-Large ────────────────────────────────────────
echo -e "${BLUE}[3/4] Starting Scheduler-Large ($LARGE_MODEL_ID) on :$SCHEDULER_LARGE_PORT...${NC}"
cd "$SWARMPILOT_ROOT"

SCHEDULER_MODEL_ID="$LARGE_MODEL_ID" \
    PREDICTOR_MODE="library" \
    PROXY_ENABLED="true" \
    PROXY_TIMEOUT="300.0" \
    SCHEDULING_STRATEGY="$SCHEDULING_ALGORITHM" \
    PLANNER_REGISTRATION_URL="http://localhost:$PLANNER_PORT" \
    SCHEDULER_SELF_URL="http://localhost:$SCHEDULER_LARGE_PORT" \
    uv run sscheduler start --port "$SCHEDULER_LARGE_PORT" \
    > "$LOG_DIR/scheduler-large.log" 2>&1 &

SCHED_LARGE_PID=$!
echo $SCHED_LARGE_PID > "$LOG_DIR/scheduler-large.pid"

sleep 2
if ! kill -0 $SCHED_LARGE_PID 2>/dev/null; then
    echo -e "${RED}Error: Scheduler-Large failed to start. See $LOG_DIR/scheduler-large.log${NC}"
    exit 1
fi
echo -e "${GREEN}Scheduler-Large started (PID: $SCHED_LARGE_PID)${NC}"
echo ""

# ── [4/4] Scheduler-Small ────────────────────────────────────────
echo -e "${BLUE}[4/4] Starting Scheduler-Small ($SMALL_MODEL_ID) on :$SCHEDULER_SMALL_PORT...${NC}"
cd "$SWARMPILOT_ROOT"

SCHEDULER_MODEL_ID="$SMALL_MODEL_ID" \
    PREDICTOR_MODE="library" \
    PROXY_ENABLED="true" \
    PROXY_TIMEOUT="300.0" \
    SCHEDULING_STRATEGY="$SCHEDULING_ALGORITHM" \
    PLANNER_REGISTRATION_URL="http://localhost:$PLANNER_PORT" \
    SCHEDULER_SELF_URL="http://localhost:$SCHEDULER_SMALL_PORT" \
    uv run sscheduler start --port "$SCHEDULER_SMALL_PORT" \
    > "$LOG_DIR/scheduler-small.log" 2>&1 &

SCHED_SMALL_PID=$!
echo $SCHED_SMALL_PID > "$LOG_DIR/scheduler-small.pid"

sleep 2
if ! kill -0 $SCHED_SMALL_PID 2>/dev/null; then
    echo -e "${RED}Error: Scheduler-Small failed to start. See $LOG_DIR/scheduler-small.log${NC}"
    exit 1
fi
echo -e "${GREEN}Scheduler-Small started (PID: $SCHED_SMALL_PID)${NC}"
echo ""

# ── Verify scheduler registration ───────────────────────────────
echo "Verifying scheduler registration with Planner..."
sleep 2
SCHEDULERS=$(curl -s "http://localhost:$PLANNER_PORT/v1/schedulers" 2>/dev/null || echo '{}')
echo "  Registered: $SCHEDULERS"
echo ""

# ── Summary ──────────────────────────────────────────────────────
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Cluster Ready (no instances yet)               ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Planner:          http://localhost:$PLANNER_PORT"
echo "  PyLet Head:       http://localhost:$PYLET_LOCAL_PORT"
echo "  Scheduler-Large:  http://localhost:$SCHEDULER_LARGE_PORT  ($LARGE_MODEL_ID)"
echo "  Scheduler-Small:  http://localhost:$SCHEDULER_SMALL_PORT  ($SMALL_MODEL_ID)"
echo "  Logs:             $LOG_DIR/"
echo ""
echo -e "${YELLOW}Next:${NC} bash examples/gallery/replay/benchmark/deploy_models.sh"
