#!/bin/bash
# Start SwarmPilot services on the head node.
# Usage: ./scripts/start_head.sh
#
# Starts: Planner (with PyLet) + one Scheduler per model.
# Reads all config from cluster.yaml via _config.py.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG="python3 $SCRIPT_DIR/_config.py"
LOG_DIR="/tmp/swarmpilot-cluster"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── Read config ───────────────────────────────────────────────
HEAD_NODE=$($CFG head_node)
PLANNER_PORT=$($CFG planner_port)
PYLET_HEAD_URL=$($CFG pylet_head_url)
PYLET_BACKEND=$($CFG pylet_backend)
PYLET_GPU=$($CFG pylet_gpu)
PYLET_CPU=$($CFG pylet_cpu)
PYLET_TIMEOUT=$($CFG pylet_timeout)
MODEL_COUNT=$($CFG model_count)

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   SwarmPilot Cluster — Start Head Node           ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Head node:    $HEAD_NODE"
echo "  Planner port: $PLANNER_PORT"
echo "  PyLet head:   $PYLET_HEAD_URL"
echo "  Models:       $MODEL_COUNT"
echo ""

# ── Port check ────────────────────────────────────────────────
check_port() {
    local port=$1
    local name=$2
    if lsof -i:"$port" > /dev/null 2>&1; then
        echo -e "${YELLOW}Warning: port $port ($name) already in use${NC}"
        echo "Run ./scripts/stop.sh first."
        return 1
    fi
    return 0
}

echo "Checking ports..."
check_port "$PLANNER_PORT" "Planner" || exit 1
for i in $(seq 0 $((MODEL_COUNT - 1))); do
    PORT=$($CFG scheduler_port.$i)
    MODEL=$($CFG model_id.$i)
    check_port "$PORT" "Scheduler ($MODEL)" || exit 1
done
echo -e "${GREEN}All ports available${NC}"
echo ""

mkdir -p "$LOG_DIR"

# ── [1] Dummy health server (for PyLet init) ─────────────────
DUMMY_PORT=9999
echo -e "${BLUE}[1/$((MODEL_COUNT + 2))] Starting dummy health server on :$DUMMY_PORT...${NC}"

python3 -c "
from http.server import HTTPServer, BaseHTTPRequestHandler
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type','application/json')
        self.end_headers()
        self.wfile.write(b'{\"status\":\"ok\"}')
    def log_message(self, *a): pass
HTTPServer(('0.0.0.0', $DUMMY_PORT), H).serve_forever()
" > "$LOG_DIR/dummy_health.log" 2>&1 &
DUMMY_PID=$!
echo $DUMMY_PID > "$LOG_DIR/dummy_health.pid"
sleep 1

if ! kill -0 $DUMMY_PID 2>/dev/null; then
    echo -e "${RED}Error: dummy health server failed to start${NC}"
    exit 1
fi
echo -e "${GREEN}Dummy health server started (PID: $DUMMY_PID)${NC}"

# ── [2] Planner ───────────────────────────────────────────────
echo -e "${BLUE}[2/$((MODEL_COUNT + 2))] Starting Planner on :$PLANNER_PORT...${NC}"
cd "$PROJECT_ROOT"

PLANNER_PORT=$PLANNER_PORT \
PYLET_ENABLED=true \
PYLET_HEAD_URL="$PYLET_HEAD_URL" \
PYLET_BACKEND="$PYLET_BACKEND" \
PYLET_GPU_COUNT="$PYLET_GPU" \
PYLET_CPU_COUNT="$PYLET_CPU" \
PYLET_DEPLOY_TIMEOUT="$PYLET_TIMEOUT" \
SCHEDULER_URL="http://localhost:$DUMMY_PORT" \
    uv run splanner start --port "$PLANNER_PORT" \
    > "$LOG_DIR/planner.log" 2>&1 &
PLANNER_PID=$!
echo $PLANNER_PID > "$LOG_DIR/planner.pid"

echo "  Waiting for Planner health check..."
for attempt in $(seq 1 30); do
    if curl -sf "http://localhost:$PLANNER_PORT/v1/health" > /dev/null 2>&1; then
        echo -e "${GREEN}  Planner started (PID: $PLANNER_PID)${NC}"
        break
    fi
    if [ "$attempt" -eq 30 ]; then
        echo -e "${RED}Error: Planner failed to start. Check $LOG_DIR/planner.log${NC}"
        exit 1
    fi
    sleep 1
done

# Stop dummy health server
kill $DUMMY_PID 2>/dev/null || true
sleep 1
kill -0 $DUMMY_PID 2>/dev/null && kill -9 $DUMMY_PID 2>/dev/null
echo -e "${GREEN}  Dummy health server stopped${NC}"
echo ""

# ── [3..N] Schedulers (one per model) ─────────────────────────
for i in $(seq 0 $((MODEL_COUNT - 1))); do
    MODEL_ID=$($CFG model_id.$i)
    SCHED_PORT=$($CFG scheduler_port.$i)
    STEP=$((i + 3))

    echo -e "${BLUE}[$STEP/$((MODEL_COUNT + 2))] Starting Scheduler ($MODEL_ID) on :$SCHED_PORT...${NC}"
    cd "$PROJECT_ROOT"

    SCHEDULER_MODEL_ID="$MODEL_ID" \
    SCHEDULER_PORT="$SCHED_PORT" \
    PLANNER_REGISTRATION_URL="http://$HEAD_NODE:$PLANNER_PORT" \
    SCHEDULER_SELF_URL="http://$HEAD_NODE:$SCHED_PORT" \
    PREDICTOR_MODE="library" \
        uv run sscheduler start --port "$SCHED_PORT" \
        > "$LOG_DIR/scheduler-$i.log" 2>&1 &
    SCHED_PID=$!
    echo $SCHED_PID > "$LOG_DIR/scheduler-$i.pid"

    sleep 2
    if ! kill -0 $SCHED_PID 2>/dev/null; then
        echo -e "${RED}Error: Scheduler ($MODEL_ID) failed. Check $LOG_DIR/scheduler-$i.log${NC}"
        exit 1
    fi
    echo -e "${GREEN}  Scheduler ($MODEL_ID) started (PID: $SCHED_PID)${NC}"
done

# ── Verify ────────────────────────────────────────────────────
echo ""
echo "Verifying scheduler registration..."
sleep 2
SCHEDULERS=$(curl -s "http://localhost:$PLANNER_PORT/v1/schedulers" 2>/dev/null || echo '{}')
echo "  Registered schedulers: $SCHEDULERS"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Head Node Ready                                ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Planner: http://$HEAD_NODE:$PLANNER_PORT"
for i in $(seq 0 $((MODEL_COUNT - 1))); do
    MODEL_ID=$($CFG model_id.$i)
    SCHED_PORT=$($CFG scheduler_port.$i)
    echo "  Scheduler ($MODEL_ID): http://$HEAD_NODE:$SCHED_PORT"
done
echo "  Logs: $LOG_DIR/"
echo ""
echo "Next: ./scripts/deploy.sh"
