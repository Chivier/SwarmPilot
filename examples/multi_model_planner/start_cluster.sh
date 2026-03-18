#!/bin/bash
# Multi-Model Planner Example - Start Cluster
# Usage: ./examples/multi_model_planner/start_cluster.sh
#
# Starts Planner + 2 per-model Schedulers (Qwen, Llama).
# Schedulers auto-register with the Planner on startup.

set -e

# Configuration
PLANNER_PORT=${PLANNER_PORT:-8002}
SCHEDULER_QWEN_PORT=${SCHEDULER_QWEN_PORT:-8010}
SCHEDULER_LLAMA_PORT=${SCHEDULER_LLAMA_PORT:-8020}
DUMMY_HEALTH_PORT=${DUMMY_HEALTH_PORT:-9999}

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="/tmp/multi_model_planner"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Multi-Model Planner Example - Startup         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# Port checker
check_port() {
    local port=$1
    local name=$2
    if lsof -i:"$port" > /dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Port $port ($name) is already in use.${NC}"
        echo "Run ./examples/multi_model_planner/stop_cluster.sh first."
        return 1
    fi
    return 0
}

echo "Checking ports..."
check_port $PLANNER_PORT "Planner" || exit 1
check_port $SCHEDULER_QWEN_PORT "Scheduler (Qwen)" || exit 1
check_port $SCHEDULER_LLAMA_PORT "Scheduler (Llama)" || exit 1
check_port $DUMMY_HEALTH_PORT "Dummy Health" || exit 1
echo -e "${GREEN}All ports available${NC}"
echo ""

mkdir -p "$LOG_DIR"

# ── Step [1/4]: Dummy Health Server ──────────────────────────────
# PyLet init check needs a reachable SCHEDULER_URL at planner boot.
echo -e "${BLUE}[1/4] Starting Dummy Health Server on port $DUMMY_HEALTH_PORT...${NC}"

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
        pass  # Suppress logging

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

# ── Step [2/4]: Planner ─────────────────────────────────────────
echo -e "${BLUE}[2/4] Starting Planner on port $PLANNER_PORT...${NC}"
cd "$PROJECT_ROOT"

PLANNER_PORT=$PLANNER_PORT \
    SCHEDULER_URL="http://localhost:$DUMMY_HEALTH_PORT" \
    uv run splanner start --port $PLANNER_PORT \
    > "$LOG_DIR/planner.log" 2>&1 &
PLANNER_PID=$!
echo $PLANNER_PID > "$LOG_DIR/planner.pid"

echo "Waiting for Planner health check..."
for attempt in $(seq 1 30); do
    if curl -s "http://localhost:$PLANNER_PORT/v1/health" > /dev/null 2>&1; then
        echo -e "${GREEN}Planner started (PID: $PLANNER_PID)${NC}"
        break
    fi
    if [ "$attempt" -eq 30 ]; then
        echo -e "${RED}Error: Planner failed to become healthy. Check $LOG_DIR/planner.log${NC}"
        exit 1
    fi
    sleep 1
done

# Stop dummy health server now that planner is initialized
if kill -0 $DUMMY_PID 2>/dev/null; then
    kill $DUMMY_PID 2>/dev/null || true
    sleep 1
    kill -0 $DUMMY_PID 2>/dev/null && kill -9 $DUMMY_PID 2>/dev/null
fi
echo -e "${GREEN}Dummy Health Server stopped${NC}"
echo ""

# ── Step [3/4]: Scheduler A (Qwen) ──────────────────────────────
echo -e "${BLUE}[3/4] Starting Scheduler (Qwen/Qwen3-8B-VL) on port $SCHEDULER_QWEN_PORT...${NC}"
cd "$PROJECT_ROOT"

SCHEDULER_MODEL_ID="Qwen/Qwen3-8B-VL" \
    PLANNER_REGISTRATION_URL="http://localhost:$PLANNER_PORT" \
    SCHEDULER_SELF_URL="http://localhost:$SCHEDULER_QWEN_PORT" \
    PREDICTOR_MODE="library" \
    uv run sscheduler start --port $SCHEDULER_QWEN_PORT \
    > "$LOG_DIR/scheduler-qwen.log" 2>&1 &
SCHEDULER_QWEN_PID=$!
echo $SCHEDULER_QWEN_PID > "$LOG_DIR/scheduler-qwen.pid"

sleep 2
if ! kill -0 $SCHEDULER_QWEN_PID 2>/dev/null; then
    echo -e "${RED}Error: Scheduler (Qwen) failed to start. Check $LOG_DIR/scheduler-qwen.log${NC}"
    exit 1
fi
echo -e "${GREEN}Scheduler (Qwen) started (PID: $SCHEDULER_QWEN_PID)${NC}"

# ── Step [4/4]: Scheduler B (Llama) ─────────────────────────────
echo -e "${BLUE}[4/4] Starting Scheduler (Llama-3.1-8B) on port $SCHEDULER_LLAMA_PORT...${NC}"
cd "$PROJECT_ROOT"

SCHEDULER_MODEL_ID="meta-llama/Llama-3.1-8B" \
    PLANNER_REGISTRATION_URL="http://localhost:$PLANNER_PORT" \
    SCHEDULER_SELF_URL="http://localhost:$SCHEDULER_LLAMA_PORT" \
    PREDICTOR_MODE="library" \
    uv run sscheduler start --port $SCHEDULER_LLAMA_PORT \
    > "$LOG_DIR/scheduler-llama.log" 2>&1 &
SCHEDULER_LLAMA_PID=$!
echo $SCHEDULER_LLAMA_PID > "$LOG_DIR/scheduler-llama.pid"

sleep 2
if ! kill -0 $SCHEDULER_LLAMA_PID 2>/dev/null; then
    echo -e "${RED}Error: Scheduler (Llama) failed to start. Check $LOG_DIR/scheduler-llama.log${NC}"
    exit 1
fi
echo -e "${GREEN}Scheduler (Llama) started (PID: $SCHEDULER_LLAMA_PID)${NC}"

# ── Verify ──────────────────────────────────────────────────────
echo ""
echo "Verifying scheduler registration with Planner..."
sleep 2
SCHEDULERS=$(curl -s "http://localhost:$PLANNER_PORT/v1/schedulers" 2>/dev/null || echo '{}')
echo "Registered schedulers: $SCHEDULERS"
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Cluster Ready                                 ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Planner:          http://localhost:$PLANNER_PORT"
echo "  Scheduler (Qwen): http://localhost:$SCHEDULER_QWEN_PORT"
echo "  Scheduler (Llama):http://localhost:$SCHEDULER_LLAMA_PORT"
echo "  Logs:             $LOG_DIR/"
echo ""
echo "Next: ./examples/multi_model_planner/deploy_model.sh"
