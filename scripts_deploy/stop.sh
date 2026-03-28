#!/bin/bash
# Stop all SwarmPilot services on the head node.
# Usage: ./scripts/stop.sh
#
# Order: terminate PyLet instances -> stop Schedulers -> stop Planner.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="python3 $SCRIPT_DIR/_config.py"
LOG_DIR="/tmp/swarmpilot-cluster"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

HEAD_NODE=$($CFG head_node)
PLANNER_PORT=$($CFG planner_port)
PLANNER_URL="http://$HEAD_NODE:$PLANNER_PORT"
MODEL_COUNT=$($CFG model_count)

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   SwarmPilot Cluster — Shutdown                  ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""

TOTAL_STEPS=$((MODEL_COUNT + 2))

# Helper: stop a process by PID file
stop_process() {
    local name=$1
    local pid_file="$LOG_DIR/$2.pid"
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null || true
            sleep 1
            # Force kill if still alive
            kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null
            echo -e "  ${GREEN}Stopped $name (PID: $PID)${NC}"
        else
            echo -e "  ${YELLOW}$name already stopped (PID: $PID)${NC}"
        fi
        rm -f "$pid_file"
    else
        echo -e "  ${YELLOW}$name: no PID file found${NC}"
    fi
}

# ── [1] Terminate PyLet instances ─────────────────────────────
echo -e "${BLUE}[1/$TOTAL_STEPS] Terminating PyLet instances...${NC}"
if curl -sf "$PLANNER_URL/v1/health" > /dev/null 2>&1; then
    RESP=$(curl -s -w "\n%{http_code}" -X POST \
        "$PLANNER_URL/v1/pylet/terminate-all" 2>/dev/null)
    HTTP_CODE=$(echo "$RESP" | tail -1)
    if [ "$HTTP_CODE" -ge 200 ] && [ "$HTTP_CODE" -lt 300 ]; then
        echo -e "  ${GREEN}All PyLet instances terminated${NC}"
    else
        echo -e "  ${YELLOW}PyLet terminate returned HTTP $HTTP_CODE (may already be empty)${NC}"
    fi
else
    echo -e "  ${YELLOW}Planner not reachable, skipping PyLet terminate${NC}"
fi

# ── [2..N] Stop Schedulers ────────────────────────────────────
for i in $(seq 0 $((MODEL_COUNT - 1))); do
    MODEL_ID=$($CFG model_id.$i)
    STEP=$((i + 2))
    echo -e "${BLUE}[$STEP/$TOTAL_STEPS] Stopping Scheduler ($MODEL_ID)...${NC}"
    stop_process "Scheduler ($MODEL_ID)" "scheduler-$i"
done

# ── [N+1] Stop Planner ───────────────────────────────────────
echo -e "${BLUE}[$TOTAL_STEPS/$TOTAL_STEPS] Stopping Planner...${NC}"
stop_process "Planner" "planner"

# Clean up dummy health PID if lingering
if [ -f "$LOG_DIR/dummy_health.pid" ]; then
    stop_process "Dummy Health" "dummy_health"
fi

echo ""
echo -e "${GREEN}All services stopped. Logs preserved at: $LOG_DIR/${NC}"
