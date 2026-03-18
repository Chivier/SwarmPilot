#!/bin/bash
# Multi-Model Planner Example - Stop Cluster
# Usage: ./examples/multi_model_planner/stop_cluster.sh

set -e

PLANNER_PORT=${PLANNER_PORT:-8002}
LOG_DIR="/tmp/multi_model_planner"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Multi-Model Planner - Shutdown                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# Try graceful PyLet termination first
echo -e "${BLUE}[1/4] Terminating managed instances...${NC}"
uv run splanner terminate --all --planner-url "http://localhost:$PLANNER_PORT" 2>/dev/null \
    && echo -e "${GREEN}Managed instances terminated${NC}" \
    || echo -e "${YELLOW}splanner terminate skipped (planner may be down)${NC}"

stop_process() {
    local name=$1
    local pid_file="$LOG_DIR/$2.pid"
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null || true
            sleep 1
            kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null
            echo -e "${GREEN}  Stopped $name (PID: $PID)${NC}"
        else
            echo -e "${YELLOW}  $name already stopped (PID: $PID)${NC}"
        fi
        rm -f "$pid_file"
    fi
}

# Kill mock instances
echo -e "${BLUE}[2/4] Stopping mock instances...${NC}"
for pid_file in "$LOG_DIR"/mock-*.pid; do
    [ -f "$pid_file" ] || continue
    stop_process "$(basename "$pid_file" .pid)" "$(basename "$pid_file" .pid)"
done

echo -e "${BLUE}[3/4] Stopping Schedulers...${NC}"
stop_process "Scheduler (Qwen)" "scheduler-qwen"
stop_process "Scheduler (Llama)" "scheduler-llama"

echo -e "${BLUE}[4/4] Stopping Planner + Dummy Health...${NC}"
stop_process "Planner" "planner"
stop_process "Dummy Health" "dummy_health"

echo ""
echo -e "${GREEN}All services stopped. Logs at: $LOG_DIR/${NC}"
