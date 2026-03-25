#!/bin/bash
# Qwen3-Next-80B-A3B Runtime Collection — Stop Cluster
# Usage: ./examples/predictor_training_playground/stop_qwen_cluster.sh

set -e

PLANNER_PORT=${PLANNER_PORT:-8002}
LOG_DIR="/tmp/qwen_cluster"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}${BOLD}║   Qwen Runtime Collection — Shutdown             ║${NC}"
echo -e "${BLUE}${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# Try graceful PyLet termination first
echo -e "${BLUE}[1/3] Terminating managed instances...${NC}"
uv run splanner terminate --all \
    --planner-url "http://localhost:$PLANNER_PORT" 2>/dev/null \
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

echo -e "${BLUE}[2/3] Stopping Scheduler...${NC}"
stop_process "Scheduler" "scheduler"

echo -e "${BLUE}[3/3] Stopping Planner (auto-stops local PyLet cluster)...${NC}"
stop_process "Planner" "planner"
stop_process "Dummy Health" "dummy_health"

echo ""
echo -e "${GREEN}All services stopped. Logs at: $LOG_DIR/${NC}"
