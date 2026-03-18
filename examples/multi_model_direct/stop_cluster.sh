#!/bin/bash
# Multi-Model Direct — Stop All Services
# Usage: ./examples/multi_model_direct/stop_cluster.sh

set -e

LOG_DIR="/tmp/multi_model_direct"
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}Multi-Model Direct — Stopping Services${NC}"

stop_by_pid() {
    local label=$1
    local pidfile="$LOG_DIR/$2.pid"
    if [ -f "$pidfile" ]; then
        PID=$(cat "$pidfile")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null || true
            sleep 0.5
            kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null
            echo -e "${GREEN}✓ Stopped $label (PID $PID)${NC}"
        fi
        rm -f "$pidfile"
    fi
}

stop_by_pid "Qwen instance 0"    "qwen-0"
stop_by_pid "Qwen instance 1"    "qwen-1"
stop_by_pid "Llama instance 0"   "llama-0"
stop_by_pid "Llama instance 1"   "llama-1"
stop_by_pid "Scheduler A (Qwen)" "scheduler-qwen"
stop_by_pid "Scheduler B (Llama)" "scheduler-llama"

echo ""
echo -e "${GREEN}All services stopped.${NC}"
echo "Logs preserved at: $LOG_DIR/"
