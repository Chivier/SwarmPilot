#!/bin/bash
# Online API Multi-Provider — Stop All Services
# Usage: ./examples/7.online_api/stop_cluster.sh

set -e

LOG_DIR="/tmp/7.online_api"
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}Online API Multi-Provider — Stopping Services${NC}"

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

# Stop schedulers first (they hold connections to mock servers)
stop_by_pid "Scheduler A (Qwen3-8B)"       "scheduler-qwen3-8b"
stop_by_pid "Scheduler B (Qwen3-Next)"    "scheduler-qwen3-next"
stop_by_pid "Scheduler C (Gemma4)"        "scheduler-gemma4"

# Stop mock API servers (parent process kills all 9 children)
stop_by_pid "Mock API servers"       "mock-servers"

echo ""
echo -e "${GREEN}All services stopped.${NC}"
echo "Logs preserved at: $LOG_DIR/"
