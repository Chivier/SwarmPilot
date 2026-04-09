#!/bin/bash
# Virtual Concurrency Test — Stop All Services
# Usage: ./examples/8.virtual_concurrency/stop_cluster.sh

set -e

LOG_DIR="/tmp/8.virtual_concurrency"
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}Virtual Concurrency Test — Stopping Services${NC}"

stop_by_pid() {
    local label=$1
    local pidfile="$LOG_DIR/$2.pid"
    if [ -f "$pidfile" ]; then
        PID=$(cat "$pidfile")
        if kill -0 "$PID" 2>/dev/null; then
            # Kill the process group to catch child processes
            kill -- -"$PID" 2>/dev/null || kill "$PID" 2>/dev/null || true
            sleep 0.5
            kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null
            echo -e "${GREEN}Stopped $label (PID $PID)${NC}"
        fi
        rm -f "$pidfile"
    fi
}

stop_by_pid "Scheduler A (Qwen3-8B)"    "scheduler-qwen3-8b"
stop_by_pid "Scheduler B (Qwen3-Next)"  "scheduler-qwen3-next"
stop_by_pid "Scheduler C (Gemma4)"      "scheduler-gemma4"
stop_by_pid "Mock API servers"           "mock-servers"

# Clean up any orphaned mock server children on our port range
for port in 9200 9201 9202 9210 9211 9212 9220 9221 9222; do
    fuser -k "$port/tcp" 2>/dev/null || true
done

echo ""
echo -e "${GREEN}All services stopped.${NC}"
echo "Logs preserved at: $LOG_DIR/"
