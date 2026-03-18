#!/bin/bash
# Single Model Example - Stop All Services
# Usage: ./examples/single_model/stop_cluster.sh

set -e

LOG_DIR="/tmp/single_model"

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}${BOLD}Single Model Example - Shutdown${NC}"
echo ""

stop_process() {
    local name=$1
    local pid_file="$LOG_DIR/$2.pid"
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null
            for _ in $(seq 1 10); do
                kill -0 "$PID" 2>/dev/null || break
                sleep 0.5
            done
            if kill -0 "$PID" 2>/dev/null; then
                kill -9 "$PID" 2>/dev/null
            fi
            echo -e "  ${GREEN}✓ Stopped $name (PID: $PID)${NC}"
        else
            echo -e "  ${YELLOW}! $name not running (PID: $PID)${NC}"
        fi
        rm -f "$pid_file"
    else
        echo -e "  ${YELLOW}! No PID file for $name${NC}"
    fi
}

echo -e "${BLUE}[1/2] Stopping instances...${NC}"
for i in 0 1 2; do
    stop_process "Instance $i" "instance-$i"
done

echo -e "${BLUE}[2/2] Stopping Scheduler...${NC}"
stop_process "Scheduler" "scheduler"

echo ""
echo -e "${GREEN}${BOLD}All services stopped.${NC}"
echo "Logs preserved at: $LOG_DIR/"
