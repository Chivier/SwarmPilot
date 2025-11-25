#!/bin/bash

# ============================================
# Stop Real Services Script
# ============================================
# Stops all services started by start_real_service.sh

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs"

echo -e "${YELLOW}Stopping all services...${NC}"

# Kill processes by pattern
kill_by_pattern() {
    local pattern="$1"
    local name="$2"

    pids=$(pgrep -f "$pattern" 2>/dev/null)
    if [[ -n "$pids" ]]; then
        echo -e "Stopping $name (PIDs: $pids)..."
        kill $pids 2>/dev/null || true
        sleep 1
        # Force kill if still running
        remaining=$(pgrep -f "$pattern" 2>/dev/null)
        if [[ -n "$remaining" ]]; then
            echo -e "Force killing $name..."
            kill -9 $remaining 2>/dev/null || true
        fi
        echo -e "${GREEN}$name stopped${NC}"
    else
        echo -e "${YELLOW}$name not running${NC}"
    fi
}

# Stop all components
kill_by_pattern "planner.*src.cli start" "Planner"
kill_by_pattern "predictor.*src.cli start" "Predictor"
kill_by_pattern "scheduler.*src.cli start" "Scheduler"
kill_by_pattern "instance.*src.cli start" "Instances"

# Clean up log files (optional)
read -p "Do you want to clean up log files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$LOG_DIR"/*.log 2>/dev/null || true
    echo -e "${GREEN}Log files cleaned${NC}"
fi

echo -e "${GREEN}All services stopped${NC}"
