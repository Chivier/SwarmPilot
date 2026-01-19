#!/bin/bash
# SwarmPilot Quick Stop - One-click cluster shutdown
# Usage: ./scripts/quick_stop.sh
#
# Stops all services started by quick_start.sh
#
# PYLET-021: Refine Quick Start docs for 5-minute setup

LOG_DIR="/tmp/swarmpilot_quickstart"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Stopping SwarmPilot Quick Start cluster...${NC}"
echo ""

# Function to stop a service by PID file
stop_service() {
    local name=$1
    local pid_file="$LOG_DIR/$2.pid"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "Stopping $name (PID: $pid)..."
            kill "$pid" 2>/dev/null
            sleep 1
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null
            fi
            echo -e "${GREEN}✓ $name stopped${NC}"
        else
            echo -e "${YELLOW}$name was not running${NC}"
        fi
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}No PID file for $name${NC}"
    fi
}

# Stop services in reverse order
stop_service "Sleep Model 2" "sleep_model_2"
stop_service "Sleep Model 1" "sleep_model_1"
stop_service "Scheduler" "scheduler"
stop_service "Predictor" "predictor"

# Also kill any orphaned processes (belt and suspenders)
echo ""
echo "Cleaning up any orphaned processes..."
pkill -f "pylet_sleep_model.py" 2>/dev/null || true
pkill -f "scheduler.*src.cli start" 2>/dev/null || true
pkill -f "mock_predictor_server" 2>/dev/null || true

# Clean up log directory
if [ -d "$LOG_DIR" ]; then
    rm -f "$LOG_DIR"/*.pid
    echo -e "${GREEN}✓ Cleaned up PID files${NC}"
fi

echo ""
echo -e "${GREEN}SwarmPilot Quick Start cluster stopped.${NC}"
echo ""
echo "Logs preserved in: $LOG_DIR/"
echo "To remove logs: rm -rf $LOG_DIR"
