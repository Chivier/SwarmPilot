#!/bin/bash
# LLM Cluster - Stop Services
# Usage: ./examples/llm_cluster/stop_cluster.sh
#
# Stops all services in proper order (planner, scheduler, predictor)
# and terminates PyLet instances via planner.

set -e

LOG_DIR="/tmp/llm_cluster"
PLANNER_PORT=${PLANNER_PORT:-8003}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     LLM Cluster - Shutdown                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Helper function to stop a service by PID file
stop_service() {
    local name=$1
    local pid_file=$2
    local step=$3

    if [ ! -f "$pid_file" ]; then
        echo -e "${YELLOW}[$step] $name not running (PID file not found)${NC}"
        return 0
    fi

    local pid=$(<"$pid_file")

    if kill -0 "$pid" 2>/dev/null; then
        echo -e "${BLUE}[$step] Stopping $name (PID: $pid)...${NC}"
        kill "$pid" 2>/dev/null || true

        # Wait for graceful shutdown
        for attempt in {1..10}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo -e "${GREEN}[$step] $name stopped${NC}"
                rm -f "$pid_file"
                return 0
            fi
            sleep 1
        done

        # Force kill if still running
        echo -e "${YELLOW}[$step] Force killing $name...${NC}"
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
        echo -e "${GREEN}[$step] $name force killed${NC}"
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}[$step] $name not running${NC}"
        rm -f "$pid_file"
    fi
}

# Step 1: Terminate PyLet instances via planner
echo -e "${BLUE}[1/4] Terminating PyLet instances via planner...${NC}"
if curl -s "http://localhost:$PLANNER_PORT/v1/health" > /dev/null 2>&1; then
    if curl -s -X POST "http://localhost:$PLANNER_PORT/v1/terminate-all" > /dev/null 2>&1; then
        echo -e "${GREEN}[1/4] PyLet instances terminated${NC}"
    else
        echo -e "${YELLOW}[1/4] Could not terminate PyLet instances via planner${NC}"
    fi
else
    echo -e "${YELLOW}[1/4] Planner not responding, skipping PyLet termination${NC}"
fi
echo ""

# Step 2: Stop Planner
stop_service "Planner" "$LOG_DIR/planner.pid" "[2/4]"
echo ""

# Step 3: Stop Scheduler
stop_service "Scheduler" "$LOG_DIR/scheduler.pid" "[3/4]"
echo ""

# Step 4: Stop Predictor
stop_service "Predictor" "$LOG_DIR/predictor.pid" "[4/4]"
echo ""

echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     LLM Cluster Services Stopped                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Logs available in: $LOG_DIR/"
