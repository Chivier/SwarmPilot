#!/bin/bash
# Mock LLM Cluster - Stop Services
# Usage: ./examples/mock_llm_cluster/stop_cluster.sh
#
# Stops all services started by start_cluster.sh and terminates PyLet instances.
#
# PYLET-022: Mock LLM Cluster Example

set -e

# Configuration
PLANNER_PORT=${PLANNER_PORT:-8002}
LOG_DIR="/tmp/mock_llm_cluster"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Mock LLM Cluster - Shutdown                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# First, terminate all PyLet instances via Planner API
echo -e "${BLUE}[1/4] Terminating PyLet instances...${NC}"
if curl -s "http://localhost:$PLANNER_PORT/health" > /dev/null 2>&1; then
    RESULT=$(curl -s -X POST "http://localhost:$PLANNER_PORT/pylet/terminate-all" 2>/dev/null || echo '{"error": "failed"}')
    if echo "$RESULT" | grep -q '"success": true' 2>/dev/null; then
        TERMINATED=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('total', 0))" 2>/dev/null || echo "?")
        echo -e "${GREEN}✓ Terminated $TERMINATED PyLet instances${NC}"
    else
        echo -e "${YELLOW}! Could not terminate PyLet instances (may already be stopped)${NC}"
    fi
else
    echo -e "${YELLOW}! Planner not responding, skipping PyLet cleanup${NC}"
fi

# Stop processes via PID files
stop_process() {
    local name=$1
    local pid_file="$LOG_DIR/$2.pid"

    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null
            # Wait for process to terminate
            for i in {1..10}; do
                if ! kill -0 "$PID" 2>/dev/null; then
                    break
                fi
                sleep 0.5
            done
            # Force kill if still running
            if kill -0 "$PID" 2>/dev/null; then
                kill -9 "$PID" 2>/dev/null
            fi
            echo -e "${GREEN}✓ Stopped $name (PID: $PID)${NC}"
        else
            echo -e "${YELLOW}! $name was not running (PID: $PID)${NC}"
        fi
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}! No PID file for $name${NC}"
    fi
}

echo -e "${BLUE}[2/4] Stopping Planner...${NC}"
stop_process "Planner" "planner"

echo -e "${BLUE}[3/4] Stopping Scheduler...${NC}"
stop_process "Scheduler" "scheduler"

echo -e "${BLUE}[4/4] Stopping Predictor...${NC}"
stop_process "Predictor" "predictor"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          Mock LLM Cluster Stopped                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Logs preserved at: $LOG_DIR/"
echo ""
echo -e "${YELLOW}Note:${NC} PyLet cluster is still running. To stop it:"
echo "  ./scripts/stop_pylet_test_cluster.sh"
