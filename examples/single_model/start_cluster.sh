#!/bin/bash
# Single Model Example - Start Scheduler
# Usage: ./examples/single_model/start_cluster.sh
#
# Starts one Scheduler for Qwen/Qwen3-8B-VL on port 8000.
# No Planner needed — instances are registered manually via deploy_model.sh.

set -e

# Configuration
SCHEDULER_PORT=${SCHEDULER_PORT:-8000}
MODEL_ID=${MODEL_ID:-"Qwen/Qwen3-8B-VL"}
LOG_DIR="/tmp/single_model"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# Project root (script is in examples/single_model/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo -e "${BLUE}${BOLD}Single Model Example - Scheduler Startup${NC}"
echo ""

if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    exit 1
fi

check_port() {
    local port=$1
    local name=$2
    if lsof -i:"$port" &> /dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Port $port ($name) already in use.${NC}"
        echo "Run ./examples/single_model/stop_cluster.sh first."
        return 1
    fi
    return 0
}

check_port $SCHEDULER_PORT "Scheduler" || exit 1
echo -e "${GREEN}✓ Port $SCHEDULER_PORT available${NC}"
echo ""

mkdir -p "$LOG_DIR"

# [1/2] Start Scheduler
echo -e "${BLUE}[1/2] Starting Scheduler on port $SCHEDULER_PORT (model: $MODEL_ID)...${NC}"
cd "$PROJECT_ROOT"
SCHEDULER_MODEL_ID="$MODEL_ID" \
    PREDICTOR_MODE="library" \
    uv run sscheduler start --port "$SCHEDULER_PORT" \
    > "$LOG_DIR/scheduler.log" 2>&1 &
SCHEDULER_PID=$!
echo "$SCHEDULER_PID" > "$LOG_DIR/scheduler.pid"

sleep 2
if ! kill -0 "$SCHEDULER_PID" 2>/dev/null; then
    echo -e "${RED}Error: Scheduler failed to start. Check $LOG_DIR/scheduler.log${NC}"
    exit 1
fi

for attempt in $(seq 1 15); do
    if curl -sf "http://localhost:$SCHEDULER_PORT/v1/health" > /dev/null 2>&1; then
        break
    fi
    if [ "$attempt" -eq 15 ]; then
        echo -e "${RED}Error: Scheduler not healthy after 15s${NC}"
        exit 1
    fi
    sleep 1
done
echo -e "${GREEN}✓ Scheduler started (PID: $SCHEDULER_PID)${NC}"

# [2/2] Summary
echo ""
echo -e "${BLUE}[2/2] ${GREEN}${BOLD}Scheduler ready!${NC}"
echo ""
echo "  Scheduler: http://localhost:$SCHEDULER_PORT"
echo "  Model:     $MODEL_ID"
echo "  Logs:      $LOG_DIR/scheduler.log"
echo ""
echo -e "${YELLOW}Next:${NC} ./examples/single_model/deploy_model.sh"
