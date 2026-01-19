#!/bin/bash
# SwarmPilot Quick Start - One-click cluster startup
# Usage: ./scripts/quick_start.sh
#
# Starts Mock Predictor, Scheduler, and 2 sleep model instances for quick testing.
# No PyLet, Docker, or trained ML models required - pure Python services.
#
# PYLET-021: Refine Quick Start docs for 5-minute setup

set -e

# Configuration
PREDICTOR_PORT=${PREDICTOR_PORT:-8001}
SCHEDULER_PORT=${SCHEDULER_PORT:-8000}
SLEEP_MODEL_PORT_1=${SLEEP_MODEL_PORT_1:-8300}
SLEEP_MODEL_PORT_2=${SLEEP_MODEL_PORT_2:-8301}
LOG_DIR="/tmp/swarmpilot_quickstart"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get project root (script is in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         SwarmPilot Quick Start Cluster                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv command not found.${NC}"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if ports are already in use
check_port() {
    local port=$1
    local name=$2
    if lsof -i:$port &> /dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Port $port ($name) is already in use.${NC}"
        echo "Run ./scripts/quick_stop.sh first or check for existing services."
        return 1
    fi
    return 0
}

echo "Checking ports..."
check_port $PREDICTOR_PORT "Mock Predictor" || exit 1
check_port $SCHEDULER_PORT "Scheduler" || exit 1
check_port $SLEEP_MODEL_PORT_1 "Sleep Model 1" || exit 1
check_port $SLEEP_MODEL_PORT_2 "Sleep Model 2" || exit 1
echo -e "${GREEN}✓ All ports available${NC}"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Install dependencies if needed
echo "Ensuring dependencies are installed..."
cd "$PROJECT_ROOT"
uv sync --quiet
echo -e "${GREEN}✓ Dependencies ready${NC}"
echo ""

# Start Mock Predictor (no trained models required)
echo -e "${BLUE}[1/4] Starting Mock Predictor on port $PREDICTOR_PORT...${NC}"
cd "$PROJECT_ROOT"
PREDICTOR_PORT=$PREDICTOR_PORT \
    uv run python -m tests.integration.e2e_pylet_benchmark.mock_predictor_server > "$LOG_DIR/predictor.log" 2>&1 &
PREDICTOR_PID=$!
echo $PREDICTOR_PID > "$LOG_DIR/predictor.pid"

# Wait for Predictor to be ready
sleep 2
if ! kill -0 $PREDICTOR_PID 2>/dev/null; then
    echo -e "${RED}Error: Mock Predictor failed to start. Check $LOG_DIR/predictor.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Mock Predictor started (PID: $PREDICTOR_PID)${NC}"

# Start Scheduler
echo -e "${BLUE}[2/4] Starting Scheduler on port $SCHEDULER_PORT...${NC}"
cd "$PROJECT_ROOT/scheduler"
PREDICTOR_URL="http://localhost:$PREDICTOR_PORT" \
    uv run python -m src.cli start --port $SCHEDULER_PORT > "$LOG_DIR/scheduler.log" 2>&1 &
SCHEDULER_PID=$!
echo $SCHEDULER_PID > "$LOG_DIR/scheduler.pid"

# Wait for Scheduler to be ready
sleep 2
if ! kill -0 $SCHEDULER_PID 2>/dev/null; then
    echo -e "${RED}Error: Scheduler failed to start. Check $LOG_DIR/scheduler.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Scheduler started (PID: $SCHEDULER_PID)${NC}"

# Start Sleep Model Instance 1
echo -e "${BLUE}[3/4] Starting Sleep Model instance 1 on port $SLEEP_MODEL_PORT_1...${NC}"
cd "$PROJECT_ROOT"
PORT=$SLEEP_MODEL_PORT_1 \
    MODEL_ID="sleep_model" \
    INSTANCE_ID="sleep_model-001" \
    SCHEDULER_URL="http://localhost:$SCHEDULER_PORT" \
    uv run python tests/integration/e2e_pylet_benchmark/pylet_sleep_model.py > "$LOG_DIR/sleep_model_1.log" 2>&1 &
SLEEP1_PID=$!
echo $SLEEP1_PID > "$LOG_DIR/sleep_model_1.pid"

sleep 1

# Start Sleep Model Instance 2
echo -e "${BLUE}[4/4] Starting Sleep Model instance 2 on port $SLEEP_MODEL_PORT_2...${NC}"
PORT=$SLEEP_MODEL_PORT_2 \
    MODEL_ID="sleep_model" \
    INSTANCE_ID="sleep_model-002" \
    SCHEDULER_URL="http://localhost:$SCHEDULER_PORT" \
    uv run python tests/integration/e2e_pylet_benchmark/pylet_sleep_model.py > "$LOG_DIR/sleep_model_2.log" 2>&1 &
SLEEP2_PID=$!
echo $SLEEP2_PID > "$LOG_DIR/sleep_model_2.pid"

# Wait for instances to register
sleep 3

# Health checks
echo ""
echo "Running health checks..."

health_check() {
    local url=$1
    local name=$2
    if curl -s "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $name is healthy${NC}"
        return 0
    else
        echo -e "${RED}✗ $name is not responding${NC}"
        return 1
    fi
}

health_check "http://localhost:$PREDICTOR_PORT/health" "Predictor"
health_check "http://localhost:$SCHEDULER_PORT/health" "Scheduler"
health_check "http://localhost:$SLEEP_MODEL_PORT_1/health" "Sleep Model 1"
health_check "http://localhost:$SLEEP_MODEL_PORT_2/health" "Sleep Model 2"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         SwarmPilot Cluster Ready!                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Services:"
echo "  Predictor:     http://localhost:$PREDICTOR_PORT"
echo "  Scheduler:     http://localhost:$SCHEDULER_PORT"
echo "  Sleep Model 1: http://localhost:$SLEEP_MODEL_PORT_1"
echo "  Sleep Model 2: http://localhost:$SLEEP_MODEL_PORT_2"
echo ""
echo "Logs: $LOG_DIR/"
echo ""
echo -e "${YELLOW}Test with:${NC}"
echo '  curl -X POST http://localhost:8000/task/submit \'
echo '    -H "Content-Type: application/json" \'
echo "    -d '{\"task_id\": \"test-001\", \"model_id\": \"sleep_model\", \"task_input\": {\"sleep_time\": 2}, \"metadata\": {}}'"
echo ""
echo -e "${YELLOW}Check result:${NC}"
echo '  curl "http://localhost:8000/task/info?task_id=test-001"'
echo ""
echo -e "${YELLOW}Stop cluster:${NC}"
echo "  ./scripts/quick_stop.sh"
