#!/bin/bash
# PyLet Benchmark Cluster - Start Services (Direct Registration)
# Usage: ./examples/pylet_benchmark/start_cluster.sh
#
# Starts Mock Predictor and Scheduler (simplest setup).
# NO planner needed - instances register directly with scheduler.
# Instances started separately via deploy_model.sh
#
# PYLET-025: Direct Scheduler Registration Example

set -e

# Configuration
PREDICTOR_PORT=${PREDICTOR_PORT:-8002}
SCHEDULER_PORT=${SCHEDULER_PORT:-8000}
LOG_DIR="/tmp/pylet_benchmark"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get project root (script is in examples/pylet_benchmark/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     PyLet Benchmark - Start Services (Direct Reg)     ║${NC}"
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
        echo "Run ./examples/pylet_benchmark/stop_cluster.sh first."
        return 1
    fi
    return 0
}

echo "Checking ports..."
check_port $PREDICTOR_PORT "Mock Predictor" || exit 1
check_port $SCHEDULER_PORT "Scheduler" || exit 1
echo -e "${GREEN}✓ Service ports available${NC}"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Install dependencies if needed
echo "Ensuring dependencies are installed..."
cd "$PROJECT_ROOT"
uv sync --extra pylet --quiet
echo -e "${GREEN}✓ Dependencies ready${NC}"
echo ""

# Start Mock Predictor
echo -e "${BLUE}[1/2] Starting Mock Predictor on port $PREDICTOR_PORT...${NC}"
cd "$PROJECT_ROOT"
PREDICTOR_PORT=$PREDICTOR_PORT \
    uv run python examples/pylet_benchmark/mock_predictor_server.py > "$LOG_DIR/predictor.log" 2>&1 &
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
echo -e "${BLUE}[2/2] Starting Scheduler on port $SCHEDULER_PORT...${NC}"
cd "$PROJECT_ROOT"

env \
    PREDICTOR_URL="http://localhost:$PREDICTOR_PORT" \
    SCHEDULER_PORT=$SCHEDULER_PORT \
    SCHEDULING_STRATEGY="round_robin" \
    SCHEDULER_LOGURU_LEVEL="INFO" \
    uv run python -m uvicorn swarmpilot.scheduler.api:app --host 0.0.0.0 --port $SCHEDULER_PORT > "$LOG_DIR/scheduler.log" 2>&1 &
SCHEDULER_PID=$!
echo $SCHEDULER_PID > "$LOG_DIR/scheduler.pid"

# Wait for Scheduler to be ready
sleep 3
if ! kill -0 $SCHEDULER_PID 2>/dev/null; then
    echo -e "${RED}Error: Scheduler failed to start. Check $LOG_DIR/scheduler.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Scheduler started (PID: $SCHEDULER_PID)${NC}"

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
health_check "http://localhost:$SCHEDULER_PORT/v1/health" "Scheduler"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Services Ready!                                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Services:"
echo "  Predictor:  http://localhost:$PREDICTOR_PORT"
echo "  Scheduler:  http://localhost:$SCHEDULER_PORT"
echo ""
echo "Logs: $LOG_DIR/"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Deploy models:    ./examples/pylet_benchmark/deploy_model.sh"
echo "2. Generate traffic: python examples/pylet_benchmark/generate_workload.py"
echo "3. Stop cluster:     ./examples/pylet_benchmark/stop_cluster.sh"
echo ""
