#!/bin/bash
# Mock LLM Cluster - Start Services
# Usage: ./examples/mock_llm_cluster/start_cluster.sh
#
# Starts Mock Predictor, Scheduler, and Planner (PyLet-enabled) for the mock LLM cluster example.
# PyLet cluster must be running separately (see scripts/start_pylet_test_cluster.sh)
#
# PYLET-022: Mock LLM Cluster Example

set -e

# Configuration
PREDICTOR_PORT=${PREDICTOR_PORT:-8001}
SCHEDULER_PORT=${SCHEDULER_PORT:-8000}
PLANNER_PORT=${PLANNER_PORT:-8002}
PYLET_HEAD_PORT=${PYLET_HEAD_PORT:-5100}
LOG_DIR="/tmp/mock_llm_cluster"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get project root (script is in examples/mock_llm_cluster/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Mock LLM Cluster - Service Startup            ║${NC}"
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
        echo "Run ./examples/mock_llm_cluster/stop_cluster.sh first."
        return 1
    fi
    return 0
}

echo "Checking ports..."
check_port $PREDICTOR_PORT "Mock Predictor" || exit 1
check_port $SCHEDULER_PORT "Scheduler" || exit 1
check_port $PLANNER_PORT "Planner" || exit 1
echo -e "${GREEN}✓ Service ports available${NC}"
echo ""

# Check if PyLet cluster is running
echo "Checking PyLet cluster..."
if ! curl -s "http://localhost:$PYLET_HEAD_PORT/workers" > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: PyLet cluster not responding on port $PYLET_HEAD_PORT${NC}"
    echo "Start PyLet cluster first with:"
    echo "  ./scripts/start_pylet_test_cluster.sh"
    echo ""
    read -p "Continue without PyLet? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    WORKERS=$(curl -s "http://localhost:$PYLET_HEAD_PORT/workers" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "?")
    echo -e "${GREEN}✓ PyLet cluster running ($WORKERS workers)${NC}"
fi
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Install dependencies if needed
echo "Ensuring dependencies are installed..."
cd "$PROJECT_ROOT"
uv sync --quiet
echo -e "${GREEN}✓ Dependencies ready${NC}"
echo ""

# Start Mock Predictor
echo -e "${BLUE}[1/3] Starting Mock Predictor on port $PREDICTOR_PORT...${NC}"
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
echo -e "${BLUE}[2/3] Starting Scheduler on port $SCHEDULER_PORT...${NC}"
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

# Start Planner with PyLet enabled
echo -e "${BLUE}[3/3] Starting Planner on port $PLANNER_PORT...${NC}"
cd "$PROJECT_ROOT/planner"

# Build custom command for mock LLM server
# MODEL_ID is passed via {model_id} placeholder, PORT is set by PyLet
# Use venv Python to ensure dependencies are available
MOCK_SERVER_PATH="$PROJECT_ROOT/examples/mock_llm_cluster/mock_llm_server.py"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
CUSTOM_CMD="MODEL_ID={model_id} $VENV_PYTHON $MOCK_SERVER_PATH"

PLANNER_PORT=$PLANNER_PORT \
    SCHEDULER_URL="http://localhost:$SCHEDULER_PORT" \
    PYLET_ENABLED=true \
    PYLET_HEAD_URL="http://localhost:$PYLET_HEAD_PORT" \
    PYLET_REUSE_CLUSTER=true \
    PYLET_GPU_COUNT=0 \
    PYLET_CPU_COUNT=1 \
    PYLET_CUSTOM_COMMAND="$CUSTOM_CMD" \
    uv run python -m uvicorn src.api:app --host 0.0.0.0 --port $PLANNER_PORT > "$LOG_DIR/planner.log" 2>&1 &
PLANNER_PID=$!
echo $PLANNER_PID > "$LOG_DIR/planner.pid"

# Wait for Planner to be ready
sleep 3
if ! kill -0 $PLANNER_PID 2>/dev/null; then
    echo -e "${RED}Error: Planner failed to start. Check $LOG_DIR/planner.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Planner started (PID: $PLANNER_PID)${NC}"

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
health_check "http://localhost:$PLANNER_PORT/health" "Planner"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          Mock LLM Cluster Services Ready!              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Services:"
echo "  Predictor: http://localhost:$PREDICTOR_PORT"
echo "  Scheduler: http://localhost:$SCHEDULER_PORT"
echo "  Planner:   http://localhost:$PLANNER_PORT"
echo "  PyLet:     http://localhost:$PYLET_HEAD_PORT"
echo ""
echo "Logs: $LOG_DIR/"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Deploy models:    ./examples/mock_llm_cluster/deploy_models.sh"
echo "2. Generate traffic: python examples/mock_llm_cluster/generate_workload.py"
echo "3. Stop cluster:     ./examples/mock_llm_cluster/stop_cluster.sh"
