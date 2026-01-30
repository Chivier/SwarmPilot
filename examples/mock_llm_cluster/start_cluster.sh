#!/bin/bash
# Mock LLM Cluster - Start Services (Multi-Scheduler)
# Usage: ./examples/mock_llm_cluster/start_cluster.sh
#
# Starts Mock Predictor, two per-model Schedulers, and Planner (PyLet-enabled)
# for the mock LLM cluster example.
# PyLet cluster must be running separately (see scripts/start_pylet_test_cluster.sh)
#
# PYLET-024: Multi-Scheduler Architecture

set -e

# Configuration
PREDICTOR_PORT=${PREDICTOR_PORT:-8001}
SCHEDULER_7B_PORT=${SCHEDULER_7B_PORT:-8010}
SCHEDULER_32B_PORT=${SCHEDULER_32B_PORT:-8020}
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
echo -e "${BLUE}║     Mock LLM Cluster - Multi-Scheduler Startup        ║${NC}"
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
check_port $SCHEDULER_7B_PORT "Scheduler (llm-7b)" || exit 1
check_port $SCHEDULER_32B_PORT "Scheduler (llm-32b)" || exit 1
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
echo -e "${BLUE}[1/4] Starting Mock Predictor on port $PREDICTOR_PORT...${NC}"
cd "$PROJECT_ROOT"
PREDICTOR_PORT=$PREDICTOR_PORT \
    uv run python "$SCRIPT_DIR/mock_predictor_server.py" > "$LOG_DIR/predictor.log" 2>&1 &
PREDICTOR_PID=$!
echo $PREDICTOR_PID > "$LOG_DIR/predictor.pid"

# Wait for Predictor to be ready
sleep 2
if ! kill -0 $PREDICTOR_PID 2>/dev/null; then
    echo -e "${RED}Error: Mock Predictor failed to start. Check $LOG_DIR/predictor.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Mock Predictor started (PID: $PREDICTOR_PID)${NC}"

# Start Planner first (schedulers need it for registration)
echo -e "${BLUE}[2/4] Starting Planner on port $PLANNER_PORT...${NC}"
cd "$PROJECT_ROOT"

# Build custom command for mock LLM server
MOCK_SERVER_PATH="$PROJECT_ROOT/examples/mock_llm_cluster/mock_llm_server.py"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
CUSTOM_CMD="MODEL_ID={model_id} $VENV_PYTHON $MOCK_SERVER_PATH"

PLANNER_PORT=$PLANNER_PORT \
    PYLET_ENABLED=true \
    PYLET_HEAD_URL="http://localhost:$PYLET_HEAD_PORT" \
    PYLET_REUSE_CLUSTER=true \
    PYLET_GPU_COUNT=0 \
    PYLET_CPU_COUNT=1 \
    PYLET_CUSTOM_COMMAND="$CUSTOM_CMD" \
    uv run python -m uvicorn swarmpilot.planner.api:app --host 0.0.0.0 --port $PLANNER_PORT > "$LOG_DIR/planner.log" 2>&1 &
PLANNER_PID=$!
echo $PLANNER_PID > "$LOG_DIR/planner.pid"

# Wait for Planner to be ready
sleep 3
if ! kill -0 $PLANNER_PID 2>/dev/null; then
    echo -e "${RED}Error: Planner failed to start. Check $LOG_DIR/planner.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Planner started (PID: $PLANNER_PID)${NC}"

# Start Scheduler for llm-7b
echo -e "${BLUE}[3/4] Starting Scheduler (llm-7b) on port $SCHEDULER_7B_PORT...${NC}"
cd "$PROJECT_ROOT"
PREDICTOR_URL="http://localhost:$PREDICTOR_PORT" \
    SCHEDULER_MODEL_ID="llm-7b" \
    PLANNER_REGISTRATION_URL="http://localhost:$PLANNER_PORT" \
    SCHEDULER_SELF_URL="http://localhost:$SCHEDULER_7B_PORT" \
    uv run python -m swarmpilot.scheduler.cli start --port $SCHEDULER_7B_PORT > "$LOG_DIR/scheduler-7b.log" 2>&1 &
SCHEDULER_7B_PID=$!
echo $SCHEDULER_7B_PID > "$LOG_DIR/scheduler-7b.pid"

# Wait for Scheduler 7b to be ready
sleep 3
if ! kill -0 $SCHEDULER_7B_PID 2>/dev/null; then
    echo -e "${RED}Error: Scheduler (llm-7b) failed to start. Check $LOG_DIR/scheduler-7b.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Scheduler (llm-7b) started (PID: $SCHEDULER_7B_PID)${NC}"

# Start Scheduler for llm-32b
echo -e "${BLUE}[4/4] Starting Scheduler (llm-32b) on port $SCHEDULER_32B_PORT...${NC}"
cd "$PROJECT_ROOT"
PREDICTOR_URL="http://localhost:$PREDICTOR_PORT" \
    SCHEDULER_MODEL_ID="llm-32b" \
    PLANNER_REGISTRATION_URL="http://localhost:$PLANNER_PORT" \
    SCHEDULER_SELF_URL="http://localhost:$SCHEDULER_32B_PORT" \
    uv run python -m swarmpilot.scheduler.cli start --port $SCHEDULER_32B_PORT > "$LOG_DIR/scheduler-32b.log" 2>&1 &
SCHEDULER_32B_PID=$!
echo $SCHEDULER_32B_PID > "$LOG_DIR/scheduler-32b.pid"

# Wait for Scheduler 32b to be ready
sleep 3
if ! kill -0 $SCHEDULER_32B_PID 2>/dev/null; then
    echo -e "${RED}Error: Scheduler (llm-32b) failed to start. Check $LOG_DIR/scheduler-32b.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Scheduler (llm-32b) started (PID: $SCHEDULER_32B_PID)${NC}"

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
health_check "http://localhost:$SCHEDULER_7B_PORT/v1/health" "Scheduler (llm-7b)"
health_check "http://localhost:$SCHEDULER_32B_PORT/v1/health" "Scheduler (llm-32b)"
health_check "http://localhost:$PLANNER_PORT/v1/health" "Planner"

# Verify scheduler registration
echo ""
echo "Verifying scheduler registration..."
REGISTERED=$(curl -s "http://localhost:$PLANNER_PORT/v1/scheduler/list" 2>/dev/null)
REG_COUNT=$(echo "$REGISTERED" | python3 -c "import sys, json; print(json.load(sys.stdin).get('total', 0))" 2>/dev/null || echo "0")

if [ "$REG_COUNT" -ge 2 ]; then
    echo -e "${GREEN}✓ $REG_COUNT schedulers registered with planner${NC}"
    echo "$REGISTERED" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for s in data.get('schedulers', []):
    print(f\"  {s['model_id']}: {s['scheduler_url']}\")
" 2>/dev/null
else
    echo -e "${YELLOW}! Only $REG_COUNT scheduler(s) registered (expected 2)${NC}"
    echo "  Check scheduler logs for registration errors."
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Mock LLM Cluster Services Ready!                  ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Services:"
echo "  Predictor:          http://localhost:$PREDICTOR_PORT"
echo "  Scheduler (llm-7b): http://localhost:$SCHEDULER_7B_PORT"
echo "  Scheduler (llm-32b):http://localhost:$SCHEDULER_32B_PORT"
echo "  Planner:            http://localhost:$PLANNER_PORT"
echo "  PyLet:              http://localhost:$PYLET_HEAD_PORT"
echo ""
echo "Logs: $LOG_DIR/"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Deploy models:    ./examples/mock_llm_cluster/deploy_models.sh"
echo "2. Generate traffic: python examples/mock_llm_cluster/generate_workload.py"
echo "3. Stop cluster:     ./examples/mock_llm_cluster/stop_cluster.sh"
