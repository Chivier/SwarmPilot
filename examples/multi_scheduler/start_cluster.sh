#!/bin/bash
# Multi-Scheduler - Start Services (Sleep Models)
# Usage: ./examples/multi_scheduler/start_cluster.sh
#
# Starts three per-model Schedulers and Planner (PyLet-enabled) for the
# multi-scheduler example with sleep models.
# PyLet cluster must be running separately (see scripts/start_pylet_test_cluster.sh)
#
# Architecture:
# - Dummy health server on :8001 (for planner PyLet init check)
# - Planner on :8003
# - Scheduler A (sleep_model_a) on :8010
# - Scheduler B (sleep_model_b) on :8011
# - Scheduler C (sleep_model_c) on :8012

set -e

# Configuration
DUMMY_HEALTH_PORT=${DUMMY_HEALTH_PORT:-8001}
SCHEDULER_A_PORT=${SCHEDULER_A_PORT:-8010}
SCHEDULER_B_PORT=${SCHEDULER_B_PORT:-8011}
SCHEDULER_C_PORT=${SCHEDULER_C_PORT:-8012}
PLANNER_PORT=${PLANNER_PORT:-8003}
PYLET_HEAD_PORT=${PYLET_HEAD_PORT:-5100}
LOG_DIR="/tmp/multi_scheduler"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get project root (script is in examples/multi_scheduler/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Multi-Scheduler - Sleep Models Startup            ║${NC}"
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
        echo "Run ./examples/multi_scheduler/stop_cluster.sh first."
        return 1
    fi
    return 0
}

echo "Checking ports..."
check_port $DUMMY_HEALTH_PORT "Dummy Health Server" || exit 1
check_port $SCHEDULER_A_PORT "Scheduler A" || exit 1
check_port $SCHEDULER_B_PORT "Scheduler B" || exit 1
check_port $SCHEDULER_C_PORT "Scheduler C" || exit 1
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

# Start Dummy Health Server on DUMMY_HEALTH_PORT for planner PyLet init check
echo -e "${BLUE}[1/5] Starting Dummy Health Server on port $DUMMY_HEALTH_PORT...${NC}"

# Python one-liner for minimal health server
python3 -c "
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{\"status\": \"ok\"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress logging

server = HTTPServer(('localhost', $DUMMY_HEALTH_PORT), HealthHandler)
print('Dummy health server started', flush=True)
sys.stdout.flush()
server.serve_forever()
" > "$LOG_DIR/dummy_health.log" 2>&1 &

DUMMY_PID=$!
echo $DUMMY_PID > "$LOG_DIR/dummy_health.pid"

# Wait for dummy server to start
sleep 1
if ! kill -0 $DUMMY_PID 2>/dev/null; then
    echo -e "${RED}Error: Dummy Health Server failed to start. Check $LOG_DIR/dummy_health.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Dummy Health Server started (PID: $DUMMY_PID)${NC}"

# Start Planner with PyLet integration (pointing to dummy health server for init check)
echo -e "${BLUE}[2/5] Starting Planner on port $PLANNER_PORT...${NC}"
cd "$PROJECT_ROOT"

# Build custom command for sleep model
SLEEP_MODEL_PATH="$PROJECT_ROOT/examples/multi_scheduler/pylet_sleep_model.py"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
CUSTOM_CMD="MODEL_ID={model_id} $VENV_PYTHON $SLEEP_MODEL_PATH"

PLANNER_PORT=$PLANNER_PORT \
    PYLET_ENABLED=true \
    PYLET_HEAD_URL="http://localhost:$PYLET_HEAD_PORT" \
    PYLET_REUSE_CLUSTER=false \
    PYLET_GPU_COUNT=0 \
    PYLET_CPU_COUNT=1 \
    PYLET_CUSTOM_COMMAND="$CUSTOM_CMD" \
    SCHEDULER_URL="http://localhost:$DUMMY_HEALTH_PORT" \
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

# Stop dummy health server now that planner is initialized
echo -e "${BLUE}Stopping Dummy Health Server...${NC}"
if kill -0 $DUMMY_PID 2>/dev/null; then
    kill $DUMMY_PID 2>/dev/null
    sleep 1
    if kill -0 $DUMMY_PID 2>/dev/null; then
        kill -9 $DUMMY_PID 2>/dev/null
    fi
    echo -e "${GREEN}✓ Dummy Health Server stopped${NC}"
fi
echo ""

# Start Scheduler A (sleep_model_a)
echo -e "${BLUE}[3/5] Starting Scheduler A (sleep_model_a) on port $SCHEDULER_A_PORT...${NC}"
cd "$PROJECT_ROOT"
SCHEDULER_MODEL_ID="sleep_model_a" \
    PLANNER_REGISTRATION_URL="http://localhost:$PLANNER_PORT" \
    SCHEDULER_SELF_URL="http://localhost:$SCHEDULER_A_PORT" \
    PREDICTOR_MODE="library" \
    uv run python -m uvicorn swarmpilot.scheduler.api:app --host 0.0.0.0 --port $SCHEDULER_A_PORT > "$LOG_DIR/scheduler-a.log" 2>&1 &
SCHEDULER_A_PID=$!
echo $SCHEDULER_A_PID > "$LOG_DIR/scheduler-a.pid"

# Wait for Scheduler A to be ready
sleep 2
if ! kill -0 $SCHEDULER_A_PID 2>/dev/null; then
    echo -e "${RED}Error: Scheduler A failed to start. Check $LOG_DIR/scheduler-a.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Scheduler A started (PID: $SCHEDULER_A_PID)${NC}"

# Start Scheduler B (sleep_model_b)
echo -e "${BLUE}[4/5] Starting Scheduler B (sleep_model_b) on port $SCHEDULER_B_PORT...${NC}"
cd "$PROJECT_ROOT"
SCHEDULER_MODEL_ID="sleep_model_b" \
    PLANNER_REGISTRATION_URL="http://localhost:$PLANNER_PORT" \
    SCHEDULER_SELF_URL="http://localhost:$SCHEDULER_B_PORT" \
    PREDICTOR_MODE="library" \
    uv run python -m uvicorn swarmpilot.scheduler.api:app --host 0.0.0.0 --port $SCHEDULER_B_PORT > "$LOG_DIR/scheduler-b.log" 2>&1 &
SCHEDULER_B_PID=$!
echo $SCHEDULER_B_PID > "$LOG_DIR/scheduler-b.pid"

# Wait for Scheduler B to be ready
sleep 2
if ! kill -0 $SCHEDULER_B_PID 2>/dev/null; then
    echo -e "${RED}Error: Scheduler B failed to start. Check $LOG_DIR/scheduler-b.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Scheduler B started (PID: $SCHEDULER_B_PID)${NC}"

# Start Scheduler C (sleep_model_c)
echo -e "${BLUE}[5/5] Starting Scheduler C (sleep_model_c) on port $SCHEDULER_C_PORT...${NC}"
cd "$PROJECT_ROOT"
SCHEDULER_MODEL_ID="sleep_model_c" \
    PLANNER_REGISTRATION_URL="http://localhost:$PLANNER_PORT" \
    SCHEDULER_SELF_URL="http://localhost:$SCHEDULER_C_PORT" \
    PREDICTOR_MODE="library" \
    uv run python -m uvicorn swarmpilot.scheduler.api:app --host 0.0.0.0 --port $SCHEDULER_C_PORT > "$LOG_DIR/scheduler-c.log" 2>&1 &
SCHEDULER_C_PID=$!
echo $SCHEDULER_C_PID > "$LOG_DIR/scheduler-c.pid"

# Wait for Scheduler C to be ready
sleep 2
if ! kill -0 $SCHEDULER_C_PID 2>/dev/null; then
    echo -e "${RED}Error: Scheduler C failed to start. Check $LOG_DIR/scheduler-c.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Scheduler C started (PID: $SCHEDULER_C_PID)${NC}"

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

health_check "http://localhost:$SCHEDULER_A_PORT/v1/health" "Scheduler A"
health_check "http://localhost:$SCHEDULER_B_PORT/v1/health" "Scheduler B"
health_check "http://localhost:$SCHEDULER_C_PORT/v1/health" "Scheduler C"
health_check "http://localhost:$PLANNER_PORT/v1/health" "Planner"

# Verify scheduler registration
echo ""
echo "Verifying scheduler registration..."
REGISTERED=$(curl -s "http://localhost:$PLANNER_PORT/v1/scheduler/list" 2>/dev/null)
REG_COUNT=$(echo "$REGISTERED" | python3 -c "import sys, json; print(json.load(sys.stdin).get('total', 0))" 2>/dev/null || echo "0")

if [ "$REG_COUNT" -ge 3 ]; then
    echo -e "${GREEN}✓ $REG_COUNT schedulers registered with planner${NC}"
    echo "$REGISTERED" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for s in data.get('schedulers', []):
    print(f\"  {s['model_id']}: {s['scheduler_url']}\")
" 2>/dev/null
else
    echo -e "${YELLOW}! Only $REG_COUNT scheduler(s) registered (expected 3)${NC}"
    echo "  Check scheduler logs for registration errors."
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Multi-Scheduler Services Ready!                   ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Services:"
echo "  Scheduler A (sleep_model_a): http://localhost:$SCHEDULER_A_PORT"
echo "  Scheduler B (sleep_model_b): http://localhost:$SCHEDULER_B_PORT"
echo "  Scheduler C (sleep_model_c): http://localhost:$SCHEDULER_C_PORT"
echo "  Planner:                      http://localhost:$PLANNER_PORT"
echo "  PyLet:                        http://localhost:$PYLET_HEAD_PORT"
echo ""
echo "Logs: $LOG_DIR/"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Deploy instances: ./examples/multi_scheduler/deploy_model.sh"
echo "2. Generate traffic: python examples/multi_scheduler/generate_workload.py"
echo "3. Stop cluster:     ./examples/multi_scheduler/stop_cluster.sh"
echo ""
