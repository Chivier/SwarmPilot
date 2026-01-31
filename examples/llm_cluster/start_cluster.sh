#!/bin/bash
# LLM Cluster - Start Services (Multi-Scheduler)
# Usage: ./examples/llm_cluster/start_cluster.sh
#
# Starts Planner (PyLet-enabled) and three per-model Schedulers
# for the LLM cluster example with 3 models (llm_fast, llm_medium, llm_slow).
# PyLet cluster must be running separately (see scripts/start_pylet_test_cluster.sh)
#
# Architecture:
# - Planner on :8003 (with PyLet)
# - Scheduler llm_fast   on :8010
# - Scheduler llm_medium on :8011
# - Scheduler llm_slow   on :8012
#
# PYLET-036: Per-Model Schedulers + Correct Startup Sequence

set -e

# Configuration
DUMMY_HEALTH_PORT=${DUMMY_HEALTH_PORT:-8099}
SCHEDULER_FAST_PORT=${SCHEDULER_FAST_PORT:-8010}
SCHEDULER_MEDIUM_PORT=${SCHEDULER_MEDIUM_PORT:-8011}
SCHEDULER_SLOW_PORT=${SCHEDULER_SLOW_PORT:-8012}
PLANNER_PORT=${PLANNER_PORT:-8003}
PYLET_HEAD_PORT=${PYLET_HEAD_PORT:-5100}
LOG_DIR="/tmp/llm_cluster"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get project root (script is in examples/llm_cluster/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     LLM Cluster - Multi-Scheduler Startup             ║${NC}"
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
        echo "Run ./examples/llm_cluster/stop_cluster.sh first."
        return 1
    fi
    return 0
}

echo "Checking ports..."
check_port $SCHEDULER_FAST_PORT "Scheduler (llm_fast)" || exit 1
check_port $SCHEDULER_MEDIUM_PORT "Scheduler (llm_medium)" || exit 1
check_port $SCHEDULER_SLOW_PORT "Scheduler (llm_slow)" || exit 1
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
uv sync --extra pylet --quiet
echo -e "${GREEN}✓ Dependencies ready${NC}"
echo ""

# Step 1: Start Dummy Health Server for planner PyLet init check
echo -e "${BLUE}[1/5] Starting Dummy Health Server on port $DUMMY_HEALTH_PORT...${NC}"

python3 -c "
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/health', '/v1/health'):
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

# Step 2: Start Planner (schedulers self-register after startup)
echo -e "${BLUE}[2/5] Starting Planner on port $PLANNER_PORT...${NC}"
cd "$PROJECT_ROOT"

# Build custom command for mock vLLM server
MOCK_VLLM_PATH="$PROJECT_ROOT/examples/llm_cluster/mock_vllm_server.py"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
CUSTOM_CMD="MODEL_ID={model_id} $VENV_PYTHON $MOCK_VLLM_PATH"

PLANNER_PORT=$PLANNER_PORT \
    PYLET_ENABLED=true \
    PYLET_HEAD_URL="http://localhost:$PYLET_HEAD_PORT" \
    PYLET_REUSE_CLUSTER=true \
    PYLET_GPU_COUNT=0 \
    PYLET_CPU_COUNT=1 \
    PYLET_CUSTOM_COMMAND="$CUSTOM_CMD" \
    SCHEDULER_URL="http://localhost:$DUMMY_HEALTH_PORT" \
    uv run python -m uvicorn swarmpilot.planner.api:app --host 0.0.0.0 --port $PLANNER_PORT \
    > "$LOG_DIR/planner.log" 2>&1 &
PLANNER_PID=$!
echo $PLANNER_PID > "$LOG_DIR/planner.pid"

# Wait for Planner to be ready
sleep 3
if ! kill -0 $PLANNER_PID 2>/dev/null; then
    echo -e "${RED}Error: Planner failed to start. Check $LOG_DIR/planner.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Planner started (PID: $PLANNER_PID)${NC}"

# Wait for planner health check
echo "Waiting for planner health check..."
for attempt in {1..30}; do
    if curl -s "http://localhost:$PLANNER_PORT/v1/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Planner is healthy${NC}"
        break
    fi
    if [ $attempt -eq 30 ]; then
        echo -e "${RED}Error: Planner failed to become healthy${NC}"
        exit 1
    fi
    sleep 1
done

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

# Step 3: Start Scheduler for llm_fast
echo -e "${BLUE}[3/5] Starting Scheduler (llm_fast) on port $SCHEDULER_FAST_PORT...${NC}"
cd "$PROJECT_ROOT"
SCHEDULER_MODEL_ID="llm_fast" \
    PLANNER_REGISTRATION_URL="http://localhost:$PLANNER_PORT" \
    SCHEDULER_SELF_URL="http://localhost:$SCHEDULER_FAST_PORT" \
    PREDICTOR_MODE="library" \
    uv run python -m uvicorn swarmpilot.scheduler.api:app --host 0.0.0.0 --port $SCHEDULER_FAST_PORT \
    > "$LOG_DIR/scheduler-fast.log" 2>&1 &
SCHEDULER_FAST_PID=$!
echo $SCHEDULER_FAST_PID > "$LOG_DIR/scheduler-fast.pid"

# Wait for Scheduler fast to be ready
sleep 2
if ! kill -0 $SCHEDULER_FAST_PID 2>/dev/null; then
    echo -e "${RED}Error: Scheduler (llm_fast) failed to start. Check $LOG_DIR/scheduler-fast.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Scheduler (llm_fast) started (PID: $SCHEDULER_FAST_PID)${NC}"

# Step 4: Start Scheduler for llm_medium
echo -e "${BLUE}[4/5] Starting Scheduler (llm_medium) on port $SCHEDULER_MEDIUM_PORT...${NC}"
cd "$PROJECT_ROOT"
SCHEDULER_MODEL_ID="llm_medium" \
    PLANNER_REGISTRATION_URL="http://localhost:$PLANNER_PORT" \
    SCHEDULER_SELF_URL="http://localhost:$SCHEDULER_MEDIUM_PORT" \
    PREDICTOR_MODE="library" \
    uv run python -m uvicorn swarmpilot.scheduler.api:app --host 0.0.0.0 --port $SCHEDULER_MEDIUM_PORT \
    > "$LOG_DIR/scheduler-medium.log" 2>&1 &
SCHEDULER_MEDIUM_PID=$!
echo $SCHEDULER_MEDIUM_PID > "$LOG_DIR/scheduler-medium.pid"

# Wait for Scheduler medium to be ready
sleep 2
if ! kill -0 $SCHEDULER_MEDIUM_PID 2>/dev/null; then
    echo -e "${RED}Error: Scheduler (llm_medium) failed to start. Check $LOG_DIR/scheduler-medium.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Scheduler (llm_medium) started (PID: $SCHEDULER_MEDIUM_PID)${NC}"

# Step 5: Start Scheduler for llm_slow
echo -e "${BLUE}[5/5] Starting Scheduler (llm_slow) on port $SCHEDULER_SLOW_PORT...${NC}"
cd "$PROJECT_ROOT"
SCHEDULER_MODEL_ID="llm_slow" \
    PLANNER_REGISTRATION_URL="http://localhost:$PLANNER_PORT" \
    SCHEDULER_SELF_URL="http://localhost:$SCHEDULER_SLOW_PORT" \
    PREDICTOR_MODE="library" \
    uv run python -m uvicorn swarmpilot.scheduler.api:app --host 0.0.0.0 --port $SCHEDULER_SLOW_PORT \
    > "$LOG_DIR/scheduler-slow.log" 2>&1 &
SCHEDULER_SLOW_PID=$!
echo $SCHEDULER_SLOW_PID > "$LOG_DIR/scheduler-slow.pid"

# Wait for Scheduler slow to be ready
sleep 2
if ! kill -0 $SCHEDULER_SLOW_PID 2>/dev/null; then
    echo -e "${RED}Error: Scheduler (llm_slow) failed to start. Check $LOG_DIR/scheduler-slow.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Scheduler (llm_slow) started (PID: $SCHEDULER_SLOW_PID)${NC}"

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

health_check "http://localhost:$SCHEDULER_FAST_PORT/v1/health" "Scheduler (llm_fast)"
health_check "http://localhost:$SCHEDULER_MEDIUM_PORT/v1/health" "Scheduler (llm_medium)"
health_check "http://localhost:$SCHEDULER_SLOW_PORT/v1/health" "Scheduler (llm_slow)"
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
echo -e "${GREEN}║     LLM Cluster Services Ready!                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Services:"
echo "  Scheduler (llm_fast):   http://localhost:$SCHEDULER_FAST_PORT"
echo "  Scheduler (llm_medium): http://localhost:$SCHEDULER_MEDIUM_PORT"
echo "  Scheduler (llm_slow):   http://localhost:$SCHEDULER_SLOW_PORT"
echo "  Planner:                http://localhost:$PLANNER_PORT"
echo "  PyLet:                  http://localhost:$PYLET_HEAD_PORT"
echo ""
echo "Logs: $LOG_DIR/"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Deploy models:    ./examples/llm_cluster/deploy_model.sh"
echo "2. Generate traffic: python examples/llm_cluster/generate_workload.py"
echo "3. Stop cluster:     ./examples/llm_cluster/stop_cluster.sh"
echo ""
