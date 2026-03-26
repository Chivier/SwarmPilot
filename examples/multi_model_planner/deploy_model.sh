#!/bin/bash
# Multi-Model Planner Example - Deploy Models
# Usage: ./examples/multi_model_planner/deploy_model.sh
#
# Two modes:
#   Mode A (PyLet): Uses `splanner serve` for automated deployment
#   Mode B (Mock):  Starts mock instances + registers via curl

set -e

PLANNER_PORT=${PLANNER_PORT:-8002}
SCHEDULER_QWEN_PORT=${SCHEDULER_QWEN_PORT:-8010}
SCHEDULER_LLAMA_PORT=${SCHEDULER_LLAMA_PORT:-8020}

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="/tmp/multi_model_planner"
MOCK_SERVER="$PROJECT_ROOT/examples/multi_model_planner/mock_vllm_server.py"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Multi-Model Planner - Deploy Models           ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# Check PyLet availability via planner /v1/pylet/status
PYLET_AVAILABLE=false
if curl -s "http://localhost:$PLANNER_PORT/v1/pylet/status" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    sys.exit(0 if d.get('pylet_enabled') else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    PYLET_AVAILABLE=true
fi

# ═══════════════════════════════════════════════════════════════
# Mode A: PyLet-managed deployment (production)
# ═══════════════════════════════════════════════════════════════
if [ "$PYLET_AVAILABLE" = true ]; then
    echo -e "${GREEN}PyLet detected — using Mode A (automated deployment)${NC}"
    echo ""

    echo -e "${BLUE}[1/2] Deploying Qwen/Qwen3-8B-VL (2 replicas)...${NC}"
    uv run splanner serve "Qwen/Qwen3-8B-VL" \
        --gpu 1 --replicas 2 \
        --planner-url "http://localhost:$PLANNER_PORT"
    echo -e "${GREEN}Qwen deployed${NC}"

    echo -e "${BLUE}[2/2] Deploying meta-llama/Llama-3.1-8B (2 replicas)...${NC}"
    uv run splanner serve "meta-llama/Llama-3.1-8B" \
        --gpu 1 --replicas 2 \
        --planner-url "http://localhost:$PLANNER_PORT"
    echo -e "${GREEN}Llama deployed${NC}"

    echo ""
    echo -e "${GREEN}All models deployed via PyLet${NC}"
    exit 0
fi

# ═══════════════════════════════════════════════════════════════
# Mode B: Mock deployment (no PyLet — demo/testing)
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}No PyLet — using Mode B (mock instances)${NC}"
echo -e "${YELLOW}Planner value: scheduler discovery via GET /v1/schedulers${NC}"
echo ""

start_mock_and_register() {
    local model_id=$1
    local port=$2
    local instance_id=$3
    local scheduler_port=$4

    MODEL_ID="$model_id" PORT=$port "$VENV_PYTHON" "$MOCK_SERVER" \
        > "$LOG_DIR/mock-${instance_id}.log" 2>&1 &
    echo $! > "$LOG_DIR/mock-${instance_id}.pid"
    sleep 1

    if ! kill -0 "$(cat "$LOG_DIR/mock-${instance_id}.pid")" 2>/dev/null; then
        echo -e "${RED}Error: Mock instance $instance_id failed to start${NC}"
        return 1
    fi

    curl -s -X POST "http://localhost:$scheduler_port/v1/instance/register" \
        -H "Content-Type: application/json" \
        -d "{
            \"instance_id\": \"$instance_id\",
            \"model_id\": \"$model_id\",
            \"endpoint\": \"http://localhost:$port\",
            \"platform_info\": {
                \"software_name\": \"mock-vllm\",
                \"software_version\": \"1.0.0\",
                \"hardware_name\": \"cpu-local\"
            }
        }" > /dev/null 2>&1

    echo -e "${GREEN}  $instance_id → :$port (registered with scheduler :$scheduler_port)${NC}"
}

echo -e "${BLUE}[1/2] Deploying Qwen/Qwen3-8B-VL mock instances...${NC}"
start_mock_and_register "Qwen/Qwen3-8B-VL" 8100 "qwen-inst-001" $SCHEDULER_QWEN_PORT
start_mock_and_register "Qwen/Qwen3-8B-VL" 8101 "qwen-inst-002" $SCHEDULER_QWEN_PORT

echo -e "${BLUE}[2/2] Deploying meta-llama/Llama-3.1-8B mock instances...${NC}"
start_mock_and_register "meta-llama/Llama-3.1-8B" 8200 "llama-inst-001" $SCHEDULER_LLAMA_PORT
start_mock_and_register "meta-llama/Llama-3.1-8B" 8201 "llama-inst-002" $SCHEDULER_LLAMA_PORT

echo ""
echo "Verifying instances..."
echo "  Qwen instances:"
curl -s "http://localhost:$SCHEDULER_QWEN_PORT/v1/instance/list" 2>/dev/null \
    | python3 -c "import sys,json; [print(f'    - {i[\"instance_id\"]} @ {i[\"endpoint\"]}') for i in json.load(sys.stdin).get('instances',[])]" 2>/dev/null || true
echo "  Llama instances:"
curl -s "http://localhost:$SCHEDULER_LLAMA_PORT/v1/instance/list" 2>/dev/null \
    | python3 -c "import sys,json; [print(f'    - {i[\"instance_id\"]} @ {i[\"endpoint\"]}') for i in json.load(sys.stdin).get('instances',[])]" 2>/dev/null || true

echo ""
echo "Scheduler discovery via Planner:"
curl -s "http://localhost:$PLANNER_PORT/v1/schedulers" 2>/dev/null | python3 -m json.tool 2>/dev/null || true
echo ""
echo -e "${GREEN}All mock instances deployed and registered${NC}"
