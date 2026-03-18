#!/bin/bash
# Multi-Model Direct — Deploy Instances & Register with Schedulers
# Usage: ./examples/multi_model_direct/deploy_model.sh
#
# Launches 4 mock vLLM instances (2 Qwen, 2 Llama) and registers each
# with the correct per-model scheduler.  No Planner involved.

set -e

SCHEDULER_QWEN_PORT=${SCHEDULER_QWEN_PORT:-8010}
SCHEDULER_LLAMA_PORT=${SCHEDULER_LLAMA_PORT:-8020}
LOG_DIR="/tmp/multi_model_direct"

GREEN='\033[0;32m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
MOCK_SERVER="$SCRIPT_DIR/mock_vllm_server.py"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

echo -e "${BOLD}Multi-Model Direct — Deploy Instances${NC}"
echo ""

mkdir -p "$LOG_DIR"

# --- [1/4] Start Qwen instances (ports 8100, 8101) --------------------------
echo -e "${BOLD}[1/4] Starting Qwen instances...${NC}"
for i in 0 1; do
    PORT=$((8100 + i))
    # Real vLLM (commented out):
    # CUDA_VISIBLE_DEVICES=$i vllm serve Qwen/Qwen3-8B-VL --port $PORT &
    MODEL_ID="Qwen/Qwen3-8B-VL" PORT=$PORT \
        "$VENV_PYTHON" "$MOCK_SERVER" > "$LOG_DIR/qwen-$i.log" 2>&1 &
    echo $! > "$LOG_DIR/qwen-$i.pid"
    echo -e "${GREEN}  qwen-vl-$i on :$PORT (PID $!)${NC}"
done

# --- [2/4] Start Llama instances (ports 8200, 8201) -------------------------
echo -e "${BOLD}[2/4] Starting Llama instances...${NC}"
for i in 0 1; do
    PORT=$((8200 + i))
    # Real vLLM (commented out):
    # CUDA_VISIBLE_DEVICES=$((i+2)) vllm serve meta-llama/Llama-3.1-8B --port $PORT &
    MODEL_ID="meta-llama/Llama-3.1-8B" PORT=$PORT \
        "$VENV_PYTHON" "$MOCK_SERVER" > "$LOG_DIR/llama-$i.log" 2>&1 &
    echo $! > "$LOG_DIR/llama-$i.pid"
    echo -e "${GREEN}  llama-$i on :$PORT (PID $!)${NC}"
done

# --- [3/4] Wait for health ---------------------------------------------------
echo -e "${BOLD}[3/4] Waiting for instances to become healthy...${NC}"
ALL_PORTS=(8100 8101 8200 8201)
for port in "${ALL_PORTS[@]}"; do
    for attempt in $(seq 1 30); do
        if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
            echo -e "${GREEN}  :$port healthy${NC}"
            break
        fi
        if [ "$attempt" -eq 30 ]; then
            echo -e "${RED}  :$port failed to start — check logs${NC}"
            exit 1
        fi
        sleep 1
    done
done

# --- [4/4] Register with schedulers -----------------------------------------
echo -e "${BOLD}[4/4] Registering instances with per-model schedulers...${NC}"

register() {
    local scheduler_url=$1
    local instance_id=$2
    local model_id=$3
    local endpoint=$4
    local hw_name=$5

    PAYLOAD=$(cat <<EOF
{
  "instance_id": "$instance_id",
  "model_id": "$model_id",
  "endpoint": "$endpoint",
  "platform_info": {
    "software_name": "vllm",
    "software_version": "0.8.0",
    "hardware_name": "$hw_name"
  }
}
EOF
)
    RESP=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "$scheduler_url/v1/instance/register" \
        -H "Content-Type: application/json" \
        -d "$PAYLOAD")

    if [ "$RESP" = "200" ]; then
        echo -e "${GREEN}  ✓ $instance_id → $scheduler_url${NC}"
    else
        echo -e "${RED}  ✗ $instance_id registration failed (HTTP $RESP)${NC}"
        return 1
    fi
}

SCHED_QWEN="http://localhost:$SCHEDULER_QWEN_PORT"
SCHED_LLAMA="http://localhost:$SCHEDULER_LLAMA_PORT"

register "$SCHED_QWEN"  "qwen-vl-0" "Qwen/Qwen3-8B-VL"       "http://localhost:8100" "GPU-0"
register "$SCHED_QWEN"  "qwen-vl-1" "Qwen/Qwen3-8B-VL"       "http://localhost:8101" "GPU-1"
register "$SCHED_LLAMA" "llama-0"   "meta-llama/Llama-3.1-8B" "http://localhost:8200" "GPU-2"
register "$SCHED_LLAMA" "llama-1"   "meta-llama/Llama-3.1-8B" "http://localhost:8201" "GPU-3"

echo ""
echo -e "${GREEN}All 4 instances deployed and registered.${NC}"
echo ""
echo "Verify:"
echo "  curl http://localhost:$SCHEDULER_QWEN_PORT/v1/instance/list | python3 -m json.tool"
echo "  curl http://localhost:$SCHEDULER_LLAMA_PORT/v1/instance/list | python3 -m json.tool"
echo ""
echo "Next: python examples/multi_model_direct/api_example.py"
