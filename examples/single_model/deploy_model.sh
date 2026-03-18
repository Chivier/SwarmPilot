#!/bin/bash
# Single Model Example - Deploy 3 Mock Instances
# Usage: ./examples/single_model/deploy_model.sh
#
# Starts 3 mock vLLM servers on ports 8100-8102, waits for health,
# then registers each with the Scheduler on port 8000.

set -e

SCHEDULER_PORT=${SCHEDULER_PORT:-8000}
SCHEDULER_URL="http://localhost:$SCHEDULER_PORT"
MODEL_ID="Qwen/Qwen3-8B-VL"
INSTANCE_PORTS=(8100 8101 8102)
LOG_DIR="/tmp/single_model"

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

echo -e "${BLUE}${BOLD}Single Model Example - Deploy 3 Instances${NC}"
echo ""

if ! curl -sf "$SCHEDULER_URL/v1/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Scheduler not running on $SCHEDULER_URL${NC}"
    echo "Run ./examples/single_model/start_cluster.sh first."
    exit 1
fi

mkdir -p "$LOG_DIR"

# ── Real vLLM (uncomment to use instead of mock) ────────────────────────
# for i in 0 1 2; do
#     PORT=${INSTANCE_PORTS[$i]} vllm serve "$MODEL_ID" \
#         --port "${INSTANCE_PORTS[$i]}" --host 0.0.0.0 --gpu-memory-utilization 0.9 \
#         > "$LOG_DIR/instance-$i.log" 2>&1 &
# done

# [1/3] Start mock instances
echo -e "${BLUE}[1/3] Starting 3 mock instances on ports ${INSTANCE_PORTS[*]}...${NC}"
for i in 0 1 2; do
    PORT=${INSTANCE_PORTS[$i]}
    MODEL_ID="$MODEL_ID" PORT="$PORT" \
        "$VENV_PYTHON" "$SCRIPT_DIR/mock_vllm_server.py" \
        > "$LOG_DIR/instance-$i.log" 2>&1 &
    INSTANCE_PID=$!
    echo "$INSTANCE_PID" > "$LOG_DIR/instance-$i.pid"
    echo -e "  Started instance $i on port $PORT (PID: $INSTANCE_PID)"
done

# [2/3] Wait for health
echo ""
echo -e "${BLUE}[2/3] Waiting for instances to become healthy...${NC}"
for i in 0 1 2; do
    PORT=${INSTANCE_PORTS[$i]}
    for attempt in $(seq 1 20); do
        if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo -e "  ${GREEN}✓ Instance $i (port $PORT) healthy${NC}"
            break
        fi
        if [ "$attempt" -eq 20 ]; then
            echo -e "  ${RED}✗ Instance $i (port $PORT) failed health check${NC}"
            exit 1
        fi
        sleep 0.5
    done
done

# [3/3] Register with Scheduler
echo ""
echo -e "${BLUE}[3/3] Registering instances with Scheduler...${NC}"
for i in 0 1 2; do
    PORT=${INSTANCE_PORTS[$i]}
    RESPONSE=$(curl -sf -X POST "$SCHEDULER_URL/v1/instance/register" \
        -H "Content-Type: application/json" \
        -d "{
            \"instance_id\": \"qwen-vl-$i\",
            \"model_id\": \"$MODEL_ID\",
            \"endpoint\": \"http://localhost:$PORT\",
            \"platform_info\": {
                \"software_name\": \"vllm\",
                \"software_version\": \"0.8.0\",
                \"hardware_name\": \"GPU-$i\"
            }
        }" 2>&1) || {
        echo -e "  ${RED}✗ Failed to register instance $i${NC}"
        exit 1
    }
    echo -e "  ${GREEN}✓ Registered qwen-vl-$i → http://localhost:$PORT${NC}"
done

echo ""
echo -e "${GREEN}${BOLD}All 3 instances deployed and registered!${NC}"
echo ""
echo "Instance list:"
curl -s "$SCHEDULER_URL/v1/instance/list" | python3 -m json.tool 2>/dev/null || true
echo ""
echo -e "${YELLOW}Next:${NC} python examples/single_model/api_example.py"
