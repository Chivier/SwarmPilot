#!/bin/bash
# Virtual Concurrency Test — Start Mock Servers + Per-Model Schedulers
# Usage: ./examples/8.virtual_concurrency/start_cluster.sh
#
# Architecture (3 models x 3 providers, concurrency 1/2/3):
#   Per model: Together(1) + Fireworks(2) + Lepton(3) = 6 virtual instances
#   Total: 18 virtual instances across 3 schedulers
#
# Mock servers on ports 9200-9222 (offset from example 7)

set -e

SCHEDULER_QWEN3_8B_PORT=${SCHEDULER_QWEN3_8B_PORT:-8010}
SCHEDULER_QWEN3_NEXT_PORT=${SCHEDULER_QWEN3_NEXT_PORT:-8020}
SCHEDULER_GEMMA4_PORT=${SCHEDULER_GEMMA4_PORT:-8030}
LOG_DIR="/tmp/8.virtual_concurrency"

export TOGETHER_API_KEY="${TOGETHER_API_KEY:-mock-together-key}"
export FIREWORKS_API_KEY="${FIREWORKS_API_KEY:-mock-fireworks-key}"
export LEPTON_API_KEY="${LEPTON_API_KEY:-mock-lepton-key}"

GREEN='\033[0;32m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

check_port() {
    local port=$1
    local name=$2
    if lsof -i:"$port" &>/dev/null 2>&1; then
        echo -e "${RED}Error: Port $port ($name) already in use.${NC}"
        echo "Run ./examples/8.virtual_concurrency/stop_cluster.sh first."
        return 1
    fi
    return 0
}

wait_for_health() {
    local url=$1
    local name=$2
    local max_attempts=${3:-30}
    for attempt in $(seq 1 "$max_attempts"); do
        if curl -sf "$url" > /dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    echo -e "${RED}$name failed to start — check logs in $LOG_DIR/${NC}"
    return 1
}

echo -e "${BOLD}Virtual Concurrency Test — Starting Cluster${NC}"
echo ""

check_port "$SCHEDULER_QWEN3_8B_PORT"   "Scheduler A" || exit 1
check_port "$SCHEDULER_QWEN3_NEXT_PORT" "Scheduler B" || exit 1
check_port "$SCHEDULER_GEMMA4_PORT"     "Scheduler C" || exit 1
echo -e "${GREEN}Scheduler ports available${NC}"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"
uv sync --quiet
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

# [1/4] Mock Servers (ports 9200-9222)
echo ""
echo -e "${BOLD}[1/4] Starting 9 mock online API servers (ports 9200-9222)...${NC}"

"$VENV_PYTHON" "$SCRIPT_DIR/mock_online_api.py" \
    > "$LOG_DIR/mock-servers.log" 2>&1 &
echo $! > "$LOG_DIR/mock-servers.pid"

wait_for_health "http://localhost:9200/health" "Mock Qwen3-8B/Together"   || exit 1
wait_for_health "http://localhost:9210/health" "Mock Qwen3-Next/Together" || exit 1
wait_for_health "http://localhost:9220/health" "Mock Gemma4/Together"     || exit 1
echo -e "${GREEN}Mock servers started (PID $(cat "$LOG_DIR/mock-servers.pid"))${NC}"

# [2/4] Scheduler A — Qwen3-8B (6 virtual instances)
echo ""
echo -e "${BOLD}[2/4] Starting Scheduler A (Qwen3-8B) on :${SCHEDULER_QWEN3_8B_PORT}...${NC}"

SCHEDULER_MODEL_ID="Qwen/Qwen3-8B" \
    PREDICTOR_MODE="library" \
    ONLINE_ENDPOINTS_CONFIG="$SCRIPT_DIR/online_endpoints_qwen3_8b.yaml" \
    uv run sscheduler start --port "$SCHEDULER_QWEN3_8B_PORT" \
    > "$LOG_DIR/scheduler-qwen3-8b.log" 2>&1 &
echo $! > "$LOG_DIR/scheduler-qwen3-8b.pid"

wait_for_health "http://localhost:$SCHEDULER_QWEN3_8B_PORT/v1/health" "Scheduler A" || exit 1
echo -e "${GREEN}Scheduler A started — 6 virtual instances (1+2+3)${NC}"

# [3/4] Scheduler B — Qwen3-Next-80B-A3B (6 virtual instances)
echo ""
echo -e "${BOLD}[3/4] Starting Scheduler B (Qwen3-Next-80B-A3B) on :${SCHEDULER_QWEN3_NEXT_PORT}...${NC}"

SCHEDULER_MODEL_ID="Qwen/Qwen3-Next-80B-A3B" \
    PREDICTOR_MODE="library" \
    ONLINE_ENDPOINTS_CONFIG="$SCRIPT_DIR/online_endpoints_qwen3_next.yaml" \
    uv run sscheduler start --port "$SCHEDULER_QWEN3_NEXT_PORT" \
    > "$LOG_DIR/scheduler-qwen3-next.log" 2>&1 &
echo $! > "$LOG_DIR/scheduler-qwen3-next.pid"

wait_for_health "http://localhost:$SCHEDULER_QWEN3_NEXT_PORT/v1/health" "Scheduler B" || exit 1
echo -e "${GREEN}Scheduler B started — 6 virtual instances (1+2+3)${NC}"

# [4/4] Scheduler C — Gemma4-41B (6 virtual instances)
echo ""
echo -e "${BOLD}[4/4] Starting Scheduler C (Gemma4-41B) on :${SCHEDULER_GEMMA4_PORT}...${NC}"

SCHEDULER_MODEL_ID="google/gemma-4-41b-it" \
    PREDICTOR_MODE="library" \
    ONLINE_ENDPOINTS_CONFIG="$SCRIPT_DIR/online_endpoints_gemma4.yaml" \
    uv run sscheduler start --port "$SCHEDULER_GEMMA4_PORT" \
    > "$LOG_DIR/scheduler-gemma4.log" 2>&1 &
echo $! > "$LOG_DIR/scheduler-gemma4.pid"

wait_for_health "http://localhost:$SCHEDULER_GEMMA4_PORT/v1/health" "Scheduler C" || exit 1
echo -e "${GREEN}Scheduler C started — 6 virtual instances (1+2+3)${NC}"

# Summary
echo ""
echo -e "${GREEN}${BOLD}Cluster started — 3 models x 6 virtual instances = 18 total${NC}"
echo ""
echo "  Scheduler A (Qwen3-8B):       http://localhost:${SCHEDULER_QWEN3_8B_PORT}   → Together(1) / Fireworks(2) / Lepton(3)"
echo "  Scheduler B (Qwen3-Next-80B): http://localhost:${SCHEDULER_QWEN3_NEXT_PORT}   → Together(1) / Fireworks(2) / Lepton(3)"
echo "  Scheduler C (Gemma4-41B):     http://localhost:${SCHEDULER_GEMMA4_PORT}   → Together(1) / Fireworks(2) / Lepton(3)"
echo ""
echo "  Mock API servers: ports 9200-9222"
echo "  Logs: $LOG_DIR/"
echo ""
echo "Next: uv run python examples/8.virtual_concurrency/test_concurrency.py"
