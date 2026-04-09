#!/bin/bash
# Online API Multi-Provider — Start Mock Servers + Per-Model Schedulers
# Usage: ./examples/7.online_api/start_cluster.sh
#
# Architecture (3 models x 3 providers = 9 online instances):
#
#   Scheduler A (:8010) → Qwen/Qwen3-8B
#       ├── Together AI   (mock :9100)
#       ├── Fireworks AI  (mock :9101)
#       └── Lepton AI     (mock :9102)
#
#   Scheduler B (:8020) → Qwen/Qwen3-Next-80B-A3B
#       ├── Together AI   (mock :9110)
#       ├── Fireworks AI  (mock :9111)
#       └── Lepton AI     (mock :9112)
#
#   Scheduler C (:8030) → google/gemma-4-41b-it
#       ├── Together AI   (mock :9120)
#       ├── Fireworks AI  (mock :9121)
#       └── Lepton AI     (mock :9122)
#
# Each Scheduler auto-registers its 3 provider endpoints at startup
# via the ONLINE_ENDPOINTS_CONFIG env var (no manual registration needed).

set -e

# --- Configuration -----------------------------------------------------------
SCHEDULER_QWEN3_8B_PORT=${SCHEDULER_QWEN3_8B_PORT:-8010}
SCHEDULER_QWEN3_NEXT_PORT=${SCHEDULER_QWEN3_NEXT_PORT:-8020}
SCHEDULER_GEMMA4_PORT=${SCHEDULER_GEMMA4_PORT:-8030}
LOG_DIR="/tmp/7.online_api"

# Dummy API keys for mock servers (any non-empty string works)
export TOGETHER_API_KEY="${TOGETHER_API_KEY:-mock-together-key}"
export FIREWORKS_API_KEY="${FIREWORKS_API_KEY:-mock-fireworks-key}"
export LEPTON_API_KEY="${LEPTON_API_KEY:-mock-lepton-key}"

# --- Colors ------------------------------------------------------------------
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# --- Project root ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# --- Helpers -----------------------------------------------------------------
check_port() {
    local port=$1
    local name=$2
    if lsof -i:"$port" &>/dev/null 2>&1; then
        echo -e "${RED}Error: Port $port ($name) already in use.${NC}"
        echo "Run ./examples/7.online_api/stop_cluster.sh first."
        return 1
    fi
    return 0
}

wait_for_health() {
    local url=$1
    local name=$2
    local max_attempts=${3:-20}
    for attempt in $(seq 1 "$max_attempts"); do
        if curl -sf "$url" > /dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    echo -e "${RED}$name failed to start — check logs${NC}"
    return 1
}

# --- Pre-flight --------------------------------------------------------------
echo -e "${BOLD}Online API Multi-Provider — Starting Cluster${NC}"
echo ""

# Check scheduler ports
check_port "$SCHEDULER_QWEN3_8B_PORT"   "Scheduler A / Qwen3-8B"       || exit 1
check_port "$SCHEDULER_QWEN3_NEXT_PORT" "Scheduler B / Qwen3-Next"     || exit 1
check_port "$SCHEDULER_GEMMA4_PORT"     "Scheduler C / Gemma4"         || exit 1
echo -e "${GREEN}✓ Scheduler ports available${NC}"

mkdir -p "$LOG_DIR"

if ! command -v uv &>/dev/null; then
    echo -e "${RED}Error: uv not found.${NC}"
    exit 1
fi

cd "$PROJECT_ROOT"
uv sync --quiet

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

# =============================================================================
# [1/4] Start Mock Online API Servers (9 endpoints)
# =============================================================================
echo ""
echo -e "${BOLD}[1/4] Starting 9 mock online API servers...${NC}"

"$VENV_PYTHON" "$SCRIPT_DIR/mock_online_api.py" \
    > "$LOG_DIR/mock-servers.log" 2>&1 &
echo $! > "$LOG_DIR/mock-servers.pid"

# Wait for at least one endpoint per model group to be healthy
wait_for_health "http://localhost:9100/health" "Mock Qwen3-8B/Together"      || exit 1
wait_for_health "http://localhost:9110/health" "Mock Qwen3-Next/Together"    || exit 1
wait_for_health "http://localhost:9120/health" "Mock Gemma4/Together"        || exit 1
echo -e "${GREEN}✓ Mock servers started (PID $(cat "$LOG_DIR/mock-servers.pid"))${NC}"

# =============================================================================
# [2/4] Scheduler A — Qwen3-8B
# =============================================================================
echo ""
echo -e "${BOLD}[2/4] Starting Scheduler A (Qwen3-8B) on :${SCHEDULER_QWEN3_8B_PORT}...${NC}"

SCHEDULER_MODEL_ID="Qwen/Qwen3-8B" \
    PREDICTOR_MODE="library" \
    ONLINE_ENDPOINTS_CONFIG="$SCRIPT_DIR/online_endpoints_qwen3_8b.yaml" \
    uv run sscheduler start --port "$SCHEDULER_QWEN3_8B_PORT" \
    > "$LOG_DIR/scheduler-qwen3-8b.log" 2>&1 &
echo $! > "$LOG_DIR/scheduler-qwen3-8b.pid"

wait_for_health "http://localhost:$SCHEDULER_QWEN3_8B_PORT/v1/health" "Scheduler A" || exit 1
echo -e "${GREEN}✓ Scheduler A started — 3 providers auto-registered${NC}"

# =============================================================================
# [3/4] Scheduler B — Qwen3-Next-80B-A3B
# =============================================================================
echo ""
echo -e "${BOLD}[3/4] Starting Scheduler B (Qwen3-Next-80B-A3B) on :${SCHEDULER_QWEN3_NEXT_PORT}...${NC}"

SCHEDULER_MODEL_ID="Qwen/Qwen3-Next-80B-A3B" \
    PREDICTOR_MODE="library" \
    ONLINE_ENDPOINTS_CONFIG="$SCRIPT_DIR/online_endpoints_qwen3_next.yaml" \
    uv run sscheduler start --port "$SCHEDULER_QWEN3_NEXT_PORT" \
    > "$LOG_DIR/scheduler-qwen3-next.log" 2>&1 &
echo $! > "$LOG_DIR/scheduler-qwen3-next.pid"

wait_for_health "http://localhost:$SCHEDULER_QWEN3_NEXT_PORT/v1/health" "Scheduler B" || exit 1
echo -e "${GREEN}✓ Scheduler B started — 3 providers auto-registered${NC}"

# =============================================================================
# [4/4] Scheduler C — Gemma4-41B
# =============================================================================
echo ""
echo -e "${BOLD}[4/4] Starting Scheduler C (Gemma4-41B) on :${SCHEDULER_GEMMA4_PORT}...${NC}"

SCHEDULER_MODEL_ID="google/gemma-4-41b-it" \
    PREDICTOR_MODE="library" \
    ONLINE_ENDPOINTS_CONFIG="$SCRIPT_DIR/online_endpoints_gemma4.yaml" \
    uv run sscheduler start --port "$SCHEDULER_GEMMA4_PORT" \
    > "$LOG_DIR/scheduler-gemma4.log" 2>&1 &
echo $! > "$LOG_DIR/scheduler-gemma4.pid"

wait_for_health "http://localhost:$SCHEDULER_GEMMA4_PORT/v1/health" "Scheduler C" || exit 1
echo -e "${GREEN}✓ Scheduler C started — 3 providers auto-registered${NC}"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}${BOLD}Cluster started — 3 models x 3 providers = 9 online endpoints${NC}"
echo ""
echo "  Scheduler A (Qwen3-8B):        http://localhost:${SCHEDULER_QWEN3_8B_PORT}   → Together / Fireworks / Lepton"
echo "  Scheduler B (Qwen3-Next-80B):  http://localhost:${SCHEDULER_QWEN3_NEXT_PORT}   → Together / Fireworks / Lepton"
echo "  Scheduler C (Gemma4-41B):      http://localhost:${SCHEDULER_GEMMA4_PORT}   → Together / Fireworks / Lepton"
echo ""
echo "  Mock API servers: ports 9100-9122"
echo "  Logs: $LOG_DIR/"
echo ""
echo -e "${YELLOW}Verify instances:${NC}"
echo "  curl -s http://localhost:${SCHEDULER_QWEN3_8B_PORT}/v1/instance/list | python3 -m json.tool"
echo ""
echo "Next: python examples/7.online_api/api_example.py"
