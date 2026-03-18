#!/bin/bash
# Multi-Model Direct — Start Per-Model Schedulers (No Planner)
# Usage: ./examples/multi_model_direct/start_cluster.sh
#
# Starts two independent schedulers — one per model:
#   Scheduler A (:8010) → Qwen/Qwen3-8B-VL
#   Scheduler B (:8020) → meta-llama/Llama-3.1-8B
#
# Core architecture rule: each scheduler serves exactly ONE model.

set -e

# --- Configuration -----------------------------------------------------------
SCHEDULER_QWEN_PORT=${SCHEDULER_QWEN_PORT:-8010}
SCHEDULER_LLAMA_PORT=${SCHEDULER_LLAMA_PORT:-8020}
LOG_DIR="/tmp/multi_model_direct"

# --- Colors ------------------------------------------------------------------
GREEN='\033[0;32m'
RED='\033[0;31m'
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
        echo "Run ./examples/multi_model_direct/stop_cluster.sh first."
        return 1
    fi
    return 0
}

# --- Pre-flight --------------------------------------------------------------
echo -e "${BOLD}Multi-Model Direct — Starting Schedulers${NC}"
echo ""

check_port "$SCHEDULER_QWEN_PORT" "Scheduler A / Qwen" || exit 1
check_port "$SCHEDULER_LLAMA_PORT" "Scheduler B / Llama" || exit 1
echo -e "${GREEN}✓ Ports available${NC}"

mkdir -p "$LOG_DIR"

if ! command -v uv &>/dev/null; then
    echo -e "${RED}Error: uv not found.${NC}"
    exit 1
fi

cd "$PROJECT_ROOT"
uv sync --quiet

# --- [1/2] Scheduler A — Qwen ------------------------------------------------
echo -e "${BOLD}[1/2] Starting Scheduler A (Qwen/Qwen3-8B-VL) on :${SCHEDULER_QWEN_PORT}...${NC}"
SCHEDULER_MODEL_ID="Qwen/Qwen3-8B-VL" \
    PREDICTOR_MODE="library" \
    uv run sscheduler start --port "$SCHEDULER_QWEN_PORT" \
    > "$LOG_DIR/scheduler-qwen.log" 2>&1 &
echo $! > "$LOG_DIR/scheduler-qwen.pid"
sleep 2

if ! kill -0 "$(cat "$LOG_DIR/scheduler-qwen.pid")" 2>/dev/null; then
    echo -e "${RED}Failed — check $LOG_DIR/scheduler-qwen.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Scheduler A started (PID $(cat "$LOG_DIR/scheduler-qwen.pid"))${NC}"

# --- [2/2] Scheduler B — Llama -----------------------------------------------
echo -e "${BOLD}[2/2] Starting Scheduler B (meta-llama/Llama-3.1-8B) on :${SCHEDULER_LLAMA_PORT}...${NC}"
SCHEDULER_MODEL_ID="meta-llama/Llama-3.1-8B" \
    PREDICTOR_MODE="library" \
    uv run sscheduler start --port "$SCHEDULER_LLAMA_PORT" \
    > "$LOG_DIR/scheduler-llama.log" 2>&1 &
echo $! > "$LOG_DIR/scheduler-llama.pid"
sleep 2

if ! kill -0 "$(cat "$LOG_DIR/scheduler-llama.pid")" 2>/dev/null; then
    echo -e "${RED}Failed — check $LOG_DIR/scheduler-llama.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Scheduler B started (PID $(cat "$LOG_DIR/scheduler-llama.pid"))${NC}"

# --- Summary -----------------------------------------------------------------
echo ""
echo -e "${GREEN}Two schedulers started — one per model (core architecture rule)${NC}"
echo ""
echo "  Scheduler A (Qwen):  http://localhost:${SCHEDULER_QWEN_PORT}"
echo "  Scheduler B (Llama): http://localhost:${SCHEDULER_LLAMA_PORT}"
echo "  Logs: $LOG_DIR/"
echo ""
echo "Next: ./examples/multi_model_direct/deploy_model.sh"
