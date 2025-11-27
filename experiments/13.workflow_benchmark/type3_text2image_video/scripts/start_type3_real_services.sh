#!/bin/bash
set -e

# ============================================================================
# Type3 Text2Image+Video - Real Cluster Service Launcher
# ============================================================================
# Multi-node service launcher for Type3 workflow (A→C→B).
#
# Architecture:
#   - Scheduler A (LLM):   29.209.114.51
#   - Scheduler C (FLUX):  29.209.114.52  (NEW for Type3)
#   - Scheduler B (T2VID): 29.209.113.228
#   - Predictor:           29.209.113.113
#   - Planner:             29.209.114.166
#   - Client:              29.209.114.166
#
# Logic:
#   - Detects local IP from bond1
#   - Starts appropriate services based on which host this is
#   - Does NOT deploy models (use manual_deploy_type3.sh)
#
# Usage:
#   bash start_type3_real_services.sh [--auto-optimize]
# ============================================================================

# Parse Arguments
AUTO_OPTIMIZE_ENABLED="False"

while [[ $# -gt 0 ]]; do
    case $1 in
        --auto-optimize)
            AUTO_OPTIMIZE_ENABLED="True"
            shift
            ;;
        --no-auto-optimize)
            AUTO_OPTIMIZE_ENABLED="False"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --auto-optimize      Enable AUTO_OPTIMIZE_ENABLED for planner"
            echo "  --no-auto-optimize   Disable AUTO_OPTIMIZE_ENABLED (default)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Note: This script only starts infrastructure services."
            echo "      Use manual_deploy_type3.sh to deploy models."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "AUTO_OPTIMIZE_ENABLED: $AUTO_OPTIMIZE_ENABLED"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

cd "$PROJECT_ROOT"
uv sync
source .venv/bin/activate

LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$LOG_DIR"

# Ports (all services use same port on different hosts)
SCHEDULER_PORT=8100
PREDICTOR_PORT=8100
PLANNER_PORT=8100

# CPU Binding
SCHEDULER_PREDICTOR_CPU_RANGE="0-7"

# Host assignments
SCHEDULER_A_HOST="29.209.114.51"    # LLM
SCHEDULER_C_HOST="29.209.114.52"    # FLUX (NEW)
SCHEDULER_B_HOST="29.209.113.228"   # T2VID
PREDICTOR_HOST="29.209.113.113"
PLANNER_HOST="29.209.114.166"
CLIENT_HOST="29.209.114.166"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Helper: Get local IP
get_local_ip() {
    if ! command -v ifconfig >/dev/null 2>&1; then
        echo "ifconfig not found" >&2
        return 1
    fi
    if ! ifconfig bond1 >/dev/null 2>&1; then
        echo "Interface bond1 not found" >&2
        return 1
    fi
    ifconfig bond1 | awk '/inet / {print $2}' | head -n1
}

# Helper: Start background process
start_bg() {
    local name="$1"
    shift
    local cmd="$1"
    shift
    local cpu_cores="$1"
    local log_file="$LOG_DIR/${name}.log"

    echo -e "${YELLOW}Starting $name...${NC}"
    if [[ -n "$cpu_cores" ]]; then
        echo -e "${BLUE}  CPU binding: $cpu_cores${NC}"
        nohup taskset -c "$cpu_cores" bash -lc "$cmd" >"$log_file" 2>&1 &
    else
        nohup bash -lc "$cmd" >"$log_file" 2>&1 &
    fi
    local pid=$!
    echo -e "${GREEN}Started $name (PID: $pid, log: $log_file)${NC}"
}

# Detect local IP
LOCAL_IP="$(get_local_ip)"
if [[ -z "$LOCAL_IP" ]]; then
    echo -e "${RED}Failed to detect local IP on bond1, aborting.${NC}"
    exit 1
fi
echo -e "${BLUE}Local IP detected: $LOCAL_IP${NC}"

# Start Planner (if this host)
if [[ "$LOCAL_IP" == "$PLANNER_HOST" ]]; then
    start_bg "planner" \
        "cd $PROJECT_ROOT/planner && \
         export PYTHONPATH=$PROJECT_ROOT && \
         export SCHEDULER_HOST=$SCHEDULER_A_HOST && \
         export SCHEDULER_PORT=$SCHEDULER_PORT && \
         export PLANNER_LOG_DIR=$LOG_DIR/planner && \
         export PLANNER_LOGURU_LEVEL=INFO && \
         export AUTO_OPTIMIZE_ENABLED=$AUTO_OPTIMIZE_ENABLED && \
         export AUTO_OPTIMIZE_INTERVAL=300 && \
         python -m src.cli start --port $PLANNER_PORT" \
        "$SCHEDULER_PREDICTOR_CPU_RANGE"
fi

# Start Predictor (if this host)
if [[ "$LOCAL_IP" == "$PREDICTOR_HOST" ]]; then
    start_bg "predictor" \
        "cd $PROJECT_ROOT/predictor && \
         export PYTHONPATH=$PROJECT_ROOT && \
         export PREDICTOR_PORT=$PREDICTOR_PORT && \
         export PREDICTOR_LOG_DIR=$LOG_DIR/predictor && \
         python -m src.cli start --port $PREDICTOR_PORT --log-level INFO" \
        "$SCHEDULER_PREDICTOR_CPU_RANGE"
fi

# Start Scheduler A (LLM)
if [[ "$LOCAL_IP" == "$SCHEDULER_A_HOST" ]]; then
    start_bg "scheduler_a" \
        "cd $PROJECT_ROOT/scheduler && \
         export PYTHONPATH=$PROJECT_ROOT && \
         export PREDICTOR_URL=http://$PREDICTOR_HOST:$PREDICTOR_PORT && \
         export SCHEDULER_HOST=$SCHEDULER_A_HOST && \
         export SCHEDULER_PORT=$SCHEDULER_PORT && \
         export SCHEDULER_LOG_DIR=$LOG_DIR/scheduler-a && \
         export SCHEDULER_LOGURU_LEVEL=INFO && \
         export PLANNER_URL=http://$PLANNER_HOST:$PLANNER_PORT && \
         export SCHEDULER_AUTO_REPORT=20.0 && \
         export PLANNER_REPORT_TIMEOUT=5.0 && \
         python -m src.cli start --port $SCHEDULER_PORT" \
        "$SCHEDULER_PREDICTOR_CPU_RANGE"
fi

# Start Scheduler C (FLUX) - NEW for Type3
if [[ "$LOCAL_IP" == "$SCHEDULER_C_HOST" ]]; then
    start_bg "scheduler_c" \
        "cd $PROJECT_ROOT/scheduler && \
         export PYTHONPATH=$PROJECT_ROOT && \
         export PREDICTOR_URL=http://$PREDICTOR_HOST:$PREDICTOR_PORT && \
         export SCHEDULER_HOST=$SCHEDULER_C_HOST && \
         export SCHEDULER_PORT=$SCHEDULER_PORT && \
         export SCHEDULER_LOG_DIR=$LOG_DIR/scheduler-c && \
         export SCHEDULER_LOGURU_LEVEL=INFO && \
         export PLANNER_URL=http://$PLANNER_HOST:$PLANNER_PORT && \
         export SCHEDULER_AUTO_REPORT=20.0 && \
         export PLANNER_REPORT_TIMEOUT=5.0 && \
         python -m src.cli start --port $SCHEDULER_PORT" \
        "$SCHEDULER_PREDICTOR_CPU_RANGE"
fi

# Start Scheduler B (T2VID)
if [[ "$LOCAL_IP" == "$SCHEDULER_B_HOST" ]]; then
    start_bg "scheduler_b" \
        "cd $PROJECT_ROOT/scheduler && \
         export PYTHONPATH=$PROJECT_ROOT && \
         export PREDICTOR_URL=http://$PREDICTOR_HOST:$PREDICTOR_PORT && \
         export SCHEDULER_HOST=$SCHEDULER_B_HOST && \
         export SCHEDULER_PORT=$SCHEDULER_PORT && \
         export SCHEDULER_LOG_DIR=$LOG_DIR/scheduler-b && \
         export SCHEDULER_LOGURU_LEVEL=INFO && \
         export PLANNER_URL=http://$PLANNER_HOST:$PLANNER_PORT && \
         export SCHEDULER_AUTO_REPORT=20.0 && \
         export PLANNER_REPORT_TIMEOUT=5.0 && \
         python -m src.cli start --port $SCHEDULER_PORT" \
        "$SCHEDULER_PREDICTOR_CPU_RANGE"
fi

sleep 10

# Health Checks
echo -e "${BLUE}Performing Health Checks...${NC}"

if [[ "$LOCAL_IP" == "$SCHEDULER_A_HOST" ]]; then
    curl -s http://$SCHEDULER_A_HOST:$SCHEDULER_PORT/health >/dev/null && \
        echo -e "${GREEN}Scheduler A (LLM) Healthy${NC}" || \
        echo -e "${RED}Scheduler A Unhealthy${NC}"
fi

if [[ "$LOCAL_IP" == "$SCHEDULER_C_HOST" ]]; then
    curl -s http://$SCHEDULER_C_HOST:$SCHEDULER_PORT/health >/dev/null && \
        echo -e "${GREEN}Scheduler C (FLUX) Healthy${NC}" || \
        echo -e "${RED}Scheduler C Unhealthy${NC}"
fi

if [[ "$LOCAL_IP" == "$SCHEDULER_B_HOST" ]]; then
    curl -s http://$SCHEDULER_B_HOST:$SCHEDULER_PORT/health >/dev/null && \
        echo -e "${GREEN}Scheduler B (T2VID) Healthy${NC}" || \
        echo -e "${RED}Scheduler B Unhealthy${NC}"
fi

if [[ "$LOCAL_IP" == "$PREDICTOR_HOST" ]]; then
    curl -s http://$PREDICTOR_HOST:$PREDICTOR_PORT/health >/dev/null && \
        echo -e "${GREEN}Predictor Healthy${NC}" || \
        echo -e "${RED}Predictor Unhealthy${NC}"
fi

if [[ "$LOCAL_IP" == "$PLANNER_HOST" ]]; then
    curl -s http://$PLANNER_HOST:$PLANNER_PORT/health >/dev/null && \
        echo -e "${GREEN}Planner Healthy${NC}" || \
        echo -e "${RED}Planner Unhealthy${NC}"
fi

echo ""
echo -e "${GREEN}Infrastructure Service Launch Completed on $LOCAL_IP${NC}"
echo -e "${BLUE}AUTO_OPTIMIZE_ENABLED: $AUTO_OPTIMIZE_ENABLED${NC}"
echo ""
echo -e "${YELLOW}Note: Models are NOT deployed by this script.${NC}"
echo -e "${YELLOW}Use manual_deploy_type3.sh to deploy models.${NC}"
echo "Logs: $LOG_DIR"
