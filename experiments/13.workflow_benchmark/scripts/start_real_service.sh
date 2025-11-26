#!/bin/bash
set -e

# ============================================
# Multi-node service launcher (Exp 13 - Workflow Benchmark)
# ============================================
# Logic:
#   - Get local IP from bond1
#   - Start Planner, Predictor, Scheduler A/B if IP matches
#   - Does NOT deploy models (use manual_deploy_type1.sh or manual_deploy_type2.sh)
#
# Usage:
#   bash start_real_service.sh [--auto-optimize]

# -----------------------------
# Parse Arguments
# -----------------------------
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
            echo "  --auto-optimize      Enable AUTO_OPTIMIZE_ENABLED for planner (default: disabled)"
            echo "  --no-auto-optimize   Disable AUTO_OPTIMIZE_ENABLED (default)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Note: This script only starts infrastructure services (planner, predictor, schedulers)."
            echo "      Use manual_deploy_type1.sh or manual_deploy_type2.sh to deploy models."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "AUTO_OPTIMIZE_ENABLED: $AUTO_OPTIMIZE_ENABLED"

# -----------------------------
# Base Paths & Config
# -----------------------------

# Navigate to project root (swarmpilot-refresh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"
uv sync
source .venv/bin/activate

LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$LOG_DIR"

# Ports
SCHEDULER_PORT=8100
PREDICTOR_PORT=8100
PLANNER_PORT=8100

# CPU Binding Config
SCHEDULER_PREDICTOR_CPU_RANGE="0-7"

# Roles & IPs
SCHEDULER_A_HOST="29.209.114.51"
SCHEDULER_B_HOST="29.209.113.228"
PREDICTOR_HOST="29.209.113.113"
PLANNER_HOST="29.209.114.166"
CLIENT_HOST="29.209.114.166"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# -----------------------------
# Helper Functions
# -----------------------------
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

# -----------------------------
# Main Logic
# -----------------------------
LOCAL_IP="$(get_local_ip)"
if [[ -z "$LOCAL_IP" ]]; then
  echo -e "${RED}Failed to detect local IP on bond1, aborting.${NC}"
  exit 1
fi
echo -e "${BLUE}Local IP detected: $LOCAL_IP${NC}"

# 1. Start Planner
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

# 2. Start Predictor
if [[ "$LOCAL_IP" == "$PREDICTOR_HOST" ]]; then
  start_bg "predictor" \
    "cd $PROJECT_ROOT/predictor && \
     export PYTHONPATH=$PROJECT_ROOT && \
     export PREDICTOR_PORT=$PREDICTOR_PORT \
     export PREDICTOR_LOG_DIR=$LOG_DIR/predictor && \
     export HTTPS_PROXY=http://hk-mmhttpproxy.woa.com:11113 && \
     python -m src.cli start --port $PREDICTOR_PORT --log-level INFO" \
    "$SCHEDULER_PREDICTOR_CPU_RANGE"
fi

# 3. Start Scheduler A
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
     export SCHEDULER_AUTO_REPORT=20.0 \
     export PLANNER_REPORT_TIMEOUT=5.0 \
     python -m src.cli start --port $SCHEDULER_PORT" \
    "$SCHEDULER_PREDICTOR_CPU_RANGE"
fi

# 4. Start Scheduler B
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
     export SCHEDULER_AUTO_REPORT=20.0 \
     export PLANNER_REPORT_TIMEOUT=5.0 \
     python -m src.cli start --port $SCHEDULER_PORT" \
    "$SCHEDULER_PREDICTOR_CPU_RANGE"
fi

sleep 10

# 仅当本机IP不是 SCHEDULER_B_HOST、SCHEDULER_A_HOST、PREDICTOR_HOST 和 CLIENT_HOST 任意一个时，启动instance
if [[ "$LOCAL_IP" != "$SCHEDULER_B_HOST" ]] && \
   [[ "$LOCAL_IP" != "$SCHEDULER_A_HOST" ]] && \
   [[ "$LOCAL_IP" != "$PREDICTOR_HOST" ]] && \
   [[ "$LOCAL_IP" != "$CLIENT_HOST" ]]; then
  for i in {0..7}; do
    INSTANCE_PORT=${INSTANCE_PORT_LIST[$i]}
    GPU_ID=${GPU_BIND_ID_LIST[$i]}
    CPU_RANGE=$(get_cpu_range $i)

    start_bg "instance_${MODEL_ID}_gpu${GPU_ID}_port${INSTANCE_PORT}" \
      "cd $PROJECT_ROOT/instance && \
       MODEL_ID=$MODEL_ID \
       CUDA_VISIBLE_DEVICES=${GPU_ID} \
       MASTER_PORT=$((INSTANCE_PORT + 20000)) \
       INSTANCE_PLATFORM_SOFTWARE_NAME=sglang \
       INSTANCE_PLATFORM_SOFTWARE_VERSION=\"0.5.5.post2\" \
       INSTANCE_PLATFORM_HARDWARE_NAME=\"NVIDIA H20\" \
       INSTANCE_PORT=${INSTANCE_PORT} \
       INSTANCE_ENDPOINT=http://${LOCAL_IP}:${INSTANCE_PORT} \
       INSTANCE_ID=${LOCAL_IP}-${INSTANCE_PORT} \
       python -m src.cli start --port ${INSTANCE_PORT}" \
      "$CPU_RANGE"
  done
fi

sleep 5

# Health Checks
echo -e "${BLUE}Performing Health Checks...${NC}"

# Check core services if running locally
if [[ "$LOCAL_IP" == "$SCHEDULER_A_HOST" ]]; then
    curl -s http://$SCHEDULER_A_HOST:$SCHEDULER_PORT/health >/dev/null && echo -e "${GREEN}Scheduler A Healthy${NC}" || echo -e "${RED}Scheduler A Unhealthy${NC}"
fi
if [[ "$LOCAL_IP" == "$SCHEDULER_B_HOST" ]]; then
    curl -s http://$SCHEDULER_B_HOST:$SCHEDULER_PORT/health >/dev/null && echo -e "${GREEN}Scheduler B Healthy${NC}" || echo -e "${RED}Scheduler B Unhealthy${NC}"
fi
if [[ "$LOCAL_IP" == "$PREDICTOR_HOST" ]]; then
    curl -s http://$PREDICTOR_HOST:$PREDICTOR_PORT/health >/dev/null && echo -e "${GREEN}Predictor Healthy${NC}" || echo -e "${RED}Predictor Unhealthy${NC}"
fi
if [[ "$LOCAL_IP" == "$PLANNER_HOST" ]]; then
    curl -s http://$PLANNER_HOST:$PLANNER_PORT/health >/dev/null && echo -e "${GREEN}Planner Healthy${NC}" || echo -e "${RED}Planner Unhealthy${NC}"
fi

# Check instance if running locally
if [[ "$LOCAL_IP" != "$SCHEDULER_B_HOST" ]] && \
   [[ "$LOCAL_IP" != "$SCHEDULER_A_HOST" ]] && \
   [[ "$LOCAL_IP" != "$PREDICTOR_HOST" ]] && \
   [[ "$LOCAL_IP" != "$CLIENT_HOST" ]]; then
  for i in {0..7}; do
    INSTANCE_PORT=${INSTANCE_PORT_LIST[$i]}
    GPU_ID=${GPU_BIND_ID_LIST[$i]}
    CPU_RANGE=$(get_cpu_range $i)

    curl -s http://$LOCAL_IP:$INSTANCE_PORT/health >/dev/null && echo -e "${GREEN}Instance ${LOCAL_IP}-${INSTANCE_PORT} Healthy${NC}" || echo -e "${RED}Instance ${LOCAL_IP}-${INSTANCE_PORT} Unhealthy${NC}"
  done
fi

echo ""
echo -e "${GREEN}Infrastructure Service Launch Completed on $LOCAL_IP${NC}"
echo -e "${BLUE}AUTO_OPTIMIZE_ENABLED: $AUTO_OPTIMIZE_ENABLED${NC}"
echo ""
echo -e "${YELLOW}Note: Models are NOT deployed by this script.${NC}"
echo -e "${YELLOW}Use manual_deploy_type1.sh or manual_deploy_type2.sh to deploy models.${NC}"
echo "Logs: $LOG_DIR"
