#!/bin/bash
set -e

# ============================================
# Multi-node service launcher (Exp 03)
# ============================================
# Logic:
#   - Get local IP from bond1
#   - Start Planner, Predictor, Scheduler A/B if IP matches
#   - Start Instances if IP matches LLM or T2Vid host lists
#   - Register instances with appropriate scheduler (or fake one for manual control)
#
# Usage:
#   bash start_real_service.sh

# -----------------------------
# Base Paths & Config
# -----------------------------

uv sync
source .venv/bin/activate
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Ports
SCHEDULER_PORT=8100
PREDICTOR_PORT=8100
PLANNER_PORT=8100
INSTANCE_PORT=8000

# CPU Binding Config
TOTAL_CORES=384
CPU_START_OFFSET=256
CORES_PER_INSTANCE=16
SCHEDULER_PREDICTOR_CPU_RANGE="0-7"

# Roles & IPs
SCHEDULER_A_HOST="29.209.114.51"
SCHEDULER_B_HOST="29.209.113.228"
PREDICTOR_HOST="29.209.113.113"
PLANNER_HOST="29.209.114.166"
CLIENT_HOST="29.209.114.166"

INSTANCE_PORT_LIST=(8200 8201 8202 8203 8204 8205 8206 8207)
GPU_BIND_ID_LIST=(0 1 2 3 4 5 6 7)

# LLM (Model A) Hosts
LLM_MODEL_HOSTS=(
  "29.209.106.237"
  "29.209.114.56"
  "29.209.114.241"
  "29.209.112.177"
  "29.209.113.235"
  "29.209.105.60"
)

# T2Vid (Model B) Hosts
T2VID_MODEL_HOSTS=(
  "29.209.113.166"
  "29.209.113.176"
  "29.209.113.169"
  "29.209.112.74"
  "29.209.115.174"
  "29.209.113.156"
)

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

in_list() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
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

get_cpu_range() {
  local idx="$1"
  local start_core=$((CPU_START_OFFSET + idx * CORES_PER_INSTANCE))
  local end_core=$((start_core + CORES_PER_INSTANCE - 1))
  echo "${start_core}-${end_core}"
}

wait_for_health() {
    local port=$1
    local timeout=60
    local interval=2
    local elapsed=0
    local url="http://localhost:$port/health"

    echo "Waiting for service at port $port to be healthy..."
    while [ $elapsed -lt $timeout ]; do
        if curl -s "$url" | grep -q "healthy"; then
            echo "Service at port $port is healthy."
            return 0
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
    done
    echo "Timeout waiting for service at port $port."
    return 1
}

start_model_via_api() {
    local port=$1
    local model_id=$2
    local scheduler_url=$3
    
    echo "Starting model $model_id on port $port with scheduler $scheduler_url..."
    response=$(curl -s -X POST "http://localhost:$port/model/start" \
        -H "Content-Type: application/json" \
        -d "{\"model_id\": \"$model_id\", \"scheduler_url\": \"$scheduler_url\"}")
        
    if echo "$response" | grep -q "success\":true"; then
        echo "Model started successfully."
    else
        echo "Failed to start model. Response: $response"
    fi
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
     export AUTO_OPTIMIZE_ENABLED=True && \
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

# 5. Start Instances
# Determine Model ID and Scheduler URL based on host list
MODEL_ID=""
SCHEDULER_URL_REAL=""

if in_list "$LOCAL_IP" "${LLM_MODEL_HOSTS[@]}"; then
  MODEL_ID="llm_service_small_model"
  SCHEDULER_URL_REAL="http://$SCHEDULER_A_HOST:$SCHEDULER_PORT"
elif in_list "$LOCAL_IP" "${T2VID_MODEL_HOSTS[@]}"; then
  MODEL_ID="t2vid"
  SCHEDULER_URL_REAL="http://$SCHEDULER_B_HOST:$SCHEDULER_PORT"
fi

if [[ -n "$MODEL_ID" ]]; then
  echo -e "${BLUE}Deploying $MODEL_ID Instances on $LOCAL_IP...${NC}"
  
  for i in {0..7}; do
    INSTANCE_PORT=${INSTANCE_PORT_LIST[$i]}
    GPU_ID=${GPU_BIND_ID_LIST[$i]}
    CPU_RANGE=$(get_cpu_range $i)
    
    # Register first 4 instances (0-3) with real scheduler, others with fake
    if [ $i -lt 4 ]; then
        SCHEDULER_URL="$SCHEDULER_URL_REAL"
        echo "  Instance $i (Port $INSTANCE_PORT): Registering with Real Scheduler"
    else
        SCHEDULER_URL="http://fake-scheduler:9999"
        echo "  Instance $i (Port $INSTANCE_PORT): NOT registering (Manual Control)"
    fi
    
    # Determine software info based on model
    SOFTWARE_NAME="sglang"
    SOFTWARE_VERSION="0.5.5.post2"
    if [[ "$MODEL_ID" == "t2vid" ]]; then
        SOFTWARE_NAME="diffuser"
        SOFTWARE_VERSION="latest"
    fi

    start_bg "instance_${MODEL_ID}_${i}" \
      "cd $PROJECT_ROOT/instance && \
       export PYTHONPATH=$PROJECT_ROOT && \
       export INSTANCE_ID=${MODEL_ID}_${LOCAL_IP}_${INSTANCE_PORT} && \
       export INSTANCE_PORT=$INSTANCE_PORT && \
       export INSTANCE_PLATFORM_SOFTWARE_NAME=$SOFTWARE_NAME && \
       export INSTANCE_PLATFORM_SOFTWARE_VERSION=\"$SOFTWARE_VERSION\" && \
       export INSTANCE_PLATFORM_HARDWARE_NAME=\"NVIDIA H20\" && \
       export CUDA_VISIBLE_DEVICES=$GPU_ID && \
       export MASTER_PORT=$((INSTANCE_PORT + 20000)) \
       export INSTANCE_LOG_DIR=$LOG_DIR/instance_${MODEL_ID}_${i} && \
       python -m src.cli start --port $INSTANCE_PORT" \
      "$CPU_RANGE"
      
    # Wait and Start Model
    if wait_for_health $INSTANCE_PORT; then
        start_model_via_api $INSTANCE_PORT $MODEL_ID $SCHEDULER_URL
    fi
  done
else
  echo -e "${YELLOW}Host $LOCAL_IP not in Model A/B lists, skipping instance deployment.${NC}"
fi

sleep 5

# Health Checks
echo -e "${BLUE}Performing Health Checks...${NC}"

# Check core services if running locally
if [[ "$LOCAL_IP" == "$SCHEDULER_A_HOST" ]]; then
    curl http://$SCHEDULER_A_HOST:$SCHEDULER_PORT/health && echo -e "${GREEN}Scheduler A Healthy${NC}" || echo -e "${RED}Scheduler A Unhealthy${NC}"
fi
if [[ "$LOCAL_IP" == "$SCHEDULER_B_HOST" ]]; then
    curl http://$SCHEDULER_B_HOST:$SCHEDULER_PORT/health && echo -e "${GREEN}Scheduler B Healthy${NC}" || echo -e "${RED}Scheduler B Unhealthy${NC}"
fi
if [[ "$LOCAL_IP" == "$PREDICTOR_HOST" ]]; then
    curl http://$PREDICTOR_HOST:$PREDICTOR_PORT/health && echo -e "${GREEN}Predictor Healthy${NC}" || echo -e "${RED}Predictor Unhealthy${NC}"
fi
if [[ "$LOCAL_IP" == "$PLANNER_HOST" ]]; then
    curl http://$PLANNER_HOST:$PLANNER_PORT/health && echo -e "${GREEN}Planner Healthy${NC}" || echo -e "${RED}Planner Unhealthy${NC}"
fi

# Check instances if running locally
if [[ -n "$MODEL_ID" ]]; then
  for i in {0..7}; do
    INSTANCE_PORT=${INSTANCE_PORT_LIST[$i]}
    curl http://$LOCAL_IP:$INSTANCE_PORT/health >/dev/null 2>&1
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}Instance $i ($INSTANCE_PORT) Healthy${NC}"
    else
        echo -e "${RED}Instance $i ($INSTANCE_PORT) Unhealthy${NC}"
    fi
  done
fi

echo ""
echo -e "${GREEN}Service Launch Sequence Completed on $LOCAL_IP${NC}"
echo "Logs: $LOG_DIR"
