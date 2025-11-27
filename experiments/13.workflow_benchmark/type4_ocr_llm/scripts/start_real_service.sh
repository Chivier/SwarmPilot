#!/bin/bash
set -e

# ============================================
# Distributed OCR+LLM Service Launcher (Type4)
# ============================================
# Logic:
#   - Get local IP from bond1
#   - Start Planner, Predictor, Scheduler A/B if IP matches
#   - Start OCR and LLM instances on same servers
#   - Register half instances with Scheduler, half with Planner
#
# Usage:
#   bash start_real_service.sh
#
# Each server runs:
#   - 16 OCR instances (CPU) on ports 9000-9015
#   - 8 LLM instances (GPU) on ports 9016-9023

# -----------------------------
# Base Paths & Config
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TYPE4_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCHMARK_ROOT="$(cd "$TYPE4_ROOT/.." && pwd)"
PROJECT_ROOT="$(cd "$BENCHMARK_ROOT/../.." && pwd)"

LOG_DIR="$BENCHMARK_ROOT/logs"
mkdir -p "$LOG_DIR"

# Activate virtual environment
cd "$PROJECT_ROOT"
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

# -----------------------------
# Role & IP Configuration (from Exp03)
# -----------------------------
# Core service hosts
SCHEDULER_A_HOST="29.209.114.51"   # Scheduler for OCR (Model A)
SCHEDULER_B_HOST="29.209.113.228"  # Scheduler for LLM (Model B)
PREDICTOR_HOST="29.209.113.113"
PLANNER_HOST="29.209.114.166"
CLIENT_HOST="29.209.114.166"

# Service ports
SCHEDULER_PORT=8100
PREDICTOR_PORT=8101
PLANNER_PORT=8202

# CPU Binding for core services
SCHEDULER_PREDICTOR_CPU_RANGE="0-7"

# Instance Hosts (OCR and LLM run on same servers)
# Each server runs: 16 OCR (CPU, ports 9000-9015) + 8 LLM (GPU, ports 9016-9023)
# Combined from Exp03 LLM and T2Vid host lists
MODEL_HOSTS=(
    "29.209.106.237"
    "29.209.114.56"
    "29.209.114.241"
    "29.209.112.177"
    "29.209.113.235"
    "29.209.105.60"
    "29.209.113.166"
    "29.209.113.176"
    "29.209.113.169"
    "29.209.112.74"
    "29.209.115.174"
    "29.209.113.156"
)

# -----------------------------
# Instance Configuration
# -----------------------------
# Port layout: 9000-9015 for OCR, 9016-9023 for LLM
INSTANCE_PORT_BASE=9000

# OCR Configuration (Instance-based, no Docker)
NUM_OCR_PER_SERVER=16
OCR_PORT_BASE=9000          # Ports 9000-9015
OCR_SOFTWARE_NAME="easyocr"
OCR_SOFTWARE_VERSION="unknown"
OCR_HARDWARE_NAME="CPU"

# LLM Configuration
NUM_LLM_PER_SERVER=8
LLM_PORT_BASE=9016          # Ports 9016-9023 (after OCR ports)
LLM_MODEL_PATH="/path/to/llm/model"  # Update this path
LLM_SOFTWARE_NAME="sglang"
LLM_SOFTWARE_VERSION="0.5.5.post2"
LLM_HARDWARE_NAME="NVIDIA H20"

# CPU Binding Config
TOTAL_CORES=384
CPU_START_OFFSET=256
CORES_PER_INSTANCE=16

# -----------------------------
# Colors
# -----------------------------
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
        # Fallback to ip command
        ip -4 addr show bond1 2>/dev/null | awk '/inet / {print $2}' | cut -d/ -f1 | head -n1
        return
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
    local cpu_cores="${1:-}"
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
    local url=$1
    local timeout=${2:-120}
    local interval=2
    local elapsed=0

    echo "Waiting for service at $url to be healthy..."
    while [ $elapsed -lt $timeout ]; do
        if curl -s "$url" | grep -q "healthy"; then
            echo "Service at $url is healthy."
            return 0
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
    done
    echo "Timeout waiting for service at $url."
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
        echo -e "${GREEN}Model started successfully.${NC}"
    else
        echo -e "${YELLOW}Model start response: $response${NC}"
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
echo ""

# -----------------------------
# 1. Start Core Services (Planner, Predictor, Schedulers)
# -----------------------------

# Start Planner
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

# Start Predictor
if [[ "$LOCAL_IP" == "$PREDICTOR_HOST" ]]; then
    start_bg "predictor" \
        "cd $PROJECT_ROOT/predictor && \
         export PYTHONPATH=$PROJECT_ROOT && \
         export PREDICTOR_PORT=$PREDICTOR_PORT && \
         export PREDICTOR_LOG_DIR=$LOG_DIR/predictor && \
         export HTTPS_PROXY=http://hk-mmhttpproxy.woa.com:11113 && \
         python -m src.cli start --port $PREDICTOR_PORT --log-level INFO" \
        "$SCHEDULER_PREDICTOR_CPU_RANGE"
fi

# Start Scheduler A (for OCR / Model A)
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

# Start Scheduler B (for LLM / Model B)
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

# Wait for core services to start
sleep 10

# -----------------------------
# 2. Start OCR and LLM Instances (on same servers)
# -----------------------------
if in_list "$LOCAL_IP" "${MODEL_HOSTS[@]}"; then
    SCHEDULER_A_URL="http://$SCHEDULER_A_HOST:$SCHEDULER_PORT"
    SCHEDULER_B_URL="http://$SCHEDULER_B_HOST:$SCHEDULER_PORT"
    PLANNER_URL="http://$PLANNER_HOST:$PLANNER_PORT"

    # =========================================
    # Phase 1: Start all instances (no model start yet)
    # =========================================
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Phase 1: Starting all instances...${NC}"
    echo -e "${BLUE}========================================${NC}"

    # -----------------------------
    # 2a. Start OCR Instances (ports 9000-9015)
    # -----------------------------
    echo -e "${BLUE}Starting OCR Instances on $LOCAL_IP (16 instances, ports 9000-9015)...${NC}"

    for i in $(seq 0 $((NUM_OCR_PER_SERVER - 1))); do
        OCR_PORT=$((OCR_PORT_BASE + i))
        CPU_RANGE=$(get_cpu_range $i)

        echo "  Starting OCR Instance $i (Port $OCR_PORT)..."

        # Start OCR instance with platform_info
        start_bg "ocr_${LOCAL_IP}_${OCR_PORT}" \
            "cd $PROJECT_ROOT/instance && \
             export PYTHONPATH=$PROJECT_ROOT && \
             export INSTANCE_ID=ocr_model_${LOCAL_IP}_${OCR_PORT} && \
             export INSTANCE_PORT=$OCR_PORT && \
             export INSTANCE_PLATFORM_SOFTWARE_NAME=\"$OCR_SOFTWARE_NAME\" && \
             export INSTANCE_PLATFORM_SOFTWARE_VERSION=\"$OCR_SOFTWARE_VERSION\" && \
             export INSTANCE_PLATFORM_HARDWARE_NAME=\"$OCR_HARDWARE_NAME\" && \
             export INSTANCE_LOG_DIR=$LOG_DIR/ocr_${i} && \
             python -m src.cli start --port $OCR_PORT" \
            "$CPU_RANGE"
    done

    # -----------------------------
    # 2b. Start LLM Instances (ports 9016-9023)
    # -----------------------------
    echo -e "${BLUE}Starting LLM Instances on $LOCAL_IP (8 instances, ports 9016-9023)...${NC}"

    for i in $(seq 0 $((NUM_LLM_PER_SERVER - 1))); do
        LLM_PORT=$((LLM_PORT_BASE + i))
        GPU_ID=$i
        # LLM instances use CPU cores after OCR instances
        CPU_RANGE=$(get_cpu_range $((NUM_OCR_PER_SERVER + i)))

        echo "  Starting LLM Instance $i (Port $LLM_PORT, GPU $GPU_ID)..."

        # Start LLM instance with platform_info
        start_bg "llm_${LOCAL_IP}_${LLM_PORT}" \
            "cd $PROJECT_ROOT/instance && \
             export PYTHONPATH=$PROJECT_ROOT && \
             export INSTANCE_ID=llm_model_${LOCAL_IP}_${LLM_PORT} && \
             export INSTANCE_PORT=$LLM_PORT && \
             export INSTANCE_PLATFORM_SOFTWARE_NAME=\"$LLM_SOFTWARE_NAME\" && \
             export INSTANCE_PLATFORM_SOFTWARE_VERSION=\"$LLM_SOFTWARE_VERSION\" && \
             export INSTANCE_PLATFORM_HARDWARE_NAME=\"$LLM_HARDWARE_NAME\" && \
             export CUDA_VISIBLE_DEVICES=$GPU_ID && \
             export MASTER_PORT=$((LLM_PORT + 20000)) && \
             export INSTANCE_LOG_DIR=$LOG_DIR/llm_${i} && \
             python -m src.cli start --port $LLM_PORT" \
            "$CPU_RANGE"
    done

    # =========================================
    # Wait for all instances to start
    # =========================================
    echo ""
    echo -e "${YELLOW}Waiting 10 seconds for all instances to initialize...${NC}"
    sleep 10

    # =========================================
    # Phase 2: Wait for health and start models via API
    # =========================================
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Phase 2: Starting models via /model/start API...${NC}"
    echo -e "${BLUE}========================================${NC}"

    # -----------------------------
    # 2c. Start OCR models via /model/start API
    # -----------------------------
    echo -e "${BLUE}Starting OCR models...${NC}"

    for i in $(seq 0 $((NUM_OCR_PER_SERVER - 1))); do
        OCR_PORT=$((OCR_PORT_BASE + i))

        # Determine registration target: first half to Scheduler, second half to Planner
        # Registration is handled internally by /model/start API
        if [ $i -lt $((NUM_OCR_PER_SERVER / 2)) ]; then
            SCHEDULER_URL="$SCHEDULER_A_URL"
            echo "  OCR Instance $i (Port $OCR_PORT): -> Scheduler A"
        else
            SCHEDULER_URL="$PLANNER_URL"
            echo "  OCR Instance $i (Port $OCR_PORT): -> Planner"
        fi

        # Wait for health and start model via API
        if wait_for_health "http://localhost:${OCR_PORT}/health" 120; then
            start_model_via_api $OCR_PORT "ocr_model" "$SCHEDULER_URL"
        else
            echo -e "${RED}OCR Instance $i failed health check${NC}"
        fi
    done

    # -----------------------------
    # 2d. Start LLM models via /model/start API
    # -----------------------------
    echo -e "${BLUE}Starting LLM models...${NC}"

    for i in $(seq 0 $((NUM_LLM_PER_SERVER - 1))); do
        LLM_PORT=$((LLM_PORT_BASE + i))

        # Determine registration target: first half to Scheduler, second half to Planner
        # Registration is handled internally by /model/start API
        if [ $i -lt $((NUM_LLM_PER_SERVER / 2)) ]; then
            SCHEDULER_URL="$SCHEDULER_B_URL"
            echo "  LLM Instance $i (Port $LLM_PORT): -> Scheduler B"
        else
            SCHEDULER_URL="$PLANNER_URL"
            echo "  LLM Instance $i (Port $LLM_PORT): -> Planner"
        fi

        # Wait for health and start model via API
        if wait_for_health "http://localhost:${LLM_PORT}/health" 180; then
            start_model_via_api $LLM_PORT "llm_model" "$SCHEDULER_URL"
        else
            echo -e "${RED}LLM Instance $i failed health check${NC}"
        fi
    done
fi

# -----------------------------
# 4. Final Health Checks
# -----------------------------
sleep 5
echo ""
echo -e "${BLUE}Performing Final Health Checks...${NC}"

# Check core services if running locally
if [[ "$LOCAL_IP" == "$SCHEDULER_A_HOST" ]]; then
    curl -s "http://$SCHEDULER_A_HOST:$SCHEDULER_PORT/health" >/dev/null && \
        echo -e "${GREEN}Scheduler A Healthy${NC}" || echo -e "${RED}Scheduler A Unhealthy${NC}"
fi
if [[ "$LOCAL_IP" == "$SCHEDULER_B_HOST" ]]; then
    curl -s "http://$SCHEDULER_B_HOST:$SCHEDULER_PORT/health" >/dev/null && \
        echo -e "${GREEN}Scheduler B Healthy${NC}" || echo -e "${RED}Scheduler B Unhealthy${NC}"
fi
if [[ "$LOCAL_IP" == "$PREDICTOR_HOST" ]]; then
    curl -s "http://$PREDICTOR_HOST:$PREDICTOR_PORT/health" >/dev/null && \
        echo -e "${GREEN}Predictor Healthy${NC}" || echo -e "${RED}Predictor Unhealthy${NC}"
fi
if [[ "$LOCAL_IP" == "$PLANNER_HOST" ]]; then
    curl -s "http://$PLANNER_HOST:$PLANNER_PORT/health" >/dev/null && \
        echo -e "${GREEN}Planner Healthy${NC}" || echo -e "${RED}Planner Unhealthy${NC}"
fi

# Check OCR and LLM instances (on same servers)
if in_list "$LOCAL_IP" "${MODEL_HOSTS[@]}"; then
    echo -e "${BLUE}Checking OCR instances (ports 9000-9015)...${NC}"
    for i in $(seq 0 $((NUM_OCR_PER_SERVER - 1))); do
        OCR_PORT=$((OCR_PORT_BASE + i))
        if curl -s "http://localhost:$OCR_PORT/health" >/dev/null 2>&1; then
            echo -e "${GREEN}  OCR Instance $i ($OCR_PORT) Healthy${NC}"
        else
            echo -e "${RED}  OCR Instance $i ($OCR_PORT) Unhealthy${NC}"
        fi
    done

    echo -e "${BLUE}Checking LLM instances (ports 9016-9023)...${NC}"
    for i in $(seq 0 $((NUM_LLM_PER_SERVER - 1))); do
        LLM_PORT=$((LLM_PORT_BASE + i))
        if curl -s "http://localhost:$LLM_PORT/health" >/dev/null 2>&1; then
            echo -e "${GREEN}  LLM Instance $i ($LLM_PORT) Healthy${NC}"
        else
            echo -e "${RED}  LLM Instance $i ($LLM_PORT) Unhealthy${NC}"
        fi
    done
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Service Launch Sequence Completed on $LOCAL_IP${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Logs directory: $LOG_DIR"
echo ""
echo "Instance Summary:"
if in_list "$LOCAL_IP" "${MODEL_HOSTS[@]}"; then
    echo "  OCR: 16 instances on ports $OCR_PORT_BASE-$((OCR_PORT_BASE + NUM_OCR_PER_SERVER - 1))"
    echo "       - 8 registered to Scheduler A"
    echo "       - 8 registered to Planner"
    echo "  LLM: 8 instances on ports $LLM_PORT_BASE-$((LLM_PORT_BASE + NUM_LLM_PER_SERVER - 1))"
    echo "       - 4 registered to Scheduler B"
    echo "       - 4 registered to Planner"
fi
