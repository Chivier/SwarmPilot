#!/bin/bash
set -e

# =============================================================================
# Configuration
# =============================================================================

# Project Root
PROJECT_ROOT="$(pwd)"
cd $PROJECT_ROOT

# Directories
LOG_DIR_BASE="logs/text2video_exp"
mkdir -p $LOG_DIR_BASE

# Ports
SCHEDULER_PORT=8100
PREDICTOR_PORT=8100
PLANNER_PORT=8100

# Instance Configuration
INSTANCE_PORT_LIST=(8200 8201 8202 8203 8204 8205 8206 8207)
GPU_BIND_ID_LIST=(0 1 2 3 4 5 6 7)
CORES_PER_INSTANCE=16
CPU_START_OFFSET=256
SCHEDULER_PREDICTOR_CPU_RANGE="0-7"

# Hosts
SCHEDULER_A_HOST="29.209.114.51"
SCHEDULER_B_HOST="29.209.113.228"
PREDICTOR_HOST="29.209.113.113"
PLANNER_HOST="29.209.114.166"

# Model Hosts
# LLM (Model A) Hosts - Based on SLEEP_MODEL_A_HOSTS
LLM_MODEL_HOSTS=(
  "29.209.114.166"
  "29.209.113.113"
  "29.209.106.237"
  "29.209.114.56"
  "29.209.114.241"
  "29.209.112.177"
  "29.209.113.235"
)

# T2Vid (Model B) Hosts - Based on SLEEP_MODEL_B_HOSTS
T2VID_MODEL_HOSTS=(
  "29.209.113.228"
  "29.209.105.60"
  "29.209.113.166"
  "29.209.113.176"
  "29.209.113.169"
  "29.209.112.74"
  "29.209.115.174"
  "29.209.113.156"
)

# =============================================================================
# Helper Functions
# =============================================================================

get_local_ip() {
    ifconfig bond1 | grep "inet " | awk '{print $2}' | head -n 1
}

is_in_list() {
    local element=$1
    shift
    local list=("$@")
    for item in "${list[@]}"; do
        if [[ "$item" == "$element" ]]; then
            return 0
        fi
    done
    return 1
}

get_cpu_range() {
  local idx="$1"
  local start_core=$((CPU_START_OFFSET + idx * CORES_PER_INSTANCE))
  local end_core=$((start_core + CORES_PER_INSTANCE - 1))
  echo "${start_core}-${end_core}"
}

start_bg() {
    local name=$1
    local cmd=$2
    local cpu_range=$3
    local log_file="$LOG_DIR_BASE/${name}.log"

    echo "Starting $name on CPUs $cpu_range..."
    echo "Log: $log_file"
    
    nohup taskset -c $cpu_range bash -c "$cmd" > "$log_file" 2>&1 &
    
    pid=$!
    echo "PID: $pid"
    echo "----------------------------------------"
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
    
    # Use curl to call /model/start
    # We use a dummy scheduler URL if we don't want to register, but the API requires a non-empty string.
    # The API will try to register. If we give a fake URL, registration will fail but model might start?
    # Wait, API says: "The scheduler_url must be provided and cannot be empty."
    # And: "Register with scheduler if enabled... Log error but don't fail model start"
    # So if we provide a fake URL, it should be fine.
    
    response=$(curl -s -X POST "http://localhost:$port/model/start" \
        -H "Content-Type: application/json" \
        -d "{\"model_id\": \"$model_id\", \"scheduler_url\": \"$scheduler_url\"}")
        
    if echo "$response" | grep -q "success\":true"; then
        echo "Model started successfully."
    else
        echo "Failed to start model. Response: $response"
    fi
}

# =============================================================================
# Main Deployment Logic
# =============================================================================

LOCAL_IP=$(get_local_ip)
echo "Local IP: $LOCAL_IP"

# 1. Start Planner
if [ "$LOCAL_IP" == "$PLANNER_HOST" ]; then
    echo "Starting Planner..."
    start_bg "planner" \
        "cd $PROJECT_ROOT/planner && \
         export PYTHONPATH=$PROJECT_ROOT && \
         export SCHEDULER_HOST=$SCHEDULER_A_HOST && \
         export SCHEDULER_PORT=$SCHEDULER_PORT && \
         export PLANNER_LOG_DIR=$LOG_DIR_BASE/planner && \
         export PLANNER_LOGURU_LEVEL=INFO && \
         export AUTO_OPTIMIZE_ENABLED=False && \
         python -m src.cli start --port $PLANNER_PORT" \
        "$SCHEDULER_PREDICTOR_CPU_RANGE"
fi

# 2. Start Predictor
if [ "$LOCAL_IP" == "$PREDICTOR_HOST" ]; then
    echo "Starting Predictor..."
    start_bg "predictor" \
        "cd $PROJECT_ROOT/predictor && \
         export PYTHONPATH=$PROJECT_ROOT && \
         export PREDICTOR_PORT=$PREDICTOR_PORT && \
         export PREDICTOR_LOG_DIR=$LOG_DIR_BASE/predictor && \
         export HTTPS_PROXY=http://hk-mmhttpproxy.woa.com:11113 && \
         python -m src.cli start --port $PREDICTOR_PORT --log-level INFO" \
        "$SCHEDULER_PREDICTOR_CPU_RANGE"
fi

# 3. Start Scheduler A
if [ "$LOCAL_IP" == "$SCHEDULER_A_HOST" ]; then
    echo "Starting Scheduler A..."
    start_bg "scheduler_a" \
        "cd $PROJECT_ROOT/scheduler && \
         export PYTHONPATH=$PROJECT_ROOT && \
         export PREDICTOR_URL=http://$PREDICTOR_HOST:$PREDICTOR_PORT && \
         export SCHEDULER_HOST=$SCHEDULER_A_HOST && \
         export SCHEDULER_PORT=$SCHEDULER_PORT && \
         export SCHEDULER_LOG_DIR=$LOG_DIR_BASE/scheduler-a && \
         export SCHEDULER_LOGURU_LEVEL=INFO && \
         export PLANNER_URL=http://$PLANNER_HOST:$PLANNER_PORT && \
         export SCHEDULER_AUTO_REPORT=20.0 \
         export PLANNER_REPORT_TIMEOUT=5.0 \
         python -m src.cli start --port $SCHEDULER_PORT" \
        "$SCHEDULER_PREDICTOR_CPU_RANGE"
fi

# 4. Start Scheduler B
if [ "$LOCAL_IP" == "$SCHEDULER_B_HOST" ]; then
    echo "Starting Scheduler B..."
    start_bg "scheduler_b" \
        "cd $PROJECT_ROOT/scheduler && \
         export PYTHONPATH=$PROJECT_ROOT && \
         export PREDICTOR_URL=http://$PREDICTOR_HOST:$PREDICTOR_PORT && \
         export SCHEDULER_HOST=$SCHEDULER_B_HOST && \
         export SCHEDULER_PORT=$SCHEDULER_PORT && \
         export SCHEDULER_LOG_DIR=$LOG_DIR_BASE/scheduler-b && \
         export SCHEDULER_LOGURU_LEVEL=INFO && \
         export PLANNER_URL=http://$PLANNER_HOST:$PLANNER_PORT && \
         export SCHEDULER_AUTO_REPORT=20.0 \
         export PLANNER_REPORT_TIMEOUT=5.0 \
         python -m src.cli start --port $SCHEDULER_PORT" \
        "$SCHEDULER_PREDICTOR_CPU_RANGE"
fi

sleep 5

# 5. Start Instances
# Check LLM Models (Model A)
if is_in_list "$LOCAL_IP" "${LLM_MODEL_HOSTS[@]}"; then
    MODEL_ID="llm_service_small_model"
    SCHEDULER_URL_REAL="http://$SCHEDULER_A_HOST:$SCHEDULER_PORT"
    
    echo "Deploying LLM Instances on $LOCAL_IP..."
    
    for i in {0..7}; do
        INSTANCE_PORT=${INSTANCE_PORT_LIST[$i]}
        GPU_ID=${GPU_BIND_ID_LIST[$i]}
        CPU_RANGE=$(get_cpu_range $i)
        
        # Register first 4 instances (0-3)
        if [ $i -lt 4 ]; then
            SCHEDULER_URL="$SCHEDULER_URL_REAL"
            echo "  Instance $i (Port $INSTANCE_PORT): Registering with Scheduler A"
        else
            SCHEDULER_URL="http://fake-scheduler:9999"
            echo "  Instance $i (Port $INSTANCE_PORT): NOT registering"
        fi
        
        start_bg "instance_${MODEL_ID}_${i}" \
            "cd $PROJECT_ROOT/instance && \
             export PYTHONPATH=$PROJECT_ROOT && \
             export INSTANCE_ID=llm_${LOCAL_IP}_${INSTANCE_PORT} && \
             export INSTANCE_PORT=$INSTANCE_PORT && \
             export CUDA_VISIBLE_DEVICES=$GPU_ID && \
             export MASTER_PORT=$((INSTANCE_PORT + 20000)) \
             export INSTANCE_LOG_DIR=$LOG_DIR_BASE/instance_llm_${i} && \
             python -m src.cli start --port $INSTANCE_PORT" \
            "$CPU_RANGE"
            
        # Wait and Start Model
        if wait_for_health $INSTANCE_PORT; then
            start_model_via_api $INSTANCE_PORT $MODEL_ID $SCHEDULER_URL
        fi
    done
fi

# Check T2Vid Models (Model B)
if is_in_list "$LOCAL_IP" "${T2VID_MODEL_HOSTS[@]}"; then
    MODEL_ID="t2vid"
    SCHEDULER_URL_REAL="http://$SCHEDULER_B_HOST:$SCHEDULER_PORT"
    
    echo "Deploying T2Vid Instances on $LOCAL_IP..."
    
    for i in {0..7}; do
        INSTANCE_PORT=${INSTANCE_PORT_LIST[$i]}
        GPU_ID=${GPU_BIND_ID_LIST[$i]}
        CPU_RANGE=$(get_cpu_range $i)
        
        # Register first 4 instances (0-3)
        if [ $i -lt 4 ]; then
            SCHEDULER_URL="$SCHEDULER_URL_REAL"
            echo "  Instance $i (Port $INSTANCE_PORT): Registering with Scheduler B"
        else
            SCHEDULER_URL="http://fake-scheduler:9999"
            echo "  Instance $i (Port $INSTANCE_PORT): NOT registering"
        fi
        
        start_bg "instance_${MODEL_ID}_${i}" \
            "cd $PROJECT_ROOT/instance && \
             export PYTHONPATH=$PROJECT_ROOT && \
             export INSTANCE_ID=t2vid_${LOCAL_IP}_${INSTANCE_PORT} && \
             export INSTANCE_PORT=$INSTANCE_PORT && \
             export CUDA_VISIBLE_DEVICES=$GPU_ID && \
             export MASTER_PORT=$((INSTANCE_PORT + 20000)) \
             export INSTANCE_LOG_DIR=$LOG_DIR_BASE/instance_t2vid_${i} && \
             python -m src.cli start --port $INSTANCE_PORT" \
            "$CPU_RANGE"
            
        # Wait and Start Model
        if wait_for_health $INSTANCE_PORT; then
            start_model_via_api $INSTANCE_PORT $MODEL_ID $SCHEDULER_URL
        fi
    done
fi

echo "Deployment script completed."
