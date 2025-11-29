#!/bin/bash

set -e

# Get the project root directory (3 levels up from this script: scripts -> 13.workflow_benchmark -> experiments -> root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

# Configuration (can be overridden via environment variables or command-line arguments)
PREDICTOR_PORT=8101
SCHEDULER_A_PORT=8100
SCHEDULER_B_PORT=8200
PLANNER_PORT=8202
INSTANCE_GROUP_A_START_PORT=8210  # Group A instances: 8210-82xx
INSTANCE_GROUP_B_START_PORT=8300  # Group B instances: 8300-83xx
N1=${N1:-4}  # Default: 4 instances in group A
N2=${N2:-2}  # Default: 2 instances in group B
MODEL_ID_A=${MODEL_ID_A:-sleep_model_a}
MODEL_ID_B=${MODEL_ID_B:-sleep_model_b}
AUTO_OPTIMIZE_ENABLED=${AUTO_OPTIMIZE_ENABLED:-False}

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command-line arguments
usage() {
    echo "Usage: $0 [N1] [N2] [MODEL_ID_A] [MODEL_ID_B]"
    echo "  N1: Number of instances in Group A (default: 4)"
    echo "  N2: Number of instances in Group B (default: 2)"
    echo "  MODEL_ID_A: Model ID for Group A (default: sleep_model_a)"
    echo "  MODEL_ID_B: Model ID for Group B (default: sleep_model_b)"
    echo ""
    echo "Examples:"
    echo "  $0                              # Use defaults (N1=4, N2=2)"
    echo "  $0 8 4                          # 8 Group A, 4 Group B instances"
    echo "  N1=10 N2=6 $0                   # Using environment variables"
    echo "  $0 10 6 llm_service_small_model t2vid  # Real models"
    exit 1
}

# Enable python jit for better performance
export PYTHON_JIT=1

# Parse positional arguments
if [ $# -ge 1 ]; then
    if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
        usage
    fi
    N1=$1
fi

if [ $# -ge 2 ]; then
    N2=$2
fi

if [ $# -ge 3 ]; then
    MODEL_ID_A=$3
fi

if [ $# -ge 4 ]; then
    MODEL_ID_B=$4
fi

# Validate inputs
if ! [[ "$N1" =~ ^[0-9]+$ ]] || [ "$N1" -lt 1 ]; then
    echo -e "${RED}Error: N1 must be a positive integer${NC}"
    usage
fi

if ! [[ "$N2" =~ ^[0-9]+$ ]] || [ "$N2" -lt 1 ]; then
    echo -e "${RED}Error: N2 must be a positive integer${NC}"
    usage
fi

# When AUTO_OPTIMIZE_ENABLED is True, force 1:1 ratio
if [[ "$AUTO_OPTIMIZE_ENABLED" == "True" ]] || [[ "$AUTO_OPTIMIZE_ENABLED" == "true" ]]; then
    ORIGINAL_N1=$N1
    ORIGINAL_N2=$N2
    TOTAL=$((N1 + N2))
    N1=$((TOTAL / 2))
    N2=$((TOTAL - N1))  # Handle odd totals by giving extra to N2

    if [ "$ORIGINAL_N1" != "$N1" ] || [ "$ORIGINAL_N2" != "$N2" ]; then
        echo ""
        echo -e "${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${YELLOW}║  AUTO_OPTIMIZE_ENABLED=True: Forcing 1:1 instance ratio    ║${NC}"
        echo -e "${YELLOW}╠════════════════════════════════════════════════════════════╣${NC}"
        echo -e "${YELLOW}║  Original: N1=${ORIGINAL_N1}, N2=${ORIGINAL_N2} (total: ${TOTAL})                          ${NC}"
        echo -e "${YELLOW}║  Adjusted: N1=${N1}, N2=${N2} (1:1 ratio)                            ${NC}"
        echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
    fi
fi

# Log directory (in experiment directory)
LOG_DIR="$EXPERIMENT_DIR/logs"
mkdir -p "$LOG_DIR"

# Helper function to check if a service is healthy
check_health() {
    local url=$1
    local max_attempts=30
    local attempt=1

    echo -n "Waiting for $url to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url/health" > /dev/null 2>&1; then
            echo -e " ${GREEN}OK${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done

    echo -e " ${RED}FAILED${NC}"
    return 1
}

# Helper function to start a service
start_service() {
    local name=$1
    local command=$2
    local port=$3  # Port number to identify the service
    local log_file="$LOG_DIR/${name}.log"
    local pid_file="$LOG_DIR/${name}.pid"

    echo -e "${YELLOW}Starting $name...${NC}"

    # Start the service in background
    nohup bash -c "$command" > "$log_file" 2>&1 &

    # Wait a moment for the process to start
    sleep 3

    # Find the actual Python process PID by port number
    local actual_pid=$(pgrep -f "python.*--port $port" | head -1)

    # Retry a few times if not found immediately
    local retry=0
    while [ -z "$actual_pid" ] && [ $retry -lt 5 ]; do
        sleep 1
        actual_pid=$(pgrep -f "python.*--port $port" | head -1)
        retry=$((retry + 1))
    done

    if [ -z "$actual_pid" ]; then
        echo -e "${RED}Failed to find PID for $name (port: $port)${NC}"
        echo -e "${YELLOW}Check log file: $log_file${NC}"
        return 1
    fi

    echo $actual_pid > "$pid_file"
    echo -e "${GREEN}Started $name (PID: $actual_pid, Port: $port)${NC}"
}

echo "========================================="
echo "Starting Unified Workflow Benchmark"
echo "========================================="
echo -e "${BLUE}Configuration:${NC}"
echo "  Group A: $N1 instances (Model: $MODEL_ID_A)"
echo "  Group B: $N2 instances (Model: $MODEL_ID_B)"
echo "  Total: $((N1 + N2)) instances"
echo "  Planner: $AUTO_OPTIMIZE_ENABLED"
echo "  Project Root: $PROJECT_ROOT"
echo "  Experiment Dir: $EXPERIMENT_DIR"
echo "========================================="

# Step 1: Start Predictor Service
echo ""
echo "Step 1: Starting Predictor Service"
start_service "predictor" \
    "cd $PROJECT_ROOT/predictor && PREDICTOR_PORT=$PREDICTOR_PORT PREDICTOR_LOG_DIR=$LOG_DIR/predictor uv run python -m src.cli start --port $PREDICTOR_PORT --log-level INFO" \
    "$PREDICTOR_PORT"

# Wait for predictor to be ready
if ! check_health "http://localhost:$PREDICTOR_PORT"; then
    echo -e "${RED}Failed to start predictor service${NC}"
    exit 1
fi

# Step 2: Start Planner Service
echo ""
echo "Step 2: Starting Planner Service"
start_service "planner" \
    "cd $PROJECT_ROOT/planner && AUTO_OPTIMIZE_ENABLED=$AUTO_OPTIMIZE_ENABLED AUTO_OPTIMIZE_INTERVAL=150 PLANNER_LOG_DIR=$LOG_DIR/planner uv run python -m uvicorn src.api:app --port $PLANNER_PORT" \
    "$PLANNER_PORT"

# Wait for planner to be ready
if ! check_health "http://localhost:$PLANNER_PORT"; then
    echo -e "${RED}Failed to start planner service${NC}"
    exit 1
fi

# Step 3: Start Scheduler A (for Group A)
echo ""
echo "Step 3: Starting Scheduler A (Group A)"
start_service "scheduler-a" \
    "cd $PROJECT_ROOT/scheduler && PREDICTOR_URL=http://localhost:$PREDICTOR_PORT PLANNER_URL=http://localhost:$PLANNER_PORT SCHEDULER_PORT=$SCHEDULER_A_PORT SCHEDULER_LOG_DIR=$LOG_DIR/scheduler-a SCHEDULER_AUTO_REPORT=5 uv run python -m src.cli start --port $SCHEDULER_A_PORT" \
    "$SCHEDULER_A_PORT"

# Wait for Scheduler A to be ready
if ! check_health "http://localhost:$SCHEDULER_A_PORT"; then
    echo -e "${RED}Failed to start Scheduler A${NC}"
    exit 1
fi

# Step 4: Start Scheduler B (for Group B)
echo ""
echo "Step 4: Starting Scheduler B (Group B)"
start_service "scheduler-b" \
    "cd $PROJECT_ROOT/scheduler && PREDICTOR_URL=http://localhost:$PREDICTOR_PORT PLANNER_URL=http://localhost:$PLANNER_PORT SCHEDULER_PORT=$SCHEDULER_B_PORT SCHEDULER_LOG_DIR=$LOG_DIR/scheduler-b SCHEDULER_AUTO_REPORT=5 uv run python -m src.cli start --port $SCHEDULER_B_PORT" \
    "$SCHEDULER_B_PORT"

# Wait for Scheduler B to be ready
if ! check_health "http://localhost:$SCHEDULER_B_PORT"; then
    echo -e "${RED}Failed to start Scheduler B${NC}"
    exit 1
fi

# Step 5: Start Instances for Group A
echo ""
echo "Step 5: Starting $N1 instances for Group A (ports $INSTANCE_GROUP_A_START_PORT-$((INSTANCE_GROUP_A_START_PORT + N1 - 1)))"

instance_pids=()
for i in $(seq 0 $((N1 - 1))); do
    instance_port=$((INSTANCE_GROUP_A_START_PORT + i))
    instance_id="instance-a-$(printf '%03d' $i)"

    if (( i % 2 == 0 )); then
        scheduler_port=$SCHEDULER_A_PORT
    else
        scheduler_port=$PLANNER_PORT
    fi

    (
        start_service "$instance_id" \
            "cd $PROJECT_ROOT/instance && SCHEDULER_URL=http://localhost:$scheduler_port INSTANCE_ID=$instance_id INSTANCE_PORT=$instance_port INSTANCE_LOG_DIR=$LOG_DIR/$instance_id uv run python -m src.cli start --port $instance_port --docker" \
            "$instance_port"
    ) &
    instance_pids+=($!)
done

# Wait for all Group A instances to start
for pid in "${instance_pids[@]}"; do
    wait "$pid" || true
done

# Health check for Group A instances (parallel)
echo -n "Health checking Group A instances..."
health_pids=()
for i in $(seq 0 $((N1 - 1))); do
    instance_port=$((INSTANCE_GROUP_A_START_PORT + i))
    ( check_health "http://localhost:$instance_port" > /dev/null 2>&1 ) &
    health_pids+=($!)
done
for pid in "${health_pids[@]}"; do
    wait "$pid" || true
done
echo -e " ${GREEN}OK${NC}"

# Step 6: Start Instances for Group B
echo ""
echo "Step 6: Starting $N2 instances for Group B (ports $INSTANCE_GROUP_B_START_PORT-$((INSTANCE_GROUP_B_START_PORT + N2 - 1)))"

instance_pids=()
for i in $(seq 0 $((N2 - 1))); do
    instance_port=$((INSTANCE_GROUP_B_START_PORT + i))
    instance_id="instance-b-$(printf '%03d' $i)"

    if (( i % 2 == 0 )); then
        scheduler_port=$SCHEDULER_B_PORT
    else
        scheduler_port=$PLANNER_PORT
    fi

    (
        start_service "$instance_id" \
            "cd $PROJECT_ROOT/instance && SCHEDULER_URL=http://localhost:$SCHEDULER_B_PORT INSTANCE_ID=$instance_id INSTANCE_PORT=$instance_port INSTANCE_LOG_DIR=$LOG_DIR/$instance_id uv run python -m src.cli start --port $instance_port --docker" \
            "$instance_port"
    ) &
    instance_pids+=($!)
done

# Wait for all Group B instances to start
for pid in "${instance_pids[@]}"; do
    wait "$pid" || true
done

# Health check for Group B instances (parallel)
echo -n "Health checking Group B instances..."
health_pids=()
for i in $(seq 0 $((N2 - 1))); do
    instance_port=$((INSTANCE_GROUP_B_START_PORT + i))
    ( check_health "http://localhost:$instance_port" > /dev/null 2>&1 ) &
    health_pids+=($!)
done
for pid in "${health_pids[@]}"; do
    wait "$pid" || true
done
echo -e " ${GREEN}OK${NC}"

# Step 7: Deploy models (local deployment)
echo ""
echo "Step 7: Deploying models locally"
"$SCRIPT_DIR/deploy_models_local.sh" \
    --scheduler-a-url "http://localhost:$SCHEDULER_A_PORT" \
    --scheduler-b-url "http://localhost:$SCHEDULER_B_PORT" \
    --planner-url "http://localhost:$PLANNER_PORT" \
    --model-id-a "$MODEL_ID_A" \
    --model-id-b "$MODEL_ID_B" \
    --n1 "$N1" \
    --n2 "$N2" \
    --port-a-start "$INSTANCE_GROUP_A_START_PORT" \
    --port-b-start "$INSTANCE_GROUP_B_START_PORT"

# Step 8: Register instances with planner for migration optimization
if [ "$AUTO_OPTIMIZE_ENABLED" == "True" ] || [ "$AUTO_OPTIMIZE_ENABLED" == "true" ]; then
    echo ""
    echo "Step 8: Registering instances with planner"
    cd "$EXPERIMENT_DIR" && uv run python "$SCRIPT_DIR/type1_redeploy_sim.py" \
        --host "localhost" \
        --n1 "$N1" \
        --n2 "$N2" \
        --scheduler-a-port "$SCHEDULER_A_PORT" \
        --scheduler-b-port "$SCHEDULER_B_PORT" \
        --planner-port "$PLANNER_PORT" \
        --port-a-start "$INSTANCE_GROUP_A_START_PORT" \
        --port-b-start "$INSTANCE_GROUP_B_START_PORT" \
        --model-id-a "$MODEL_ID_A" \
        --model-id-b "$MODEL_ID_B"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Instances registered with planner successfully${NC}"
    else
        echo -e "${YELLOW}Warning: Failed to register instances with planner${NC}"
        echo -e "${YELLOW}You can manually run: python $SCRIPT_DIR/type1_redeploy_sim.py${NC}"
    fi
fi

# Summary
echo ""
echo "========================================="
echo -e "${GREEN}All services started successfully!${NC}"
echo "========================================="
echo -e "${BLUE}Service URLs:${NC}"
echo "  Predictor:   http://localhost:$PREDICTOR_PORT"
echo "  Planner:     http://localhost:$PLANNER_PORT"
echo "  Scheduler A: http://localhost:$SCHEDULER_A_PORT (Model: $MODEL_ID_A, $N1 instances)"
echo "  Scheduler B: http://localhost:$SCHEDULER_B_PORT (Model: $MODEL_ID_B, $N2 instances)"
echo ""
echo -e "${BLUE}Log Directory:${NC} $LOG_DIR"
echo ""
echo "Use './scripts/stop_all_services.sh' to shutdown all services."
echo "========================================="
