#!/bin/bash

set -e

# Get the project root directory (2 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration (can be overridden via command-line arguments)
PREDICTOR_PORT=8101
SCHEDULER_A_PORT=8100
SCHEDULER_B_PORT=8200
INSTANCE_GROUP_A_START_PORT=8210  # Group A instances: 8210-82xx
INSTANCE_GROUP_B_START_PORT=8300  # Group B instances: 8300-83xx
N1=${N1:-10}  # Default: 10 instances in group A
N2=${N2:-6}   # Default: 6 instances in group B
MODEL_ID="sleep_model"

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command-line arguments
usage() {
    echo "Usage: $0 [N1] [N2]"
    echo "  N1: Number of instances in Group A (default: 10)"
    echo "  N2: Number of instances in Group B (default: 6)"
    echo ""
    echo "Example: $0 8 8  # Start with 8 instances in each group"
    exit 1
}

if [ $# -ge 1 ]; then
    if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
        usage
    fi
    N1=$1
fi

if [ $# -ge 2 ]; then
    N2=$2
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

# Log directory
LOG_DIR="./logs"
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
    sleep 2

    # Find the actual Python process PID by port number
    # Pattern: python3 (not uv) + src.cli start + --port parameter
    local actual_pid=$(pgrep -f "python3.*src\.cli start.*--port $port" | head -1)

    # Retry a few times if not found immediately
    local retry=0
    while [ -z "$actual_pid" ] && [ $retry -lt 5 ]; do
        sleep 1
        actual_pid=$(pgrep -f "python3.*src\.cli start.*--port $port" | head -1)
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
echo "Starting 02.multi_model_no_dep Experiment"
echo "========================================="
echo -e "${BLUE}Configuration:${NC}"
echo "  Group A: $N1 instances (Scheduler A on port $SCHEDULER_A_PORT)"
echo "  Group B: $N2 instances (Scheduler B on port $SCHEDULER_B_PORT)"
echo "  Total: $((N1 + N2)) instances"
echo "========================================="

# Pre-flight: Clean up any existing Docker containers
echo ""
echo "Pre-flight: Cleaning up existing Docker containers..."
existing_containers=$(docker ps -a --filter "ancestor=sleep_model" -q)
if [ -n "$existing_containers" ]; then
    echo "Found existing containers, removing them..."
    docker rm -f $existing_containers 2>/dev/null
    echo -e "${GREEN}Cleaned up existing containers${NC}"
else
    echo "No existing containers found"
fi

# Step 1: Start Predictor Service
echo ""
echo "Step 1: Starting Predictor Service"
start_service "predictor" \
    "cd $PROJECT_ROOT/predictor && PREDICTOR_PORT=$PREDICTOR_PORT PREDICTOR_LOG_DIR=$SCRIPT_DIR/logs/predictor uv run python -m src.cli start --port $PREDICTOR_PORT --log-level INFO" \
    "$PREDICTOR_PORT"

# Wait for predictor to be ready
if ! check_health "http://localhost:$PREDICTOR_PORT"; then
    echo -e "${RED}Failed to start predictor service${NC}"
    exit 1
fi

# Step 2: Start Scheduler A (for Group A)
echo ""
echo "Step 2: Starting Scheduler A (Group A)"
start_service "scheduler-a" \
    "cd $PROJECT_ROOT/scheduler && PREDICTOR_URL=http://localhost:$PREDICTOR_PORT SCHEDULER_PORT=$SCHEDULER_A_PORT SCHEDULER_LOG_DIR=$SCRIPT_DIR/logs/scheduler-a SCHEDULER_LOGURU_LEVEL=\"INFO\" uv run python -m src.cli start --port $SCHEDULER_A_PORT" \
    "$SCHEDULER_A_PORT"

# Wait for scheduler A to be ready
if ! check_health "http://localhost:$SCHEDULER_A_PORT"; then
    echo -e "${RED}Failed to start scheduler A service${NC}"
    exit 1
fi

# Step 3: Start Scheduler B (for Group B)
echo ""
echo "Step 3: Starting Scheduler B (Group B)"
start_service "scheduler-b" \
    "cd $PROJECT_ROOT/scheduler && PREDICTOR_URL=http://localhost:$PREDICTOR_PORT SCHEDULER_PORT=$SCHEDULER_B_PORT SCHEDULER_LOG_DIR=$SCRIPT_DIR/logs/scheduler-b SCHEDULER_LOGURU_LEVEL=\"INFO\" uv run python -m src.cli start --port $SCHEDULER_B_PORT" \
    "$SCHEDULER_B_PORT"

# Wait for scheduler B to be ready
if ! check_health "http://localhost:$SCHEDULER_B_PORT"; then
    echo -e "${RED}Failed to start scheduler B service${NC}"
    exit 1
fi

# Step 4: Build sleep-model Docker image (if not already built)
echo ""
echo "Step 4: Building sleep-model Docker image"
if ! docker images | grep -q "sleep_model"; then
    echo "Building sleep_model image..."
    cd $PROJECT_ROOT/instance
    chmod +x build_sleep_model.sh
    ./build_sleep_model.sh
else
    echo -e "${GREEN}sleep_model image already exists${NC}"
fi

# Step 5: Start Group A Instance Services (parallel)
echo ""
echo "Step 5: Starting Group A Instance Services ($N1 instances, parallel)"
pids=()
for i in $(seq 0 $((N1 - 1))); do
    instance_id="instance-$(printf '%03d' $i)"
    instance_port=$((INSTANCE_GROUP_A_START_PORT + i))

    start_service "$instance_id" \
        "cd $PROJECT_ROOT/instance && SCHEDULER_URL=http://localhost:$SCHEDULER_A_PORT INSTANCE_ID=$instance_id INSTANCE_PORT=$instance_port INSTANCE_LOG_DIR=$SCRIPT_DIR/logs/instance_$instance_port uv run python -m src.cli start --port $instance_port" \
        "$instance_port" &
    pids+=($!)
done

# Wait for all instance services to start
for pid in "${pids[@]}"; do
    wait $pid
done
echo -e "${GREEN}All Group A instance services started${NC}"

# Health check all Group A instances (parallel)
echo "Checking health of Group A instances..."
pids=()
for i in $(seq 0 $((N1 - 1))); do
    instance_port=$((INSTANCE_GROUP_A_START_PORT + i))
    (
        if ! check_health "http://localhost:$instance_port"; then
            echo -e "${RED}Failed health check for instance on port $instance_port${NC}"
            exit 1
        fi
    ) &
    pids+=($!)
done

# Wait for all health checks
failed=0
for pid in "${pids[@]}"; do
    if ! wait $pid; then
        failed=1
    fi
done

if [ $failed -eq 1 ]; then
    echo -e "${RED}Some Group A instances failed to start${NC}"
    exit 1
fi
echo -e "${GREEN}All Group A instances healthy${NC}"

# Step 6: Start Group B Instance Services (parallel)
echo ""
echo "Step 6: Starting Group B Instance Services ($N2 instances, parallel)"
pids=()
for i in $(seq 0 $((N2 - 1))); do
    global_index=$((N1 + i))
    instance_id="instance-$(printf '%03d' $global_index)"
    instance_port=$((INSTANCE_GROUP_B_START_PORT + i))

    start_service "$instance_id" \
        "cd $PROJECT_ROOT/instance && SCHEDULER_URL=http://localhost:$SCHEDULER_B_PORT INSTANCE_ID=$instance_id INSTANCE_PORT=$instance_port INSTANCE_LOG_DIR=$SCRIPT_DIR/logs/instance_$instance_port uv run python -m src.cli start --port $instance_port" \
        "$instance_port" &
    pids+=($!)
done

# Wait for all instance services to start
for pid in "${pids[@]}"; do
    wait $pid
done
echo -e "${GREEN}All Group B instance services started${NC}"

# Health check all Group B instances (parallel)
echo "Checking health of Group B instances..."
pids=()
for i in $(seq 0 $((N2 - 1))); do
    instance_port=$((INSTANCE_GROUP_B_START_PORT + i))
    (
        if ! check_health "http://localhost:$instance_port"; then
            echo -e "${RED}Failed health check for instance on port $instance_port${NC}"
            exit 1
        fi
    ) &
    pids+=($!)
done

# Wait for all health checks
failed=0
for pid in "${pids[@]}"; do
    if ! wait $pid; then
        failed=1
    fi
done

if [ $failed -eq 1 ]; then
    echo -e "${RED}Some Group B instances failed to start${NC}"
    exit 1
fi
echo -e "${GREEN}All Group B instances healthy${NC}"

# Step 7: Start sleep-model on Group A instances
echo ""
echo "Step 7: Starting sleep-model on Group A instances (parallel)"
pids=()
for i in $(seq 0 $((N1 - 1))); do
    instance_port=$((INSTANCE_GROUP_A_START_PORT + i))
    instance_id="instance-$(printf '%03d' $i)"

    echo "Starting model on $instance_id..."
    (
        response=$(curl -s -X POST "http://localhost:$instance_port/model/start" \
            -H "Content-Type: application/json" \
            -d "{\"model_id\": \"$MODEL_ID\", \"parameters\": {}}")

        if echo "$response" | grep -q "success\|started"; then
            echo -e "$instance_id: ${GREEN}OK${NC}"
        else
            echo -e "$instance_id: ${RED}FAILED${NC}"
            echo "Response: $response"
        fi
    ) &
    pids+=($!)
done

# Wait for all parallel model starts to complete
for pid in "${pids[@]}"; do
    wait $pid
done
echo -e "${GREEN}Group A model starts completed${NC}"

# Step 8: Start sleep-model on Group B instances
echo ""
echo "Step 8: Starting sleep-model on Group B instances (parallel)"
pids=()
for i in $(seq 0 $((N2 - 1))); do
    global_index=$((N1 + i))
    instance_port=$((INSTANCE_GROUP_B_START_PORT + i))
    instance_id="instance-$(printf '%03d' $global_index)"

    echo "Starting model on $instance_id..."
    (
        response=$(curl -s -X POST "http://localhost:$instance_port/model/start" \
            -H "Content-Type: application/json" \
            -d "{\"model_id\": \"$MODEL_ID\", \"parameters\": {}}")

        if echo "$response" | grep -q "success\|started"; then
            echo -e "$instance_id: ${GREEN}OK${NC}"
        else
            echo -e "$instance_id: ${RED}FAILED${NC}"
            echo "Response: $response"
        fi
    ) &
    pids+=($!)
done

# Wait for all parallel model starts to complete
for pid in "${pids[@]}"; do
    wait $pid
done
echo -e "${GREEN}Group B model starts completed${NC}"

# Final status
echo ""
echo "========================================="
echo -e "${GREEN}All services started successfully!${NC}"
echo "========================================="
echo ""
echo "Service Status:"
echo "  Predictor:    http://localhost:$PREDICTOR_PORT"
echo "  Scheduler A:  http://localhost:$SCHEDULER_A_PORT (Group A: $N1 instances)"
echo "  Scheduler B:  http://localhost:$SCHEDULER_B_PORT (Group B: $N2 instances)"
echo ""
echo "Instance Groups:"
echo "  Group A: $N1 instances on ports $INSTANCE_GROUP_A_START_PORT-$((INSTANCE_GROUP_A_START_PORT + N1 - 1)) (registers to Scheduler A)"
echo "  Group B: $N2 instances on ports $INSTANCE_GROUP_B_START_PORT-$((INSTANCE_GROUP_B_START_PORT + N2 - 1)) (registers to Scheduler B)"
echo ""
echo "Logs are available in: $LOG_DIR/"
echo ""
echo "To stop all services, run: ./stop_all_services.sh"
echo ""
