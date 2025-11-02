#!/bin/bash

set -e

# Get the project root directory (2 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
PREDICTOR_PORT=8101
SCHEDULER_PORT=8100
INSTANCE_START_PORT=8200
NUM_INSTANCES=16
MODEL_ID="sleep_model"

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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
echo "Starting 01.quick-start-up Experiment"
echo "========================================="

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

# Step 2: Start Scheduler Service
echo ""
echo "Step 2: Starting Scheduler Service"
start_service "scheduler" \
    "cd $PROJECT_ROOT/scheduler && PREDICTOR_URL=http://localhost:$PREDICTOR_PORT SCHEDULER_PORT=$SCHEDULER_PORT SCHEDULER_LOG_DIR=$SCRIPT_DIR/logs/scheduler SCHEDULER_LOGURU_LEVEL="INFO" uv run python -m src.cli start --port $SCHEDULER_PORT" \
    "$SCHEDULER_PORT"

# Wait for scheduler to be ready
if ! check_health "http://localhost:$SCHEDULER_PORT"; then
    echo -e "${RED}Failed to start scheduler service${NC}"
    exit 1
fi

# Step 3: Build sleep-model Docker image (if not already built)
echo ""
echo "Step 3: Building sleep-model Docker image"
if ! docker images | grep -q "sleep_model"; then
    echo "Building sleep_model image..."
    cd $PROJECT_ROOT/instance
    chmod +x build_sleep_model.sh
    ./build_sleep_model.sh
else
    echo -e "${GREEN}sleep_model image already exists${NC}"
fi

# Step 4: Start Instance Services
echo ""
echo "Step 4: Starting $NUM_INSTANCES Instance Services"
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    instance_id="instance-$(printf '%03d' $i)"
    instance_port=$((INSTANCE_START_PORT + i))

    start_service "$instance_id" \
        "cd $PROJECT_ROOT/instance && SCHEDULER_URL=http://localhost:$SCHEDULER_PORT INSTANCE_ID=$instance_id INSTANCE_PORT=$instance_port INSTANCE_LOG_DIR=$SCRIPT_DIR/logs/instance_$instance_port uv run python -m src.cli start --port $instance_port" \
        "$instance_port"

    # Wait for instance to be ready
    if ! check_health "http://localhost:$instance_port"; then
        echo -e "${RED}Failed to start $instance_id${NC}"
        exit 1
    fi
done

# Step 5: Start sleep-model on each instance
echo ""
echo "Step 5: Starting sleep-model on all instances"
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    instance_port=$((INSTANCE_START_PORT + i))
    instance_id="instance-$(printf '%03d' $i)"

    echo -n "Starting model on $instance_id..."
    response=$(curl -s -X POST "http://localhost:$instance_port/model/start" \
        -H "Content-Type: application/json" \
        -d "{\"model_id\": \"$MODEL_ID\", \"parameters\": {}}")

    if echo "$response" | grep -q "success\|started"; then
        echo -e " ${GREEN}OK${NC}"
    else
        echo -e " ${RED}FAILED${NC}"
        echo "Response: $response"
    fi

    # Small delay between starts
    sleep 0.5
done


# Final status
echo ""
echo "========================================="
echo -e "${GREEN}All services started successfully!${NC}"
echo "========================================="
echo ""
echo "Service Status:"
echo "  Predictor:  http://localhost:$PREDICTOR_PORT"
echo "  Scheduler:  http://localhost:$SCHEDULER_PORT"
echo "  Instances:  $NUM_INSTANCES instances on ports $INSTANCE_START_PORT-$((INSTANCE_START_PORT + NUM_INSTANCES - 1))"
echo ""
echo "Logs are available in: $LOG_DIR/"
echo ""
echo "To stop all services, run: ./stop_all_services.sh"
echo ""
