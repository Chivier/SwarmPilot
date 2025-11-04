#!/bin/bash

set -e

# Get the project root directory (2 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration for Experiment 08
PREDICTOR_PORT=8101
SCHEDULER_A_PORT=8100
SCHEDULER_B_PORT=8200
INSTANCE_START_PORT=8210
TOTAL_INSTANCES=16
MODEL_ID="sleep_model"

# Phase 1 initial distribution (ratio 1:3 for n=3)
PHASE1_A_INSTANCES=4   # instance-000 to instance-003 on Scheduler A
PHASE1_B_INSTANCES=12  # instance-004 to instance-015 on Scheduler B

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
    local port=$3
    local log_file="$LOG_DIR/${name}.log"
    local pid_file="$LOG_DIR/${name}.pid"

    echo -e "${YELLOW}Starting $name...${NC}"

    # Start the service in background
    nohup bash -c "$command" > "$log_file" 2>&1 &

    # Wait a moment for the process to start
    sleep 2

    # Find the actual Python process PID by port number
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
echo "Starting Experiment 08 Services"
echo "========================================="
echo -e "${BLUE}Configuration:${NC}"
echo "  Phase 1 Distribution:"
echo "    Scheduler A: $PHASE1_A_INSTANCES instances (ports $INSTANCE_START_PORT-$((INSTANCE_START_PORT + PHASE1_A_INSTANCES - 1)))"
echo "    Scheduler B: $PHASE1_B_INSTANCES instances (ports $((INSTANCE_START_PORT + PHASE1_A_INSTANCES))-$((INSTANCE_START_PORT + TOTAL_INSTANCES - 1)))"
echo "  Total: $TOTAL_INSTANCES instances"
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

# Step 2: Start Scheduler A
echo ""
echo "Step 2: Starting Scheduler A"
start_service "scheduler-a" \
    "cd $PROJECT_ROOT/scheduler && PREDICTOR_URL=http://localhost:$PREDICTOR_PORT SCHEDULER_PORT=$SCHEDULER_A_PORT SCHEDULER_LOG_DIR=$SCRIPT_DIR/logs/scheduler-a SCHEDULER_LOGURU_LEVEL=\"INFO\" uv run python -m src.cli start --port $SCHEDULER_A_PORT" \
    "$SCHEDULER_A_PORT"

# Wait for scheduler A to be ready
if ! check_health "http://localhost:$SCHEDULER_A_PORT"; then
    echo -e "${RED}Failed to start scheduler A service${NC}"
    exit 1
fi

# Step 3: Start Scheduler B
echo ""
echo "Step 3: Starting Scheduler B"
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

# Step 5: Start instances for Scheduler A (Phase 1: 4 instances, parallel)
echo ""
echo "Step 5: Starting instances for Scheduler A ($PHASE1_A_INSTANCES instances, parallel)"
pids=()
for i in $(seq 0 $((PHASE1_A_INSTANCES - 1))); do
    instance_id="instance-$(printf '%03d' $i)"
    instance_port=$((INSTANCE_START_PORT + i))

    start_service "$instance_id" \
        "cd $PROJECT_ROOT/instance && SCHEDULER_URL=http://localhost:$SCHEDULER_A_PORT INSTANCE_ID=$instance_id INSTANCE_PORT=$instance_port INSTANCE_LOG_DIR=$SCRIPT_DIR/logs/instance_$instance_port uv run python -m src.cli start --port $instance_port" \
        "$instance_port" &
    pids+=($!)
done

# Wait for all Scheduler A instance services to start
for pid in "${pids[@]}"; do
    wait $pid
done
echo -e "${GREEN}All Scheduler A instance services started${NC}"

# Health check all Scheduler A instances (parallel)
echo "Checking health of Scheduler A instances..."
pids=()
for i in $(seq 0 $((PHASE1_A_INSTANCES - 1))); do
    instance_port=$((INSTANCE_START_PORT + i))
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
    echo -e "${RED}Some Scheduler A instances failed to start${NC}"
    exit 1
fi
echo -e "${GREEN}All Scheduler A instances healthy${NC}"

# Step 6: Start instances for Scheduler B (Phase 1: 12 instances, parallel)
echo ""
echo "Step 6: Starting instances for Scheduler B ($PHASE1_B_INSTANCES instances, parallel)"
pids=()
for i in $(seq 0 $((PHASE1_B_INSTANCES - 1))); do
    global_index=$((PHASE1_A_INSTANCES + i))
    instance_id="instance-$(printf '%03d' $global_index)"
    instance_port=$((INSTANCE_START_PORT + global_index))

    start_service "$instance_id" \
        "cd $PROJECT_ROOT/instance && SCHEDULER_URL=http://localhost:$SCHEDULER_B_PORT INSTANCE_ID=$instance_id INSTANCE_PORT=$instance_port INSTANCE_LOG_DIR=$SCRIPT_DIR/logs/instance_$instance_port uv run python -m src.cli start --port $instance_port" \
        "$instance_port" &
    pids+=($!)
done

# Wait for all Scheduler B instance services to start
for pid in "${pids[@]}"; do
    wait $pid
done
echo -e "${GREEN}All Scheduler B instance services started${NC}"

# Health check all Scheduler B instances (parallel)
echo "Checking health of Scheduler B instances..."
pids=()
for i in $(seq 0 $((PHASE1_B_INSTANCES - 1))); do
    global_index=$((PHASE1_A_INSTANCES + i))
    instance_port=$((INSTANCE_START_PORT + global_index))
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
    echo -e "${RED}Some Scheduler B instances failed to start${NC}"
    exit 1
fi
echo -e "${GREEN}All Scheduler B instances healthy${NC}"

# Step 7: Start sleep-model on all instances (parallel)
echo ""
echo "Step 7: Starting sleep-model on all instances (parallel)"
pids=()
for i in $(seq 0 $((TOTAL_INSTANCES - 1))); do
    instance_id="instance-$(printf '%03d' $i)"
    instance_port=$((INSTANCE_START_PORT + i))

    echo "Starting model on $instance_id (port $instance_port)..."
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
echo -e "${GREEN}All model starts completed${NC}"

# Final status
echo ""
echo "========================================="
echo -e "${GREEN}All services started successfully!${NC}"
echo "========================================="
echo ""
echo "Service Status:"
echo "  Predictor:    http://localhost:$PREDICTOR_PORT"
echo "  Scheduler A:  http://localhost:$SCHEDULER_A_PORT ($PHASE1_A_INSTANCES instances)"
echo "  Scheduler B:  http://localhost:$SCHEDULER_B_PORT ($PHASE1_B_INSTANCES instances)"
echo ""
echo "Phase 1 Instance Distribution:"
echo "  Scheduler A: instances 000-003 (ports $INSTANCE_START_PORT-$((INSTANCE_START_PORT + 3)))"
echo "  Scheduler B: instances 004-015 (ports $((INSTANCE_START_PORT + 4))-$((INSTANCE_START_PORT + 15)))"
echo ""
echo "Logs are available in: $LOG_DIR/"
echo ""
echo "To stop all services, run: ./stop_all_services.sh"
echo ""
