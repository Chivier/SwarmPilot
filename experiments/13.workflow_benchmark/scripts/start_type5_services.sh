#!/bin/bash
#
# Start services for Type5 OOD Recovery Experiment (Simulation Mode)
#
# This is a simplified setup that only starts:
#   - Predictor service (for runtime prediction)
#   - Scheduler A (single scheduler for all tasks)
#   - Group A instances (single model group)
#
# No planner is needed since OOD experiment tests predictor retraining behavior.
#
# Usage:
#   ./start_type5_services.sh           # Default: 4 instances
#   ./start_type5_services.sh 8         # 8 instances
#   N1=10 ./start_type5_services.sh     # Using environment variable

set -e

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

# Configuration (can be overridden via environment variables)
PREDICTOR_PORT=8101
SCHEDULER_PORT=8100
INSTANCE_START_PORT=8210
N1=${N1:-4}  # Default: 4 instances
MODEL_ID=${MODEL_ID:-sleep_model_a}

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command-line arguments
usage() {
    echo "Usage: $0 [N1] [MODEL_ID]"
    echo "  N1: Number of instances (default: 4)"
    echo "  MODEL_ID: Model ID (default: sleep_model_a)"
    echo ""
    echo "Examples:"
    echo "  $0                              # Use defaults (N1=4)"
    echo "  $0 8                            # 8 instances"
    echo "  N1=10 $0                        # Using environment variables"
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
    MODEL_ID=$2
fi

# Validate inputs
if ! [[ "$N1" =~ ^[0-9]+$ ]] || [ "$N1" -lt 1 ]; then
    echo -e "${RED}Error: N1 must be a positive integer${NC}"
    usage
fi

# Log directory
LOG_DIR="$EXPERIMENT_DIR/logs_type5"
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
echo "Type5 OOD Recovery Experiment - Services"
echo "========================================="
echo -e "${BLUE}Configuration:${NC}"
echo "  Instances: $N1 (Model: $MODEL_ID)"
echo "  Scheduler: http://localhost:$SCHEDULER_PORT"
echo "  Predictor: http://localhost:$PREDICTOR_PORT"
echo "  Project Root: $PROJECT_ROOT"
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

# Step 2: Start Scheduler (single scheduler for OOD experiment)
echo ""
echo "Step 2: Starting Scheduler"
start_service "scheduler" \
    "cd $PROJECT_ROOT/scheduler && PREDICTOR_URL=http://localhost:$PREDICTOR_PORT SCHEDULER_PORT=$SCHEDULER_PORT SCHEDULER_LOG_DIR=$LOG_DIR/scheduler SCHEDULER_AUTO_REPORT=5 uv run python -m src.cli start --port $SCHEDULER_PORT" \
    "$SCHEDULER_PORT"

# Wait for Scheduler to be ready
if ! check_health "http://localhost:$SCHEDULER_PORT"; then
    echo -e "${RED}Failed to start Scheduler${NC}"
    exit 1
fi

# Step 3: Start Instances
echo ""
echo "Step 3: Starting $N1 instances (ports $INSTANCE_START_PORT-$((INSTANCE_START_PORT + N1 - 1)))"

instance_pids=()
for i in $(seq 0 $((N1 - 1))); do
    instance_port=$((INSTANCE_START_PORT + i))
    instance_id="instance-$(printf '%03d' $i)"

    (
        start_service "$instance_id" \
            "cd $PROJECT_ROOT/instance && SCHEDULER_URL=http://localhost:$SCHEDULER_PORT INSTANCE_ID=$instance_id INSTANCE_PORT=$instance_port INSTANCE_LOG_DIR=$LOG_DIR/$instance_id uv run python -m src.cli start --port $instance_port --docker" \
            "$instance_port"
    ) &
    instance_pids+=($!)
done

# Wait for all instances to start
for pid in "${instance_pids[@]}"; do
    wait "$pid" || true
done

# Health check for instances (parallel)
echo -n "Health checking instances..."
health_pids=()
for i in $(seq 0 $((N1 - 1))); do
    instance_port=$((INSTANCE_START_PORT + i))
    ( check_health "http://localhost:$instance_port" > /dev/null 2>&1 ) &
    health_pids+=($!)
done
for pid in "${health_pids[@]}"; do
    wait "$pid" || true
done
echo -e " ${GREEN}OK${NC}"

# Step 4: Deploy model to instances
echo ""
echo "Step 4: Deploying model to instances"

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}Warning: jq not installed, using simple JSON construction${NC}"
    USE_JQ=false
else
    USE_JQ=true
fi

# Deploy model to each instance (call /model/start on instance, not scheduler)
deploy_pids=()
for i in $(seq 0 $((N1 - 1))); do
    instance_port=$((INSTANCE_START_PORT + i))
    (
        instance_url="http://localhost:$instance_port"

        # Build JSON payload
        if [ "$USE_JQ" = true ]; then
            json_payload=$(jq -n \
                --arg model_id "$MODEL_ID" \
                --arg scheduler_url "http://localhost:$SCHEDULER_PORT" \
                '{model_id: $model_id, scheduler_url: $scheduler_url}')
        else
            json_payload="{\"model_id\":\"$MODEL_ID\",\"scheduler_url\":\"http://localhost:$SCHEDULER_PORT\"}"
        fi

        response=$(curl -s -X POST "$instance_url/model/start" \
            -H "Content-Type: application/json" \
            -d "$json_payload")

        if echo "$response" | grep -q "success\|started"; then
            echo -e "  Instance $instance_port: ${GREEN}OK${NC}"
        else
            echo -e "  Instance $instance_port: ${RED}FAILED${NC}"
            echo "    Response: $response"
        fi
    ) &
    deploy_pids+=($!)
done

# Wait for all deployments to complete
for pid in "${deploy_pids[@]}"; do
    wait "$pid" || true
done

echo -e "${GREEN}Model deployment completed${NC}"

# Summary
echo ""
echo "========================================="
echo -e "${GREEN}All services started successfully!${NC}"
echo "========================================="
echo -e "${BLUE}Service URLs:${NC}"
echo "  Predictor: http://localhost:$PREDICTOR_PORT"
echo "  Scheduler: http://localhost:$SCHEDULER_PORT"
echo "  Model: $MODEL_ID ($N1 instances)"
echo ""
echo -e "${BLUE}Log Directory:${NC} $LOG_DIR"
echo ""
echo -e "${BLUE}Run OOD experiment:${NC}"
echo "  # Recovery mode"
echo "  uv run python -m type5_ood_recovery.simulation.test_ood_sim \\"
echo "      --num-tasks 100 --qps 2.0 --scheduler-url http://localhost:$SCHEDULER_PORT"
echo ""
echo "  # Baseline mode (no recovery)"
echo "  uv run python -m type5_ood_recovery.simulation.test_ood_sim \\"
echo "      --num-tasks 100 --qps 2.0 --no-recovery --scheduler-url http://localhost:$SCHEDULER_PORT"
echo ""
echo "Use './scripts/stop_type5_services.sh' to shutdown services."
echo "========================================="
