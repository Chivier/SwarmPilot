#!/bin/bash

set -e

# ============================================================================
# Type3 Text2Image+Video Workflow - Simulation Mode Service Launcher
# ============================================================================
# This script starts all services required for the Type3 workflow simulation:
#   - Predictor (port 8101)
#   - Planner (port 8202)
#   - Scheduler A (LLM, port 8100)
#   - Scheduler C (FLUX, port 8300)
#   - Scheduler B (T2VID, port 8200)
#   - Instance Group A (sleep_model_a, ports 8210-82xx)
#   - Instance Group C (sleep_model_c, ports 8400-84xx)
#   - Instance Group B (sleep_model_b, ports 8500-85xx)
#
# Usage:
#   ./start_type3_sim_services.sh [N1] [N2] [N3]
#   N1: Number of Group A (LLM) instances (default: 4)
#   N2: Number of Group C (FLUX) instances (default: 2)
#   N3: Number of Group B (T2VID) instances (default: 2)
#
# Example:
#   ./start_type3_sim_services.sh 4 2 2
#   N1=6 N2=3 N3=3 ./start_type3_sim_services.sh
# ============================================================================

# Get directory paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCHMARK_DIR="$(cd "$EXPERIMENT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$BENCHMARK_DIR/../.." && pwd)"

# Configuration
PREDICTOR_PORT=8101
PLANNER_PORT=8202
SCHEDULER_A_PORT=8100   # LLM
SCHEDULER_C_PORT=8300   # FLUX
SCHEDULER_B_PORT=8200   # T2VID

INSTANCE_GROUP_A_START_PORT=8210  # Group A (LLM): 8210-82xx
INSTANCE_GROUP_C_START_PORT=8400  # Group C (FLUX): 8400-84xx
INSTANCE_GROUP_B_START_PORT=8500  # Group B (T2VID): 8500-85xx

N1=${N1:-4}  # Default: 4 LLM instances
N2=${N2:-2}  # Default: 2 FLUX instances
N3=${N3:-2}  # Default: 2 T2VID instances

MODEL_ID_A=${MODEL_ID_A:-sleep_model_a}
MODEL_ID_C=${MODEL_ID_C:-sleep_model_c}
MODEL_ID_B=${MODEL_ID_B:-sleep_model_b}

AUTO_OPTIMIZE_ENABLED=${AUTO_OPTIMIZE_ENABLED:-True}

# Enable Python JIT
export PYTHON_JIT=1

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse command-line arguments
usage() {
    echo "Usage: $0 [N1] [N2] [N3]"
    echo "  N1: Number of Group A (LLM) instances (default: 4)"
    echo "  N2: Number of Group C (FLUX) instances (default: 2)"
    echo "  N3: Number of Group B (T2VID) instances (default: 2)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Use defaults (N1=4, N2=2, N3=2)"
    echo "  $0 6 3 3                     # Custom instance counts"
    echo "  N1=8 N2=4 N3=4 $0            # Using environment variables"
    exit 1
}

if [ $# -ge 1 ]; then
    if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
        usage
    fi
    N1=$1
fi
if [ $# -ge 2 ]; then N2=$2; fi
if [ $# -ge 3 ]; then N3=$3; fi

# Validate inputs
for var in N1 N2 N3; do
    val=${!var}
    if ! [[ "$val" =~ ^[0-9]+$ ]] || [ "$val" -lt 1 ]; then
        echo -e "${RED}Error: $var must be a positive integer${NC}"
        usage
    fi
done

# Log directory
LOG_DIR="$EXPERIMENT_DIR/logs"
mkdir -p "$LOG_DIR"

# Helper: Check health
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

# Helper: Start service
start_service() {
    local name=$1
    local command=$2
    local port=$3
    local log_file="$LOG_DIR/${name}.log"
    local pid_file="$LOG_DIR/${name}.pid"

    echo -e "${YELLOW}Starting $name...${NC}"
    nohup bash -c "$command" > "$log_file" 2>&1 &
    sleep 3

    local actual_pid=$(pgrep -f "python.*--port $port" | head -1)
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
echo "Type3 Text2Image+Video Simulation"
echo "========================================="
echo -e "${BLUE}Configuration:${NC}"
echo "  Group A (LLM):   $N1 instances (Model: $MODEL_ID_A)"
echo "  Group C (FLUX):  $N2 instances (Model: $MODEL_ID_C)"
echo "  Group B (T2VID): $N3 instances (Model: $MODEL_ID_B)"
echo "  Total: $((N1 + N2 + N3)) instances"
echo "  Auto-Optimize: $AUTO_OPTIMIZE_ENABLED"
echo "  Project Root: $PROJECT_ROOT"
echo "========================================="

# Step 1: Start Predictor
echo ""
echo "Step 1: Starting Predictor Service"
start_service "predictor" \
    "cd $PROJECT_ROOT/predictor && PREDICTOR_PORT=$PREDICTOR_PORT PREDICTOR_LOG_DIR=$LOG_DIR/predictor uv run python -m src.cli start --port $PREDICTOR_PORT --log-level INFO" \
    "$PREDICTOR_PORT"
check_health "http://localhost:$PREDICTOR_PORT" || exit 1

# Step 2: Start Planner
echo ""
echo "Step 2: Starting Planner Service"
start_service "planner" \
    "cd $PROJECT_ROOT/planner && AUTO_OPTIMIZE_ENABLED=$AUTO_OPTIMIZE_ENABLED AUTO_OPTIMIZE_INTERVAL=150 PLANNER_LOG_DIR=$LOG_DIR/planner uv run python -m uvicorn src.api:app --port $PLANNER_PORT" \
    "$PLANNER_PORT"
check_health "http://localhost:$PLANNER_PORT" || exit 1

# Step 3: Start Scheduler A (LLM)
echo ""
echo "Step 3: Starting Scheduler A (LLM)"
start_service "scheduler-a" \
    "cd $PROJECT_ROOT/scheduler && PREDICTOR_URL=http://localhost:$PREDICTOR_PORT PLANNER_URL=http://localhost:$PLANNER_PORT SCHEDULER_PORT=$SCHEDULER_A_PORT SCHEDULER_LOG_DIR=$LOG_DIR/scheduler-a SCHEDULER_AUTO_REPORT=5 uv run python -m src.cli start --port $SCHEDULER_A_PORT" \
    "$SCHEDULER_A_PORT"
check_health "http://localhost:$SCHEDULER_A_PORT" || exit 1

# Step 4: Start Scheduler C (FLUX)
echo ""
echo "Step 4: Starting Scheduler C (FLUX)"
start_service "scheduler-c" \
    "cd $PROJECT_ROOT/scheduler && PREDICTOR_URL=http://localhost:$PREDICTOR_PORT PLANNER_URL=http://localhost:$PLANNER_PORT SCHEDULER_PORT=$SCHEDULER_C_PORT SCHEDULER_LOG_DIR=$LOG_DIR/scheduler-c SCHEDULER_AUTO_REPORT=5 uv run python -m src.cli start --port $SCHEDULER_C_PORT" \
    "$SCHEDULER_C_PORT"
check_health "http://localhost:$SCHEDULER_C_PORT" || exit 1

# Step 5: Start Scheduler B (T2VID)
echo ""
echo "Step 5: Starting Scheduler B (T2VID)"
start_service "scheduler-b" \
    "cd $PROJECT_ROOT/scheduler && PREDICTOR_URL=http://localhost:$PREDICTOR_PORT PLANNER_URL=http://localhost:$PLANNER_PORT SCHEDULER_PORT=$SCHEDULER_B_PORT SCHEDULER_LOG_DIR=$LOG_DIR/scheduler-b SCHEDULER_AUTO_REPORT=5 uv run python -m src.cli start --port $SCHEDULER_B_PORT" \
    "$SCHEDULER_B_PORT"
check_health "http://localhost:$SCHEDULER_B_PORT" || exit 1

# Step 6: Start Group A Instances (LLM)
echo ""
echo "Step 6: Starting $N1 instances for Group A (LLM, ports $INSTANCE_GROUP_A_START_PORT-$((INSTANCE_GROUP_A_START_PORT + N1 - 1)))"
instance_pids=()
for i in $(seq 0 $((N1 - 1))); do
    instance_port=$((INSTANCE_GROUP_A_START_PORT + i))
    instance_id="instance-a-$(printf '%03d' $i)"
    scheduler_port=$( (( i % 2 == 0 )) && echo $SCHEDULER_A_PORT || echo $PLANNER_PORT )
    (
        start_service "$instance_id" \
            "cd $PROJECT_ROOT/instance && SCHEDULER_URL=http://localhost:$scheduler_port INSTANCE_ID=$instance_id INSTANCE_PORT=$instance_port INSTANCE_LOG_DIR=$LOG_DIR/$instance_id uv run python -m src.cli start --port $instance_port --docker" \
            "$instance_port"
    ) &
    instance_pids+=($!)
done
for pid in "${instance_pids[@]}"; do wait "$pid" || true; done

# Health check Group A
echo -n "Health checking Group A instances..."
for i in $(seq 0 $((N1 - 1))); do
    check_health "http://localhost:$((INSTANCE_GROUP_A_START_PORT + i))" > /dev/null 2>&1 &
done
wait
echo -e " ${GREEN}OK${NC}"

# Step 7: Start Group C Instances (FLUX)
echo ""
echo "Step 7: Starting $N2 instances for Group C (FLUX, ports $INSTANCE_GROUP_C_START_PORT-$((INSTANCE_GROUP_C_START_PORT + N2 - 1)))"
instance_pids=()
for i in $(seq 0 $((N2 - 1))); do
    instance_port=$((INSTANCE_GROUP_C_START_PORT + i))
    instance_id="instance-c-$(printf '%03d' $i)"
    scheduler_port=$( (( i % 2 == 0 )) && echo $SCHEDULER_C_PORT || echo $PLANNER_PORT )
    (
        start_service "$instance_id" \
            "cd $PROJECT_ROOT/instance && SCHEDULER_URL=http://localhost:$scheduler_port INSTANCE_ID=$instance_id INSTANCE_PORT=$instance_port INSTANCE_LOG_DIR=$LOG_DIR/$instance_id uv run python -m src.cli start --port $instance_port --docker" \
            "$instance_port"
    ) &
    instance_pids+=($!)
done
for pid in "${instance_pids[@]}"; do wait "$pid" || true; done

# Health check Group C
echo -n "Health checking Group C instances..."
for i in $(seq 0 $((N2 - 1))); do
    check_health "http://localhost:$((INSTANCE_GROUP_C_START_PORT + i))" > /dev/null 2>&1 &
done
wait
echo -e " ${GREEN}OK${NC}"

# Step 8: Start Group B Instances (T2VID)
echo ""
echo "Step 8: Starting $N3 instances for Group B (T2VID, ports $INSTANCE_GROUP_B_START_PORT-$((INSTANCE_GROUP_B_START_PORT + N3 - 1)))"
instance_pids=()
for i in $(seq 0 $((N3 - 1))); do
    instance_port=$((INSTANCE_GROUP_B_START_PORT + i))
    instance_id="instance-b-$(printf '%03d' $i)"
    scheduler_port=$( (( i % 2 == 0 )) && echo $SCHEDULER_B_PORT || echo $PLANNER_PORT )
    (
        start_service "$instance_id" \
            "cd $PROJECT_ROOT/instance && SCHEDULER_URL=http://localhost:$scheduler_port INSTANCE_ID=$instance_id INSTANCE_PORT=$instance_port INSTANCE_LOG_DIR=$LOG_DIR/$instance_id uv run python -m src.cli start --port $instance_port --docker" \
            "$instance_port"
    ) &
    instance_pids+=($!)
done
for pid in "${instance_pids[@]}"; do wait "$pid" || true; done

# Health check Group B
echo -n "Health checking Group B instances..."
for i in $(seq 0 $((N3 - 1))); do
    check_health "http://localhost:$((INSTANCE_GROUP_B_START_PORT + i))" > /dev/null 2>&1 &
done
wait
echo -e " ${GREEN}OK${NC}"

# Step 9: Deploy models
echo ""
echo "Step 9: Deploying models locally"
"$SCRIPT_DIR/deploy_type3_models_local.sh" \
    --scheduler-a-url "http://localhost:$SCHEDULER_A_PORT" \
    --scheduler-c-url "http://localhost:$SCHEDULER_C_PORT" \
    --scheduler-b-url "http://localhost:$SCHEDULER_B_PORT" \
    --planner-url "http://localhost:$PLANNER_PORT" \
    --model-id-a "$MODEL_ID_A" \
    --model-id-c "$MODEL_ID_C" \
    --model-id-b "$MODEL_ID_B" \
    --n1 "$N1" \
    --n2 "$N2" \
    --n3 "$N3" \
    --port-a-start "$INSTANCE_GROUP_A_START_PORT" \
    --port-c-start "$INSTANCE_GROUP_C_START_PORT" \
    --port-b-start "$INSTANCE_GROUP_B_START_PORT"

# Summary
echo ""
echo "========================================="
echo -e "${GREEN}All services started successfully!${NC}"
echo "========================================="
echo -e "${BLUE}Service URLs:${NC}"
echo "  Predictor:   http://localhost:$PREDICTOR_PORT"
echo "  Planner:     http://localhost:$PLANNER_PORT"
echo "  Scheduler A: http://localhost:$SCHEDULER_A_PORT (LLM, $N1 instances)"
echo "  Scheduler C: http://localhost:$SCHEDULER_C_PORT (FLUX, $N2 instances)"
echo "  Scheduler B: http://localhost:$SCHEDULER_B_PORT (T2VID, $N3 instances)"
echo ""
echo -e "${BLUE}Log Directory:${NC} $LOG_DIR"
echo ""
echo "Use './scripts/stop_type3_services.sh' to shutdown all services."
echo "========================================="
