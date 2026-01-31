#!/bin/bash
# PyLet Benchmark - Deploy Sleep Model Instances (Direct Registration)
# Usage: ./examples/pylet_benchmark/deploy_model.sh
#
# Deploys sleep model instances directly:
# - Starts instance processes listening on sequential ports
# - Registers each instance with the scheduler via /v1/instance/register API
# - No planner needed (simplest setup)
#
# Default distribution:
#   sleep_model_a: 4 instances
#   sleep_model_b: 3 instances
#   sleep_model_c: 3 instances
#   Total: 10 instances
#
# PYLET-025: Direct Scheduler Registration Example

set -e

# Configuration
SCHEDULER_PORT=${SCHEDULER_PORT:-8000}
INSTANCE_PORT_START=8100
LOG_DIR="/tmp/pylet_benchmark"

# Model distribution
declare -A MODEL_DISTRIBUTION=(
    [sleep_model_a]=4
    [sleep_model_b]=3
    [sleep_model_c]=3
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get project root (script is in examples/pylet_benchmark/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     PyLet Benchmark - Deploy Model Instances          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Calculate totals
TOTAL_INSTANCES=0
for model_id in "${!MODEL_DISTRIBUTION[@]}"; do
    count=${MODEL_DISTRIBUTION[$model_id]}
    TOTAL_INSTANCES=$((TOTAL_INSTANCES + count))
done

echo -e "${CYAN}Configuration:${NC}"
echo "  Scheduler:     http://localhost:$SCHEDULER_PORT"
echo "  Total Instances: $TOTAL_INSTANCES"
echo "  Instance Port Start: $INSTANCE_PORT_START"
echo ""
echo -e "${CYAN}Model Distribution:${NC}"
for model_id in "${!MODEL_DISTRIBUTION[@]}"; do
    count=${MODEL_DISTRIBUTION[$model_id]}
    echo "  $model_id: $count instances"
done
echo ""

# Check if Scheduler is running
echo "Checking Scheduler status..."
if ! curl -s "http://localhost:$SCHEDULER_PORT/v1/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Scheduler not responding on port $SCHEDULER_PORT${NC}"
    echo "Start services first: ./examples/pylet_benchmark/start_cluster.sh"
    exit 1
fi
echo -e "${GREEN}✓ Scheduler is healthy${NC}"
echo ""

# Track deployed instances
instance_index=0
deployed_instances=0

# Deploy instances for each model
for model_id in "${!MODEL_DISTRIBUTION[@]}"; do
    count=${MODEL_DISTRIBUTION[$model_id]}

    echo -e "${BLUE}Deploying $count instances of $model_id...${NC}"

    for i in $(seq 1 $count); do
        instance_id="${model_id}-$(printf "%03d" $((i-1)))"
        instance_port=$((INSTANCE_PORT_START + instance_index))

        # Prepare log file
        instance_log="$LOG_DIR/instance_${instance_id}.log"
        mkdir -p "$(dirname "$instance_log")"

        # Start instance process
        echo -n "  Starting $instance_id on port $instance_port..."

        cd "$PROJECT_ROOT"
        PYTHONUNBUFFERED=1 \
            PORT=$instance_port \
            MODEL_ID=$model_id \
            INSTANCE_ID=$instance_id \
            SCHEDULER_URL="" \
            LOG_LEVEL="INFO" \
            uv run python "$PROJECT_ROOT/examples/pylet_benchmark/pylet_sleep_model.py" > "$instance_log" 2>&1 &

        instance_pid=$!
        echo $instance_pid > "$LOG_DIR/instance_${instance_id}.pid"

        # Brief wait for startup
        sleep 1

        # Check if process is still alive
        if ! kill -0 $instance_pid 2>/dev/null; then
            echo -e " ${RED}FAILED${NC}"
            echo "    Process exited. Check log: $instance_log"
            exit 1
        fi

        # Wait for instance to be ready
        echo -n " waiting for health..."
        for attempt in {1..10}; do
            if curl -s "http://localhost:$instance_port/health" > /dev/null 2>&1; then
                echo -n " registering..."
                break
            fi
            if [ $attempt -eq 10 ]; then
                echo -e " ${RED}TIMEOUT${NC}"
                kill -9 $instance_pid 2>/dev/null || true
                exit 1
            fi
            sleep 0.5
        done

        # Register instance with scheduler
        registration_data=$(cat <<EOF
{
    "instance_id": "$instance_id",
    "model_id": "$model_id",
    "endpoint": "http://localhost:$instance_port",
    "platform_info": {
        "software_name": "python",
        "software_version": "3.11",
        "hardware_name": "cpu"
    }
}
EOF
)

        registration_response=$(curl -s -X POST \
            "http://localhost:$SCHEDULER_PORT/v1/instance/register" \
            -H "Content-Type: application/json" \
            -d "$registration_data")

        if echo "$registration_response" | grep -q '"success"\s*:\s*true' 2>/dev/null; then
            echo -e " ${GREEN}OK${NC}"
            deployed_instances=$((deployed_instances + 1))
        else
            echo -e " ${RED}REGISTRATION FAILED${NC}"
            echo "    Response: $registration_response"
            kill -9 $instance_pid 2>/dev/null || true
            exit 1
        fi

        instance_index=$((instance_index + 1))
    done
done

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          Deployment Complete!                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Instances deployed:${NC} $deployed_instances / $TOTAL_INSTANCES"
echo ""

# Verify deployment by checking scheduler
echo "Verifying instance registration..."
instances_response=$(curl -s "http://localhost:$SCHEDULER_PORT/v1/instance/list" 2>/dev/null)
registered_count=$(echo "$instances_response" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('instances', [])))" 2>/dev/null || echo "0")
echo -e "${GREEN}✓ Scheduler reports $registered_count instances registered${NC}"

echo ""
echo -e "${YELLOW}Next step:${NC}"
echo "  Generate traffic: python examples/pylet_benchmark/generate_workload.py"
echo ""
