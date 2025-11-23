#!/bin/bash

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
SCHEDULER_A_URL="http://localhost:8100"
SCHEDULER_B_URL="http://localhost:8200"
PLANNER_URL="http://localhost:8202"
MODEL_ID_A="sleep_model_a"
MODEL_ID_B="sleep_model_b"
N1=4
N2=2
PORT_A_START=8210
PORT_B_START=8300
MODEL_PATH_A=""
MODEL_PATH_B=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scheduler-a-url)
            SCHEDULER_A_URL="$2"
            shift 2
            ;;
        --scheduler-b-url)
            SCHEDULER_B_URL="$2"
            shift 2
            ;;
        --planner-url)
            PLANNER_URL="$2"
            shift 2
            ;;
        --model-id-a)
            MODEL_ID_A="$2"
            shift 2
            ;;
        --model-id-b)
            MODEL_ID_B="$2"
            shift 2
            ;;
        --n1)
            N1="$2"
            shift 2
            ;;
        --n2)
            N2="$2"
            shift 2
            ;;
        --port-a-start)
            PORT_A_START="$2"
            shift 2
            ;;
        --port-b-start)
            PORT_B_START="$2"
            shift 2
            ;;
        --model-path-a)
            MODEL_PATH_A="$2"
            shift 2
            ;;
        --model-path-b)
            MODEL_PATH_B="$2"
            shift 2
            ;;
        --model-path)
            # If both use the same path
            MODEL_PATH_A="$2"
            MODEL_PATH_B="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --scheduler-a-url URL     Scheduler A URL (default: http://localhost:8100)"
            echo "  --scheduler-b-url URL     Scheduler B URL (default: http://localhost:8200)"
            echo "  --planner-url URL         Planner URL (default: http://localhost:8202)"
            echo "  --model-id-a ID           Model ID for Group A (default: sleep_model_a)"
            echo "  --model-id-b ID           Model ID for Group B (default: sleep_model_b)"
            echo "  --n1 NUM                  Number of Group A instances (default: 4)"
            echo "  --n2 NUM                  Number of Group B instances (default: 2)"
            echo "  --port-a-start PORT       Starting port for Group A (default: 8210)"
            echo "  --port-b-start PORT       Starting port for Group B (default: 8300)"
            echo "  --model-path PATH         Set model path for both groups"
            echo "  --model-path-a PATH       Set model path for Group A only"
            echo "  --model-path-b PATH       Set model path for Group B only"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --model-id-a llm_service_small_model --model-id-b t2vid --n1 6 --n2 3"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display configuration
echo "========================================="
echo -e "${BLUE}Deployment Configuration:${NC}"
echo "========================================="
echo "Scheduler A URL: $SCHEDULER_A_URL"
echo "Scheduler B URL: $SCHEDULER_B_URL"
echo "Planner URL: $PLANNER_URL"
echo ""
echo "Group A: $MODEL_ID_A ($N1 instances, ports $PORT_A_START-$((PORT_A_START + N1 - 1)))"
if [ -n "$MODEL_PATH_A" ]; then
    echo "  Model Path: $MODEL_PATH_A"
fi
echo ""
echo "Group B: $MODEL_ID_B ($N2 instances, ports $PORT_B_START-$((PORT_B_START + N2 - 1)))"
if [ -n "$MODEL_PATH_B" ]; then
    echo "  Model Path: $MODEL_PATH_B"
fi
echo "========================================="

# Check if jq is available for JSON generation
if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: jq is required but not installed.${NC}"
    echo "Please install jq: sudo apt-get install jq (or brew install jq on macOS)"
    exit 1
fi

# Array to track background processes
pids=()

# Deploy models to Group A instances
echo ""
echo "Deploying $MODEL_ID_A to Group A instances..."

# First half of Group A instances: register to Scheduler A
for i in $(seq 0 $((N1 / 2 - 1))); do
    instance_port=$((PORT_A_START + i))
    (
        instance_url="http://localhost:$instance_port"

        # Build JSON payload
        if [ -n "$MODEL_PATH_A" ]; then
            json_payload=$(jq -n \
                --arg model_id "$MODEL_ID_A" \
                --arg scheduler_url "$SCHEDULER_A_URL" \
                --arg model_path "$MODEL_PATH_A" \
                '{model_id: $model_id, scheduler_url: $scheduler_url, parameters: {MODEL_PATH: $model_path}}')
        else
            json_payload=$(jq -n \
                --arg model_id "$MODEL_ID_A" \
                --arg scheduler_url "$SCHEDULER_A_URL" \
                '{model_id: $model_id, scheduler_url: $scheduler_url}')
        fi

        response=$(curl -s -X POST "$instance_url/model/start" \
            -H "Content-Type: application/json" \
            -d "$json_payload")

        if echo "$response" | grep -q "success\|started"; then
            echo -e "  Instance $instance_port → Scheduler A: ${GREEN}OK${NC}"
        else
            echo -e "  Instance $instance_port → Scheduler A: ${RED}FAILED${NC}"
            echo "  Response: $response"
        fi
    ) &
    pids+=($!)
done

# Second half of Group A instances: register to Planner
for i in $(seq $((N1 / 2)) $((N1 - 1))); do
    instance_port=$((PORT_A_START + i))
    (
        instance_url="http://localhost:$instance_port"

        # Build JSON payload
        if [ -n "$MODEL_PATH_A" ]; then
            json_payload=$(jq -n \
                --arg model_id "$MODEL_ID_A" \
                --arg scheduler_url "$PLANNER_URL" \
                --arg model_path "$MODEL_PATH_A" \
                '{model_id: $model_id, scheduler_url: $scheduler_url, parameters: {MODEL_PATH: $model_path}}')
        else
            json_payload=$(jq -n \
                --arg model_id "$MODEL_ID_A" \
                --arg scheduler_url "$PLANNER_URL" \
                '{model_id: $model_id, scheduler_url: $scheduler_url}')
        fi

        response=$(curl -s -X POST "$instance_url/model/start" \
            -H "Content-Type: application/json" \
            -d "$json_payload")

        if echo "$response" | grep -q "success\|started"; then
            echo -e "  Instance $instance_port → Planner: ${GREEN}OK${NC}"
        else
            echo -e "  Instance $instance_port → Planner: ${RED}FAILED${NC}"
            echo "  Response: $response"
        fi
    ) &
    pids+=($!)
done

# Deploy models to Group B instances
echo ""
echo "Deploying $MODEL_ID_B to Group B instances..."

# First half of Group B instances: register to Scheduler B
for i in $(seq 0 $((N2 / 2 - 1))); do
    instance_port=$((PORT_B_START + i))
    (
        instance_url="http://localhost:$instance_port"

        # Build JSON payload
        if [ -n "$MODEL_PATH_B" ]; then
            json_payload=$(jq -n \
                --arg model_id "$MODEL_ID_B" \
                --arg scheduler_url "$SCHEDULER_B_URL" \
                --arg model_path "$MODEL_PATH_B" \
                '{model_id: $model_id, scheduler_url: $scheduler_url, parameters: {MODEL_PATH: $model_path}}')
        else
            json_payload=$(jq -n \
                --arg model_id "$MODEL_ID_B" \
                --arg scheduler_url "$SCHEDULER_B_URL" \
                '{model_id: $model_id, scheduler_url: $scheduler_url}')
        fi

        response=$(curl -s -X POST "$instance_url/model/start" \
            -H "Content-Type: application/json" \
            -d "$json_payload")

        if echo "$response" | grep -q "success\|started"; then
            echo -e "  Instance $instance_port → Scheduler B: ${GREEN}OK${NC}"
        else
            echo -e "  Instance $instance_port → Scheduler B: ${RED}FAILED${NC}"
            echo "  Response: $response"
        fi
    ) &
    pids+=($!)
done

# Second half of Group B instances: register to Planner
for i in $(seq $((N2 / 2)) $((N2 - 1))); do
    instance_port=$((PORT_B_START + i))
    (
        instance_url="http://localhost:$instance_port"

        # Build JSON payload
        if [ -n "$MODEL_PATH_B" ]; then
            json_payload=$(jq -n \
                --arg model_id "$MODEL_ID_B" \
                --arg scheduler_url "$PLANNER_URL" \
                --arg model_path "$MODEL_PATH_B" \
                '{model_id: $model_id, scheduler_url: $scheduler_url, parameters: {MODEL_PATH: $model_path}}')
        else
            json_payload=$(jq -n \
                --arg model_id "$MODEL_ID_B" \
                --arg scheduler_url "$PLANNER_URL" \
                '{model_id: $model_id, scheduler_url: $scheduler_url}')
        fi

        response=$(curl -s -X POST "$instance_url/model/start" \
            -H "Content-Type: application/json" \
            -d "$json_payload")

        if echo "$response" | grep -q "success\|started"; then
            echo -e "  Instance $instance_port → Planner: ${GREEN}OK${NC}"
        else
            echo -e "  Instance $instance_port → Planner: ${RED}FAILED${NC}"
            echo "  Response: $response"
        fi
    ) &
    pids+=($!)
done

# Wait for all parallel model starts to complete
echo ""
echo "Waiting for all deployments to complete..."
for pid in "${pids[@]}"; do
    wait $pid
done

echo ""
echo "========================================="
echo -e "${GREEN}Model deployment completed!${NC}"
echo "========================================="
echo ""
echo "Deployment Summary:"
echo "  Group A: $N1 instances of $MODEL_ID_A"
echo "    - $(( (N1 + 1) / 2 )) registered to Scheduler A"
echo "    - $(( N1 / 2 )) registered to Planner"
echo ""
echo "  Group B: $N2 instances of $MODEL_ID_B"
echo "    - $(( (N2 + 1) / 2 )) registered to Scheduler B"
echo "    - $(( N2 / 2 )) registered to Planner"
echo "========================================="
