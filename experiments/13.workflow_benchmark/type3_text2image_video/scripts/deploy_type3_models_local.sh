#!/bin/bash

set -e

# ============================================================================
# Type3 Text2Image+Video - Local Model Deployment Script
# ============================================================================
# Deploys models to three groups of instances:
#   - Group A: LLM (sleep_model_a) → Scheduler A
#   - Group C: FLUX (sleep_model_c) → Scheduler C
#   - Group B: T2VID (sleep_model_b) → Scheduler B
#
# Half of each group registers to their scheduler, half to the planner.
# ============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
SCHEDULER_A_URL="http://localhost:8100"
SCHEDULER_C_URL="http://localhost:8300"
SCHEDULER_B_URL="http://localhost:8200"
PLANNER_URL="http://localhost:8202"

MODEL_ID_A="sleep_model_a"
MODEL_ID_C="sleep_model_c"
MODEL_ID_B="sleep_model_b"

N1=4  # LLM instances
N2=2  # FLUX instances
N3=2  # T2VID instances

PORT_A_START=8210
PORT_C_START=8400
PORT_B_START=8500

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scheduler-a-url) SCHEDULER_A_URL="$2"; shift 2 ;;
        --scheduler-c-url) SCHEDULER_C_URL="$2"; shift 2 ;;
        --scheduler-b-url) SCHEDULER_B_URL="$2"; shift 2 ;;
        --planner-url) PLANNER_URL="$2"; shift 2 ;;
        --model-id-a) MODEL_ID_A="$2"; shift 2 ;;
        --model-id-c) MODEL_ID_C="$2"; shift 2 ;;
        --model-id-b) MODEL_ID_B="$2"; shift 2 ;;
        --n1) N1="$2"; shift 2 ;;
        --n2) N2="$2"; shift 2 ;;
        --n3) N3="$2"; shift 2 ;;
        --port-a-start) PORT_A_START="$2"; shift 2 ;;
        --port-c-start) PORT_C_START="$2"; shift 2 ;;
        --port-b-start) PORT_B_START="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --scheduler-a-url URL    Scheduler A URL (default: http://localhost:8100)"
            echo "  --scheduler-c-url URL    Scheduler C URL (default: http://localhost:8300)"
            echo "  --scheduler-b-url URL    Scheduler B URL (default: http://localhost:8200)"
            echo "  --planner-url URL        Planner URL (default: http://localhost:8202)"
            echo "  --model-id-a ID          Model ID for Group A (default: sleep_model_a)"
            echo "  --model-id-c ID          Model ID for Group C (default: sleep_model_c)"
            echo "  --model-id-b ID          Model ID for Group B (default: sleep_model_b)"
            echo "  --n1 NUM                 Number of Group A instances (default: 4)"
            echo "  --n2 NUM                 Number of Group C instances (default: 2)"
            echo "  --n3 NUM                 Number of Group B instances (default: 2)"
            echo "  --port-a-start PORT      Starting port for Group A (default: 8210)"
            echo "  --port-c-start PORT      Starting port for Group C (default: 8400)"
            echo "  --port-b-start PORT      Starting port for Group B (default: 8500)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check jq
if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: jq is required but not installed.${NC}"
    exit 1
fi

echo "========================================="
echo -e "${BLUE}Type3 Model Deployment:${NC}"
echo "========================================="
echo "Group A (LLM):   $MODEL_ID_A ($N1 instances, ports $PORT_A_START-$((PORT_A_START + N1 - 1)))"
echo "Group C (FLUX):  $MODEL_ID_C ($N2 instances, ports $PORT_C_START-$((PORT_C_START + N2 - 1)))"
echo "Group B (T2VID): $MODEL_ID_B ($N3 instances, ports $PORT_B_START-$((PORT_B_START + N3 - 1)))"
echo "========================================="

pids=()

# Deploy helper function
deploy_model() {
    local instance_port=$1
    local model_id=$2
    local scheduler_url=$3
    local group_name=$4

    local instance_url="http://localhost:$instance_port"
    local json_payload=$(jq -n \
        --arg model_id "$model_id" \
        --arg scheduler_url "$scheduler_url" \
        '{model_id: $model_id, scheduler_url: $scheduler_url}')

    local response=$(curl -s -X POST "$instance_url/model/start" \
        -H "Content-Type: application/json" \
        -d "$json_payload")

    if echo "$response" | grep -q "success\|started"; then
        echo -e "  $group_name Instance $instance_port → ${scheduler_url##*/}: ${GREEN}OK${NC}"
    else
        echo -e "  $group_name Instance $instance_port → ${scheduler_url##*/}: ${RED}FAILED${NC}"
        echo "  Response: $response"
    fi
}

# Deploy Group A (LLM)
echo ""
echo "Deploying $MODEL_ID_A to Group A instances..."
for i in $(seq 0 $((N1 - 1))); do
    instance_port=$((PORT_A_START + i))
    if (( i < (N1 + 1) / 2 )); then
        scheduler_url=$SCHEDULER_A_URL
    else
        scheduler_url=$PLANNER_URL
    fi
    ( deploy_model "$instance_port" "$MODEL_ID_A" "$scheduler_url" "A" ) &
    pids+=($!)
done

# Deploy Group C (FLUX)
echo ""
echo "Deploying $MODEL_ID_C to Group C instances..."
for i in $(seq 0 $((N2 - 1))); do
    instance_port=$((PORT_C_START + i))
    if (( i < (N2 + 1) / 2 )); then
        scheduler_url=$SCHEDULER_C_URL
    else
        scheduler_url=$PLANNER_URL
    fi
    ( deploy_model "$instance_port" "$MODEL_ID_C" "$scheduler_url" "C" ) &
    pids+=($!)
done

# Deploy Group B (T2VID)
echo ""
echo "Deploying $MODEL_ID_B to Group B instances..."
for i in $(seq 0 $((N3 - 1))); do
    instance_port=$((PORT_B_START + i))
    if (( i < (N3 + 1) / 2 )); then
        scheduler_url=$SCHEDULER_B_URL
    else
        scheduler_url=$PLANNER_URL
    fi
    ( deploy_model "$instance_port" "$MODEL_ID_B" "$scheduler_url" "B" ) &
    pids+=($!)
done

# Wait for all deployments
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
echo "  Group A (LLM):   $N1 instances of $MODEL_ID_A"
echo "    - $(( (N1 + 1) / 2 )) → Scheduler A"
echo "    - $(( N1 / 2 )) → Planner"
echo ""
echo "  Group C (FLUX):  $N2 instances of $MODEL_ID_C"
echo "    - $(( (N2 + 1) / 2 )) → Scheduler C"
echo "    - $(( N2 / 2 )) → Planner"
echo ""
echo "  Group B (T2VID): $N3 instances of $MODEL_ID_B"
echo "    - $(( (N3 + 1) / 2 )) → Scheduler B"
echo "    - $(( N3 / 2 )) → Planner"
echo "========================================="
