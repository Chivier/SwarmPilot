#!/bin/bash
set -e

# ============================================================================
# Type3 Text2Image+Video - Real Cluster Model Deployment
# ============================================================================
# Deploys real models for the Type3 workflow:
#   - Group A: LLM (llm_service_small_model) → Scheduler A
#   - Group C: FLUX (t2img) → Scheduler C
#   - Group B: T2VID (t2vid) → Scheduler B
#
# This script should be run on the client node after all services are started.
# ============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Host configuration (must match start_type3_real_services.sh)
SCHEDULER_A_HOST="29.209.114.51"    # LLM
SCHEDULER_C_HOST="29.209.114.52"    # FLUX
SCHEDULER_B_HOST="29.209.113.228"   # T2VID
PLANNER_HOST="29.209.114.166"

SCHEDULER_PORT=8100

# Model IDs
MODEL_ID_A=${MODEL_ID_A:-"llm_service_small_model"}
MODEL_ID_C=${MODEL_ID_C:-"t2img"}
MODEL_ID_B=${MODEL_ID_B:-"t2vid"}

# Model paths (optional, for custom model locations)
MODEL_PATH_A=${MODEL_PATH_A:-""}
MODEL_PATH_C=${MODEL_PATH_C:-""}
MODEL_PATH_B=${MODEL_PATH_B:-""}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-id-a) MODEL_ID_A="$2"; shift 2 ;;
        --model-id-c) MODEL_ID_C="$2"; shift 2 ;;
        --model-id-b) MODEL_ID_B="$2"; shift 2 ;;
        --model-path-a) MODEL_PATH_A="$2"; shift 2 ;;
        --model-path-c) MODEL_PATH_C="$2"; shift 2 ;;
        --model-path-b) MODEL_PATH_B="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model-id-a ID      Model ID for LLM (default: llm_service_small_model)"
            echo "  --model-id-c ID      Model ID for FLUX (default: t2img)"
            echo "  --model-id-b ID      Model ID for T2VID (default: t2vid)"
            echo "  --model-path-a PATH  Model path for LLM (optional)"
            echo "  --model-path-c PATH  Model path for FLUX (optional)"
            echo "  --model-path-b PATH  Model path for T2VID (optional)"
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
echo -e "${BLUE}Type3 Real Cluster Model Deployment${NC}"
echo "========================================="
echo "Group A (LLM):   $MODEL_ID_A → Scheduler A ($SCHEDULER_A_HOST)"
echo "Group C (FLUX):  $MODEL_ID_C → Scheduler C ($SCHEDULER_C_HOST)"
echo "Group B (T2VID): $MODEL_ID_B → Scheduler B ($SCHEDULER_B_HOST)"
echo "========================================="
echo ""

# Deploy helper function
deploy_to_instances() {
    local host=$1
    local model_id=$2
    local scheduler_url=$3
    local planner_url=$4
    local model_path=$5
    local group_name=$6

    echo -e "${YELLOW}Deploying $model_id to $group_name instances on $host...${NC}"

    # Build base payload
    if [ -n "$model_path" ]; then
        json_payload=$(jq -n \
            --arg model_id "$model_id" \
            --arg scheduler_url "$scheduler_url" \
            --arg model_path "$model_path" \
            '{model_id: $model_id, scheduler_url: $scheduler_url, parameters: {MODEL_PATH: $model_path}}')
    else
        json_payload=$(jq -n \
            --arg model_id "$model_id" \
            --arg scheduler_url "$scheduler_url" \
            '{model_id: $model_id, scheduler_url: $scheduler_url}')
    fi

    # Deploy to each instance on the host (ports 8110-8117 for 8 GPU instances)
    local pids=()
    for port in 8110 8111 8112 8113 8114 8115 8116 8117; do
        (
            local instance_url="http://$host:$port"

            # Alternate between scheduler and planner
            local target_url
            if (( (port - 8110) % 2 == 0 )); then
                target_url=$scheduler_url
            else
                target_url=$planner_url
            fi

            # Update payload with correct scheduler URL
            if [ -n "$model_path" ]; then
                local payload=$(jq -n \
                    --arg model_id "$model_id" \
                    --arg scheduler_url "$target_url" \
                    --arg model_path "$model_path" \
                    '{model_id: $model_id, scheduler_url: $scheduler_url, parameters: {MODEL_PATH: $model_path}}')
            else
                local payload=$(jq -n \
                    --arg model_id "$model_id" \
                    --arg scheduler_url "$target_url" \
                    '{model_id: $model_id, scheduler_url: $scheduler_url}')
            fi

            local response=$(curl -s -X POST "$instance_url/model/start" \
                -H "Content-Type: application/json" \
                -d "$payload" 2>/dev/null || echo "CONNECTION_FAILED")

            if echo "$response" | grep -q "success\|started"; then
                echo -e "  $group_name:$port → ${target_url##*/}: ${GREEN}OK${NC}"
            elif echo "$response" | grep -q "CONNECTION_FAILED"; then
                echo -e "  $group_name:$port: ${YELLOW}Not running (skipped)${NC}"
            else
                echo -e "  $group_name:$port: ${RED}FAILED${NC}"
            fi
        ) &
        pids+=($!)
    done

    # Wait for all deployments
    for pid in "${pids[@]}"; do
        wait $pid 2>/dev/null || true
    done
}

# Deploy Group A (LLM)
deploy_to_instances \
    "$SCHEDULER_A_HOST" \
    "$MODEL_ID_A" \
    "http://$SCHEDULER_A_HOST:$SCHEDULER_PORT" \
    "http://$PLANNER_HOST:$SCHEDULER_PORT" \
    "$MODEL_PATH_A" \
    "A"

echo ""

# Deploy Group C (FLUX)
deploy_to_instances \
    "$SCHEDULER_C_HOST" \
    "$MODEL_ID_C" \
    "http://$SCHEDULER_C_HOST:$SCHEDULER_PORT" \
    "http://$PLANNER_HOST:$SCHEDULER_PORT" \
    "$MODEL_PATH_C" \
    "C"

echo ""

# Deploy Group B (T2VID)
deploy_to_instances \
    "$SCHEDULER_B_HOST" \
    "$MODEL_ID_B" \
    "http://$SCHEDULER_B_HOST:$SCHEDULER_PORT" \
    "http://$PLANNER_HOST:$SCHEDULER_PORT" \
    "$MODEL_PATH_B" \
    "B"

echo ""
echo "========================================="
echo -e "${GREEN}Model deployment completed!${NC}"
echo "========================================="
echo ""
echo "Models deployed:"
echo "  - $MODEL_ID_A on Scheduler A ($SCHEDULER_A_HOST)"
echo "  - $MODEL_ID_C on Scheduler C ($SCHEDULER_C_HOST)"
echo "  - $MODEL_ID_B on Scheduler B ($SCHEDULER_B_HOST)"
echo "========================================="
