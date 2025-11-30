#!/bin/bash

set -e

# ============================================
# Manual Model Deployment for Type2 (Deep Research) Workflow - Redeploy Version
# ============================================
# Type2 Workflow: A -> n*B1 -> n*B2 -> Merge
#   - Model A (llm_service_small_model) -> Scheduler A / Planner (for A and Merge tasks)
#   - Model B (llm_service_large_model) -> Scheduler B / Planner (for B1/B2 tasks)
#
# Registration Strategy (Planner Style):
#   - Ports 8200-8203: Register to Scheduler
#   - Ports 8204-8207: Register to Planner
#
# Usage:
#   bash manual_deploy_type2_redeploy.sh --model-path-a /path/to/small_llm --model-path-b /path/to/large_llm

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse command line arguments
MODEL_PATH_A=""
MODEL_PATH_B=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path-a)
            MODEL_PATH_A="$2"
            shift 2
            ;;
        --model-path-b)
            MODEL_PATH_B="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH_A="$2"
            MODEL_PATH_B="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Type2 (Deep Research) Model Deployment with Planner Registration (Redeploy Version):"
            echo "  Model A: llm_service_small_model (for A and Merge tasks)"
            echo "  Model B: llm_service_large_model (for B1/B2 tasks)"
            echo ""
            echo "Registration Strategy:"
            echo "  Ports 8200-8203 -> Scheduler (A to Scheduler A, B to Scheduler B)"
            echo "  Ports 8204-8207 -> Planner"
            echo ""
            echo "Options:"
            echo "  --model-path PATH         Set model path for both Group A and B"
            echo "  --model-path-a PATH       Set model path for Group A (Small LLM)"
            echo "  --model-path-b PATH       Set model path for Group B (Large LLM)"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --model-path-a /path/to/small_llm --model-path-b /path/to/large_llm"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate model paths
if [[ -z "$MODEL_PATH_A" ]]; then
    echo "Error: --model-path-a or --model-path must be specified"
    exit 1
fi

if [[ -z "$MODEL_PATH_B" ]]; then
    echo "Error: --model-path-b or --model-path must be specified"
    exit 1
fi

# Display configuration
echo -e "${GREEN}Type2 (Deep Research) Deployment Configuration (Redeploy):${NC}"
echo "  Group A (Small LLM) model path: $MODEL_PATH_A"
echo "  Group B (Large LLM) model path: $MODEL_PATH_B"
echo ""
echo "Registration Strategy:"
echo "  Ports 8200-8203 -> Scheduler"
echo "  Ports 8204-8207 -> Planner"
echo ""

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: jq is required but not installed.${NC}"
    exit 1
fi

# Ports
SCHEDULER_PORT=8100
PLANNER_PORT=8100

# Role IPs
SCHEDULER_A_HOST="29.209.114.51"
SCHEDULER_B_HOST="29.209.113.228"
PLANNER_HOST="29.209.114.166"

# Ports for scheduler (first half)
SCHEDULER_PORT_LIST=(8200 8201 8202 8203)

# Ports for planner (second half)
PLANNER_PORT_LIST=(8204 8205 8206 8207)

# All instance ports (for health check)
INSTANCE_PORT_LIST=(8200 8201 8202 8203 8204 8205 8206 8207)

# Group A Hosts (llm_service_small_model for A and Merge tasks)
GROUP_A_HOSTS=(
  29.209.106.237
  29.209.114.56
  29.209.114.241
  29.209.112.177
  29.209.113.235
  29.209.105.60
)

# Group B Hosts (llm_service_large_model for B1/B2 tasks)
GROUP_B_HOSTS=(
  29.209.113.166
  29.209.113.176
  29.209.113.169
  29.209.112.74
  29.209.115.174
  29.209.113.156
)

# Health check for all hosts
echo -e "${YELLOW}Performing health checks...${NC}"

for host in "${GROUP_A_HOSTS[@]}"; do
    for port in "${INSTANCE_PORT_LIST[@]}"; do
        response=$(curl -s -X GET "http://$host:$port/health")
        if echo "$response" | grep -q "healthy"; then
            echo -e "$host:$port: ${GREEN}OK${NC}"
        else
            echo -e "$host:$port: ${RED}FAILED${NC}"
        fi
    done
done

for host in "${GROUP_B_HOSTS[@]}"; do
    for port in "${INSTANCE_PORT_LIST[@]}"; do
        response=$(curl -s -X GET "http://$host:$port/health")
        if echo "$response" | grep -q "healthy"; then
            echo -e "$host:$port: ${GREEN}OK${NC}"
        else
            echo -e "$host:$port: ${RED}FAILED${NC}"
        fi
    done
done

echo -e "${GREEN}Health checks completed${NC}"

# Deploy models
pids=()

# ============================================
# Group A: Deploy llm_service_small_model
# ============================================
echo -e "${YELLOW}Deploying llm_service_small_model on Group A hosts...${NC}"

# First half: register to Scheduler A
for host in "${GROUP_A_HOSTS[@]}"; do
    for instance_port in "${SCHEDULER_PORT_LIST[@]}"; do
        (
            instance_id="${host}:${instance_port}"
            json_payload=$(jq -n \
                --arg model_id "llm_service_small_model" \
                --arg scheduler_url "http://$SCHEDULER_A_HOST:$SCHEDULER_PORT" \
                --arg model_path "$MODEL_PATH_A" \
                --arg software_name "sglang" \
                --arg software_version "0.5.5.post2" \
                '{model_id: $model_id, scheduler_url: $scheduler_url, parameters: {MODEL_PATH: $model_path, software_name: $software_name, software_version: $software_version}, standby: false}')

            response=$(curl -s -X POST "http://$host:$instance_port/model/start" \
                -H "Content-Type: application/json" \
                -d "$json_payload")

            if echo "$response" | grep -q "success\|started"; then
                echo -e "$instance_id (llm_service_small_model -> scheduler): ${GREEN}OK${NC}"
            else
                echo -e "$instance_id (llm_service_small_model -> scheduler): ${RED}FAILED${NC}"
                echo "Response: $response"
            fi
        ) &
        pids+=($!)
    done
done

# Second half: register to Planner
for host in "${GROUP_A_HOSTS[@]}"; do
    for instance_port in "${PLANNER_PORT_LIST[@]}"; do
        (
            instance_id="${host}:${instance_port}"
            json_payload=$(jq -n \
                --arg model_id "llm_service_small_model" \
                --arg scheduler_url "http://$PLANNER_HOST:$PLANNER_PORT" \
                --arg model_path "$MODEL_PATH_A" \
                --arg software_name "sglang" \
                --arg software_version "0.5.5.post2" \
                '{model_id: $model_id, scheduler_url: $scheduler_url, parameters: {MODEL_PATH: $model_path, software_name: $software_name, software_version: $software_version}, standby: false}')

            response=$(curl -s -X POST "http://$host:$instance_port/model/start" \
                -H "Content-Type: application/json" \
                -d "$json_payload")

            if echo "$response" | grep -q "success\|started"; then
                echo -e "$instance_id (llm_service_small_model -> planner): ${GREEN}OK${NC}"
            else
                echo -e "$instance_id (llm_service_small_model -> planner): ${RED}FAILED${NC}"
                echo "Response: $response"
            fi
        ) &
        pids+=($!)
    done
done

# ============================================
# Group B: Deploy llm_service_large_model
# ============================================
echo -e "${YELLOW}Deploying llm_service_large_model on Group B hosts...${NC}"

# First half: register to Scheduler B
for host in "${GROUP_B_HOSTS[@]}"; do
    for instance_port in "${SCHEDULER_PORT_LIST[@]}"; do
        (
            instance_id="${host}:${instance_port}"
            json_payload=$(jq -n \
                --arg model_id "llm_service_large_model" \
                --arg scheduler_url "http://$SCHEDULER_B_HOST:$SCHEDULER_PORT" \
                --arg model_path "$MODEL_PATH_B" \
                --arg software_name "sglang" \
                --arg software_version "0.5.5.post2" \
                '{model_id: $model_id, scheduler_url: $scheduler_url, parameters: {MODEL_PATH: $model_path, software_name: $software_name, software_version: $software_version}, standby: false}')

            response=$(curl -s -X POST "http://$host:$instance_port/model/start" \
                -H "Content-Type: application/json" \
                -d "$json_payload")

            if echo "$response" | grep -q "success\|started"; then
                echo -e "$instance_id (llm_service_large_model -> scheduler): ${GREEN}OK${NC}"
            else
                echo -e "$instance_id (llm_service_large_model -> scheduler): ${RED}FAILED${NC}"
                echo "Response: $response"
            fi
        ) &
        pids+=($!)
    done
done

# Second half: register to Planner
for host in "${GROUP_B_HOSTS[@]}"; do
    for instance_port in "${PLANNER_PORT_LIST[@]}"; do
        (
            instance_id="${host}:${instance_port}"
            json_payload=$(jq -n \
                --arg model_id "llm_service_large_model" \
                --arg scheduler_url "http://$PLANNER_HOST:$PLANNER_PORT" \
                --arg model_path "$MODEL_PATH_B" \
                --arg software_name "sglang" \
                --arg software_version "0.5.5.post2" \
                '{model_id: $model_id, scheduler_url: $scheduler_url, parameters: {MODEL_PATH: $model_path, software_name: $software_name, software_version: $software_version}, standby: false}')

            response=$(curl -s -X POST "http://$host:$instance_port/model/start" \
                -H "Content-Type: application/json" \
                -d "$json_payload")

            if echo "$response" | grep -q "success\|started"; then
                echo -e "$instance_id (llm_service_large_model -> planner): ${GREEN}OK${NC}"
            else
                echo -e "$instance_id (llm_service_large_model -> planner): ${RED}FAILED${NC}"
                echo "Response: $response"
            fi
        ) &
        pids+=($!)
    done
done

# Wait for all deployments
for pid in "${pids[@]}"; do
    wait $pid
done

echo -e "${GREEN}Type2 (Deep Research) model deployment completed${NC}"
echo ""
echo "Summary:"
echo "  Group A (llm_service_small_model - for A and Merge tasks):"
echo "    - Ports 8200-8203 -> Scheduler A ($SCHEDULER_A_HOST:$SCHEDULER_PORT)"
echo "    - Ports 8204-8207 -> Planner ($PLANNER_HOST:$PLANNER_PORT)"
echo "  Group B (llm_service_large_model - for B1/B2 tasks):"
echo "    - Ports 8200-8203 -> Scheduler B ($SCHEDULER_B_HOST:$SCHEDULER_PORT)"
echo "    - Ports 8204-8207 -> Planner ($PLANNER_HOST:$PLANNER_PORT)"
