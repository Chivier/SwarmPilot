#!/bin/bash

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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
            # If both use the same path
            MODEL_PATH_A="$2"
            MODEL_PATH_B="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model-path PATH         Set model path for both Group A and B"
            echo "  --model-path-a PATH       Set model path for Group A only"
            echo "  --model-path-b PATH       Set model path for Group B only"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --model-path /path/to/model"
            echo "  $0 --model-path-a /path/to/model_a --model-path-b /path/to/model_b"
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
echo -e "${GREEN}Deployment Configuration:${NC}"
echo "  Group A model path: $MODEL_PATH_A"
echo "  Group B model path: $MODEL_PATH_B"
echo ""

# Check if jq is available for JSON generation
if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: jq is required but not installed.${NC}"
    echo "Please install jq: sudo apt-get install jq (or brew install jq on macOS)"
    exit 1
fi

# 固定端口
SCHEDULER_PORT=8100
PREDICTOR_PORT=8100
PLANNER_PORT=8100
INSTANCE_PORT=8000

# 角色 IP
SCHEDULER_A_HOST="29.209.114.51" # 0
SCHEDULER_B_HOST="29.209.113.228" # 8
PREDICTOR_HOST="29.209.113.113" # 2
PLANNER_HOST="29.209.114.166" # 1
CLIENT_HOST="29.209.114.166" # 1

# Ports for scheduler (first half)
SCHEDULER_PORT_LIST=(
    8200
    8201
    8202
    8203
)

# Ports for planner (second half)
PLANNER_PORT_LIST=(
    8204
    8205
    8206
    8207
)

# All instance ports (for health check)
INSTANCE_PORT_LIST=(
    8200
    8201
    8202
    8203
    8204
    8205
    8206
    8207
)

# Check service health

SLEEP_MODEL_A_HOSTS=(
  29.209.106.237

)

# sleep_model_b 对应的机器
SLEEP_MODEL_B_HOSTS=(
  29.209.114.56
  29.209.114.241
  29.209.112.177
  29.209.113.235
  29.209.105.60
  29.209.113.166
  29.209.113.176
  29.209.113.169
  29.209.112.74
  29.209.115.174
  29.209.113.156
)


# Do health check for all hosts by request its /health endpoint

for host in "${SLEEP_MODEL_A_HOSTS[@]}"; do
    for port in "${INSTANCE_PORT_LIST[@]}"; do
        response=$(curl -s -X GET "http://$host:$port/health")
        if echo "$response" | grep -q "healthy"; then
            echo -e "$host:$port: ${GREEN}OK${NC}"
        else
            echo -e "$host:$port: ${RED}FAILED${NC}"
        fi
    done
done

for host in "${SLEEP_MODEL_B_HOSTS[@]}"; do
    for port in "${INSTANCE_PORT_LIST[@]}"; do
        response=$(curl -s -X GET "http://$host:$port/health")
        if echo "$response" | grep -q "healthy"; then
            echo -e "$host: ${GREEN}OK${NC}"
        else
            echo -e "$host: ${RED}FAILED${NC}"
        fi
    done
done

# If any of the health checks failed, exit with error
if [ $? -ne 0 ]; then
    echo -e "${RED}Some hosts failed to start${NC}"
    exit 1
fi  

echo -e "${GREEN}All hosts are healthy${NC}"

# Start sleep-model on Group A instances (parallel)
# First half: register to scheduler
for host in "${SLEEP_MODEL_A_HOSTS[@]}"; do
    for instance_port in "${SCHEDULER_PORT_LIST[@]}"; do
        (
            instance_id="${host}:${instance_port}"
            # Build JSON payload with model_path in parameters
            # Note: The key should be "MODEL_PATH" (not "MODEL_MODEL_PATH") because
            # subprocess_manager._build_env_vars will add "MODEL_" prefix, resulting in "MODEL_MODEL_PATH"
            
            # Group A uses llm_service_small_model
            json_payload=$(jq -n \
                --arg model_id "llm_service_small_model" \
                --arg scheduler_url "http://$SCHEDULER_A_HOST:$SCHEDULER_PORT" \
                --arg model_path "$MODEL_PATH_A" \
                '{model_id: $model_id, scheduler_url: $scheduler_url, parameters: {MODEL_PATH: $model_path}}')

            response=$(curl -s -X POST "http://$host:$instance_port/model/start" \
                -H "Content-Type: application/json" \
                -d "$json_payload")

            if echo "$response" | grep -q "success\|started"; then
                echo -e "$instance_id (scheduler): ${GREEN}OK${NC}"
            else
                echo -e "$instance_id (scheduler): ${RED}FAILED${NC}"
                echo "Response: $response"
            fi
        ) &
        pids+=($!)
    done
done

# Second half: register to planner
for host in "${SLEEP_MODEL_A_HOSTS[@]}"; do
    for instance_port in "${PLANNER_PORT_LIST[@]}"; do
        (
            instance_id="${host}:${instance_port}"
            json_payload=$(jq -n \
                --arg model_id "llm_service_small_model" \
                --arg scheduler_url "http://$PLANNER_HOST:$PLANNER_PORT" \
                --arg model_path "$MODEL_PATH_A" \
                '{model_id: $model_id, scheduler_url: $scheduler_url, parameters: {MODEL_PATH: $model_path}}')

            response=$(curl -s -X POST "http://$host:$instance_port/model/start" \
                -H "Content-Type: application/json" \
                -d "$json_payload")

            if echo "$response" | grep -q "success\|started"; then
                echo -e "$instance_id (planner): ${GREEN}OK${NC}"
            else
                echo -e "$instance_id (planner): ${RED}FAILED${NC}"
                echo "Response: $response"
            fi
        ) &
        pids+=($!)
    done
done

# Start sleep-model on Group B instances (parallel)
# First half: register to scheduler
for host in "${SLEEP_MODEL_B_HOSTS[@]}"; do
    for instance_port in "${SCHEDULER_PORT_LIST[@]}"; do
        (
            instance_id="${host}:${instance_port}"
            # Group B uses t2vid
            json_payload=$(jq -n \
                --arg model_id "t2vid" \
                --arg scheduler_url "http://$SCHEDULER_B_HOST:$SCHEDULER_PORT" \
                --arg model_path "$MODEL_PATH_B" \
                --arg software_name "diffuser" \
                --arg software_version "latest" \
                '{model_id: $model_id, scheduler_url: $scheduler_url, parameters: {MODEL_PATH: $model_path, software_name: $software_name, software_version: $software_version}}')

            response=$(curl -s -X POST "http://$host:$instance_port/model/start" \
                -H "Content-Type: application/json" \
                -d "$json_payload")

            if echo "$response" | grep -q "success\|started"; then
                echo -e "$instance_id (scheduler): ${GREEN}OK${NC}"
            else
                echo -e "$instance_id (scheduler): ${RED}FAILED${NC}"
                echo "Response: $response"
            fi
        ) &
        pids+=($!)
    done
done

# Second half: register to planner
for host in "${SLEEP_MODEL_B_HOSTS[@]}"; do
    for instance_port in "${PLANNER_PORT_LIST[@]}"; do
        (
            instance_id="${host}:${instance_port}"
            json_payload=$(jq -n \
                --arg model_id "t2vid" \
                --arg scheduler_url "http://$PLANNER_HOST:$PLANNER_PORT" \
                --arg model_path "$MODEL_PATH_B" \
                --arg software_name "diffuser" \
                --arg software_version "latest" \
                '{model_id: $model_id, scheduler_url: $scheduler_url, parameters: {MODEL_PATH: $model_path, software_name: $software_name, software_version: $software_version}}')

            response=$(curl -s -X POST "http://$host:$instance_port/model/start" \
                -H "Content-Type: application/json" \
                -d "$json_payload")

            if echo "$response" | grep -q "success\|started"; then
                echo -e "$instance_id (planner): ${GREEN}OK${NC}"
            else
                echo -e "$instance_id (planner): ${RED}FAILED${NC}"
                echo "Response: $response"
            fi
        ) &
        pids+=($!)
    done
done

# Wait for all parallel model starts to complete
for pid in "${pids[@]}"; do
    wait $pid
done
echo -e "${GREEN}All model starts completed${NC}"
