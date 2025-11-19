#!/bin/bash

set -e

# 固定端口
SCHEDULER_PORT=8100
PREDICTOR_PORT=8100
INSTANCE_PORT=8000

# 角色 IP
SCHEDULER_A_HOST="29.209.114.51"
SCHEDULER_B_HOST="29.209.113.228"
PREDICTOR_HOST="29.209.114.166"

# Check service health

SLEEP_MODEL_A_HOSTS=(
  29.209.114.51
  29.209.114.166
  29.209.113.113
  29.209.106.237
  29.209.114.56
  29.209.114.241
  29.209.112.177
  29.209.113.235
)

# sleep_model_b 对应的机器
SLEEP_MODEL_B_HOSTS=(
  29.209.113.228
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
    response=$(curl -s -X GET "http://$host:8000/health")
    if echo "$response" | grep -q "healthy"; then
        echo -e "$host: ${GREEN}OK${NC}"
    else
        echo -e "$host: ${RED}FAILED${NC}"
    fi
done

for host in "${SLEEP_MODEL_B_HOSTS[@]}"; do
    response=$(curl -s -X GET "http://$host:8000/health")
    if echo "$response" | grep -q "healthy"; then
        echo -e "$host: ${GREEN}OK${NC}"
    else
        echo -e "$host: ${RED}FAILED${NC}"
    fi
done

# If any of the health checks failed, exit with error
if [ $? -ne 0 ]; then
    echo -e "${RED}Some hosts failed to start${NC}"
    exit 1
fi  

echo -e "${GREEN}All hosts are healthy${NC}"

# Start sleep-model on Group A instances (parallel)
for host in "${SLEEP_MODEL_A_HOSTS[@]}"; do
    (
        response=$(curl -s -X POST "http://$host:$instance_port/model/start" \
            -H "Content-Type: application/json" \
            -d "{\"model_id\": \"sleep_model_a\", \"scheduler_url\": \"http://$SCHEDULER_A_HOST:$SCHEDULER_PORT\", \"parameters\": {}}")

        if echo "$response" | grep -q "success\|started"; then
            echo -e "$instance_id: ${GREEN}OK${NC}"
        else
            echo -e "$instance_id: ${RED}FAILED${NC}"
            echo "Response: $response"
        fi
    ) &
    pids+=($!)
done

# Start sleep-model on Group B instances (parallel)
for host in "${SLEEP_MODEL_B_HOSTS[@]}"; do
    (
        response=$(curl -s -X POST "http://$host:$instance_port/model/start" \
            -H "Content-Type: application/json" \
            -d "{\"model_id\": \"sleep_model_b\", \"scheduler_url\": \"http://$SCHEDULER_B_HOST:$SCHEDULER_PORT\", \"parameters\": {}}")

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