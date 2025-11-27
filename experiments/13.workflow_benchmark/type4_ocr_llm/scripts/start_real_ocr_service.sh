#!/bin/bash
#
# Start Real OCR+LLM Services for Type4 Workflow
#
# This script starts the OCR and LLM model instances on the local server.
# Configuration:
#   - 16 OCR instances (CPU-only, ports 9000-9015)
#   - 8 LLM instances (GPU, ports 9100-9107)
#   - Half of each type registered to scheduler, half to planner
#
# Usage:
#   ./start_real_ocr_service.sh
#
# Environment Variables:
#   OCR_COUNT: Number of OCR instances (default: 16)
#   LLM_COUNT: Number of LLM instances (default: 8)
#   OCR_BASE_PORT: Starting port for OCR instances (default: 9000)
#   LLM_BASE_PORT: Starting port for LLM instances (default: 9100)
#   SCHEDULER_URL: Scheduler URL for registration (default: http://127.0.0.1:8100)
#   PLANNER_URL: Planner URL for registration (default: http://127.0.0.1:8103)

set -e

# Configuration
OCR_COUNT=${OCR_COUNT:-16}
LLM_COUNT=${LLM_COUNT:-8}
OCR_BASE_PORT=${OCR_BASE_PORT:-9000}
LLM_BASE_PORT=${LLM_BASE_PORT:-9100}
SCHEDULER_URL=${SCHEDULER_URL:-"http://127.0.0.1:8100"}
PLANNER_URL=${PLANNER_URL:-"http://127.0.0.1:8103"}

# Docker image names
OCR_IMAGE="ocr_model:latest"
LLM_IMAGE="llm_service_small_model:latest"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
DOCKERS_DIR="${PROJECT_ROOT}/instance/dockers"

echo "=============================================="
echo "Starting Type4 OCR+LLM Real Services"
echo "=============================================="
echo "OCR instances: ${OCR_COUNT} (ports ${OCR_BASE_PORT}-$((OCR_BASE_PORT + OCR_COUNT - 1)))"
echo "LLM instances: ${LLM_COUNT} (ports ${LLM_BASE_PORT}-$((LLM_BASE_PORT + LLM_COUNT - 1)))"
echo "Scheduler URL: ${SCHEDULER_URL}"
echo "Planner URL: ${PLANNER_URL}"
echo "=============================================="

# Function to start OCR instances
start_ocr_instances() {
    echo ""
    echo "Starting OCR instances..."

    # Calculate split: half to scheduler, half to planner
    local half=$((OCR_COUNT / 2))

    for i in $(seq 0 $((OCR_COUNT - 1))); do
        local port=$((OCR_BASE_PORT + i))
        local instance_id="ocr-instance-${i}"
        local container_name="ocr_model_${i}"

        # Determine registration target
        local register_to=""
        if [ $i -lt $half ]; then
            register_to="${SCHEDULER_URL}"
            echo "  [${i}] OCR instance on port ${port} -> Scheduler"
        else
            register_to="${PLANNER_URL}"
            echo "  [${i}] OCR instance on port ${port} -> Planner"
        fi

        # Start container
        docker run -d \
            --name "${container_name}" \
            -p "${port}:8000" \
            -e MODEL_ID="ocr_model" \
            -e INSTANCE_ID="${instance_id}" \
            -e LOG_LEVEL="INFO" \
            -e OCR_DEFAULT_LANGUAGES="en" \
            -e OCR_GPU_ENABLED="false" \
            --restart unless-stopped \
            "${OCR_IMAGE}" || {
                echo "Warning: Failed to start ${container_name}, may already exist"
            }
    done
}

# Function to start LLM instances
start_llm_instances() {
    echo ""
    echo "Starting LLM instances..."

    # Calculate split: half to scheduler, half to planner
    local half=$((LLM_COUNT / 2))

    for i in $(seq 0 $((LLM_COUNT - 1))); do
        local port=$((LLM_BASE_PORT + i))
        local instance_id="llm-instance-${i}"
        local container_name="llm_model_${i}"
        local gpu_id=$((i % 8))  # Cycle through GPUs 0-7

        # Determine registration target
        local register_to=""
        if [ $i -lt $half ]; then
            register_to="${SCHEDULER_URL}"
            echo "  [${i}] LLM instance on port ${port} (GPU ${gpu_id}) -> Scheduler"
        else
            register_to="${PLANNER_URL}"
            echo "  [${i}] LLM instance on port ${port} (GPU ${gpu_id}) -> Planner"
        fi

        # Start container with GPU
        docker run -d \
            --name "${container_name}" \
            --gpus "device=${gpu_id}" \
            -p "${port}:8000" \
            -e MODEL_ID="llm_service_small_model" \
            -e INSTANCE_ID="${instance_id}" \
            -e LOG_LEVEL="INFO" \
            --restart unless-stopped \
            "${LLM_IMAGE}" || {
                echo "Warning: Failed to start ${container_name}, may already exist"
            }
    done
}

# Function to register instances with scheduler/planner
register_instances() {
    echo ""
    echo "Waiting for instances to start..."
    sleep 10

    echo "Registering instances..."

    local half_ocr=$((OCR_COUNT / 2))
    local half_llm=$((LLM_COUNT / 2))

    # Register OCR instances
    for i in $(seq 0 $((OCR_COUNT - 1))); do
        local port=$((OCR_BASE_PORT + i))
        local instance_url="http://localhost:${port}"

        if [ $i -lt $half_ocr ]; then
            # Register to scheduler
            curl -s -X POST "${SCHEDULER_URL}/instance/register" \
                -H "Content-Type: application/json" \
                -d "{\"model_id\": \"ocr_model\", \"instance_url\": \"${instance_url}\", \"instance_id\": \"ocr-instance-${i}\"}" \
                > /dev/null && echo "  Registered OCR-${i} to Scheduler" || echo "  Failed to register OCR-${i}"
        else
            # Register to planner
            curl -s -X POST "${PLANNER_URL}/instance/register" \
                -H "Content-Type: application/json" \
                -d "{\"model_id\": \"ocr_model\", \"instance_url\": \"${instance_url}\", \"instance_id\": \"ocr-instance-${i}\"}" \
                > /dev/null && echo "  Registered OCR-${i} to Planner" || echo "  Failed to register OCR-${i}"
        fi
    done

    # Register LLM instances
    for i in $(seq 0 $((LLM_COUNT - 1))); do
        local port=$((LLM_BASE_PORT + i))
        local instance_url="http://localhost:${port}"

        if [ $i -lt $half_llm ]; then
            # Register to scheduler
            curl -s -X POST "${SCHEDULER_URL}/instance/register" \
                -H "Content-Type: application/json" \
                -d "{\"model_id\": \"llm_service_small_model\", \"instance_url\": \"${instance_url}\", \"instance_id\": \"llm-instance-${i}\"}" \
                > /dev/null && echo "  Registered LLM-${i} to Scheduler" || echo "  Failed to register LLM-${i}"
        else
            # Register to planner
            curl -s -X POST "${PLANNER_URL}/instance/register" \
                -H "Content-Type: application/json" \
                -d "{\"model_id\": \"llm_service_small_model\", \"instance_url\": \"${instance_url}\", \"instance_id\": \"llm-instance-${i}\"}" \
                > /dev/null && echo "  Registered LLM-${i} to Planner" || echo "  Failed to register LLM-${i}"
        fi
    done
}

# Main execution
echo ""
echo "Building Docker images if needed..."
cd "${DOCKERS_DIR}/ocr_model" && docker build -t "${OCR_IMAGE}" . 2>/dev/null || echo "OCR image already exists or build skipped"
# LLM image should already exist from other workflows

start_ocr_instances
start_llm_instances
register_instances

echo ""
echo "=============================================="
echo "All services started!"
echo "=============================================="
echo ""
echo "OCR instances: http://localhost:${OCR_BASE_PORT} - http://localhost:$((OCR_BASE_PORT + OCR_COUNT - 1))"
echo "LLM instances: http://localhost:${LLM_BASE_PORT} - http://localhost:$((LLM_BASE_PORT + LLM_COUNT - 1))"
echo ""
echo "To stop all services, run: ./stop_real_ocr_service.sh"
