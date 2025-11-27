#!/bin/bash
#
# Stop Real OCR+LLM Services for Type4 Workflow
#
# This script stops all OCR and LLM model instances started by start_real_ocr_service.sh.
#
# Usage:
#   ./stop_real_ocr_service.sh

set -e

# Configuration
OCR_COUNT=${OCR_COUNT:-16}
LLM_COUNT=${LLM_COUNT:-8}

echo "=============================================="
echo "Stopping Type4 OCR+LLM Real Services"
echo "=============================================="

# Stop OCR containers
echo ""
echo "Stopping OCR instances..."
for i in $(seq 0 $((OCR_COUNT - 1))); do
    container_name="ocr_model_${i}"
    docker stop "${container_name}" 2>/dev/null && docker rm "${container_name}" 2>/dev/null && \
        echo "  Stopped ${container_name}" || echo "  ${container_name} not running"
done

# Stop LLM containers
echo ""
echo "Stopping LLM instances..."
for i in $(seq 0 $((LLM_COUNT - 1))); do
    container_name="llm_model_${i}"
    docker stop "${container_name}" 2>/dev/null && docker rm "${container_name}" 2>/dev/null && \
        echo "  Stopped ${container_name}" || echo "  ${container_name} not running"
done

echo ""
echo "=============================================="
echo "All services stopped!"
echo "=============================================="
