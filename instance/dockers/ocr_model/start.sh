#!/bin/bash
set -e

echo "Starting OCR Model Container..."
echo "Default Languages: ${OCR_DEFAULT_LANGUAGES:-en}"
echo "GPU Enabled: ${OCR_GPU_ENABLED:-false}"

# Start the HTTP service using uv
exec uv run python main.py
