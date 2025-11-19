#!/bin/bash
set -e

echo "Starting LLM Service Large Model Container..."

# Start the HTTP service using uv
exec uv run python main.py

