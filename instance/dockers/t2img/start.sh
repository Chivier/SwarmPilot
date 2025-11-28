#!/bin/bash
set -e

echo "Starting T2Img (Text-to-Image) Service Container..."

# Start the HTTP service using uv
exec uv run python main.py
