#!/bin/sh
set -e

echo "Starting Sleep Model Container..."

# Start the HTTP service
exec python main.py
