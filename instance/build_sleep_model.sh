#!/bin/bash
set -e

echo "Building Sleep Model Docker Image..."
echo ""

cd dockers/sleep_model

echo "Building Docker image..."
docker-compose build

echo ""
echo "✓ Sleep Model image built successfully!"
echo ""
echo "You can now test it with:"
echo "  python test_docker.py"
