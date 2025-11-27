#!/bin/bash

# ============================================================================
# Type3 Text2Image+Video - Service Shutdown Script
# ============================================================================
# Stops all services started by start_type3_sim_services.sh or
# start_type3_real_services.sh.
#
# This script:
#   1. Stops instances (Group A, C, B) in parallel
#   2. Stops schedulers (A, C, B) in parallel
#   3. Stops planner and predictor
#   4. Cleans up any remaining processes
#   5. Stops Docker containers for model services
# ============================================================================

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$EXPERIMENT_DIR/logs"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo "Stopping Type3 Text2Image+Video Services"
echo "========================================="

# Stop service by PID file
stop_service() {
    local name=$1
    local pid_file="$LOG_DIR/${name}.pid"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo -n "Stopping $name (PID: $pid)..."

            # Graceful shutdown
            kill -TERM $pid 2>/dev/null

            # Wait up to 10 seconds
            local count=0
            while ps -p $pid > /dev/null 2>&1 && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
            done

            # Force kill if needed
            if ps -p $pid > /dev/null 2>&1; then
                echo -e " ${YELLOW}Force killing...${NC}"
                kill -9 $pid 2>/dev/null
                sleep 1

                if ps -p $pid > /dev/null 2>&1; then
                    echo -e " ${RED}Failed to stop${NC}"
                else
                    echo -e " ${GREEN}OK${NC}"
                fi
            else
                echo -e " ${GREEN}OK${NC}"
            fi
        else
            echo -e "${YELLOW}$name (PID: $pid) not running${NC}"
        fi
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}No PID file found for $name${NC}"
    fi
}

if [ -d "$LOG_DIR" ]; then
    pids=()

    # Stop all instances (Group A, C, B) in parallel
    echo "Stopping instance services (parallel)..."
    for pid_file in "$LOG_DIR"/instance-*.pid; do
        if [ -f "$pid_file" ]; then
            instance_name=$(basename "$pid_file" .pid)
            stop_service "$instance_name" &
            pids+=($!)
        fi
    done

    for pid in "${pids[@]}"; do
        wait $pid
    done
    echo -e "${GREEN}All instances stopped${NC}"

    # Stop all three schedulers in parallel
    echo ""
    echo "Stopping scheduler services (parallel)..."
    pids=()
    stop_service "scheduler-a" &
    pids+=($!)
    stop_service "scheduler-c" &
    pids+=($!)
    stop_service "scheduler-b" &
    pids+=($!)

    for pid in "${pids[@]}"; do
        wait $pid
    done
    echo -e "${GREEN}All schedulers stopped${NC}"

    # Stop planner
    echo ""
    stop_service "planner"

    # Stop predictor
    echo ""
    stop_service "predictor"
else
    echo -e "${YELLOW}Log directory not found. No services to stop.${NC}"
fi

# Cleanup any remaining processes
echo ""
echo "Cleaning up any remaining processes..."
pkill -f "predictor.*start" 2>/dev/null && echo -e "${GREEN}Cleaned up predictor processes${NC}" || echo "No predictor processes found"
pkill -f "planner.*uvicorn" 2>/dev/null && echo -e "${GREEN}Cleaned up planner processes${NC}" || echo "No planner processes found"
pkill -f "scheduler.*start" 2>/dev/null && echo -e "${GREEN}Cleaned up scheduler processes${NC}" || echo "No scheduler processes found"
pkill -f "instance.*start" 2>/dev/null && echo -e "${GREEN}Cleaned up instance processes${NC}" || echo "No instance processes found"

# Stop Docker containers
echo ""
echo "Stopping Docker containers..."
docker stop $(docker ps -a --filter "name=sleep_model" -q) 2>/dev/null && \
    docker rm $(docker ps -a --filter "name=sleep_model" -q) 2>/dev/null || \
    echo "No sleep_model containers found"

docker stop $(docker ps -a --filter "name=llm_service" -q) 2>/dev/null && \
    docker rm $(docker ps -a --filter "name=llm_service" -q) 2>/dev/null || \
    echo "No llm_service containers found"

docker stop $(docker ps -a --filter "name=t2vid" -q) 2>/dev/null && \
    docker rm $(docker ps -a --filter "name=t2vid" -q) 2>/dev/null || \
    echo "No t2vid containers found"

docker stop $(docker ps -a --filter "name=t2img" -q) 2>/dev/null && \
    docker rm $(docker ps -a --filter "name=t2img" -q) 2>/dev/null || \
    echo "No t2img containers found"

echo ""
echo "========================================="
echo -e "${GREEN}All Type3 services stopped successfully!${NC}"
echo "========================================="
echo ""
