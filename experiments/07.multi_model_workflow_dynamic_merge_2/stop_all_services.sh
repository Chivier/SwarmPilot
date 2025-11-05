#!/bin/bash

# Configuration
LOG_DIR="./logs"

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Stopping 02.multi_model_no_dep Experiment"
echo "========================================="

# Function to stop a service by PID file
stop_service() {
    local name=$1
    local pid_file="$LOG_DIR/${name}.pid"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo -n "Stopping $name (PID: $pid)..."

            # First try graceful shutdown (SIGTERM)
            kill -TERM $pid 2>/dev/null

            # Wait for process to stop (max 10 seconds)
            local count=0
            while ps -p $pid > /dev/null 2>&1 && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
            done

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

# Stop all services
if [ -d "$LOG_DIR" ]; then
    pids=()

    # Stop instances first (in parallel)
    echo "Stopping instance services (parallel)..."
    for pid_file in "$LOG_DIR"/instance-*.pid; do
        if [ -f "$pid_file" ]; then
            instance_name=$(basename "$pid_file" .pid)
            stop_service "$instance_name" &
            pids+=($!)
        fi
    done

    # Wait for all instances to stop
    for pid in "${pids[@]}"; do
        wait $pid
    done
    echo -e "${GREEN}All instances stopped${NC}"

    # Stop both schedulers (in parallel)
    echo ""
    echo "Stopping scheduler services (parallel)..."
    pids=()
    stop_service "scheduler-a" &
    pids+=($!)
    stop_service "scheduler-b" &
    pids+=($!)

    # Wait for schedulers to stop
    for pid in "${pids[@]}"; do
        wait $pid
    done
    echo -e "${GREEN}All schedulers stopped${NC}"

    # Stop predictor
    echo ""
    stop_service "predictor"
else
    echo -e "${YELLOW}Log directory not found. No services to stop.${NC}"
fi

# Clean up any remaining processes (fallback)
echo ""
echo "Cleaning up any remaining processes..."
pkill -f "spredictor start" 2>/dev/null && echo -e "${GREEN}Cleaned up predictor processes${NC}" || echo "No predictor processes found"
pkill -f "sscheduler start" 2>/dev/null && echo -e "${GREEN}Cleaned up scheduler processes${NC}" || echo "No scheduler processes found"
pkill -f "sinstance start" 2>/dev/null && echo -e "${GREEN}Cleaned up instance processes${NC}" || echo "No instance processes found"

# Stop all Docker containers for sleep_model
echo ""
echo "Stopping Docker containers..."
docker stop $(docker ps -a --filter "name=sleep_model" -q) && docker rm $(docker ps -a --filter "name=sleep_model" -q)

echo ""
echo "========================================="
echo -e "${GREEN}All services stopped successfully!${NC}"
echo "========================================="
echo ""
