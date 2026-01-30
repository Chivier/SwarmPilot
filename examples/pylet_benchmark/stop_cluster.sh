#!/bin/bash
# PyLet Benchmark Cluster - Stop Services
# Usage: ./examples/pylet_benchmark/stop_cluster.sh
#
# Stops all services (Scheduler, Predictor) and any running instances.

set -e

LOG_DIR="/tmp/pylet_benchmark"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     PyLet Benchmark - Stopping Services               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to stop process by PID file
stop_service() {
    local name=$1
    local pid_file=$2

    if [ ! -f "$pid_file" ]; then
        echo -e "${YELLOW}! $name PID file not found: $pid_file${NC}"
        return 0
    fi

    local pid=$(cat "$pid_file" 2>/dev/null || echo "")
    if [ -z "$pid" ]; then
        echo -e "${YELLOW}! $name PID file empty${NC}"
        return 0
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
        echo -e "${YELLOW}! $name already stopped (PID $pid not found)${NC}"
        rm -f "$pid_file"
        return 0
    fi

    echo "Stopping $name (PID: $pid)..."
    if kill -TERM "$pid" 2>/dev/null; then
        # Wait a bit for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo -e "${GREEN}✓ $name stopped${NC}"
                rm -f "$pid_file"
                return 0
            fi
            sleep 0.5
        done

        # Force kill if still running
        echo "Force killing $name..."
        kill -9 "$pid" 2>/dev/null || true
        sleep 0.5
        echo -e "${GREEN}✓ $name force stopped${NC}"
        rm -f "$pid_file"
    else
        echo -e "${RED}✗ Failed to stop $name${NC}"
        return 1
    fi
}

# Stop main services
echo -e "${BLUE}Stopping main services...${NC}"
stop_service "Scheduler" "$LOG_DIR/scheduler.pid"
stop_service "Predictor" "$LOG_DIR/predictor.pid"

# Stop all instance processes
echo -e "${BLUE}Stopping instance processes...${NC}"
instance_pids=()
if compgen -G "$LOG_DIR"/instance_*.pid > /dev/null 2>&1; then
    for pid_file in "$LOG_DIR"/instance_*.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file" 2>/dev/null || echo "")
            if [ -n "$pid" ]; then
                instance_pids+=("$pid")
            fi
        fi
    done
fi

if [ ${#instance_pids[@]} -gt 0 ]; then
    echo "Found ${#instance_pids[@]} instance(s)"
    for pid in "${instance_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 1
    # Force kill any remaining
    for pid in "${instance_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    echo -e "${GREEN}✓ Instance processes stopped${NC}"
else
    echo -e "${YELLOW}! No instance PID files found${NC}"
fi

# Clean up PID files
rm -f "$LOG_DIR"/instance_*.pid

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     All services stopped                              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Logs available in:${NC} $LOG_DIR/"
echo ""
