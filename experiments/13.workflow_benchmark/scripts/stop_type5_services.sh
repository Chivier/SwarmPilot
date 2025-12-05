#!/bin/bash
#
# Stop services for Type5 OOD Recovery Experiment
#
# Stops all services started by start_type5_services.sh:
#   - Predictor service
#   - Scheduler
#   - All instances
#
# Usage:
#   ./stop_type5_services.sh

set -e

# Get the experiment directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Log directory (where PIDs are stored)
LOG_DIR="$EXPERIMENT_DIR/logs_type5"

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Stopping Type5 OOD Recovery Services"
echo "========================================="

# Stop function
stop_service() {
    local name=$1
    local pid_file="$LOG_DIR/${name}.pid"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}Stopping $name (PID: $pid)...${NC}"
            kill "$pid" 2>/dev/null || true
            sleep 1
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
            echo -e "${GREEN}Stopped $name${NC}"
        else
            echo -e "${YELLOW}$name (PID: $pid) not running${NC}"
        fi
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}No PID file found for $name${NC}"
    fi
}

# Stop instances first
echo ""
echo "Stopping instances..."
for pid_file in "$LOG_DIR"/instance-*.pid; do
    if [ -f "$pid_file" ]; then
        name=$(basename "$pid_file" .pid)
        stop_service "$name"
    fi
done

# Stop scheduler
echo ""
echo "Stopping scheduler..."
stop_service "scheduler"

# Stop predictor
echo ""
echo "Stopping predictor..."
stop_service "predictor"

# Kill any remaining processes on known ports
echo ""
echo "Cleaning up any remaining processes..."

# Predictor port
if lsof -ti:8101 > /dev/null 2>&1; then
    echo -e "${YELLOW}Killing process on port 8101 (predictor)${NC}"
    kill $(lsof -ti:8101) 2>/dev/null || true
fi

# Scheduler port
if lsof -ti:8100 > /dev/null 2>&1; then
    echo -e "${YELLOW}Killing process on port 8100 (scheduler)${NC}"
    kill $(lsof -ti:8100) 2>/dev/null || true
fi

# Instance ports (8210-8299)
for port in $(seq 8210 8250); do
    if lsof -ti:$port > /dev/null 2>&1; then
        echo -e "${YELLOW}Killing process on port $port${NC}"
        kill $(lsof -ti:$port) 2>/dev/null || true
    fi
done

echo ""
echo "========================================="
echo -e "${GREEN}All Type5 services stopped${NC}"
echo "========================================="
