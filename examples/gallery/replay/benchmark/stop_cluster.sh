#!/bin/bash
# Real-Model Benchmark — Stop Cluster
# Usage: bash examples/gallery/replay/benchmark/stop_cluster.sh [config.yaml]

set -e

source "$(dirname "$0")/_parse_config.sh" "$1"

mkdir -p "$LOG_DIR"

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Real-Model Benchmark — Shutdown                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# Helper: send SIGTERM, wait 0.5s, SIGKILL if still alive, remove PID file.
stop_process() {
    local name=$1
    local pid_file="$LOG_DIR/$2.pid"
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null || true
            sleep 0.5
            kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null || true
            echo -e "${GREEN}  Stopped $name (PID: $PID)${NC}"
        else
            echo -e "${YELLOW}  $name already stopped (PID: $PID)${NC}"
        fi
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}  $name — no PID file found, skipping${NC}"
    fi
}

# [1/3] Gracefully terminate managed instances via Planner.
echo -e "${BLUE}[1/3] Terminating managed instances...${NC}"
cd "$SWARMPILOT_ROOT" && \
    uv run splanner terminate --all --planner-url "http://localhost:$PLANNER_PORT" 2>/dev/null \
    && echo -e "${GREEN}Managed instances terminated${NC}" \
    || echo -e "${YELLOW}splanner terminate skipped (planner may be down)${NC}"

# [2/3] Stop Schedulers.
echo -e "${BLUE}[2/3] Stopping Schedulers...${NC}"
stop_process "Scheduler-Large" "scheduler-large"
stop_process "Scheduler-Small" "scheduler-small"

# [3/3] Stop Planner + Dummy Health.
# Killing the Planner also stops its local PyLet subprocess tree.
echo -e "${BLUE}[3/3] Stopping Planner + Dummy Health...${NC}"
stop_process "Planner" "planner"
stop_process "Dummy Health" "dummy_health"

# Clean up any orphaned PyLet processes on configured ports.
for port in "$PYLET_LOCAL_PORT" "$PYLET_LOCAL_WORKER_PORT_START"; do
    PID=$(lsof -ti:"$port" 2>/dev/null || true)
    if [ -n "$PID" ]; then
        kill "$PID" 2>/dev/null || true
        echo -e "${GREEN}  Killed orphaned process on :$port (PID: $PID)${NC}"
    fi
done

echo ""
echo -e "${GREEN}All services stopped. Logs at: $LOG_DIR/${NC}"
