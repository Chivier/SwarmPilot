#!/bin/bash
# Start PyLet head node.
# Usage: ./scripts_deploy/start_pylet_head.sh
#
# Run this on the head node BEFORE start_head.sh.
# The head node coordinates the PyLet cluster — workers connect to it.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="python3 $SCRIPT_DIR/_config.py"
LOG_DIR="/tmp/swarmpilot-cluster"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── Read config ───────────────────────────────────────────────
HEAD_NODE=$($CFG head_node)
PYLET_PORT=$($CFG pylet_head_port)

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   PyLet Cluster — Start Head Node                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Head node: $HEAD_NODE"
echo "  Port:      $PYLET_PORT"
echo ""

# ── Port check ────────────────────────────────────────────────
if lsof -i:"$PYLET_PORT" > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: port $PYLET_PORT already in use.${NC}"
    echo "PyLet head may already be running."
    exit 1
fi

mkdir -p "$LOG_DIR"

# ── Start PyLet head ──────────────────────────────────────────
echo -e "${BLUE}Starting PyLet head on :$PYLET_PORT ...${NC}"

python3 -c "
import pylet
pylet.start(port=$PYLET_PORT, block=True)
" > "$LOG_DIR/pylet-head.log" 2>&1 &
PYLET_PID=$!
echo $PYLET_PID > "$LOG_DIR/pylet-head.pid"

# Wait for head to be ready (GET /workers is PyLet's status endpoint)
echo "  Waiting for PyLet head..."
for attempt in $(seq 1 30); do
    if curl -sf "http://localhost:$PYLET_PORT/workers" > /dev/null 2>&1; then
        echo -e "${GREEN}  PyLet head started (PID: $PYLET_PID)${NC}"
        break
    fi
    if ! kill -0 $PYLET_PID 2>/dev/null; then
        echo -e "${RED}Error: PyLet head process died. Check $LOG_DIR/pylet-head.log${NC}"
        exit 1
    fi
    if [ "$attempt" -eq 30 ]; then
        echo -e "${RED}Error: PyLet head failed to start. Check $LOG_DIR/pylet-head.log${NC}"
        exit 1
    fi
    sleep 1
done

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   PyLet Head Ready                               ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "  URL: http://$HEAD_NODE:$PYLET_PORT"
echo "  PID: $PYLET_PID"
echo "  Log: $LOG_DIR/pylet-head.log"
echo ""
echo "Next: run start_pylet_worker.sh on each worker node."
