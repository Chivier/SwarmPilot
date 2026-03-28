#!/bin/bash
# Start PyLet worker on a compute node.
# Usage: ./scripts_deploy/start_pylet_worker.sh
#
# Run this on each worker node (including the head node if it also
# serves as a compute node). The worker connects to the PyLet head
# and advertises its GPU/CPU/memory resources.

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
PYLET_HEAD_URL="http://$HEAD_NODE:$PYLET_PORT"
WORKER_PORT=$($CFG pylet_worker_port)
WORKER_CPU=$($CFG pylet_worker_cpu)
WORKER_GPU=$($CFG pylet_worker_gpu)
WORKER_MEM=$($CFG pylet_worker_mem)

# Detect current hostname/IP for display
THIS_HOST=$(hostname -I 2>/dev/null | awk '{print $1}' || hostname)

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   PyLet Cluster — Start Worker Node              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "  This node:    $THIS_HOST"
echo "  PyLet head:   $PYLET_HEAD_URL"
echo "  Worker port:  $WORKER_PORT"
echo "  Resources:    ${WORKER_CPU} CPU, ${WORKER_GPU} GPU, ${WORKER_MEM} MB RAM"
echo ""

# ── Check head reachability ───────────────────────────────────
echo "Checking PyLet head at $PYLET_HEAD_URL ..."
HEAD_OK=false
for attempt in $(seq 1 10); do
    if curl -sf "$PYLET_HEAD_URL" > /dev/null 2>&1 || \
       curl -sf "$PYLET_HEAD_URL/health" > /dev/null 2>&1; then
        HEAD_OK=true
        break
    fi
    sleep 1
done

if [ "$HEAD_OK" = false ]; then
    echo -e "${RED}Error: cannot reach PyLet head at $PYLET_HEAD_URL${NC}"
    echo "Make sure start_pylet_head.sh has been run on the head node."
    exit 1
fi
echo -e "${GREEN}PyLet head is reachable${NC}"
echo ""

mkdir -p "$LOG_DIR"

# ── Start PyLet worker ────────────────────────────────────────
echo -e "${BLUE}Starting PyLet worker...${NC}"

python3 -c "
import pylet
pylet.start(
    address='$PYLET_HEAD_URL',
    port=$WORKER_PORT,
    cpu=$WORKER_CPU,
    gpu=$WORKER_GPU,
    memory=$WORKER_MEM,
    block=True,
)
" > "$LOG_DIR/pylet-worker.log" 2>&1 &
WORKER_PID=$!
echo $WORKER_PID > "$LOG_DIR/pylet-worker.pid"

# Verify process is alive after a brief startup
sleep 3
if ! kill -0 $WORKER_PID 2>/dev/null; then
    echo -e "${RED}Error: PyLet worker failed to start. Check $LOG_DIR/pylet-worker.log${NC}"
    cat "$LOG_DIR/pylet-worker.log" | tail -20
    exit 1
fi

echo -e "${GREEN}  PyLet worker started (PID: $WORKER_PID)${NC}"
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   PyLet Worker Ready                             ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Node:      $THIS_HOST"
echo "  PID:       $WORKER_PID"
echo "  Log:       $LOG_DIR/pylet-worker.log"
echo "  Resources: ${WORKER_CPU} CPU, ${WORKER_GPU} GPU, ${WORKER_MEM} MB"
echo ""
echo "Worker is connected to head at $PYLET_HEAD_URL."
echo "Repeat this on all other worker nodes."
