#!/bin/bash
# Unified PyLet startup script — auto-detects head vs worker role.
# Usage: ./scripts_deploy/start_pylet.sh
#
# Checks local IPs against cluster.yaml head_node:
#   - If this machine IS the head node → start head + worker
#   - If this machine is NOT the head  → wait for head, then start worker
#
# Run this same script on every node in the cluster.

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

# ── Detect local IPs ─────────────────────────────────────────
# Collect all local IPv4 addresses via ip or ifconfig
LOCAL_IPS=""
if command -v ip > /dev/null 2>&1; then
    LOCAL_IPS=$(ip -4 addr show | grep -oP 'inet \K[\d.]+')
elif command -v ifconfig > /dev/null 2>&1; then
    LOCAL_IPS=$(ifconfig | grep -oP 'inet \K[\d.]+' 2>/dev/null \
             || ifconfig | grep 'inet ' | awk '{print $2}' | sed 's/addr://')
fi
# Always include localhost
LOCAL_IPS="$LOCAL_IPS 127.0.0.1"

# Check if this machine is the head node
IS_HEAD=false
for ip in $LOCAL_IPS; do
    if [ "$ip" = "$HEAD_NODE" ]; then
        IS_HEAD=true
        break
    fi
done

# Check if this machine is in the cluster node list
NODE_HOSTS=$($CFG node_hosts)
# Also include head_node (head may not be listed in nodes separately)
ALLOWED_IPS="$NODE_HOSTS $HEAD_NODE"

IN_CLUSTER=false
MATCHED_IP=""
for local_ip in $LOCAL_IPS; do
    for allowed_ip in $ALLOWED_IPS; do
        if [ "$local_ip" = "$allowed_ip" ]; then
            IN_CLUSTER=true
            MATCHED_IP="$local_ip"
            break 2
        fi
    done
done

THIS_HOST=$(hostname -I 2>/dev/null | awk '{print $1}' || hostname)

if [ "$IN_CLUSTER" = false ]; then
    echo -e "${RED}Error: This node ($THIS_HOST) is not in the cluster config.${NC}"
    echo "  Local IPs:    $LOCAL_IPS"
    echo "  Allowed nodes: $ALLOWED_IPS"
    echo "  Edit cluster.yaml to add this node, or run on a configured node."
    exit 1
fi

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   PyLet Cluster — Auto Start                     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "  This node:  $THIS_HOST ($MATCHED_IP)"
echo "  Head node:  $HEAD_NODE"
if [ "$IS_HEAD" = true ]; then
    echo -e "  Role:       ${GREEN}HEAD + WORKER${NC}"
else
    echo -e "  Role:       ${BLUE}WORKER${NC}"
fi
echo ""

mkdir -p "$LOG_DIR"

# ── Head node: start PyLet head ───────────────────────────────
if [ "$IS_HEAD" = true ]; then
    # Port check
    if lsof -i:"$PYLET_PORT" > /dev/null 2>&1; then
        echo -e "${YELLOW}PyLet head port $PYLET_PORT already in use, skipping head start.${NC}"
    else
        echo -e "${BLUE}[1/2] Starting PyLet head on :$PYLET_PORT ...${NC}"

        python3 -c "
import pylet
pylet.start(port=$PYLET_PORT, block=True)
" > "$LOG_DIR/pylet-head.log" 2>&1 &
        PYLET_PID=$!
        echo $PYLET_PID > "$LOG_DIR/pylet-head.pid"

        # Wait for head to be ready
        echo "  Waiting for PyLet head..."
        for attempt in $(seq 1 60); do
            if curl -sf "http://localhost:$PYLET_PORT/workers" > /dev/null 2>&1; then
                echo -e "${GREEN}  PyLet head started (PID: $PYLET_PID)${NC}"
                break
            fi
            if ! kill -0 $PYLET_PID 2>/dev/null; then
                echo -e "${RED}Error: PyLet head process died. Check $LOG_DIR/pylet-head.log${NC}"
                exit 1
            fi
            if [ "$attempt" -eq 60 ]; then
                echo -e "${RED}Error: PyLet head failed to start. Check $LOG_DIR/pylet-head.log${NC}"
                exit 1
            fi
            sleep 1
        done
        echo ""
    fi
fi

# ── All nodes: wait for head to be reachable ──────────────────
if [ "$IS_HEAD" != true ]; then
    echo -e "${BLUE}Waiting for PyLet head at $PYLET_HEAD_URL ...${NC}"
    MAX_WAIT=300
    WAITED=0
    while true; do
        if curl -sf "$PYLET_HEAD_URL/workers" > /dev/null 2>&1; then
            echo -e "${GREEN}  PyLet head is online${NC}"
            break
        fi
        WAITED=$((WAITED + 5))
        if [ "$WAITED" -ge "$MAX_WAIT" ]; then
            echo -e "${RED}Error: PyLet head not reachable after ${MAX_WAIT}s${NC}"
            echo "  Ensure start_pylet.sh is running on $HEAD_NODE"
            exit 1
        fi
        echo "  Head not ready yet, retrying in 5s... (${WAITED}s/${MAX_WAIT}s)"
        sleep 5
    done
    echo ""
fi

# ── All nodes: start PyLet worker ─────────────────────────────
STEP_LABEL="[2/2]"
[ "$IS_HEAD" != true ] && STEP_LABEL="[1/1]"

echo -e "${BLUE}${STEP_LABEL} Starting PyLet worker...${NC}"
echo "  Resources: ${WORKER_CPU} CPU, ${WORKER_GPU} GPU, ${WORKER_MEM} MB RAM"

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

# Verify process is alive
sleep 3
if ! kill -0 $WORKER_PID 2>/dev/null; then
    echo -e "${RED}Error: PyLet worker failed to start. Check $LOG_DIR/pylet-worker.log${NC}"
    tail -20 "$LOG_DIR/pylet-worker.log"
    exit 1
fi

echo -e "${GREEN}  PyLet worker started (PID: $WORKER_PID)${NC}"
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
if [ "$IS_HEAD" = true ]; then
    echo -e "${GREEN}║   PyLet Head + Worker Ready                      ║${NC}"
else
    echo -e "${GREEN}║   PyLet Worker Ready                             ║${NC}"
fi
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Node:      $THIS_HOST"
echo "  Head:      $PYLET_HEAD_URL"
echo "  Worker:    PID $WORKER_PID"
echo "  Resources: ${WORKER_CPU} CPU, ${WORKER_GPU} GPU, ${WORKER_MEM} MB"
echo "  Logs:      $LOG_DIR/pylet-{head,worker}.log"
