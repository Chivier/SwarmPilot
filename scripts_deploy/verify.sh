#!/bin/bash
# Verify all model instances are active across both Planner and Schedulers.
# Usage: ./scripts_deploy/verify.sh [--timeout 300] [--interval 10]
#
# Polls until every expected instance reaches "active" status,
# or exits with error on timeout / failed instances.
# Can be run after deploy.sh or independently.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="python3 $SCRIPT_DIR/_config.py"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── Parse arguments ───────────────────────────────────────────
TIMEOUT=300
INTERVAL=10

while [[ $# -gt 0 ]]; do
    case $1 in
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        *) echo "Usage: $0 [--timeout SECONDS] [--interval SECONDS]"; exit 1 ;;
    esac
done

# ── Read config ───────────────────────────────────────────────
HEAD_NODE=$($CFG head_node)
PLANNER_PORT=$($CFG planner_port)
PLANNER_URL="http://$HEAD_NODE:$PLANNER_PORT"
MODEL_COUNT=$($CFG model_count)

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   SwarmPilot Cluster — Verify Deployment         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Timeout:  ${TIMEOUT}s"
echo "  Interval: ${INTERVAL}s"
echo ""

# ── Pre-flight ────────────────────────────────────────────────
if ! curl -sf "$PLANNER_URL/v1/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Planner not reachable at $PLANNER_URL${NC}"
    exit 1
fi

# Build expected model list
declare -A EXPECTED_REPLICAS
for i in $(seq 0 $((MODEL_COUNT - 1))); do
    MODEL_ID=$($CFG model_id.$i)
    REPLICAS=$($CFG replicas.$i)
    EXPECTED_REPLICAS["$MODEL_ID"]=$REPLICAS
    echo "  Expected: $MODEL_ID = $REPLICAS replicas"
done
echo ""

# ── Polling loop ──────────────────────────────────────────────
ELAPSED=0

while true; do
    ALL_READY=true
    HAS_FAILED=false
    STATUS_LINES=""

    # --- Check 1: Planner-side (PyLet managed instances) ------
    PYLET_STATUS=$(curl -s "$PLANNER_URL/v1/pylet/status" 2>/dev/null || echo '{}')
    PLANNER_RESULT=$(echo "$PYLET_STATUS" | python3 -c "
import sys, json

d = json.load(sys.stdin)
instances = d.get('active_instances', [])

# Count by model and status
by_model = {}
failed = []
for inst in instances:
    mid = inst.get('model_id', '?')
    st = inst.get('status', '?')
    by_model.setdefault(mid, {'active': 0, 'other': 0, 'total': 0})
    by_model[mid]['total'] += 1
    if st == 'active':
        by_model[mid]['active'] += 1
    else:
        by_model[mid]['other'] += 1
    if st == 'failed':
        failed.append(inst.get('instance_id', '?'))

for mid, counts in by_model.items():
    print(f'PLANNER|{mid}|{counts[\"active\"]}|{counts[\"total\"]}')

for fid in failed:
    print(f'FAILED|{fid}')
" 2>/dev/null || echo "")

    # --- Check 2: Scheduler-side (registered instances) -------
    SCHEDULER_RESULT=""
    for i in $(seq 0 $((MODEL_COUNT - 1))); do
        MODEL_ID=$($CFG model_id.$i)
        SCHED_PORT=$($CFG scheduler_port.$i)
        SCHED_URL="http://$HEAD_NODE:$SCHED_PORT"

        SCHED_LINE=$(curl -s "$SCHED_URL/v1/instance/list" 2>/dev/null | python3 -c "
import sys, json

d = json.load(sys.stdin)
instances = d.get('instances', [])
active = sum(1 for i in instances if i.get('status') == 'active')
total = len(instances)
print(f'SCHEDULER|$MODEL_ID|{active}|{total}')
" 2>/dev/null || echo "SCHEDULER|$MODEL_ID|0|0")
        SCHEDULER_RESULT="${SCHEDULER_RESULT}${SCHED_LINE}\n"
    done

    # --- Evaluate results -------------------------------------
    echo -e "${BLUE}[${ELAPSED}s/${TIMEOUT}s] Checking...${NC}"

    # Check for failed instances (immediate exit)
    FAILED_IDS=$(echo "$PLANNER_RESULT" | grep "^FAILED|" | cut -d'|' -f2)
    if [ -n "$FAILED_IDS" ]; then
        echo -e "${RED}FAILED instances detected:${NC}"
        echo "$FAILED_IDS" | while read -r fid; do
            echo -e "  ${RED}- $fid${NC}"
        done
        HAS_FAILED=true
    fi

    # Display per-model status from both sides
    for i in $(seq 0 $((MODEL_COUNT - 1))); do
        MODEL_ID=$($CFG model_id.$i)
        EXPECTED=${EXPECTED_REPLICAS["$MODEL_ID"]}

        # Planner side
        P_LINE=$(echo "$PLANNER_RESULT" | grep "^PLANNER|$MODEL_ID|" || echo "PLANNER|$MODEL_ID|0|0")
        P_ACTIVE=$(echo "$P_LINE" | cut -d'|' -f3)
        P_TOTAL=$(echo "$P_LINE" | cut -d'|' -f4)

        # Scheduler side
        S_LINE=$(echo -e "$SCHEDULER_RESULT" | grep "^SCHEDULER|$MODEL_ID|" || echo "SCHEDULER|$MODEL_ID|0|0")
        S_ACTIVE=$(echo "$S_LINE" | cut -d'|' -f3)
        S_TOTAL=$(echo "$S_LINE" | cut -d'|' -f4)

        # Status indicator
        if [ "$S_ACTIVE" = "$EXPECTED" ] && [ "$P_ACTIVE" = "$EXPECTED" ]; then
            ICON="${GREEN}OK${NC}"
        else
            ICON="${YELLOW}..${NC}"
            ALL_READY=false
        fi

        echo -e "  [$ICON] $MODEL_ID"
        echo -e "        Planner:   ${P_ACTIVE}/${EXPECTED} active (${P_TOTAL} total)"
        echo -e "        Scheduler: ${S_ACTIVE}/${EXPECTED} active (${S_TOTAL} total)"
    done
    echo ""

    # --- Exit conditions --------------------------------------
    if [ "$HAS_FAILED" = true ]; then
        echo -e "${RED}Deployment has FAILED instances. Check logs:${NC}"
        echo "  curl -s $PLANNER_URL/v1/pylet/status | python3 -m json.tool"
        exit 1
    fi

    if [ "$ALL_READY" = true ]; then
        echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║   All instances ACTIVE                           ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
        exit 0
    fi

    # --- Timeout check ----------------------------------------
    ELAPSED=$((ELAPSED + INTERVAL))
    if [ "$ELAPSED" -gt "$TIMEOUT" ]; then
        echo -e "${RED}Timeout after ${TIMEOUT}s. Not all instances are active.${NC}"
        echo ""
        echo "Troubleshooting:"
        echo "  # Planner-side details (shows deploying/failed/etc.)"
        echo "  curl -s $PLANNER_URL/v1/pylet/status | python3 -m json.tool"
        echo ""
        echo "  # Per-model Scheduler instances"
        for i in $(seq 0 $((MODEL_COUNT - 1))); do
            SCHED_PORT=$($CFG scheduler_port.$i)
            echo "  curl -s http://$HEAD_NODE:$SCHED_PORT/v1/instance/list | python3 -m json.tool"
        done
        echo ""
        echo "  # PyLet worker status"
        PYLET_HEAD_URL=$($CFG pylet_head_url)
        echo "  curl -s $PYLET_HEAD_URL/workers | python3 -m json.tool"
        exit 1
    fi

    sleep "$INTERVAL"
done
