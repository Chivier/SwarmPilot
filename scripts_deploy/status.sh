#!/bin/bash
# Check SwarmPilot cluster status.
# Usage: ./scripts/status.sh
#
# Can be run from any node that can reach the head node.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="python3 $SCRIPT_DIR/_config.py"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

HEAD_NODE=$($CFG head_node)
PLANNER_PORT=$($CFG planner_port)
PLANNER_URL="http://$HEAD_NODE:$PLANNER_PORT"
MODEL_COUNT=$($CFG model_count)

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   SwarmPilot Cluster Status                      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ── Planner ───────────────────────────────────────────────────
echo -n "  Planner ($PLANNER_URL): "
if curl -sf "$PLANNER_URL/v1/health" > /dev/null 2>&1; then
    echo -e "${GREEN}UP${NC}"
else
    echo -e "${RED}DOWN${NC}"
fi

# ── PyLet ─────────────────────────────────────────────────────
echo -n "  PyLet: "
PYLET_STATUS=$(curl -s "$PLANNER_URL/v1/pylet/status" 2>/dev/null || echo '{}')
echo "$PYLET_STATUS" | python3 -c "
import sys, json
d = json.load(sys.stdin)
enabled = d.get('pylet_enabled', False)
initialized = d.get('pylet_initialized', False)
active = d.get('active_instances', '?')
if enabled and initialized:
    print(f'\033[0;32mENABLED (active: {active})\033[0m')
elif enabled:
    print(f'\033[1;33mENABLED (not initialized)\033[0m')
else:
    print('\033[0;31mDISABLED\033[0m')
" 2>/dev/null || echo -e "${YELLOW}UNKNOWN${NC}"

# ── Schedulers ────────────────────────────────────────────────
echo ""
echo "  Schedulers:"
for i in $(seq 0 $((MODEL_COUNT - 1))); do
    MODEL_ID=$($CFG model_id.$i)
    SCHED_PORT=$($CFG scheduler_port.$i)
    SCHED_URL="http://$HEAD_NODE:$SCHED_PORT"
    EXPECTED=$($CFG replicas.$i)

    echo -n "    $MODEL_ID (:$SCHED_PORT): "
    if curl -sf "$SCHED_URL/v1/health" > /dev/null 2>&1; then
        echo -e "${GREEN}UP${NC}"
    else
        echo -e "${RED}DOWN${NC}"
        continue
    fi

    # Instance details
    INST_DATA=$(curl -s "$SCHED_URL/v1/instance/list" 2>/dev/null || echo '{}')
    echo "$INST_DATA" | python3 -c "
import sys, json
d = json.load(sys.stdin)
instances = d.get('instances', [])
expected = $EXPECTED
count = len(instances)
color = '\033[0;32m' if count == expected else '\033[1;33m'
print(f'      Instances: {color}{count}/{expected}\033[0m')
for inst in instances:
    iid = inst.get('instance_id', '?')
    ep = inst.get('endpoint', '?')
    status = inst.get('status', '?')
    print(f'        - {iid} @ {ep} [{status}]')
" 2>/dev/null || echo "      (could not parse instance list)"
done

# ── Scheduler registry (Planner view) ────────────────────────
echo ""
echo "  Scheduler Registry (Planner):"
curl -s "$PLANNER_URL/v1/schedulers" 2>/dev/null \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
if isinstance(d, dict):
    items = d.get('schedulers', d)
    if isinstance(items, dict):
        for model, url in items.items():
            print(f'    {model} -> {url}')
    elif isinstance(items, list):
        for s in items:
            mid = s.get('model_id', '?')
            url = s.get('scheduler_url', '?')
            print(f'    {mid} -> {url}')
    if not items:
        print('    (none)')
else:
    print(f'    {d}')
" 2>/dev/null || echo "    (could not reach Planner)"

echo ""
