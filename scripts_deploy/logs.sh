#!/bin/bash
# View PyLet instance logs for debugging.
# Usage:
#   ./scripts_deploy/logs.sh                  # list all instances, pick interactively
#   ./scripts_deploy/logs.sh <pylet_id>       # show logs for specific instance
#   ./scripts_deploy/logs.sh --model <name>   # list instances for a model
#   ./scripts_deploy/logs.sh --failed         # show logs of all failed instances

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="python3 $SCRIPT_DIR/_config.py"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

HEAD_NODE=$($CFG head_node)
PYLET_PORT=$($CFG pylet_head_port)
PYLET_URL="http://$HEAD_NODE:$PYLET_PORT"
PLANNER_PORT=$($CFG planner_port)
PLANNER_URL="http://$HEAD_NODE:$PLANNER_PORT"

# ── Helper: fetch and decode logs for a pylet_id ─────────────
fetch_logs() {
    local pid="$1"
    local raw
    raw=$(curl -s "$PYLET_URL/instances/$pid/logs" 2>/dev/null)
    if [ -z "$raw" ] || echo "$raw" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'detail' in d:
        print(d['detail'], file=sys.stderr)
        sys.exit(1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        echo -e "${RED}Failed to fetch logs for $pid${NC}" >&2
        return 1
    fi

    echo "$raw" | python3 -c "
import sys, json, base64
d = json.load(sys.stdin)
content = d.get('content', '')
if content:
    print(base64.b64decode(content).decode(errors='replace'))
else:
    print('(empty logs)')
" 2>/dev/null
}

# ── Helper: list instances from PyLet + Planner ──────────────
list_instances() {
    local model_filter="$1"

    # Get Planner-managed instances (has model_id and status)
    local planner_data
    planner_data=$(curl -s "$PLANNER_URL/v1/pylet/status" 2>/dev/null || echo '{}')

    # Get PyLet instances (has runtime details)
    local pylet_data
    pylet_data=$(curl -s "$PYLET_URL/instances" 2>/dev/null || echo '[]')

    python3 -c "
import sys, json

planner = json.loads('''$planner_data''')
pylet_raw = json.loads('''$pylet_data''')

# Build PyLet instance lookup by id
pylet_map = {}
if isinstance(pylet_raw, list):
    for inst in pylet_raw:
        pylet_map[inst.get('id', '')] = inst

# Merge Planner + PyLet data
managed = planner.get('active_instances', [])
model_filter = '''$model_filter'''

# Also check PyLet instances not in Planner (e.g., failed early)
planner_ids = {m.get('pylet_id', '') for m in managed}

rows = []
for m in managed:
    mid = m.get('model_id', '?')
    if model_filter and model_filter.lower() not in mid.lower():
        continue
    pid = m.get('pylet_id', '?')
    status = m.get('status', '?')
    endpoint = m.get('endpoint', '') or ''
    error = m.get('error', '') or ''
    rows.append((pid, mid, status, endpoint, error))

# Add PyLet-only instances (not in Planner managed list)
for pid, inst in pylet_map.items():
    if pid in planner_ids:
        continue
    mid = ''
    for lbl_key in ('model_id',):
        mid = inst.get('labels', {}).get(lbl_key, '')
        if mid:
            break
    if not mid:
        mid = inst.get('name', '?')
    if model_filter and model_filter.lower() not in mid.lower():
        continue
    status = inst.get('status', '?')
    endpoint = ''
    error = inst.get('failure_reason', '') or ''
    rows.append((pid, mid, f'pylet:{status}', endpoint, error))

if not rows:
    print('No instances found.')
    sys.exit(0)

# Print table
header = f'{'ID':<38} {'MODEL':<45} {'STATUS':<18} {'ERROR'}'
print(header)
print('-' * len(header))
for pid, mid, status, endpoint, error in sorted(rows, key=lambda r: (r[1], r[2])):
    short_err = (error[:40] + '...') if len(error) > 40 else error
    print(f'{pid:<38} {mid:<45} {status:<18} {short_err}')
print()
print(f'Total: {len(rows)} instance(s)')
" 2>/dev/null
}

# ── Parse arguments ───────────────────────────────────────────
MODE="interactive"
PYLET_ID=""
MODEL_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODE="list"
            MODEL_FILTER="$2"
            shift 2
            ;;
        --failed)
            MODE="failed"
            shift
            ;;
        --help|-h)
            echo "Usage:"
            echo "  $0                    List all instances"
            echo "  $0 <pylet_id>         Show logs for a specific instance"
            echo "  $0 --model <name>     List instances matching model name"
            echo "  $0 --failed           Show logs of all failed instances"
            exit 0
            ;;
        *)
            PYLET_ID="$1"
            MODE="single"
            shift
            ;;
    esac
done

# ── Check PyLet head ──────────────────────────────────────────
if ! curl -sf "$PYLET_URL/workers" > /dev/null 2>&1; then
    echo -e "${RED}Error: PyLet head not reachable at $PYLET_URL${NC}"
    exit 1
fi

# ── Mode: single instance logs ────────────────────────────────
if [ "$MODE" = "single" ]; then
    echo -e "${BLUE}Logs for instance: $PYLET_ID${NC}"
    echo ""

    # Show instance details first
    DETAIL=$(curl -s "$PYLET_URL/instances/$PYLET_ID" 2>/dev/null)
    echo "$DETAIL" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(f'  Name:    {d.get(\"name\", \"?\")}')
    print(f'  Status:  {d.get(\"status\", \"?\")}')
    labels = d.get('labels', {})
    print(f'  Model:   {labels.get(\"model_id\", \"?\")}')
    fr = d.get('failure_reason', '')
    if fr:
        print(f'  Failure: {fr}')
    print()
except Exception:
    pass
" 2>/dev/null

    echo -e "${CYAN}--- stdout/stderr ---${NC}"
    fetch_logs "$PYLET_ID"
    exit $?
fi

# ── Mode: failed instance logs ────────────────────────────────
if [ "$MODE" = "failed" ]; then
    echo -e "${BLUE}Fetching logs for failed instances...${NC}"
    echo ""

    # Get failed from Planner
    PLANNER_DATA=$(curl -s "$PLANNER_URL/v1/pylet/status" 2>/dev/null || echo '{}')

    # Get failed/cancelled from PyLet directly
    PYLET_DATA=$(curl -s "$PYLET_URL/instances" 2>/dev/null || echo '[]')

    FAILED_IDS=$(python3 -c "
import sys, json

planner = json.loads('''$PLANNER_DATA''')
pylet_raw = json.loads('''$PYLET_DATA''')

ids = set()

# Planner-side failed
for inst in planner.get('active_instances', []):
    if inst.get('status') == 'failed':
        ids.add(inst.get('pylet_id', ''))

# PyLet-side failed/cancelled (not in Planner if they failed early)
if isinstance(pylet_raw, list):
    for inst in pylet_raw:
        st = inst.get('status', '').lower()
        if st in ('failed', 'cancelled', 'error'):
            ids.add(inst.get('id', ''))

for pid in sorted(ids):
    if pid:
        print(pid)
" 2>/dev/null)

    if [ -z "$FAILED_IDS" ]; then
        echo "No failed instances found."
        echo ""
        echo "Tip: instances that failed very early may have been cleaned up."
        echo "Check Planner logs: tail -100 /tmp/swarmpilot-cluster/planner.log"
        exit 0
    fi

    echo "$FAILED_IDS" | while read -r pid; do
        echo -e "${RED}════════════════════════════════════════════════════${NC}"
        echo -e "${RED}Instance: $pid${NC}"

        # Details
        DETAIL=$(curl -s "$PYLET_URL/instances/$pid" 2>/dev/null)
        echo "$DETAIL" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    labels = d.get('labels', {})
    print(f'  Model:   {labels.get(\"model_id\", d.get(\"name\", \"?\"))}')
    print(f'  Status:  {d.get(\"status\", \"?\")}')
    fr = d.get('failure_reason', '')
    if fr:
        print(f'  Failure: {fr}')
except Exception:
    pass
" 2>/dev/null

        echo -e "${CYAN}--- logs ---${NC}"
        fetch_logs "$pid" || true
        echo ""
    done
    exit 0
fi

# ── Mode: list / interactive ──────────────────────────────────
echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   PyLet Instance Logs                            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""

list_instances "$MODEL_FILTER"

echo ""
echo "Usage:"
echo "  $0 <pylet_id>       View logs for a specific instance"
echo "  $0 --failed         View logs of all failed instances"
echo "  $0 --model 80B      Filter instances by model name"
