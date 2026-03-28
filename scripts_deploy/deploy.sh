#!/bin/bash
# Deploy all model instances via PyLet.
# Usage: ./scripts/deploy.sh
#
# Reads models + replicas from cluster.yaml and calls
# Planner's /v1/pylet/deploy_manually endpoint.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="python3 $SCRIPT_DIR/_config.py"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── Read config ───────────────────────────────────────────────
HEAD_NODE=$($CFG head_node)
PLANNER_PORT=$($CFG planner_port)
PLANNER_URL="http://$HEAD_NODE:$PLANNER_PORT"
MODELS_JSON=$($CFG models_json)
MODEL_COUNT=$($CFG model_count)

echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   SwarmPilot Cluster — Deploy Instances          ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ── Pre-flight checks ────────────────────────────────────────
echo "Checking Planner at $PLANNER_URL ..."
if ! curl -sf "$PLANNER_URL/v1/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Planner not reachable. Run start_head.sh first.${NC}"
    exit 1
fi
echo -e "${GREEN}Planner is healthy${NC}"

echo "Checking PyLet status..."
PYLET_STATUS=$(curl -s "$PLANNER_URL/v1/pylet/status" 2>/dev/null)
PYLET_ENABLED=$(echo "$PYLET_STATUS" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('true' if d.get('pylet_enabled') else 'false')
" 2>/dev/null || echo "false")

if [ "$PYLET_ENABLED" != "true" ]; then
    echo -e "${RED}Error: PyLet is not enabled on the Planner.${NC}"
    echo "  Check PYLET_ENABLED and PYLET_HEAD_URL in start_head.sh"
    exit 1
fi
echo -e "${GREEN}PyLet is enabled${NC}"
echo ""

# ── Show deployment plan ──────────────────────────────────────
echo "Deployment plan:"
TOTAL=0
for i in $(seq 0 $((MODEL_COUNT - 1))); do
    MODEL_ID=$($CFG model_id.$i)
    REPLICAS=$($CFG replicas.$i)
    TOTAL=$((TOTAL + REPLICAS))
    echo "  $MODEL_ID: $REPLICAS replicas"
done
echo "  Total: $TOTAL instances"
echo ""

# ── Deploy via PyLet ──────────────────────────────────────────
echo -e "${BLUE}Deploying via /v1/pylet/deploy_manually ...${NC}"
echo "  target_state: $MODELS_JSON"
echo ""

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
    "$PLANNER_URL/v1/pylet/deploy_manually" \
    -H "Content-Type: application/json" \
    -d "{
        \"target_state\": $MODELS_JSON,
        \"wait_for_ready\": true
    }")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)

if [ "$HTTP_CODE" -ge 200 ] && [ "$HTTP_CODE" -lt 300 ]; then
    echo -e "${GREEN}Deployment successful (HTTP $HTTP_CODE)${NC}"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
else
    echo -e "${RED}Deployment failed (HTTP $HTTP_CODE)${NC}"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
    exit 1
fi

echo ""

# ── Verify all instances are active ────────────────────────────
echo "Running verification..."
echo ""
exec "$SCRIPT_DIR/verify.sh" --timeout 600 --interval 10
