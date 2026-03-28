#!/bin/bash
# Deploy all model instances via PyLet.
# Usage: ./scripts_deploy/deploy.sh
#
# Deploys each model separately via `splanner serve` to support
# per-model GPU count (e.g. 1 GPU for 8B, 2 GPUs for 80B).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
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
PYLET_ENABLED=$(curl -s "$PLANNER_URL/v1/pylet/status" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('true' if d.get('pylet_enabled') else 'false')
" 2>/dev/null || echo "false")

if [ "$PYLET_ENABLED" != "true" ]; then
    echo -e "${RED}Error: PyLet is not enabled on the Planner.${NC}"
    exit 1
fi
echo -e "${GREEN}PyLet is enabled${NC}"
echo ""

# ── Show deployment plan ──────────────────────────────────────
echo "Deployment plan:"
TOTAL_GPU=0
for i in $(seq 0 $((MODEL_COUNT - 1))); do
    MODEL_ID=$($CFG model_id.$i)
    REPLICAS=$($CFG replicas.$i)
    GPU=$($CFG gpu.$i)
    MODEL_GPU=$((REPLICAS * GPU))
    TOTAL_GPU=$((TOTAL_GPU + MODEL_GPU))
    echo "  $MODEL_ID: $REPLICAS replicas × $GPU GPU = $MODEL_GPU GPUs"
done
echo "  Total GPU usage: $TOTAL_GPU"
echo ""

# ── Deploy each model separately ──────────────────────────────
cd "$PROJECT_ROOT"

for i in $(seq 0 $((MODEL_COUNT - 1))); do
    MODEL_ID=$($CFG model_id.$i)
    REPLICAS=$($CFG replicas.$i)
    GPU=$($CFG gpu.$i)
    STEP=$((i + 1))

    echo -e "${BLUE}[$STEP/$MODEL_COUNT] Deploying $MODEL_ID ($REPLICAS replicas, $GPU GPU each)...${NC}"

    uv run splanner serve "$MODEL_ID" \
        --gpu "$GPU" \
        --replicas "$REPLICAS" \
        --planner-url "$PLANNER_URL"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  $MODEL_ID deployed${NC}"
    else
        echo -e "${RED}  $MODEL_ID deployment failed${NC}"
        exit 1
    fi
    echo ""
done

# ── Verify all instances are active ────────────────────────────
echo "Running verification..."
echo ""
exec "$SCRIPT_DIR/verify.sh" --timeout 600 --interval 10
