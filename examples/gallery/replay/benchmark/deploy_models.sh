#!/bin/bash
# Real-Model Benchmark — Deploy vLLM Instances
# Usage: bash examples/gallery/replay/benchmark/deploy_models.sh [config.yaml]
set -e
source "$(dirname "$0")/_parse_config.sh" "$1"

PLANNER_URL="http://localhost:$PLANNER_PORT"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        Real-Model Benchmark — Deploy vLLM Instances      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${YELLOW}Large model:${NC}  $LARGE_MODEL_ID"
echo -e "  ${YELLOW}  replicas:${NC}   $REPLICAS_LARGE  (${GPU_PER_INSTANCE_LARGE} GPU each)"
echo -e "  ${YELLOW}Small model:${NC}  $SMALL_MODEL_ID"
echo -e "  ${YELLOW}  replicas:${NC}   $REPLICAS_SMALL  (${GPU_PER_INSTANCE_SMALL} GPU each)"
echo -e "  ${YELLOW}Planner:${NC}      $PLANNER_URL"
echo ""

cd "$SWARMPILOT_ROOT"

# ── [1/3] Deploy large model ───────────────────────────────────────
echo -e "${BLUE}[1/3]${NC} Deploying large model: ${YELLOW}${LARGE_MODEL_ID}${NC}"
echo -e "      replicas=${REPLICAS_LARGE}, gpu=${GPU_PER_INSTANCE_LARGE}"
echo ""
uv run splanner serve "$LARGE_MODEL_ID" \
    --gpu "$GPU_PER_INSTANCE_LARGE" \
    --replicas "$REPLICAS_LARGE" \
    --planner-url "$PLANNER_URL" \
    | while read -r line; do echo "  $line"; done
echo ""
echo -e "  ${GREEN}Large model deployment submitted.${NC}"
echo ""

# ── [2/3] Deploy small model ───────────────────────────────────────
echo -e "${BLUE}[2/3]${NC} Deploying small model: ${YELLOW}${SMALL_MODEL_ID}${NC}"
echo -e "      replicas=${REPLICAS_SMALL}, gpu=${GPU_PER_INSTANCE_SMALL}"
echo ""
uv run splanner serve "$SMALL_MODEL_ID" \
    --gpu "$GPU_PER_INSTANCE_SMALL" \
    --replicas "$REPLICAS_SMALL" \
    --planner-url "$PLANNER_URL" \
    | while read -r line; do echo "  $line"; done
echo ""
echo -e "  ${GREEN}Small model deployment submitted.${NC}"
echo ""

# ── [3/3] Verify deployments ───────────────────────────────────────
echo -e "${BLUE}[3/3]${NC} Verifying deployments..."
echo ""

echo -e "  ${YELLOW}splanner ps:${NC}"
uv run splanner ps --planner-url "$PLANNER_URL" \
    | while read -r line; do echo "    $line"; done
echo ""

echo -e "  ${YELLOW}Large scheduler instances${NC} (port ${SCHEDULER_LARGE_PORT}):"
curl -s "http://localhost:${SCHEDULER_LARGE_PORT}/v1/instance/list" \
    | "$VENV_PYTHON" -c "
import json, sys
data = json.load(sys.stdin)
instances = data if isinstance(data, list) else data.get('instances', [])
if not instances:
    print('  (no instances)')
else:
    for inst in instances:
        print(f'  {inst}')
" | while read -r line; do echo "    $line"; done
echo ""

echo -e "  ${YELLOW}Small scheduler instances${NC} (port ${SCHEDULER_SMALL_PORT}):"
curl -s "http://localhost:${SCHEDULER_SMALL_PORT}/v1/instance/list" \
    | "$VENV_PYTHON" -c "
import json, sys
data = json.load(sys.stdin)
instances = data if isinstance(data, list) else data.get('instances', [])
if not instances:
    print('  (no instances)')
else:
    for inst in instances:
        print(f'  {inst}')
" | while read -r line; do echo "    $line"; done
echo ""

# ── Next steps ─────────────────────────────────────────────────────
echo -e "${GREEN}Deployment complete.${NC}"
echo ""
echo -e "${YELLOW}Next:${NC} run the benchmark"
echo ""
echo "  uv run python examples/gallery/replay/benchmark/benchmark_runner.py \\"
echo "    --data data/mcp-atlas.jsonl \\"
echo "    --config examples/gallery/replay/benchmark/config.yaml"
echo ""
