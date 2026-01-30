#!/bin/bash
# Multi-Scheduler - Deploy Models via Planner
# Usage: ./examples/multi_scheduler/deploy_model.sh [instances_per_model]
#
# Deploys sleep models using the Planner's /v1/deploy_manually endpoint.
# Each model gets equal instances (default: 4 per model = 12 total).
#
# Example:
#    ./examples/multi_scheduler/deploy_model.sh 2    # 2 per model = 6 total
#    ./examples/multi_scheduler/deploy_model.sh 8    # 8 per model = 24 total

set -e

# Configuration
PLANNER_PORT=${PLANNER_PORT:-8003}
SCHEDULER_A_PORT=${SCHEDULER_A_PORT:-8010}
SCHEDULER_B_PORT=${SCHEDULER_B_PORT:-8011}
SCHEDULER_C_PORT=${SCHEDULER_C_PORT:-8012}
INSTANCES_PER_MODEL=${1:-4}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Multi-Scheduler - Model Deployment           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo "  Instances per model: $INSTANCES_PER_MODEL"
echo "  Models: sleep_model_a, sleep_model_b, sleep_model_c"
echo "  Total instances: $((INSTANCES_PER_MODEL * 3))"
echo ""

# Check if Planner is running
echo "Checking Planner status..."
if ! curl -s "http://localhost:$PLANNER_PORT/v1/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Planner not responding on port $PLANNER_PORT${NC}"
    echo "Start services first: ./examples/multi_scheduler/start_cluster.sh"
    exit 1
fi
echo -e "${GREEN}✓ Planner is healthy${NC}"

# Check scheduler registration
echo "Checking scheduler registration..."
REGISTERED=$(curl -s "http://localhost:$PLANNER_PORT/v1/scheduler/list" 2>/dev/null)
REG_COUNT=$(echo "$REGISTERED" | python3 -c "import sys, json; print(json.load(sys.stdin).get('total', 0))" 2>/dev/null || echo "0")

if [ "$REG_COUNT" -ge 3 ]; then
    echo -e "${GREEN}✓ $REG_COUNT schedulers registered${NC}"
    echo "$REGISTERED" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for s in data.get('schedulers', []):
    print(f\"  {s['model_id']}: {s['scheduler_url']}\")
" 2>/dev/null
else
    echo -e "${RED}Error: Only $REG_COUNT scheduler(s) registered (expected 3)${NC}"
    echo "  Check scheduler logs for registration errors."
    exit 1
fi
echo ""

# Deploy using /v1/deploy_manually with target_state
echo -e "${BLUE}Deploying instances via /v1/deploy_manually...${NC}"
echo ""

TARGET_STATE=$(cat <<EOF
{
  "sleep_model_a": $INSTANCES_PER_MODEL,
  "sleep_model_b": $INSTANCES_PER_MODEL,
  "sleep_model_c": $INSTANCES_PER_MODEL
}
EOF
)

DEPLOY_REQUEST=$(cat <<EOF
{
  "target_state": $TARGET_STATE,
  "wait_for_ready": true
}
EOF
)

echo -e "${CYAN}Deploy request:${NC}"
echo "  target_state:"
echo "    sleep_model_a: $INSTANCES_PER_MODEL"
echo "    sleep_model_b: $INSTANCES_PER_MODEL"
echo "    sleep_model_c: $INSTANCES_PER_MODEL"
echo "  wait_for_ready: true"
echo ""

# Call /v1/deploy_manually
RESPONSE=$(curl -s -X POST "http://localhost:$PLANNER_PORT/v1/deploy_manually" \
    -H "Content-Type: application/json" \
    -d "$DEPLOY_REQUEST")

# Parse response
SUCCESS=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null || echo "false")

if [ "$SUCCESS" != "True" ] && [ "$SUCCESS" != "true" ]; then
    echo -e "${RED}Deployment failed!${NC}"
    ERROR=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('error', 'Unknown error'))" 2>/dev/null || echo "$RESPONSE")
    echo "Error: $ERROR"
    exit 1
fi

# Extract deployment info
ADDED=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('added_count', 0))" 2>/dev/null)
ACTIVE=$(echo "$RESPONSE" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('active_instances', [])))" 2>/dev/null)

echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          Deployment Complete!                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Deployment Result:${NC}"
echo "  Added instances: $ADDED"
echo "  Total active instances: $ACTIVE"
echo ""

# Verify via per-model schedulers
echo -e "${BLUE}Verifying instances in schedulers...${NC}"
INSTANCES_A=$(curl -s "http://localhost:$SCHEDULER_A_PORT/v1/instance/list" 2>/dev/null)
COUNT_A=$(echo "$INSTANCES_A" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('instances', [])))" 2>/dev/null || echo "?")
echo -e "${GREEN}✓ Scheduler A (sleep_model_a) reports $COUNT_A instances${NC}"

INSTANCES_B=$(curl -s "http://localhost:$SCHEDULER_B_PORT/v1/instance/list" 2>/dev/null)
COUNT_B=$(echo "$INSTANCES_B" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('instances', [])))" 2>/dev/null || echo "?")
echo -e "${GREEN}✓ Scheduler B (sleep_model_b) reports $COUNT_B instances${NC}"

INSTANCES_C=$(curl -s "http://localhost:$SCHEDULER_C_PORT/v1/instance/list" 2>/dev/null)
COUNT_C=$(echo "$INSTANCES_C" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('instances', [])))" 2>/dev/null || echo "?")
echo -e "${GREEN}✓ Scheduler C (sleep_model_c) reports $COUNT_C instances${NC}"
echo ""

echo -e "${CYAN}Instance Distribution:${NC}"
echo "  sleep_model_a: $COUNT_A instances"
echo "  sleep_model_b: $COUNT_B instances"
echo "  sleep_model_c: $COUNT_C instances"
echo "  Total:         $((COUNT_A + COUNT_B + COUNT_C)) instances"
echo ""

echo -e "${YELLOW}Next step:${NC}"
echo "  Generate traffic: python examples/multi_scheduler/generate_workload.py"
echo ""
