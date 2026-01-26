#!/bin/bash
# Mock LLM Cluster - Deploy Models via Planner
# Usage: ./examples/mock_llm_cluster/deploy_models.sh [total_instances]
#
# Deploys mock LLM models (7B and 32B) using the Planner's /deploy endpoint.
# The /deploy endpoint runs the optimization algorithm and deploys via PyLet.
# Allocation is calculated based on:
# - Traffic ratio (1:5 - 32B gets 5x more requests)
# - Model throughput (7B: ~5 req/s, 32B: ~1 req/s due to latency)
#
# PYLET-022: Mock LLM Cluster Example

set -e

# Configuration
PLANNER_PORT=${PLANNER_PORT:-8002}
SCHEDULER_PORT=${SCHEDULER_PORT:-8000}
TOTAL_INSTANCES=${1:-16}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Mock LLM Cluster - Model Deployment           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo "  Total Instances: $TOTAL_INSTANCES"
echo "  Traffic Ratio:   1:5 (7B:32B)"
echo "  7B Latency:      ~200ms (throughput ~5 req/s)"
echo "  32B Latency:     ~1000ms (throughput ~1 req/s)"
echo ""

# Check if Planner is running
echo "Checking Planner status..."
if ! curl -s "http://localhost:$PLANNER_PORT/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Planner not responding on port $PLANNER_PORT${NC}"
    echo "Start services first: ./examples/mock_llm_cluster/start_cluster.sh"
    exit 1
fi
echo -e "${GREEN}✓ Planner is healthy${NC}"

# Check PyLet status
echo "Checking PyLet status..."
PYLET_STATUS=$(curl -s "http://localhost:$PLANNER_PORT/status" 2>/dev/null)
if echo "$PYLET_STATUS" | grep -q '"initialized": false' 2>/dev/null; then
    echo -e "${RED}Error: PyLet not initialized${NC}"
    echo "Ensure PyLet cluster is running and restart the Planner."
    exit 1
fi
echo -e "${GREEN}✓ PyLet is initialized${NC}"
echo ""

# Calculate optimal deployment using /deploy
#
# Traffic ratio 1:5 means:
#   7B: 16.67% of traffic (1/6)
#   32B: 83.33% of traffic (5/6)
#
# Capacity matrix B (throughput per instance):
#   B[i][j] = throughput of instance i when running model j
#   Each instance can run either model: [5.0, 1.0] (5 req/s for 7B, 1 req/s for 32B)
#
# The /deploy endpoint runs the optimizer and deploys the result via PyLet.

echo -e "${BLUE}Calling /deploy to optimize and deploy instances...${NC}"
echo ""

# Build capacity matrix for $TOTAL_INSTANCES instances
# Each instance can run either model, with different throughputs
B_MATRIX="["
for i in $(seq 1 $TOTAL_INSTANCES); do
    if [ $i -gt 1 ]; then
        B_MATRIX+=","
    fi
    B_MATRIX+="[5.0, 1.0]"  # Each instance: 5 req/s for 7B, 1 req/s for 32B
done
B_MATRIX+="]"

# Target traffic distribution (normalized QPS)
# 1:5 ratio -> [1/6, 5/6] * 100 for percentage-like values
TARGET="[16.67, 83.33]"

DEPLOY_REQUEST=$(cat <<EOF
{
    "M": $TOTAL_INSTANCES,
    "N": 2,
    "B": $B_MATRIX,
    "target": $TARGET,
    "a": 1.0,
    "model_ids": ["llm-7b", "llm-32b"],
    "algorithm": "simulated_annealing",
    "objective_method": "ratio_difference",
    "wait_for_ready": true
}
EOF
)

echo -e "${CYAN}Deploy request:${NC}"
echo "  Models: llm-7b, llm-32b"
echo "  Target ratio: 1:5 (16.67% : 83.33%)"
echo "  Instances: $TOTAL_INSTANCES"
echo "  Algorithm: simulated_annealing"
echo ""

# Call /deploy (runs optimizer then deploys via PyLet)
RESPONSE=$(curl -s -X POST "http://localhost:$PLANNER_PORT/deploy" \
    -H "Content-Type: application/json" \
    -d "$DEPLOY_REQUEST")

# Parse response
SUCCESS=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('deployment_success', False))" 2>/dev/null || echo "false")

if [ "$SUCCESS" != "True" ] && [ "$SUCCESS" != "true" ]; then
    echo -e "${RED}Deployment failed!${NC}"
    ERROR=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('error', 'Unknown error'))" 2>/dev/null || echo "$RESPONSE")
    echo "Error: $ERROR"
    exit 1
fi

# Extract deployment info
DEPLOYMENT=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('deployment', []))" 2>/dev/null)
SCORE=$(echo "$RESPONSE" | python3 -c "import sys, json; print(f\"{json.load(sys.stdin).get('score', 0):.4f}\")" 2>/dev/null)
CAPACITY=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('service_capacity', []))" 2>/dev/null)
ADDED=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('added_count', 0))" 2>/dev/null)

# Count instances per model
COUNT_7B=$(echo "$RESPONSE" | python3 -c "import sys, json; d=json.load(sys.stdin).get('deployment', []); print(sum(1 for x in d if x==0))" 2>/dev/null)
COUNT_32B=$(echo "$RESPONSE" | python3 -c "import sys, json; d=json.load(sys.stdin).get('deployment', []); print(sum(1 for x in d if x==1))" 2>/dev/null)

echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          Deployment Complete!                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Optimization Result:${NC}"
echo "  Objective Score: $SCORE (lower is better)"
echo "  Service Capacity: $CAPACITY"
echo ""
echo -e "${CYAN}Instance Allocation:${NC}"
echo "  llm-7b:  $COUNT_7B instances (~5 req/s each)"
echo "  llm-32b: $COUNT_32B instances (~1 req/s each)"
echo "  Total:   $((COUNT_7B + COUNT_32B)) instances"
echo ""

# Verify via scheduler
echo -e "${BLUE}Verifying instances in scheduler...${NC}"
INSTANCES=$(curl -s "http://localhost:$SCHEDULER_PORT/instance/list" 2>/dev/null)
TOTAL=$(echo "$INSTANCES" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('instances', [])))" 2>/dev/null || echo "?")
echo -e "${GREEN}✓ Scheduler reports $TOTAL registered instances${NC}"
echo ""

echo -e "${YELLOW}Next step:${NC}"
echo "  Generate traffic: python examples/mock_llm_cluster/generate_workload.py"
