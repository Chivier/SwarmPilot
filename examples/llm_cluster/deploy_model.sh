#!/bin/bash
# LLM Cluster - Deploy Models via Planner Optimizer
# Usage: ./examples/llm_cluster/deploy_model.sh [num_instances]
#
# Deploys 3 LLM models using planner optimizer:
# - llm_fast (runtime 1x, QPS ratio 5)
# - llm_medium (runtime 5x, QPS ratio 1)
# - llm_slow (runtime 20x, QPS ratio 3)
#
# Default: 32 total instances distributed optimally
#
# PYLET-036: Per-Model Schedulers + Correct Startup Sequence

PLANNER_PORT=${PLANNER_PORT:-8003}
SCHEDULER_FAST_PORT=${SCHEDULER_FAST_PORT:-8010}
SCHEDULER_MEDIUM_PORT=${SCHEDULER_MEDIUM_PORT:-8011}
SCHEDULER_SLOW_PORT=${SCHEDULER_SLOW_PORT:-8012}
NUM_INSTANCES=${1:-32}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     LLM Cluster - Deploy Models with Optimizer        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Configuration
MODEL_IDS='["llm_fast", "llm_medium", "llm_slow"]'
NUM_MODELS=3

# Capacity matrix B: inverse of runtime ratios
# Runtime ratio 1:5:20 -> Capacity 20:4:1 (normalized)
# Each worker can run any model with different capacities
CAPACITY_MATRIX=$(python3 << EOF
import json
num_instances = $NUM_INSTANCES
num_models = 3
# Normalized capacities: [20.0, 4.0, 1.0] for each instance
capacity_per_instance = [20.0, 4.0, 1.0]
# Create M x N matrix (each worker can run any model)
matrix = [capacity_per_instance for _ in range(num_instances)]
print(json.dumps(matrix))
EOF
)

# Target distribution from QPS ratios 5:1:3
TARGET_DIST=$(python3 << 'EOF'
import json
qps_ratios = [5.0, 1.0, 3.0]
total = sum(qps_ratios)
target = [q / total for q in qps_ratios]
print(json.dumps(target))
EOF
)

echo -e "${CYAN}Configuration:${NC}"
echo "  Total Instances: $NUM_INSTANCES"
echo "  Models: 3 (llm_fast, llm_medium, llm_slow)"
echo "  Runtime Ratio: 1:5:20"
echo "  QPS Ratio: 5:1:3"
echo "  Target Distribution: $(echo $TARGET_DIST | python3 -c 'import sys, json; d=json.load(sys.stdin); print(", ".join([f"{x*100:.1f}%" for x in d]))')"
echo ""

# Health check planner
echo "Checking planner health..."
if ! curl -s "http://localhost:$PLANNER_PORT/v1/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Planner not responding on port $PLANNER_PORT${NC}"
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
    echo -e "${YELLOW}Warning: Only $REG_COUNT scheduler(s) registered (expected 3)${NC}"
    echo "  Instances may not be routed to the correct scheduler."
fi
echo ""

# Build deployment request
DEPLOY_REQUEST=$(python3 << EOF
import json

deploy_req = {
    "M": $NUM_INSTANCES,
    "N": $NUM_MODELS,
    "B": $CAPACITY_MATRIX,
    "target": $TARGET_DIST,
    "a": 0.5,
    "model_ids": ["llm_fast", "llm_medium", "llm_slow"],
    "algorithm": "simulated_annealing",
    "objective_method": "relative_error",
    "wait_for_ready": True,
}

print(json.dumps(deploy_req))
EOF
)

echo -e "${BLUE}Deploying models via planner optimizer...${NC}"

# Call planner /v1/deploy endpoint
RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "$DEPLOY_REQUEST" \
    "http://localhost:$PLANNER_PORT/v1/deploy")

# Check if deployment was successful
DEPLOYMENT_SUCCESS=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('deployment_success', False))" 2>/dev/null)

if [ "$DEPLOYMENT_SUCCESS" != "True" ]; then
    echo -e "${RED}Deployment failed:${NC}"
    echo "$RESPONSE" | python3 -m json.tool
    exit 1
fi

echo -e "${GREEN}✓ Deployment successful${NC}"
echo ""

# Extract and display results
python3 << EOF
import json
import sys

response = json.loads('''$RESPONSE''')

print("Deployment Results:")
print("")

# Optimization score
score = response.get('score', 'N/A')
print(f"Optimization Score: {score}")

# Deployment array
deployment = response.get('deployment', [])
print(f"Deployment Array: {deployment}")

# Service capacity
service_capacity = response.get('service_capacity', [])
print(f"Service Capacity: {service_capacity}")

# Active instances summary
active_instances = response.get('active_instances', [])
print(f"\nDeployed Instances: {len(active_instances)}")

# Count by model
model_counts = {}
for inst in active_instances:
    model_id = inst.get('model_id', 'unknown')
    model_counts[model_id] = model_counts.get(model_id, 0) + 1

print("\nInstances by Model:")
for model_id, count in sorted(model_counts.items()):
    print(f"  {model_id}: {count} instances")

# Show expected capacity
print("\nExpected Capacity:")
for i, model_id in enumerate(['llm_fast', 'llm_medium', 'llm_slow']):
    if model_counts.get(model_id, 0) > 0:
        # Capacity per model based on count
        cap_per_instance = [20.0, 4.0, 1.0][i]
        total_cap = model_counts[model_id] * cap_per_instance
        print(f"  {model_id}: {total_cap:.0f} units ({model_counts[model_id]} instances × {cap_per_instance:.0f})")
EOF

echo ""

# Verify instances in each per-model scheduler
echo -e "${BLUE}Verifying instances in per-model schedulers...${NC}"

INSTANCES_FAST=$(curl -s "http://localhost:$SCHEDULER_FAST_PORT/v1/instance/list" 2>/dev/null)
TOTAL_FAST=$(echo "$INSTANCES_FAST" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('instances', [])))" 2>/dev/null || echo "?")
echo -e "${GREEN}✓ Scheduler (llm_fast) reports $TOTAL_FAST instances${NC}"

INSTANCES_MEDIUM=$(curl -s "http://localhost:$SCHEDULER_MEDIUM_PORT/v1/instance/list" 2>/dev/null)
TOTAL_MEDIUM=$(echo "$INSTANCES_MEDIUM" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('instances', [])))" 2>/dev/null || echo "?")
echo -e "${GREEN}✓ Scheduler (llm_medium) reports $TOTAL_MEDIUM instances${NC}"

INSTANCES_SLOW=$(curl -s "http://localhost:$SCHEDULER_SLOW_PORT/v1/instance/list" 2>/dev/null)
TOTAL_SLOW=$(echo "$INSTANCES_SLOW" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('instances', [])))" 2>/dev/null || echo "?")
echo -e "${GREEN}✓ Scheduler (llm_slow) reports $TOTAL_SLOW instances${NC}"

echo ""
echo -e "${GREEN}✓ Deployment complete!${NC}"
echo ""
echo -e "${YELLOW}Next step:${NC}"
echo "  Generate traffic: python examples/llm_cluster/generate_workload.py --total-qps 10 --duration 60"
