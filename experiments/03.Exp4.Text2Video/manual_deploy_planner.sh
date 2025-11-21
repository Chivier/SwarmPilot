#!/bin/bash
# Manual deployment with planner split: half instances to schedulers, half to planner.
# Mirrors the behavior of exp07 manual_deploy_planner but adapted for exp03 local topology.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults (override via env or flags)
SCHEDULER_A_URL="${SCHEDULER_A_URL:-http://localhost:8100}"
SCHEDULER_B_URL="${SCHEDULER_B_URL:-http://localhost:8200}"
PLANNER_URL="${PLANNER_URL:-http://localhost:8202}"
MODEL_ID_A="${MODEL_ID_A:-sleep_model_a}"
MODEL_ID_B="${MODEL_ID_B:-sleep_model_b}"
N1="${N1:-4}"
N2="${N2:-2}"
PORT_A_START="${PORT_A_START:-8210}"
PORT_B_START="${PORT_B_START:-8300}"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
  cat <<EOF
Usage: manual_deploy_planner.sh [options]
  --scheduler-a-url URL   Scheduler A URL (default: $SCHEDULER_A_URL)
  --scheduler-b-url URL   Scheduler B URL (default: $SCHEDULER_B_URL)
  --planner-url URL       Planner URL (default: $PLANNER_URL)
  --model-id-a ID         Model for Group A (default: $MODEL_ID_A)
  --model-id-b ID         Model for Group B (default: $MODEL_ID_B)
  --n1 N                  Group A instance count (default: $N1)
  --n2 N                  Group B instance count (default: $N2)
  --port-a-start P        Group A start port (default: $PORT_A_START)
  --port-b-start P        Group B start port (default: $PORT_B_START)
  -h|--help               Show this help
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scheduler-a-url) SCHEDULER_A_URL="$2"; shift 2 ;;
    --scheduler-b-url) SCHEDULER_B_URL="$2"; shift 2 ;;
    --planner-url) PLANNER_URL="$2"; shift 2 ;;
    --model-id-a) MODEL_ID_A="$2"; shift 2 ;;
    --model-id-b) MODEL_ID_B="$2"; shift 2 ;;
    --n1) N1="$2"; shift 2 ;;
    --n2) N2="$2"; shift 2 ;;
    --port-a-start) PORT_A_START="$2"; shift 2 ;;
    --port-b-start) PORT_B_START="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

echo "Starting deployment for Group A: $MODEL_ID_A, $N1 instances, starting from port $PORT_A_START"
echo "                        Group B: $MODEL_ID_B, $N2 instances, starting from port $PORT_B_START"
echo "Scheduler A URL: $SCHEDULER_A_URL"
echo "Scheduler B URL: $SCHEDULER_B_URL"
echo "Planner URL: $PLANNER_URL"


deploy_split() {
  local group="$1" model_id="$2" count="$3" start_port="$4" scheduler_url="$5" planner_url="$6"
  local half=$((count / 2))
  echo -e "${YELLOW}Deploying ${group}: first ${half} -> scheduler, remaining -> planner${NC}"
  local status=0
  local pids=()
  for i in $(seq 0 $((count - 1))); do
    port=$((start_port + i))
    target_url=$scheduler_url
    if [ "$i" -ge "$half" ]; then
      target_url=$planner_url
    fi
    payload=$(cat <<EOF
{"model_id":"${model_id}","scheduler_url":"${target_url}","parameters":{}}
EOF
)
    (
      resp=$(curl -s -X POST "http://localhost:${port}/model/start" \
        -H "Content-Type: application/json" \
        -d "${payload}")
      if echo "$resp" | grep -q "success\|started"; then
        echo -e "  ${group}-${i} (port ${port} -> ${target_url}): ${GREEN}OK${NC}"
      else
        echo -e "  ${group}-${i} (port ${port} -> ${target_url}): ${RED}FAILED${NC} -> ${resp}"
        exit 1
      fi
    ) &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

deploy_split "GroupA" "$MODEL_ID_A" "$N1" "$PORT_A_START" "$SCHEDULER_A_URL" "$PLANNER_URL"
deploy_split "GroupB" "$MODEL_ID_B" "$N2" "$PORT_B_START" "$SCHEDULER_B_URL" "$PLANNER_URL"

echo -e "${GREEN}Waiting for 10 seconds...${NC}"
sleep 10

echo -e "${GREEN}Initial split deployment complete. Triggering planner redeploy...${NC}"
uv run "$SCRIPT_DIR/redeploy.py" \
  --scheduler-a-url "$SCHEDULER_A_URL" \
  --scheduler-b-url "$SCHEDULER_B_URL" \
  --planner-url "$PLANNER_URL" \
  --model-id-a "$MODEL_ID_A" \
  --model-id-b "$MODEL_ID_B" \
  --n1 "$N1" \
  --n2 "$N2" \
  --port-a-start "$PORT_A_START" \
  --port-b-start "$PORT_B_START"

echo -e "${GREEN}Manual deploy + initial redeploy done.${NC}"
