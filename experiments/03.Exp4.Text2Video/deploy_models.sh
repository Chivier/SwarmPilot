#!/bin/bash
# Deploy models to already-running instances for Exp03 Text2Video (A/B groups).
# Assumes instances are up and registered to schedulers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults (override via env or CLI)
SCHEDULER_A_URL="${SCHEDULER_A_URL:-http://localhost:8100}"
SCHEDULER_B_URL="${SCHEDULER_B_URL:-http://localhost:8200}"
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
Usage: [env|flags] deploy_models.sh

Env/Flags (all optional):
  --scheduler-a-url URL     Scheduler A URL (default: $SCHEDULER_A_URL)
  --scheduler-b-url URL     Scheduler B URL (default: $SCHEDULER_B_URL)
  --model-id-a ID           Model ID for Group A (default: $MODEL_ID_A)
  --model-id-b ID           Model ID for Group B (default: $MODEL_ID_B)
  --n1 N                    Number of Group A instances (default: $N1)
  --n2 N                    Number of Group B instances (default: $N2)
  --port-a-start P          Starting port for Group A instances (default: $PORT_A_START)
  --port-b-start P          Starting port for Group B instances (default: $PORT_B_START)
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scheduler-a-url) SCHEDULER_A_URL="$2"; shift 2 ;;
    --scheduler-b-url) SCHEDULER_B_URL="$2"; shift 2 ;;
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

deploy_group() {
  local group="$1" model_id="$2" count="$3" start_port="$4" scheduler_url="$5"
  echo -e "${YELLOW}Deploying ${group} (${model_id}) to ${count} instances...${NC}"
  local pids=()
  local status=0
  for i in $(seq 0 $((count - 1))); do
    port=$((start_port + i))
    instance_id="${group}-${i}"
    payload=$(cat <<EOF
{"model_id":"${model_id}","scheduler_url":"${scheduler_url}","parameters":{}}
EOF
)
    (
      resp=$(curl -s -X POST "http://localhost:${port}/model/start" \
        -H "Content-Type: application/json" \
        -d "${payload}")
      if echo "$resp" | grep -q "success\|started"; then
        echo -e "  ${instance_id} (port ${port}): ${GREEN}OK${NC}"
      else
        echo -e "  ${instance_id} (port ${port}): ${RED}FAILED${NC} -> ${resp}"
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

deploy_group "GroupA" "$MODEL_ID_A" "$N1" "$PORT_A_START" "$SCHEDULER_A_URL"
deploy_group "GroupB" "$MODEL_ID_B" "$N2" "$PORT_B_START" "$SCHEDULER_B_URL"

echo -e "${GREEN}Deployment finished.${NC}"
