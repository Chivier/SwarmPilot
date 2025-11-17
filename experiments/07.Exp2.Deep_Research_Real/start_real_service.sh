#!/bin/bash
set -e

# ============================================
# Multi-node service launcher
# ============================================
# 逻辑说明：
#   - 使用 ifconfig 从 bond1 获取本机 IP
#   - 若本机 IP == 29.209.114.51：启动 scheduler，端口 8100
#   - 若本机 IP == 29.209.114.166：启动 predictor，端口 8100
#   - 无论本机 IP 为多少：均启动 instance，端口 8000
#   - 根据本机 IP 所属列表，选择 MODEL_ID = sleep_model_a 或 sleep_model_b
#
# 启动命令沿用原脚本模式：
#   uv run python -m src.cli start --port <PORT>

# -----------------------------
# 基础路径与配置
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# 固定端口
SCHEDULER_PORT=8100
PREDICTOR_PORT=8100
INSTANCE_PORT=8000

# CPU 绑定配置 (384核心系统,使用后128个核心: 256-383)
TOTAL_CORES=384
CPU_START_OFFSET=256  # 从第256个核心开始
CORES_PER_INSTANCE=16 # 每个instance使用16个核心

# Scheduler和Predictor绑定到前8个核心
SCHEDULER_PREDICTOR_CPU_RANGE="0-7"

# 角色 IP
SCHEDULER_A_HOST="29.209.114.51"
SCHEDULER_B_HOST="29.209.113.228"
PREDICTOR_HOST="29.209.114.166"

INSTANCE_PORT_LIST=(
    8200
    8201
    8202
    8203
    8204
    8205
    8206
    8207
)

GPU_BIND_ID_LIST=(
    0
    1
    2
    3
    4
    5
    6
    7
)

# sleep_model_a 对应的机器
SLEEP_MODEL_A_HOSTS=(
  29.209.114.51
  29.209.114.166
  29.209.113.113
  29.209.106.237
  29.209.114.56
  29.209.114.241
  29.209.112.177
  29.209.113.235
)

# sleep_model_b 对应的机器
SLEEP_MODEL_B_HOSTS=(
  29.209.113.228
  29.209.105.60
  29.209.113.166
  29.209.113.176
  29.209.113.169
  29.209.112.74
  29.209.115.174
  29.209.113.156
)

# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# -----------------------------
# 工具函数
# -----------------------------
get_local_ip() {
  # 从 bond1 提取 IPv4 地址
  if ! command -v ifconfig >/dev/null 2>&1; then
    echo "ifconfig not found" >&2
    return 1
  fi

  if ! ifconfig bond1 >/dev/null 2>&1; then
    echo "Interface bond1 not found" >&2
    return 1
  fi

  ifconfig bond1 | awk '/inet / {print $2}' | head -n1
}

in_list() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

start_bg() {
  # start_bg <name> <command> [cpu_cores]
  local name="$1"
  shift
  local cmd="$1"
  shift
  local cpu_cores="$1"  # 可选参数: CPU核心列表 (e.g., "256-271")

  local log_file="$LOG_DIR/${name}.log"

  echo -e "${YELLOW}Starting $name...${NC}"

  if [[ -n "$cpu_cores" ]]; then
    # 使用 taskset 绑定到指定CPU核心
    echo -e "${BLUE}  CPU binding: $cpu_cores${NC}"
    nohup taskset -c "$cpu_cores" bash -lc "$cmd" >"$log_file" 2>&1 &
  else
    # 不绑定CPU核心
    nohup bash -lc "$cmd" >"$log_file" 2>&1 &
  fi

  local pid=$!
  echo -e "${GREEN}Started $name (PID: $pid, log: $log_file)${NC}"
}

get_cpu_range() {
  # get_cpu_range <instance_index>
  # 返回该instance应该绑定的CPU核心范围
  local idx="$1"
  local start_core=$((CPU_START_OFFSET + idx * CORES_PER_INSTANCE))
  local end_core=$((start_core + CORES_PER_INSTANCE - 1))
  echo "${start_core}-${end_core}"
}

# -----------------------------
# 主逻辑
# -----------------------------
LOCAL_IP="$(get_local_ip)"
if [[ -z "$LOCAL_IP" ]]; then
  echo -e "${RED}Failed to detect local IP on bond1, aborting.${NC}"
  exit 1
fi

echo -e "${BLUE}Local IP detected: $LOCAL_IP${NC}"

# 根据 IP 决定 MODEL_ID
MODEL_ID=""
if in_list "$LOCAL_IP" "${SLEEP_MODEL_A_HOSTS[@]}"; then
  MODEL_ID="sleep_model_a"
elif in_list "$LOCAL_IP" "${SLEEP_MODEL_B_HOSTS[@]}"; then
  MODEL_ID="sleep_model_b"
else
  # 不在列表中的机器，给一个默认值（可按需修改）
  MODEL_ID="sleep_model_a"
  echo -e "${YELLOW}Host $LOCAL_IP not in A/B lists, defaulting MODEL_ID=$MODEL_ID${NC}"
fi

# -----------------------------
# 启动 scheduler / predictor
# -----------------------------

if [[ "$LOCAL_IP" == "$PREDICTOR_HOST" ]]; then
  # 只有 predictor 机器启动 predictor（8100）
  start_bg "predictor" \
    "cd $PROJECT_ROOT/predictor && \
     PREDICTOR_PORT=$PREDICTOR_PORT \
     PREDICTOR_LOG_DIR=$SCRIPT_DIR/logs/predictor uv run python -m src.cli start --port $PREDICTOR_PORT --log-level INFO" \
    "$SCHEDULER_PREDICTOR_CPU_RANGE"
fi

if [[ "$LOCAL_IP" == "$SCHEDULER_A_HOST" ]]; then
  # 只有 scheduler 机器启动 scheduler（8100）
  start_bg "scheduler" \
    "cd $PROJECT_ROOT/scheduler && \
     PREDICTOR_URL=http://$PREDICTOR_HOST:$PREDICTOR_PORT \
     SCHEDULER_PORT=$SCHEDULER_PORT SCHEDULER_LOG_DIR=$SCRIPT_DIR/logs/scheduler-a SCHEDULER_LOGURU_LEVEL=\"INFO\" uv run python -m src.cli start --port $SCHEDULER_PORT" \
    "$SCHEDULER_PREDICTOR_CPU_RANGE"
fi

if [[ "$LOCAL_IP" == "$SCHEDULER_B_HOST" ]]; then
  # 只有 scheduler 机器启动 scheduler（8100）
  start_bg "scheduler" \
    "cd $PROJECT_ROOT/scheduler && \
     PREDICTOR_URL=http://$PREDICTOR_HOST:$PREDICTOR_PORT \
     SCHEDULER_PORT=$SCHEDULER_PORT SCHEDULER_LOG_DIR=$SCRIPT_DIR/logs/scheduler-b SCHEDULER_LOGURU_LEVEL=\"INFO\" uv run python -m src.cli start --port $SCHEDULER_PORT" \
    "$SCHEDULER_PREDICTOR_CPU_RANGE"
fi

# -----------------------------
# 所有机器统一启动 instance:8000
# -----------------------------

contains_element() {
    local element match="$1"
    shift
    for element; do
        [[ "$element" == "$match" ]] && return 0
    done
    return 1
}


for i in {0..7}; do
  INSTANCE_PORT=${INSTANCE_PORT_LIST[$i]}
  GPU_ID=${GPU_BIND_ID_LIST[$i]}
  CPU_RANGE=$(get_cpu_range $i)

  start_bg "instance_${MODEL_ID}_gpu${GPU_ID}_port${INSTANCE_PORT}" \
    "cd $PROJECT_ROOT/instance && \
     MODEL_ID=$MODEL_ID \
     CUDA_VISIBLE_DEVICES=${GPU_ID} \
     uv run python -m src.cli start --port ${INSTANCE_PORT}" \
    "$CPU_RANGE"
done


echo ""
echo -e "${GREEN}All services that should run on $LOCAL_IP have been launched.${NC}"
echo -e "Scheduler host:  $SCHEDULER_HOST (port $SCHEDULER_PORT)"
echo -e "Predictor host:  $PREDICTOR_HOST (port $PREDICTOR_PORT)"
echo -e "Instance model:  $MODEL_ID (port $INSTANCE_PORT)"
echo "Logs: $LOG_DIR"