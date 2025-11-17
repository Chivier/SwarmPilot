#!/bin/bash
# 测试CPU绑定逻辑

CPU_START_OFFSET=256
CORES_PER_INSTANCE=16
SCHEDULER_PREDICTOR_CPU_RANGE="0-7"

get_cpu_range() {
  local idx="$1"
  local start_core=$((CPU_START_OFFSET + idx * CORES_PER_INSTANCE))
  local end_core=$((start_core + CORES_PER_INSTANCE - 1))
  echo "${start_core}-${end_core}"
}

echo "CPU绑定配置测试 (384核心系统)"
echo "=============================================="
echo "总核心数: 384"
echo ""
echo "服务组件的CPU核心分配:"
echo "=============================================="
echo "Scheduler:  CPU cores $SCHEDULER_PREDICTOR_CPU_RANGE (8个核心)"
echo "Predictor:  CPU cores $SCHEDULER_PREDICTOR_CPU_RANGE (8个核心)"
echo ""
echo "Instance的CPU核心分配 (使用后128个核心: 256-383):"
echo "=============================================="
echo "每个instance使用: 16个核心"
echo "可启动instance数: 8个"
echo ""

for i in {0..7}; do
  CPU_RANGE=$(get_cpu_range $i)
  echo "Instance $i (GPU $i, Port 82$((200+i))): CPU cores $CPU_RANGE"
done

echo ""
echo "验证:"
echo "- Scheduler/Predictor: 0-7 (共享8个核心)"
echo "- Instance 0: 256-271 (16个核心)"
echo "- Instance 7: 368-383 (16个核心)"
echo "- Instance总共使用: 256 到 383 = 128个核心 ✓"
echo "- 核心8-255 保留给系统和其他任务"
