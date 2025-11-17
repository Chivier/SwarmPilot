# CPU 核心绑定说明

## 配置概述

脚本已更新,在384核心系统上为所有服务组件绑定专用CPU核心:

- **Scheduler/Predictor**: 绑定到核心 0-7 (共8个核心,可共享)
- **Instance (8个)**: 每个绑定到16个专用核心 (256-383)

## 核心分配策略

- **系统总核心数**: 384个 (0-383)
- **Scheduler/Predictor**: 0-7 (前8个核心)
- **保留核心**: 8-255 (用于系统和其他任务)
- **Instance核心范围**: 256-383 (后128个核心)
- **每个instance**: 16个连续核心
- **instance总数**: 8个

## 核心分配表

### Scheduler/Predictor

| 组件      | Port | CPU Cores | 核心数量 | 说明 |
|----------|------|-----------|---------|------|
| Scheduler A | 8100 | 0-7    | 8       | 共享核心 |
| Scheduler B | 8100 | 0-7    | 8       | 共享核心 |
| Predictor   | 8100 | 0-7    | 8       | 共享核心 |

### Instance

| Instance | GPU ID | Port | CPU Cores | 核心数量 |
|----------|--------|------|-----------|---------|
| 0        | 0      | 8200 | 256-271   | 16      |
| 1        | 1      | 8201 | 272-287   | 16      |
| 2        | 2      | 8202 | 288-303   | 16      |
| 3        | 3      | 8203 | 304-319   | 16      |
| 4        | 4      | 8204 | 320-335   | 16      |
| 5        | 5      | 8205 | 336-351   | 16      |
| 6        | 6      | 8206 | 352-367   | 16      |
| 7        | 7      | 8207 | 368-383   | 16      |

## 实现细节

### 使用的工具

脚本使用Linux `taskset` 命令进行CPU亲和性绑定:

```bash
taskset -c <cpu_range> <command>
```

### 关键配置参数

在 `start_real_service.sh` 中:

```bash
# CPU 绑定配置
TOTAL_CORES=384                         # 系统总核心数
CPU_START_OFFSET=256                    # Instance从第256个核心开始
CORES_PER_INSTANCE=16                   # 每个instance使用16个核心
SCHEDULER_PREDICTOR_CPU_RANGE="0-7"    # Scheduler/Predictor使用核心0-7
```

### 核心分配函数

```bash
get_cpu_range() {
  local idx="$1"
  local start_core=$((CPU_START_OFFSET + idx * CORES_PER_INSTANCE))
  local end_core=$((start_core + CORES_PER_INSTANCE - 1))
  echo "${start_core}-${end_core}"
}
```

## 优势

1. **性能隔离**: 每个instance有专用的CPU核心,避免核心争抢
2. **优先级分配**: Scheduler/Predictor使用前8个核心,通常这些核心具有更好的缓存局部性
3. **NUMA优化**: 可以根据需要调整CPU_START_OFFSET以对齐NUMA节点
4. **可预测性**: 固定的核心分配使性能更加稳定
5. **资源控制**:
   - 核心 0-7: Scheduler/Predictor (轻量级任务)
   - 核心 8-255: 系统和其他任务保留
   - 核心 256-383: Instance专用 (计算密集型任务)

## 验证

运行测试脚本查看核心分配:

```bash
bash test_cpu_binding.sh
```

## 启动服务

```bash
bash start_real_service.sh
```

启动后可以使用以下命令验证CPU绑定:

```bash
# 查看进程的CPU亲和性
ps aux | grep "python -m src.cli start" | grep -v grep | awk '{print $2}' | while read pid; do
    echo "PID: $pid"
    taskset -cp $pid
done
```

## 注意事项

1. **taskset 依赖**: 确保系统已安装 `util-linux` 包 (通常默认安装)
2. **权限**: 不需要root权限即可设置CPU亲和性
3. **NUMA**: 在多NUMA节点系统上,可能需要进一步优化核心选择以提升性能
4. **调整**: 可以通过修改配置参数来改变核心分配策略

## 如何调整配置

### 修改使用的核心范围

编辑 `start_real_service.sh`:

```bash
CPU_START_OFFSET=128  # 改为从第128个核心开始
```

### 修改每个instance的核心数

```bash
CORES_PER_INSTANCE=32  # 改为每个instance使用32个核心
```

### 启动更多instance

修改循环范围和相应的配置数组:

```bash
for i in {0..15}; do  # 启动16个instance
  # ...
done
```

同时更新 `INSTANCE_PORT_LIST` 和 `GPU_BIND_ID_LIST` 数组。
