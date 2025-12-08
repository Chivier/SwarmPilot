# Timeout控制指南

## 概述

测试脚本中的timeout控制等待工作流完成的最长时间。如果在timeout时间内工作流没有完成，测试将自动停止。

## Timeout参数使用

### 1. **命令行参数（推荐）**

使用`--timeout`参数指定超时时间（单位：分钟）：

```bash
# 默认timeout（20分钟）
python3 test_dynamic_workflow.py --strategies min_time --num-workflows 100

# 自定义timeout为30分钟
python3 test_dynamic_workflow.py --strategies min_time --num-workflows 100 --timeout 30

# 更长的timeout（60分钟）用于大型实验
python3 test_dynamic_workflow.py --num-workflows 500 --timeout 60

# 更短的timeout（5分钟）用于快速测试
python3 test_dynamic_workflow.py --num-workflows 10 --timeout 5
```

### 2. **在代码中调用**

如果直接在Python代码中调用main函数：

```python
from test_dynamic_workflow import main

# 使用默认timeout（20分钟）
main(num_workflows=100, strategies=["min_time"])

# 自定义timeout
main(num_workflows=100, strategies=["min_time"], timeout_minutes=30)
```

## Timeout配置详情

### 主要Timeout设置

| 参数位置 | 默认值 | 说明 |
|---------|--------|------|
| `--timeout` | 20分钟 | 工作流完成的最长等待时间 |
| `test_strategy_workflow()` | 10分钟 | 函数默认值（被命令行参数覆盖） |

### 其他内部Timeout设置

这些timeout是内部使用的，一般不需要修改：

| 类型 | 值 | 位置 | 说明 |
|------|------|------|------|
| HTTP请求超时 | 5秒 | 任务提交 | 单个HTTP请求的超时时间 |
| WebSocket接收超时 | 1秒 | 消息接收 | 每次等待消息的超时（会自动重试）|
| 线程join超时 | 10秒 | 线程停止 | 等待线程结束的超时 |
| Queue获取超时 | 0.5-1秒 | 队列操作 | 从队列获取数据的超时（会自动重试）|

## Timeout计算建议

### 如何选择合适的timeout？

根据以下因素估算所需timeout：

1. **工作流数量**：`num_workflows`
2. **任务执行时间**：
   - A任务：1-3秒（快）或 7-10秒（慢），平均约5秒
   - B1任务（查询）：根据trace数据，平均约4秒
   - B2任务（评估）：根据trace数据，平均约4秒
   - Merge任务：根据trace数据，平均约90-100秒
3. **Fanout值**：每个workflow的B任务数量（3-8个）
4. **调度策略**：min_time通常更快，round_robin可能较慢
5. **QPS限制**：较低的QPS会延长总时间

### 估算公式

```
预估时间 = 提交时间 + 执行时间 + 缓冲时间

提交时间 ≈ num_workflows / QPS
执行时间 ≈ max(A时间, B1时间, B2时间) + Merge时间
缓冲时间 ≈ 执行时间 × 0.2（20%缓冲）

推荐timeout = 预估时间 × 1.5（50%余量）
```

### 示例计算

**场景1：快速测试**
```
num_workflows = 10
QPS = 8
A时间 ≈ 5秒
B任务 ≈ 8秒（假设fanout=5）
Merge ≈ 90秒

预估时间 = (10/8) + 5 + 8 + 90 ≈ 105秒 ≈ 2分钟
推荐timeout = 2 × 1.5 = 3分钟
```

**场景2：标准实验**
```
num_workflows = 100
QPS = 8
预估时间 ≈ (100/8) + 5 + 8 + 90 ≈ 115秒 ≈ 2分钟
但考虑到100个workflow串行执行：
总时间 ≈ 100 × (5 + 8 + 90) / 并行度 ≈ 10-15分钟
推荐timeout = 15 × 1.5 ≈ 20分钟 ✅（当前默认值）
```

**场景3：大型实验**
```
num_workflows = 500
QPS = 8
预估总时间 ≈ 50-75分钟
推荐timeout = 75 × 1.5 ≈ 120分钟（2小时）
```

## 使用建议

### 快速测试
```bash
# 10个workflow，5分钟timeout
python3 test_dynamic_workflow.py --num-workflows 10 --timeout 5 --strategies min_time
```

### 标准实验（推荐）
```bash
# 100个workflow，默认20分钟timeout
python3 test_dynamic_workflow.py --num-workflows 100 --strategies min_time
```

### 大规模实验
```bash
# 500个workflow，60分钟timeout
python3 test_dynamic_workflow.py --num-workflows 500 --timeout 60

# 多策略对比，120分钟timeout
python3 test_dynamic_workflow.py --num-workflows 200 --timeout 120 \
    --strategies min_time round_robin probabilistic
```

### Min_time策略（带MAPE误差）
```bash
# min_time策略可能需要更长时间（因为预测误差）
python3 test_dynamic_workflow.py --num-workflows 100 --timeout 30 --strategies min_time
```

## Timeout触发后会发生什么？

当达到timeout时间后：

1. ✅ **已完成的工作流**：会被正常统计和保存
2. ✅ **部分完成的工作流**：可能缺少Merge任务数据
3. ✅ **所有线程**：会被停止（调用stop()方法）
4. ✅ **结果文件**：会保存当前已采集的数据
5. ⚠️ **实验继续**：如果有多个策略，会继续测试下一个策略

### 日志输出示例

```
[INFO] Step 14: Waiting for target workflows to complete (timeout: 20 minutes)
[INFO] Will stop when 100 target workflows complete (out of 100 total)
[INFO] Waiting... 50/100 target workflows completed, 50/100 total
[INFO] Waiting... 80/100 target workflows completed, 80/100 total
[WARN] Timeout reached! Only 80/100 workflows completed.
[INFO] Step 15: Stopping all threads
```

## 监控进度

测试运行时，每10秒会输出进度信息：

```
[INFO] Waiting... 45/100 target workflows completed, 45/100 total
```

- **第一个数字**：已完成的目标工作流（用于统计）
- **第二个数字**：目标工作流总数
- **第三个数字**：所有工作流（包括warmup）

## 故障排查

### Timeout过早触发

**症状**：实验在完成前就超时

**解决方案**：
```bash
# 增加timeout
python3 test_dynamic_workflow.py --timeout 30  # 改为30分钟

# 或减少工作流数量
python3 test_dynamic_workflow.py --num-workflows 50 --timeout 20
```

### Timeout太长浪费时间

**症状**：实验早就完成了，但等了很久才结束

**解决方案**：
```bash
# 减少timeout
python3 test_dynamic_workflow.py --timeout 10  # 改为10分钟
```

### 所有工作流都没完成

**症状**：timeout后没有任何结果

**可能原因**：
1. Scheduler没有运行
2. 任务提交失败（检查exp_runtime是否为0）
3. WebSocket连接失败

**检查方法**：
```bash
# 检查scheduler是否运行
curl http://localhost:8100/health
curl http://localhost:8200/health

# 查看日志
tail -f logs/scheduler_a.log
tail -f logs/scheduler_b.log
```

## 完整参数示例

```bash
# 最小化配置（快速测试）
python3 test_dynamic_workflow.py \
    --num-workflows 10 \
    --timeout 5 \
    --strategies min_time

# 标准配置
python3 test_dynamic_workflow.py \
    --num-workflows 100 \
    --qps 8.0 \
    --timeout 20 \
    --strategies min_time round_robin probabilistic

# 高级配置（带QPS限制和warmup）
python3 test_dynamic_workflow.py \
    --num-workflows 200 \
    --qps 10.0 \
    --gqps 25.0 \
    --timeout 45 \
    --warmup 0.2 \
    --strategies min_time

# 大规模实验
python3 test_dynamic_workflow.py \
    --num-workflows 500 \
    --qps 20.0 \
    --timeout 120 \
    --continuous \
    --strategies min_time round_robin probabilistic
```

## 总结

- ✅ 使用`--timeout`参数轻松控制超时时间
- ✅ 默认20分钟适合大多数标准实验（100个workflow）
- ✅ 快速测试用5-10分钟，大型实验用60-120分钟
- ✅ 根据`num_workflows`、QPS和任务复杂度调整timeout
- ✅ 监控进度日志，及时调整timeout配置
