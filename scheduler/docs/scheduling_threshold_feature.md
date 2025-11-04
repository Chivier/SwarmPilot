# 调度阈值控制功能

## 概述

该功能实现了基于任务负载的自动调度暂停/恢复机制。当系统负载过高时，调度器会暂停新任务的派发，将任务积压在scheduler中；当负载降低后自动恢复调度。

## 核心特性

### 1. 自动暂停调度

**暂停条件：**
- 已提交任务数（PENDING + RUNNING）≥ 10 × 活跃实例数
- **且** 所有实例都在忙碌（pending_tasks + running_count > 0）

**行为：**
- 继续接受任务提交（/task/submit 返回成功）
- 任务保持 PENDING 状态，不派发到 instance
- 返回消息："Task queued - scheduling paused due to high load"

### 2. 自动恢复调度

**恢复条件（滞后机制）：**
- 已提交任务数 < 8 × 活跃实例数

**行为：**
- 设置 `scheduling_paused = False`
- 自动批量派发积压的 PENDING 任务
- 按任务提交时间顺序派发

### 3. 滞后机制

为避免频繁切换，使用不同的暂停/恢复阈值：
- **暂停阈值**：10× 实例数
- **恢复阈值**：8× 实例数
- **滞后区间**：[8x, 10x) - 在此区间内状态保持不变

### 4. 两种触发方式

#### a) 后台监控（每秒检查）
- 启动时自动创建后台任务 `monitor_scheduling_threshold()`
- 每秒检查一次条件
- 在状态转换时记录日志

#### b) Callback触发（实时响应）
- 每次任务完成回调时检查恢复条件
- 更快响应负载下降

## 实现架构

### 文件修改

**scheduler/src/api.py** - 主要修改文件

1. **全局状态变量** (line 187-188)
   ```python
   scheduling_paused: bool = False
   scheduling_lock = asyncio.Lock()
   ```

2. **辅助函数** (line 196-323)
   - `get_submitted_task_count()` - 统计 PENDING + RUNNING 任务
   - `check_all_instances_busy()` - 检查所有实例是否忙碌
   - `should_pause_scheduling()` - 判断是否应暂停
   - `should_resume_scheduling()` - 判断是否应恢复
   - `dispatch_pending_tasks_batch()` - 批量派发待处理任务

3. **后台监控任务** (line 326-374)
   - `monitor_scheduling_threshold()` - 持续监控并控制状态

4. **修改 /task/submit 端点** (line 731-759)
   - 在派发前检查 `scheduling_paused`
   - 如果暂停，跳过派发步骤

5. **修改 /callback/task_result 端点** (line 1033-1045)
   - 在处理回调后检查恢复条件
   - 如满足条件则恢复并派发

6. **启动监控任务** (line 1415-1417)
   - 在 `startup_event()` 中启动后台监控

7. **新增监控端点** (line 1249-1356)
   - `GET /scheduling/status` - 查看当前状态和指标
   - `POST /scheduling/pause` - 手动暂停调度
   - `POST /scheduling/resume` - 手动恢复调度

### 测试文件

**scheduler/tests/test_scheduling_threshold.py** - 完整单元测试
- 10个测试用例全部通过
- 覆盖所有核心功能

## API 使用说明

### 1. 查看调度状态

```bash
GET http://localhost:8000/scheduling/status
```

**响应示例：**
```json
{
  "success": true,
  "scheduling_paused": false,
  "metrics": {
    "active_instances": 3,
    "submitted_tasks": 25,
    "pending_tasks": 15,
    "running_tasks": 10
  },
  "thresholds": {
    "pause_threshold": 30,
    "resume_threshold": 24
  },
  "conditions": {
    "all_instances_busy": true,
    "should_pause": false,
    "should_resume": true
  }
}
```

### 2. 手动暂停调度

```bash
POST http://localhost:8000/scheduling/pause
```

**响应：**
```json
{
  "success": true,
  "message": "Scheduling paused successfully",
  "scheduling_paused": true
}
```

### 3. 手动恢复调度

```bash
POST http://localhost:8000/scheduling/resume
```

**响应：**
```json
{
  "success": true,
  "message": "Scheduling resumed successfully - dispatching pending tasks",
  "scheduling_paused": false
}
```

## 日志输出

### 暂停时
```
2025-11-04 14:00:00 | WARNING | Pausing scheduling: 30 tasks >= 30 threshold (3 instances x 10), all instances busy
```

### 恢复时（后台监控）
```
2025-11-04 14:01:00 | INFO | Resuming scheduling: 20 tasks < 24 threshold (3 instances x 8)
2025-11-04 14:01:00 | INFO | Starting batch dispatch of 15 pending tasks
2025-11-04 14:01:01 | INFO | Batch dispatch completed: 15/15 tasks dispatched
```

### 恢复时（callback触发）
```
2025-11-04 14:01:00 | INFO | Resuming scheduling (callback trigger): 20 tasks < 24 threshold (3 instances x 8)
```

### 任务排队时
```
2025-11-04 14:00:05 | INFO | Scheduling paused - task task_123 queued (not dispatched). Current load: 32 tasks for 3 instances
```

## 测试验证

运行单元测试：
```bash
cd scheduler
uv run pytest tests/test_scheduling_threshold.py -v
```

**测试覆盖：**
- ✓ get_submitted_task_count() 功能
- ✓ should_pause_scheduling() - 低于阈值场景
- ✓ should_pause_scheduling() - 高于阈值但未全忙场景
- ✓ should_resume_scheduling() - 低于恢复阈值场景
- ✓ should_resume_scheduling() - 高于恢复阈值场景
- ✓ 滞后机制验证
- ✓ dispatch_pending_tasks_batch() 功能
- ✓ GET /scheduling/status 端点
- ✓ POST /scheduling/pause 端点
- ✓ POST /scheduling/resume 端点

## 设计亮点

1. **双重触发机制** - 后台监控 + callback触发，确保快速响应
2. **滞后机制** - 避免在阈值边界频繁切换状态
3. **异步无阻塞** - 使用 asyncio.Lock 保证线程安全
4. **渐进式派发** - 恢复时批量派发任务，每次派发前检查是否需要再次暂停
5. **完整的可观测性** - 详细日志 + 监控端点
6. **向后兼容** - 不影响现有功能，仅增强调度逻辑

## 边界情况处理

1. **实例数为0** - 跳过检查，不触发暂停/恢复
2. **无待处理任务** - 恢复时 dispatch_pending_tasks_batch() 立即返回
3. **派发过程中再次达到阈值** - 立即停止派发，保留剩余任务
4. **并发提交** - 使用 asyncio.Lock 保证状态一致性

## 配置参数

当前硬编码的阈值：
- **暂停倍数**：10（可配置为 config.scheduling.pause_multiplier）
- **恢复倍数**：8（可配置为 config.scheduling.resume_multiplier）
- **监控间隔**：1秒（可配置为 config.scheduling.monitor_interval）

## 未来扩展

1. 将阈值倍数移到配置文件
2. 支持按模型ID设置不同的阈值
3. 添加 Prometheus 指标导出
4. 支持更复杂的暂停条件（如平均等待时间）
5. 实现优先级队列（高优先级任务优先派发）

## 兼容性说明

- **Python**: 3.8+
- **FastAPI**: 0.100+
- **向后兼容**: 完全兼容现有API，不影响现有功能
- **性能影响**: 每秒一次检查 + 每次callback检查，开销极小（<1ms）
