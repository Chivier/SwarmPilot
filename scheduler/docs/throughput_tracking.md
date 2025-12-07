# Scheduler Throughput Tracking

## 功能概述

Scheduler 自动收集每个 instance 的任务执行时间，通过滑动窗口计算平均值，随 `/submit_target` 一起上报给 Planner 的 `/submit_throughput` 接口。

## 数据流

```
TaskDispatcher.handle_task_result()
       │ execution_time_ms
       ▼
ThroughputTracker.record_execution_time()
       │ 滑动窗口存储
       ▼
PlannerReporter._report_to_planner()
       │ 计算平均值 (ms → s)
       ▼
Planner /submit_throughput
```

## 新增文件

| 文件 | 说明 |
|------|------|
| `src/throughput_tracker.py` | 滑动窗口吞吐量跟踪器 |
| `tests/test_throughput_tracker.py` | 单元测试 (14 cases) |
| `tests/test_planner_reporter_throughput.py` | 集成测试 (8 cases) |

## 修改文件

| 文件 | 变更 |
|------|------|
| `src/config.py` | `PlannerReportConfig.throughput_window_size = 20` |
| `src/task_dispatcher.py` | 添加 `throughput_tracker` 参数，完成时记录执行时间 |
| `src/planner_reporter.py` | 添加 `_report_throughput()` 方法 |
| `src/api.py` | 初始化并连接组件 |

## 配置

滑动窗口大小固定为 20，与 `/submit_target` 共享 Planner 报告配置：

```bash
PLANNER_URL=http://planner:8000
SCHEDULER_AUTO_REPORT=10  # 报告间隔(秒)
```

---

# Work Stealing 机制

## 窃取数量计算

Work stealing 窃取数量基于 scheduler 中所有活跃任务数计算：

```python
# 1. 统计 scheduler 中未完成的任务（PENDING + RUNNING）
pending_count = task_registry.get_count_by_status(TaskStatus.PENDING)
running_count = task_registry.get_count_by_status(TaskStatus.RUNNING)
total_active_tasks = pending_count + running_count

# 2. 计算平均任务数（包含新 instance）
avg = total_active_tasks / (num_donors + 1)

# 3. 选择 donor 数量
num_donors_to_select = min(ceil(avg), num_donors)

# 4. 随机选择该数量的 instance 并行执行窃取
```

## 任务选择策略

窃取从队列前端选择任务（FIFO），即最先入队的任务优先被窃取。

**保护机制**: Instance 会保留至少一个活跃任务（RUNNING 或 QUEUED），防止被完全掏空。

## 并行执行

使用 `asyncio.gather` 并行向所有选中的 donor 发起 `/task/fetch` 请求。

## 日志格式

所有日志以 `[WorkStealing]` 前缀标识：

```
[WorkStealing] Started for new instance inst-new (model=model_a)
[WorkStealing] Found 3 potential donors for instance inst-new: [inst-1, inst-2, inst-3]
[WorkStealing] Selecting 2 donors (avg_tasks=1.5, total_pending=6): [inst-1, inst-3]
[WorkStealing] Stolen task task-001 from inst-1 → inst-new
[WorkStealing] Summary for inst-new: fetched=2, resubmitted=2, failed=0
[WorkStealing] Instance inst-new status → ACTIVE
```

## 日志级别

全部使用 `INFO` 级别，确保生产环境可见。
