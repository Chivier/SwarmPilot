# 持续请求模式实现状态

## ✅ 已完成

### 1. 核心基础设施 (common.py)
- ✅ `WorkflowState` 添加 `is_target_for_stats` 字段
- ✅ `WorkflowCompletionEvent` 添加 `is_target_for_stats` 字段
- ✅ `force_clear_scheduler_tasks()` 函数：强制清理 scheduler
- ✅ `calculate_makespan()` 函数：计算 makespan 指标
- ✅ `print_continuous_mode_summary()` 函数：打印持续模式统计摘要

### 2. 实验 04 完整实现 ✅
**文件**: `/chivier-disk/yanweiye/Projects/swarmpilot-refresh/experiments/04.multi_model_workflow_dynamic/test_dynamic_workflow.py`

#### 修改点：
1. **函数签名** (line 1455):
   - 添加 `continuous_mode: bool = False` 参数
   - 添加 `target_workflows: Optional[int] = None` 参数

2. **Workflow States 初始化** (line 1610-1644):
   ```python
   # 标记前 target_workflows 个非 warmup workflow 为目标
   if continuous_mode and target_workflows is not None:
       # 前 target_workflows 个: is_target_for_stats=True
       # 其余: is_target_for_stats=False (overflow)
   ```

3. **WorkflowCompletionEvent 创建** (line 1003):
   - 添加 `is_target_for_stats=workflow.is_target_for_stats`

4. **Scheduler 强制清理** (line 1721-1732):
   ```python
   if continuous_mode:
       time.sleep(5.0)  # 等待 5 秒
       force_clear_scheduler_tasks(SCHEDULER_A_URL)
       force_clear_scheduler_tasks(SCHEDULER_B_URL)
   ```

5. **统计输出** (line 1747-1752):
   ```python
   if continuous_mode:
       makespan_metrics = calculate_makespan(monitor.completed_workflows)
       print_continuous_mode_summary(strategy, makespan_metrics, a_metrics, b_metrics, wf_metrics)
   else:
       print_metrics_summary(strategy, a_metrics, b_metrics, wf_metrics)
   ```

6. **Main 函数** (line 1777):
   - 添加 `continuous_mode: bool = False` 参数
   - 如果 continuous_mode=True, 设置 `NUM_WORKFLOWS = 2 * num_workflows`
   - 传递 `continuous_mode` 和 `target_workflows` 给 `test_strategy_workflow()`

### 3. 封装器更新 ✅
**文件**: `continuous_wrapper.py`

- ✅ 简化为直接调用 `experiment_module.main(continuous_mode=True)`
- ✅ 实验内部处理所有持续模式逻辑
- ✅ 不再在 wrapper 中手动倍增任务数

### 4. 统一入口 ✅
**文件**: `unified_workflow.py`

- ✅ 添加 `--continuous` 命令行参数
- ✅ 集成 continuous_wrapper
- ✅ 传递 `continuous` 参数到所有实验

### 5. 文档 ✅
**文件**: `README.md`

- ✅ 完整的持续模式使用说明
- ✅ 输出示例
- ✅ 与标准模式的对比表

---

## ⏳ 待完成

### 实验 05 持续模式支持
**文件**: `/chivier-disk/yanweiye/Projects/swarmpilot-refresh/experiments/05.multi_model_workflow_dynamic_parallel/test_dynamic_workflow.py`

需要进行与实验 04 相同的修改（参考上述实验 04 的 6 个修改点）。

### 实验 06 持续模式支持
**文件**: `/chivier-disk/yanweiye/Projects/swarmpilot-refresh/experiments/06.multi_model_workflow_dynamic_merge/test_dynamic_workflow_merge.py`

需要进行与实验 04 类似的修改，注意：
- 有 merge task，需要在 merge 完成时也传递 `is_target_for_stats`
- WorkflowStateMerge 已有 `is_target_for_stats` 字段（继承自 WorkflowState）

### 实验 07 持续模式支持
**文件**: `/chivier-disk/yanweiye/Projects/swarmpilot-refresh/experiments/07.multi_model_workflow_dynamic_merge_2/test_dynamic_workflow_merge_2.py`

需要进行与实验 04 类似的修改，注意：
- 有 B1/B2 split 和 merge task
- WorkflowStateB1B2 已有 `is_target_for_stats` 字段（继承自 WorkflowStateMerge）

---

## 📋 修改模板（适用于实验 05/06/07）

### 步骤 1: 修改 test_strategy_workflow() 函数签名
```python
def test_strategy_workflow(
    ...
    timeout_minutes: int = 10,
    continuous_mode: bool = False,
    target_workflows: Optional[int] = None
) -> Dict:
```

### 步骤 2: 修改 Workflow States 初始化
在 "Step 5: Initialize workflow states" 部分，添加：
```python
# In continuous mode, determine which workflows are targets for statistics
if continuous_mode and target_workflows is not None:
    logger.info(f"Continuous mode: marking first {target_workflows} non-warmup workflows as targets")

target_count = 0  # Count of non-warmup workflows marked as targets
for i in range(total_workflows):
    workflow_id = f"wf-{strategy}-{i:04d}"
    is_warmup = i < num_warmup_workflows

    # Determine if this is a target workflow for statistics
    is_target_for_stats = True  # Default: all workflows are targets
    if continuous_mode and target_workflows is not None:
        if is_warmup:
            is_target_for_stats = False  # Warmup workflows are never targets
        else:
            # Mark first target_workflows non-warmup workflows as targets
            if target_count < target_workflows:
                is_target_for_stats = True
                target_count += 1
            else:
                is_target_for_stats = False  # Overflow workflows

    workflow_states[workflow_id] = WorkflowState(  # 或 WorkflowStateMerge 或 WorkflowStateB1B2
        ...
        is_target_for_stats=is_target_for_stats
    )
```

### 步骤 3: 修改 WorkflowCompletionEvent 创建
在 `_push_workflow_completion()` 方法中：
```python
event = WorkflowCompletionEvent(
    ...
    is_target_for_stats=workflow.is_target_for_stats
)
```

### 步骤 4: 添加 Scheduler 强制清理
在 "Step 13: Stop all threads" 之后：
```python
# Step 13.5: Continuous mode cleanup
if continuous_mode:
    logger.info("Step 13.5: Continuous mode cleanup")
    logger.info("Waiting 5 seconds before force-clearing schedulers...")
    time.sleep(5.0)

    from common import force_clear_scheduler_tasks
    logger.info("Force-clearing Scheduler A...")
    force_clear_scheduler_tasks(SCHEDULER_A_URL)
    logger.info("Force-clearing Scheduler B...")
    force_clear_scheduler_tasks(SCHEDULER_B_URL)
    logger.info("Schedulers cleared successfully")
```

### 步骤 5: 修改统计输出
在 "Step 14: Collect results" 部分：
```python
# Print summary
if continuous_mode:
    from common import calculate_makespan, print_continuous_mode_summary
    makespan_metrics = calculate_makespan(monitor.completed_workflows)
    print_continuous_mode_summary(strategy, makespan_metrics, a_metrics, b_metrics, wf_metrics)
else:
    print_metrics_summary(strategy, a_metrics, b_metrics, wf_metrics)
```

### 步骤 6: 修改 main() 函数
```python
def main(num_workflows: int = 100, qps_a: float = 8.0, gqps: Optional[float] = None,
         warmup_ratio: float = 0.0, seed: int = 42, strategies: List[str] = None,
         continuous_mode: bool = False):
    ...

    # Experiment parameters
    if continuous_mode:
        NUM_WORKFLOWS = 2 * num_workflows  # Generate 2x workflows in continuous mode
        TARGET_WORKFLOWS = num_workflows  # Track first num_workflows
        logger.info(f"CONTINUOUS MODE: Generating {NUM_WORKFLOWS} workflows, tracking first {TARGET_WORKFLOWS}")
    else:
        NUM_WORKFLOWS = num_workflows
        TARGET_WORKFLOWS = None

    ...

    # In test call:
    results = test_strategy_workflow(
        ...
        continuous_mode=continuous_mode,
        target_workflows=TARGET_WORKFLOWS
    )
```

---

## 🧪 测试方法

### 实验 04 测试（已完成实现）
```bash
# 测试持续模式
python unified_workflow.py --experiment 04-ocr --continuous --num-workflows 10 --strategies min_time

# 预期输出：
# - 生成 20 个 workflows
# - 标记前 10 个为 target
# - 输出包含 makespan
# - 分 A/B 模型统计
# - 5 秒后强制清理 scheduler
```

### 验证要点
1. ✅ 日志显示 "CONTINUOUS MODE: Generating 20 workflows, tracking first 10"
2. ✅ 输出使用 `print_continuous_mode_summary` 格式
3. ✅ Makespan 指标正确计算
4. ✅ 显示 "Target workflows: 10, Overflow workflows: 10"
5. ✅ 日志显示 "Force-clearing Scheduler A/B"

---

## 📊 输出示例

```
================================================================================
Continuous Request Mode Results: min_time
================================================================================

Makespan:
  Total time (first target → last target):  125.43s
  First target workflow submitted at:        14:23:15.123
  Last target workflow completed at:         14:25:20.553

Workflow Counts:
  Total workflows submitted:     20
  Warmup workflows:              0
  Target workflows (tracked):    10
  Overflow workflows (extra):    10

A Model Tasks (Scheduler A):
  Completed:  20
  Failed:     0
  Avg time:   2.45s
  P95:        9.12s
  P99:        9.87s

B Model Tasks (Scheduler B):
  Completed:  110
  Failed:     0
  Avg time:   5.18s
  P95:        9.45s
  P99:        9.92s

Target Workflows (First 10 non-warmup):
  Avg fanout: 5.5 B tasks per A task
  Avg time:   11.23s
  Median:     10.87s
  P95:        18.45s
  P99:        19.12s

Overflow Workflows: 10 workflows submitted but not tracked in statistics
================================================================================
```

---

## 🎯 总结

**实验 04** 的持续模式已经完整实现，包括：
- ✅ 任务标记
- ✅ Makespan 计算
- ✅ 分模型统计
- ✅ 强制清理 scheduler
- ✅ 专用统计输出

**实验 05/06/07** 需要应用相同的修改模板。所有基础设施已就绪，只需要按照上述 6 个步骤修改各实验文件即可。
