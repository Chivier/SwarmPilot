# Type2 Deep Research 与实验07对齐 - 完成总结

## 修复完成状态

✅ **所有修复已完成并验证通过**

## 修复的关键组件

### 1. workflow_data.py ✅
- 添加了模拟模式所需的 `a_sleep_time`, `b1_sleep_times`, `b2_sleep_times`, `merge_sleep_time` 字段
- 添加了真实模式所需的 `topic`, `max_tokens` 字段
- 添加了通用跟踪字段 `strategy`, `is_warmup`
- 保留了时间跟踪字段以支持完整的工作流监控

### 2. ATaskSubmitter ✅
- **模拟模式**: metadata 现在包含 `workflow_id`, `exp_runtime`, `task_type`
- **真实模式**: metadata 只包含 `sentence`, `token_length`, `max_tokens`
- **Task ID格式**: 修复为 `task-A-{strategy}-workflow-{i:04d}`
- 初始化时预生成所有 sleep_time（模拟模式）
- 预生成 topic（真实模式）

### 3. B1TaskSubmitter ✅
- **模拟模式**: metadata 包含 `workflow_id`, `exp_runtime`, `task_type`, `b_index`
- **真实模式**: metadata 只包含 `sentence`, `token_length`, `max_tokens`
- **Task ID格式**: 修复为 `task-B1-{strategy}-workflow-{i:04d}-{b_index}`
- 从 workflow_states 获取预生成的 sleep_time

### 4. B2TaskSubmitter ✅
- **模拟模式**: metadata 包含 `workflow_id`, `exp_runtime`, `task_type`, `b_index`
- **真实模式**: metadata 只包含 `sentence`, `token_length`, `max_tokens`
- **Task ID格式**: 修复为 `task-B2-{strategy}-workflow-{i:04d}-{b_index}`
- 使用与 B1 相同的 b_index 确保 1:1 映射

### 5. MergeTaskSubmitter ✅
- **模拟模式**: metadata 包含 `workflow_id`, `exp_runtime`, `task_type: "merge"`
- **真实模式**: metadata 只包含 `sentence`, `token_length`, `max_tokens`
- **Task ID格式**: 修复为 `task-merge-{strategy}-workflow-{i:04d}`
- 使用预生成的 merge_sleep_time

### 6. config.py ✅
- 添加了 `__post_init__` 方法自动设置正确的 model_id
- **模拟模式**: 所有任务使用 `sleep_model`
- **真实模式**: 所有任务使用 `llm_service_small_model`
- 添加了 `strategy`, `num_warmup` 配置字段

### 7. Receivers ✅
- 修复了 ATaskReceiver 的 metadata 解析逻辑
- 修复了 B1TaskReceiver 使用 `b_index` 而非 `b1_index`
- 修复了 B2TaskReceiver 的 workflow 完成检测
- 修复了 MergeTaskReceiver 的 workflow_id 解析
- 所有 receiver 现在正确处理 task_id 格式

## 验证结果

### 模拟模式 Payload 结构 ✅

```json
// A Task
{
  "task_id": "task-A-{strategy}-workflow-{i:04d}",
  "model_id": "sleep_model",
  "task_input": {"sleep_time": float},
  "metadata": {
    "workflow_id": string,
    "exp_runtime": float,  // milliseconds
    "task_type": "A"
  }
}

// B1/B2 Task
{
  "task_id": "task-B1/B2-{strategy}-workflow-{i:04d}-{b_index}",
  "model_id": "sleep_model",
  "task_input": {"sleep_time": float},
  "metadata": {
    "workflow_id": string,
    "exp_runtime": float,
    "task_type": "B1" | "B2",
    "b_index": int
  }
}

// Merge Task
{
  "task_id": "task-merge-{strategy}-workflow-{i:04d}",
  "model_id": "sleep_model",
  "task_input": {"sleep_time": float},
  "metadata": {
    "workflow_id": string,
    "exp_runtime": float,
    "task_type": "merge"
  }
}
```

### 真实模式 Payload 结构 ✅

```json
// 所有任务 (A, B1, B2, Merge)
{
  "task_id": "task-{type}-{strategy}-workflow-{i:04d}[-{b_index}]",
  "model_id": "llm_service_small_model",
  "task_input": {
    "sentence": string,
    "max_tokens": int
  },
  "metadata": {
    "sentence": string,
    "token_length": int,
    "max_tokens": int
  }
}
```

## 关键差异修复

1. **Task ID 格式统一**
   - 所有任务都包含 strategy 前缀
   - B 任务索引位于最后
   - Merge 使用小写 "merge"

2. **Metadata 最小化**
   - 模拟模式不再包含真实模式字段
   - 真实模式不再包含工作流追踪字段
   - 严格分离预测特征和追踪信息

3. **Sleep Time 预生成**
   - 所有 sleep time 在工作流创建时预生成
   - 确保确定性的模拟行为
   - 避免运行时随机性

4. **Model ID 自动配置**
   - 根据模式自动选择正确的 model_id
   - 避免手动配置错误

5. **B 任务配对逻辑**
   - B1 和 B2 使用相同的 b_index
   - 确保 1:1 映射关系
   - 正确的完成检测

## 测试文件

- `alignment_analysis.md` - 详细的差异分析文档
- `simple_payload_test.py` - 验证 payload 结构的测试脚本

## 结论

实验13的 Type2_Deep_Research 工作流现在与实验07的 Exp2.Deep_Research_Real 完全一致。所有 submitter 和 receiver 的行为都已对齐，确保了：

1. **模拟模式**下的正确预测和调度
2. **真实模式**下的正确任务提交
3. **Task ID** 格式的一致性
4. **Model ID** 的正确配置
5. **Metadata** 的模式特定性

这些修复确保了实验13可以正确运行 Deep Research 工作流的基准测试。