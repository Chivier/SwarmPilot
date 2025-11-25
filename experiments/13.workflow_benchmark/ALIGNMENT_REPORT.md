# 实验13 Workflow Benchmark 对齐报告

## 完成状态：✅ 全部对齐完成

---

## Type1 Text2Video 对齐状态

### 参考实验：03.Exp4.Text2Video

### 对齐内容：
1. **Task ID格式** ✅
   - A1: `task-A1-{strategy}-workflow-{i:04d}`
   - A2: `task-A2-{strategy}-workflow-{i:04d}`
   - B: `task-B{iteration}-{strategy}-workflow-{i:04d}`

2. **Model ID配置** ✅
   - 模拟模式：A1/A2使用`sleep_model_a`，B使用`sleep_model_b`
   - 真实模式：A1/A2使用`llm_service_small_model`，B使用`t2vid`

3. **Payload结构** ✅

   **模拟模式：**
   ```json
   {
     "task_id": "task-{type}-{strategy}-workflow-{num}",
     "model_id": "sleep_model_a/b",
     "task_input": {"sleep_time": float},
     "metadata": {
       "workflow_id": string,
       "exp_runtime": float,
       "task_type": string,
       // B任务额外字段：
       "frame_count": int,
       "b_iteration": int,
       "max_b_loops": int
     }
   }
   ```

   **真实模式：**
   - A1/A2: `task_input: {sentence, max_tokens}`, `metadata: {sentence, token_length, max_tokens}`
   - B: `task_input: {prompt, negative_prompt, frames}`, `metadata: {positive_prompt_length, negative_prompt_length, frames}`

### 验证文件：
- `type1_text2video/simple_payload_test.py` - 独立验证脚本
- `type1_text2video/test_payload_consistency.py` - 完整一致性测试
- `type1_text2video/alignment_summary.md` - 详细修复文档

---

## Type2 Deep Research 对齐状态

### 参考实验：07.Exp2.Deep_Research_Real

### 对齐内容：
1. **Task ID格式** ✅
   - A: `task-A-{strategy}-workflow-{i:04d}`
   - B1: `task-B1-{strategy}-workflow-{i:04d}-{b_index}`
   - B2: `task-B2-{strategy}-workflow-{i:04d}-{b_index}`
   - Merge: `task-merge-{strategy}-workflow-{i:04d}`

2. **Model ID配置** ✅
   - 模拟模式：所有任务使用`sleep_model`
   - 真实模式：所有任务使用`llm_service_small_model`

3. **Payload结构** ✅

   **模拟模式：**
   ```json
   {
     "task_id": "task-{type}-{strategy}-workflow-{num}",
     "model_id": "sleep_model",
     "task_input": {"sleep_time": float},
     "metadata": {
       "workflow_id": string,
       "exp_runtime": float,
       "task_type": string,
       // B1/B2任务额外字段：
       "b_index": int
     }
   }
   ```

   **真实模式：**
   - 所有任务：`task_input: {sentence, max_tokens}`, `metadata: {sentence, token_length, max_tokens}`

### 验证文件：
- `type2_deep_research/simple_payload_test.py` - 独立验证脚本
- `type2_deep_research/test_alignment_verification.py` - 完整验证脚本
- `type2_deep_research/alignment_summary.md` - 详细修复文档

---

## 关键改进总结

### 1. 架构差异处理
- **订阅模式**：实验07使用task_ids订阅，实验13使用model_id订阅
- **解决方案**：确保model_id在submitter和receiver之间严格匹配

### 2. Metadata严格分离
- **模拟模式**：只包含workflow追踪和预测所需的最小字段
- **真实模式**：只包含LLM/模型预测特征，不包含workflow追踪

### 3. Sleep Time预生成
- 所有sleep time在workflow创建时预生成
- 确保分布式系统中的确定性行为
- 避免运行时随机性导致的不一致

### 4. 自动配置
- 通过`__post_init__`方法自动设置model_id
- 根据mode自动选择正确的模型配置
- 减少手动配置错误

---

## 测试验证

### 执行的验证：
1. ✅ Task ID格式验证
2. ✅ Model ID一致性验证
3. ✅ Metadata结构验证
4. ✅ Sleep time预生成验证
5. ✅ Submitter和Receiver匹配验证

### 验证结果：
- Type1 Text2Video：**完全对齐** ✅
- Type2 Deep Research：**完全对齐** ✅

---

## 使用说明

### 运行模拟模式实验：
```bash
# Type1 Text2Video
cd experiments/13.workflow_benchmark/type1_text2video
python -m simulation.test_workflow_sim

# Type2 Deep Research
cd experiments/13.workflow_benchmark/type2_deep_research
python -m simulation.test_workflow_sim
```

### 运行真实模式实验：
```bash
# Type1 Text2Video
cd experiments/13.workflow_benchmark/type1_text2video
python -m real.test_workflow_real

# Type2 Deep Research
cd experiments/13.workflow_benchmark/type2_deep_research
python -m real.test_workflow_real
```

### 验证对齐：
```bash
# Type1验证
cd type1_text2video
python3 simple_payload_test.py

# Type2验证
cd type2_deep_research
uv run python test_alignment_verification.py
```

---

## 结论

实验13的两个workflow benchmark（Type1_Text2Video和Type2_Deep_Research）已经与参考实验（03和07）完全对齐。所有关键组件包括：

- ✅ Task ID格式
- ✅ Model ID配置
- ✅ Payload结构
- ✅ Metadata分离
- ✅ Submitter/Receiver匹配

现在可以进行准确的基准测试和性能对比实验。

---

**最后更新时间：** 2024-11-23
**验证状态：** 已通过所有测试