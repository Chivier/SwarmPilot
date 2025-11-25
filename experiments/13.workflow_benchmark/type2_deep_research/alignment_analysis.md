# Type2 Deep Research 与实验07对齐分析

## 工作流模式

**实验07 Deep Research (A → n*B1 → n*B2 → merge A)**:
- 初始A任务提交研究主题
- A任务生成n个B1任务（详细研究）
- 每个B1任务触发对应的B2任务（分析总结）
- 所有B2任务完成后，提交一个merge A任务（综合报告）
- merge A完成标志工作流结束

## 关键差异对比

### 1. Task ID格式差异

**实验07格式**:
```
A: task-A-{strategy}-workflow-{i:04d}
B1: task-B1-{strategy}-workflow-{i:04d}-{b_index}
B2: task-B2-{strategy}-workflow-{i:04d}-{b_index}
Merge: task-merge-{strategy}-workflow-{i:04d}
```

**Type2当前格式** (需要修复):
```
A: {workflow_id}-A
B1: {workflow_id}-B1-{b1_index}
B2: {workflow_id}-B2-{b2_index}
Merge: {workflow_id}-merge
```

### 2. Metadata结构差异

#### 模拟模式 Metadata

**实验07格式**:
```python
# A任务
metadata = {
    "workflow_id": workflow_id,
    "exp_runtime": sleep_time * 1000.0,  # 必须有
    "task_type": "A"
}

# B1/B2任务
metadata = {
    "workflow_id": workflow_id,
    "exp_runtime": sleep_time * 1000.0,  # 必须有
    "task_type": "B1",  # 或 "B2"
    "b_index": b_index  # B任务索引
}

# Merge任务
metadata = {
    "workflow_id": workflow_id,
    "exp_runtime": sleep_time * 1000.0,  # 必须有
    "task_type": "merge"
}
```

**Type2当前格式** (缺少exp_runtime):
```python
# 缺少 exp_runtime 字段！
metadata = {
    "workflow_id": workflow_id,
    "fanout_count": fanout_count,  # 不应该在模拟模式
    "task_type": "A"
}
```

#### 真实模式 Metadata

**实验07格式**:
```python
# LLM任务
metadata = {
    "sentence": sentence,
    "token_length": token_length,
    "max_tokens": max_tokens
}
```

**Type2格式** (包含额外字段):
```python
metadata = {
    "workflow_id": workflow_id,  # 不应该在真实模式
    "fanout_count": fanout_count,  # 不应该在真实模式
    "task_type": "A",  # 不应该在真实模式
    "sentence": sentence,
    "token_length": token_length,
    "max_tokens": max_tokens
}
```

### 3. Model ID配置

**实验07**:
- A任务: `llm_service_small_model`（真实）或 `sleep_model`（模拟）
- B1/B2任务: `llm_service_small_model`（真实）或 `sleep_model`（模拟）
- Merge任务: `llm_service_small_model`（真实）或 `sleep_model`（模拟）

**Type2**:
- 需要根据模式自动设置正确的model_id

### 4. WorkflowTaskData结构

**实验07包含的字段**:
```python
@dataclass
class WorkflowTaskData:
    task_id: str
    workflow_id: str
    task_type: str  # "A", "B1", "B2", or "merge"
    sleep_time: float
    exp_runtime: float  # Expected runtime in milliseconds
    b_index: Optional[int] = None  # Index for B1/B2 pairing
    is_warmup: bool = False
    # LLM service fields
    sentence: Optional[str] = None
    max_tokens: Optional[int] = None
```

**Type2需要添加**:
- `sleep_time` 字段（模拟模式）
- `exp_runtime` 字段（模拟模式）
- `sentence` 字段（真实模式）
- `max_tokens` 字段（真实模式）
- `strategy` 字段（用于task_id）
- `is_warmup` 字段

### 5. 配置文件差异

**实验07**:
- 使用硬编码的scheduler URL
- 使用环境变量或配置文件设置

**Type2**:
- 需要__post_init__方法自动设置model_id
- 需要添加strategy字段

## 需要修复的问题总结

1. **Task ID格式统一**
   - 所有task ID必须包含strategy前缀
   - 格式: `task-{type}-{strategy}-workflow-{i:04d}`

2. **Metadata清理**
   - 模拟模式: 只包含 `workflow_id`, `exp_runtime`, `task_type` (和B任务的 `b_index`)
   - 真实模式: 只包含 `sentence`, `token_length`, `max_tokens`

3. **WorkflowTaskData增强**
   - 添加所有缺失字段
   - 支持预生成sleep_time

4. **Model ID自动配置**
   - 根据mode自动设置正确的model_id

5. **B1/B2配对逻辑**
   - 确保B1和B2使用相同的b_index
   - B2只在对应B1完成后提交

6. **Merge任务触发**
   - 所有B2任务完成后触发merge任务
   - Merge任务完成标志工作流结束